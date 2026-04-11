"""
Agent — hybrid query pipeline for the NCCS Query Assistant.

Fixed execution order:
  1. get_schema_context  — hardcoded, always runs first
  2. get_sql_template    — hardcoded, always runs first
  3. get_cancer_info     — LLM-decided (offered as a tool; LLM calls it if relevant)
  4. LLM: generate SQL   — LLM generates SQL from all accumulated context
  5. validate_sql_query  — hardcoded loop until SQL passes (max 3 tries)
  6. get_data            — hardcoded, always runs after validation passes
  7. LLM: summarise results
  8. Evaluation logging of final query and metrics
"""

import json
import re
import time
from typing import Any, Dict, Generator, List, Optional

from ContextConfiguration import (
    CANCER_INFO_TOOL,
    CONTEXT_SYSTEM,
    SQL_GEN_PROMPT,
    SUMMARY_SYSTEM,
    SUMMARY_USER_TEMPLATE,
    SQL_VALIDATION_FIX_USER_TEMPLATE,
)
from retrieval.llm import ollama_chat, OLLAMA_MODEL
from retrieval.graph.outputParser import parse_data_json, extract_final_text, extract_data_json
from retrieval.graph.tool.vectorRag import get_cancer_info, get_schema_context, get_sql_template
from retrieval.graph.tool.SQLvalidator import validate_sql_query
from retrieval.graph.tool.get_data import get_data
from retrieval.graph.tool.evaluation_update import evaluate_live_query


# ---------------------------------------------------------------------------
# SQL extraction from raw LLM text
# ---------------------------------------------------------------------------

def _extract_sql(text: str) -> str:
    """Pull a SELECT statement out of an LLM response."""
    # Strip ```sql ... ``` fences
    m = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fall back to first SELECT … found in the text
    m = re.search(r"(SELECT\b.+)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Validation success check
# ---------------------------------------------------------------------------

def _sql_passed(validation_result: str) -> bool:
    """True when the validator says the SQL is safe to execute."""
    lower = validation_result.lower()
    return "safety issues" not in lower and (
        "sql is valid" in lower
        or "sql passed safety checks" in lower
        or "proceed to get_data" in lower
    )


# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------

def _build_response(messages: list) -> Dict[str, Any]:
    raw_data = extract_data_json(messages)
    final_text = extract_final_text(messages) or ""

    if raw_data is None:
        return {
            "status": "error",
            "message": "Could not retrieve data for your question.",
            "reasons": ["get_data was not reached"],
        }

    if raw_data.startswith("EXECUTION_ERROR"):
        return {
            "status": "error",
            "message": "Query execution failed.",
            "reasons": [raw_data],
        }

    if raw_data.startswith("Query executed successfully but returned no rows"):
        return {
            "status": "ok",
            "message": final_text or "No results found.",
            "columns": [],
            "rows": [],
            "row_count": 0,
        }

    data = parse_data_json(messages)
    if not data:
        return {
            "status": "ok",
            "message": final_text,
            "metric": "Answer",
            "value": final_text,
        }

    columns = list(data[0].keys())
    rows = [[row.get(col) for col in columns] for row in data]

    if len(columns) == 1 and len(rows) == 1:
        return {
            "status": "ok",
            "message": final_text,
            "metric": columns[0],
            "value": rows[0][0],
            "columns": columns,
            "rows": rows,
            "row_count": 1,
        }

    return {
        "status": "ok",
        "message": final_text,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
    }


# ---------------------------------------------------------------------------
# SQL extractor for evaluation logging
# ---------------------------------------------------------------------------

def get_latest_sql(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and re.search(r"\bSELECT\b", content, re.IGNORECASE):
                return _extract_sql(content)
    return "No SQL found"


# ---------------------------------------------------------------------------
# Main streaming generator
# ---------------------------------------------------------------------------

def stream_question_agent(
    question: str,
    model: Optional[str] = None,
    history: Optional[List[Dict]] = None,
) -> Generator[str, None, None]:
    """
    Deterministic agent flow that yields SSE events.

    Event types:
      {"type": "step_call",   "tool": <name>}
      {"type": "step_result", "tool": <name>, "snippet": <str>}
      {"type": "done",        ...response payload...}
      {"type": "error",       "message": ..., "reasons": [...]}
    """
    def sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    def snip(text: str, n: int = 200) -> str:
        return (text[:n] + "…") if len(text) > n else text

    start_time = time.perf_counter()
    all_messages: List[dict] = []   # used by _build_response
    sql = ""
    validation_tries = 0
    input_tokens = 0
    output_tokens = 0

    # ------------------------------------------------------------------
    # STEP 1 — Schema context + SQL templates (hardcoded, always)
    # ------------------------------------------------------------------
    try:
        yield sse({"type": "step_call", "tool": "get_schema_context"})
        schema_data = get_schema_context(question)
        yield sse({"type": "step_result", "tool": "get_schema_context", "snippet": snip(schema_data)})

        yield sse({"type": "step_call", "tool": "get_sql_template"})
        sql_template_data = get_sql_template(question)
        yield sse({"type": "step_result", "tool": "get_sql_template", "snippet": snip(sql_template_data)})
    except Exception as e:
        yield sse({"type": "error", "status": "error", "message": "Failed to load schema context.", "reasons": [str(e)]})
        return

    # ------------------------------------------------------------------
    # STEP 2 — LLM decides whether to call get_cancer_info
    # ------------------------------------------------------------------
    system_content = (
        CONTEXT_SYSTEM
        + f"\n\nDatabase schema:\n{schema_data}"
        + f"\n\nSQL examples (few-shot):\n{sql_template_data}"
    )
    gen_messages = [{"role": "system", "content": system_content}]

    # Inject recent conversation history
    if history:
        for msg in history[:-1][-6:]:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user" and isinstance(content, str):
                gen_messages.append({"role": "user", "content": content})
            elif role == "assistant" and isinstance(content, dict):
                prev_sql = content.get("final_sql")
                if prev_sql:
                    gen_messages.append({"role": "assistant", "content": f"Previous SQL used:\n{prev_sql}"})

    gen_messages.append({"role": "user", "content": question})

    # Mini tool-calling loop — only get_cancer_info is offered
    try:
        for _ in range(3):
            resp = ollama_chat(gen_messages, tools=CANCER_INFO_TOOL, model=model)
            assistant_msg = resp["message"]
            if hasattr(assistant_msg, "__dict__"):
                assistant_msg = dict(assistant_msg)
            input_tokens += resp.get("prompt_eval_count", 0) or 0
            output_tokens += resp.get("eval_count", 0) or 0
            gen_messages.append(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls") or []
            if not tool_calls:
                break  # LLM decided no cancer info is needed

            for tc in tool_calls:
                fn_obj = tc.get("function", {})
                if hasattr(fn_obj, "__dict__"):
                    fn_obj = dict(fn_obj)
                if fn_obj.get("name") == "get_cancer_info":
                    args = fn_obj.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    yield sse({"type": "step_call", "tool": "get_cancer_info"})
                    cancer_result = get_cancer_info(**args)
                    yield sse({"type": "step_result", "tool": "get_cancer_info", "snippet": snip(cancer_result)})
                    gen_messages.append({
                        "role": "tool",
                        "name": "get_cancer_info",
                        "content": cancer_result,
                    })
    except Exception as e:
        yield sse({"type": "error", "status": "error", "message": "Context gathering failed.", "reasons": [str(e)]})
        return

    # ------------------------------------------------------------------
    # STEP 3 — Generate SQL (LLM call, no tools)
    # ------------------------------------------------------------------
    try:
        gen_messages.append({"role": "user", "content": SQL_GEN_PROMPT})
        resp = ollama_chat(gen_messages, model = model)   # no tools — pure SQL generation
        raw = resp["message"].get("content") or ""
        input_tokens += resp.get("prompt_eval_count", 0) or 0
        output_tokens += resp.get("eval_count", 0) or 0
        sql = _extract_sql(raw)
        gen_messages.append({"role": "assistant", "content": raw})
        all_messages.append({"role": "assistant", "content": raw})
    except Exception as e:
        yield sse({"type": "error", "status": "error", "message": "SQL generation failed.", "reasons": [str(e)]})
        return

    # ------------------------------------------------------------------
    # STEP 5 — Validate loop (LLM fixes until SQL passes, safety cap at 10)
    # ------------------------------------------------------------------
    last_validation = ""
    validated = False
    while validation_tries < 10:
        try:
            yield sse({"type": "step_call", "tool": "validate_sql_query"})
            last_validation = validate_sql_query(sql)
            validation_tries += 1
            yield sse({"type": "step_result", "tool": "validate_sql_query", "snippet": snip(last_validation)})

            if _sql_passed(last_validation):
                validated = True
                break

            # Validation failed — ask LLM to fix and loop again
            gen_messages.append({
                "role": "user",
                "content": SQL_VALIDATION_FIX_USER_TEMPLATE.format(
                    validation_text=last_validation,
                    sql=sql,
                ),
            })
            resp = ollama_chat(gen_messages, model=model)
            raw = resp["message"].get("content") or ""
            input_tokens += resp.get("prompt_eval_count", 0) or 0
            output_tokens += resp.get("eval_count", 0) or 0
            sql = _extract_sql(raw)
            gen_messages.append({"role": "assistant", "content": raw})
            all_messages.append({"role": "assistant", "content": raw})

        except Exception as e:
            yield sse({"type": "error", "status": "error", "message": "SQL validation failed.", "reasons": [str(e)]})
            return

    if not validated:
        final_error = {
            "type": "done",
            "status": "error",
            "message": f"Could not generate a valid SQL query after {validation_tries} attempts.",
            "reasons": [last_validation],
            "final_sql": sql,
            "validation_tries": validation_tries,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": 0.0,
            "prompt_cost": 0.0,
            "completion_cost": 0.0,
        }

        try:
            eval_row = evaluate_live_query(
                prompt=question,
                model=model or OLLAMA_MODEL,
                generated_sql=sql,
                latency=time.perf_counter() - start_time,
                metrics=final_error,
            )
            final_error.update({
                "complexity_score": eval_row.get("Complexity Score"),
                "complexity_level": eval_row.get("Complexity Level"),
                "semantic_score": eval_row.get("Semantic Score"),
                "generated_explanation": eval_row.get("Generated Explanation"),
            })
        except Exception as log_err:
            print(f"[eval] Failed to log invalid query: {log_err}")

        yield sse(final_error)
        return

    # ------------------------------------------------------------------
    # STEP 6 — Execute query
    # ------------------------------------------------------------------
    try:
        yield sse({"type": "step_call", "tool": "get_data"})
        data_result = get_data(sql)
        yield sse({"type": "step_result", "tool": "get_data", "snippet": snip(data_result)})
        all_messages.append({"role": "tool", "name": "get_data", "content": data_result})
    except Exception as e:
        yield sse({"type": "error", "status": "error", "message": "Query execution failed.", "reasons": [str(e)]})
        return

    # ------------------------------------------------------------------
    # STEP 7 — Summarise results (LLM call)
    # ------------------------------------------------------------------
    try:
        summary_resp = ollama_chat([
            {"role": "system", "content": SUMMARY_SYSTEM},
            {
                "role": "user",
                "content": SUMMARY_USER_TEMPLATE.format(
                    question=question,
                    data=data_result[:3000],
                ),
            },
        ], model = model)
        final_text = summary_resp["message"].get("content") or ""
        input_tokens += summary_resp.get("prompt_eval_count", 0) or 0
        output_tokens += summary_resp.get("eval_count", 0) or 0
        all_messages.append({"role": "assistant", "content": final_text})
    except Exception:
        final_text = ""

    # ------------------------------------------------------------------
    # Emit final SSE event
    # ------------------------------------------------------------------
    latency = time.perf_counter() - start_time
    final = _build_response(all_messages)
    final.update({
        "type": "done",
        "final_sql": sql,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost": 0.0,
        "prompt_cost": 0.0,
        "completion_cost": 0.0,
        "validation_tries": validation_tries,
    })

    try:
        eval_row = evaluate_live_query(
            prompt=question,
            model=model or OLLAMA_MODEL,
            generated_sql=sql,
            latency=latency,
            metrics=final,
        )

        final.update({
            "complexity_score": eval_row.get("Complexity Score"),
            "complexity_level": eval_row.get("Complexity Level"),
            "semantic_score": eval_row.get("Semantic Score"),
            "generated_explanation": eval_row.get("Generated Explanation"),
        })
    except Exception as log_err:
        print(f"[eval] Failed to log query: {log_err}")

    yield sse(final)
