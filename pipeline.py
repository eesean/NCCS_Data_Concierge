import json
import re
import time
from typing import Any, Dict, Generator, List, Optional

from retrieval.llm import ollama_chat, OLLAMA_MODEL
from retrieval.graph.node.reasoner import TOOLS, TOOL_FN_MAP
from retrieval.graph.outputParser import parse_data_json, extract_final_text, extract_data_json
from retrieval.graph.tool.vectorRag import get_schema_context, get_sql_template
from retrieval.graph.tool.evaluation_update import evaluate_live_query


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYS_PROMPT = """/no_think
You are an expert data analyst assistant for the Singapore National Cancer Centre dataset (NCCS).

Your workflow:
1. If the user's question involves a specific cancer type, ALWAYS call get_cancer_info first to retrieve the correct ICD-10 codes and SQL_FILTER.
2. Call get_schema_context to understand the table/column structure if needed.
3. Call get_sql_template to find similar example queries if helpful.
4. Write the DuckDB SQL query. Use ONLY column names and tables from the schema context.
   - Copy the SQL_FILTER from get_cancer_info verbatim — do NOT invent codes such as ICDO3.
   - Use ICD10 column predicates exactly as returned by get_cancer_info.
5. Call validate_sql_query to verify the SQL before executing.
6. Call get_data to execute the validated SQL and retrieve results.
7. Summarise the results clearly for the user.

Rules:
- Never use ICDO3 or any column not present in the schema.
- Use the tools provided — do not guess or hallucinate column names or table names.
- If validate_sql_query returns errors, fix the SQL and validate again before calling get_data.
"""


# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------

def _build_response(messages: list) -> Dict[str, Any]:
    """Convert the agent message history into the response dict the frontend expects."""
    raw_data = extract_data_json(messages)
    final_text = extract_final_text(messages) or ""

    if raw_data is None:
        return {
            "status": "error",
            "message": "Could not retrieve data for your question.",
            "reasons": ["get_data was not reached by the agent"],
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

    is_scalar = len(columns) == 1 and len(rows) == 1
    if is_scalar:
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
# SQL extractor (for evaluation logging)
# ---------------------------------------------------------------------------

def get_latest_sql(all_messages: list) -> str:
    """Extract the most recent SQL query from the message history."""
    for msg in reversed(all_messages):
        if isinstance(msg, dict):
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                fn = tc.get("function", {})
                if fn.get("name") in ("get_data", "validate_sql_query"):
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    sql = args.get("sql", "")
                    if sql:
                        return sql

    sql_pattern = re.compile(r"SQL:\s*(.*?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
    for msg in reversed(all_messages):
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, str):
                match = sql_pattern.search(content)
                if match:
                    return match.group(1).strip()

    return "No SQL query found in history."


# ---------------------------------------------------------------------------
# Helper: normalise Ollama message objects → plain dicts
# ---------------------------------------------------------------------------

def _to_dict(msg: Any) -> dict:
    """Convert an Ollama Message object (or plain dict) to a plain dict."""
    if isinstance(msg, dict):
        return msg
    if hasattr(msg, "__dict__"):
        d = dict(msg.__dict__)
    elif hasattr(msg, "model_dump"):
        d = msg.model_dump()
    else:
        d = {"role": str(getattr(msg, "role", "assistant")), "content": str(msg)}

    # Normalise tool_calls entries
    if d.get("tool_calls"):
        converted = []
        for tc in d["tool_calls"]:
            if not isinstance(tc, dict):
                tc = dict(tc.__dict__) if hasattr(tc, "__dict__") else {}
            if "function" in tc and not isinstance(tc["function"], dict):
                tc = dict(tc)
                tc["function"] = dict(tc["function"].__dict__) if hasattr(tc["function"], "__dict__") else tc["function"]
            converted.append(tc)
        d["tool_calls"] = converted

    return d


# ---------------------------------------------------------------------------
# Main streaming generator
# ---------------------------------------------------------------------------

def stream_question_agent(
    question: str,
    model: Optional[str] = None,
    history: Optional[List[Dict]] = None,
) -> Generator[str, None, None]:
    """
    Generator that runs the Ollama native tool-calling agent and yields SSE events.

    Event types emitted:
      {"type": "step_call",   "tool": <name>}
      {"type": "step_result", "tool": <name>, "snippet": <str>}
      {"type": "done",        ...response payload...}
      {"type": "error",       "message": ..., "reasons": [...]}
    """
    def sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Build initial message list
    # ------------------------------------------------------------------
    try:
        schema_data = get_schema_context(question)
        sql_template_data = get_sql_template(question)

        system_content = (
            _SYS_PROMPT
            + f"\n\nDatabase schema context:\n{schema_data}\n\n"
            + f"Relevant SQL examples for similar questions:\n{sql_template_data}"
        )

        messages: List[dict] = [{"role": "system", "content": system_content}]

        # Inject recent conversation history (last 3 exchanges)
        if history:
            for msg in history[:-1][-6:]:
                role = msg.get("role")
                content = msg.get("content")
                if role == "user" and isinstance(content, str):
                    messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    if isinstance(content, dict):
                        sql = content.get("final_sql")
                        if sql:
                            messages.append({
                                "role": "assistant",
                                "content": f"Previous SQL query used:\n{sql}",
                            })

        messages.append({"role": "user", "content": question})

    except Exception as e:
        yield sse({"type": "error", "status": "error", "message": "Agent setup failed.", "reasons": [str(e)]})
        return

    # Emit schema retrieval as the first visible step
    yield sse({"type": "step_call", "tool": "get_schema_context"})
    snippet = schema_data[:200] + "…" if len(schema_data) > 200 else schema_data
    yield sse({"type": "step_result", "tool": "get_schema_context", "snippet": snippet})

    # ------------------------------------------------------------------
    # Tool-calling loop — drive it here so we can yield SSE per step
    # ------------------------------------------------------------------
    all_messages: List[dict] = list(messages)
    validation_tries = 0
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    try:
        for _ in range(10):  # max iterations
            response = ollama_chat(all_messages, tools=TOOLS)
            assistant_msg = _to_dict(response["message"])

            # Token tracking from Ollama response
            if "prompt_eval_count" in response:
                input_tokens += response.get("prompt_eval_count", 0)
            if "eval_count" in response:
                output_tokens += response.get("eval_count", 0)

            all_messages.append(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls") or []
            if not tool_calls:
                break  # final answer reached

            for tc in tool_calls:
                fn_obj = tc.get("function", {})
                fn_name = fn_obj.get("name", "")
                args = fn_obj.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                yield sse({"type": "step_call", "tool": fn_name})

                fn = TOOL_FN_MAP.get(fn_name)
                if fn is None:
                    result = f"ERROR: unknown tool '{fn_name}'"
                else:
                    try:
                        result = fn(**args)
                    except Exception as exc:
                        result = f"ERROR: {exc}"

                if fn_name == "validate_sql_query":
                    validation_tries += 1

                content = str(result)
                snippet = content[:200] + "…" if len(content) > 200 else content
                yield sse({"type": "step_result", "tool": fn_name, "snippet": snippet})

                all_messages.append({
                    "role": "tool",
                    "name": fn_name,
                    "content": content,
                })

    except Exception as e:
        yield sse({"type": "error", "status": "error", "message": "Agent execution failed.", "reasons": [str(e)]})
        return

    total_tokens = input_tokens + output_tokens
    latency = time.perf_counter() - start_time

    final_sql = get_latest_sql(all_messages)
    final = _build_response(all_messages)

    effective_model = model or OLLAMA_MODEL

    final["type"] = "done"
    final["final_sql"] = final_sql
    final["input_tokens"] = input_tokens
    final["output_tokens"] = output_tokens
    final["total_tokens"] = total_tokens
    final["cost"] = 0.0               # local inference — no API cost
    final["prompt_cost"] = 0.0
    final["completion_cost"] = 0.0
    final["validation_tries"] = validation_tries

    try:
        evaluate_live_query(
            prompt=question,
            model=effective_model,
            generated_sql=final_sql,
            latency=latency,
            metrics=final,
        )
    except Exception as log_err:
        print(f"Failed to log live query: {log_err}")

    yield sse(final)
