import json
import os
import re
import time
from typing import Any, Dict, Generator, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from retrieval.graph.node.workflow import build_graph
from retrieval.graph.outputParser import parse_data_json, extract_final_text, extract_data_json
from retrieval.graph.tool.vectorRag import get_schema_context, get_sql_template
from retrieval.graph.tool.evaluation_update import evaluate_live_query
from retrieval.llm import OLLAMA_DEFAULT_MODEL


def _resolve_model(requested: Optional[str]) -> str:
    """
    Return the Ollama model tag to use for inference.

    If the caller supplies a model that looks like an Ollama tag (no '/' in the
    name, e.g. 'qwen2.5:7b') it is used directly. Otherwise fall back to the
    server's OLLAMA_DEFAULT_MODEL env setting.
    """
    if requested and "/" not in requested:
        return requested
    return OLLAMA_DEFAULT_MODEL


def _build_response(messages: list) -> Dict[str, Any]:
    """
    Convert a list of agent messages into the response dict the frontend expects.
    Does not include 'steps' — callers add that separately.
    """
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


def _get_schema_and_messages(question: str) -> list:
    """Fetch schema + SQL template and wrap as initial LangGraph messages."""
    schema_data = get_schema_context(question)
    sql_template_data = get_sql_template(question)

    system_content = (
        f"You have access to the following database schema:\n{schema_data}\n\n"
        f"Relevant SQL examples (few-shot) for similar questions:\n{sql_template_data}"
    )
    return [SystemMessage(content=system_content), HumanMessage(content=question)]


def stream_question_agent(
    question: str,
    model: Optional[str] = None,
    history: Optional[List[Dict]] = None,
) -> Generator[str, None, None]:
    """
    Generator that runs the local-Ollama ReAct agent and yields SSE events.

    Args:
        question: Natural-language query from the user.
        model:    Ollama model tag (e.g. 'qwen2.5:7b'). Falls back to
                  OLLAMA_DEFAULT_MODEL when omitted or when an OpenRouter slug
                  is accidentally passed.
        history:  Prior conversation turns for multi-turn context.

    Event types emitted:
      {"type": "step_call",   "tool": <name>}
      {"type": "step_result", "tool": <name>, "snippet": <str>}
      {"type": "done",        ...response payload...}
      {"type": "error",       "message": ..., "reasons": [...]}
    """
    def sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    start_time = time.perf_counter()

    try:
        graph_model = _resolve_model(model)
        schema_messages = _get_schema_and_messages(question)

        if history:
            cleaned_history = []
            for msg in history[:-1][-6:]:
                role = msg.get("role")
                content = msg.get("content")
                if role == "user" and isinstance(content, str):
                    cleaned_history.append(HumanMessage(content=content))
                elif role == "assistant" and isinstance(content, dict):
                    sql = content.get("final_sql")
                    if sql:
                        cleaned_history.append(
                            SystemMessage(content=f"Previous SQL query:\n{sql}")
                        )
            schema_messages = (
                [schema_messages[0]] + cleaned_history + [schema_messages[-1]]
            )

        graph = build_graph(graph_model)
        initial_state = {"query": question, "final_answer": "", "messages": schema_messages}

    except Exception as e:
        yield sse({"type": "error", "status": "error", "message": "Agent setup failed.", "reasons": [str(e)]})
        return

    all_messages: List[Any] = list(schema_messages)
    validation_tries = 0

    try:
        yield sse({"type": "step_call", "tool": "get_schema_context"})
        schema_content = str(schema_messages[0].content)
        snippet = schema_content[:200] + "…" if len(schema_content) > 200 else schema_content
        yield sse({"type": "step_result", "tool": "get_schema_context", "snippet": snippet})

        for chunk in graph.stream(
            initial_state,
            stream_mode="updates",
            config={"recursion_limit": 20},
        ):
            for _node, node_output in chunk.items():
                new_msgs = node_output.get("messages", [])
                all_messages.extend(new_msgs)

                for msg in new_msgs:
                    msg_type = type(msg).__name__
                    if msg_type == "AIMessage" and getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            yield sse({"type": "step_call", "tool": tc.get("name")})
                    elif msg_type == "ToolMessage":
                        tool_name = getattr(msg, "name", "unknown")
                        if tool_name == "validate_sql_query":
                            validation_tries += 1
                        content = str(msg.content)
                        snippet = content[:200] + "…" if len(content) > 200 else content
                        yield sse({"type": "step_result", "tool": tool_name, "snippet": snippet})

    except Exception as e:
        yield sse({"type": "error", "status": "error", "message": "Agent execution failed.", "reasons": [str(e)]})
        return

    # Token usage (ChatOllama populates usage_metadata when available)
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    for msg in all_messages:
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            meta = msg.usage_metadata
            input_tokens += meta.get("input_tokens", 0)
            output_tokens += meta.get("output_tokens", 0)
            total_tokens += meta.get("total_tokens", 0)

    final_sql = get_latest_sql(all_messages)
    final = _build_response(all_messages)

    final["type"] = "done"
    final["final_sql"] = final_sql
    final["input_tokens"] = input_tokens
    final["output_tokens"] = output_tokens
    final["total_tokens"] = total_tokens
    final["cost"] = 0.0
    final["prompt_cost"] = 0.0
    final["completion_cost"] = 0.0
    final["validation_tries"] = validation_tries

    latency = time.perf_counter() - start_time

    try:
        evaluate_live_query(
            prompt=question,
            model=graph_model,
            generated_sql=final_sql,
            latency=latency,
            metrics=final,
        )
    except Exception as log_err:
        print(f"Failed to log live query: {log_err}")

    yield sse(final)


def get_latest_sql(all_messages: list) -> str:
    """
    Extract the most recently used SQL from the message history.
    Prefers structured tool_calls; falls back to regex on message content.
    """
    for msg in reversed(all_messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") in ["get_data", "sql_db_query", "query_sql_db", "execute_sql"]:
                    return str(tc.get("args", {}).get("sql", ""))

    sql_pattern = re.compile(r"SQL:\s*(.*?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
    for msg in reversed(all_messages):
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            match = sql_pattern.search(content)
            if match:
                return match.group(1).strip()

    return "No SQL query found in history."
