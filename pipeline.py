import json
from typing import Any, Dict, Generator, List, Optional, Tuple

from langchain_core.messages import ToolMessage

from retrieval.graph.node.workflow import build_graph
from retrieval.graph.outputParser import parse_data_json, extract_final_text, extract_data_json
from retrieval.graph.tool.vectorRag import get_schema_context
from retrieval.llm import DEFAULT_MODEL


def _build_response(messages: list) -> Dict[str, Any]:
    """
    Convert a list of agent messages into the response dict shape the frontend expects.
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
    """Call get_schema_context once and return messages with schema for the graph."""
    schema_result = get_schema_context(question)
    schema_msg = ToolMessage(
        content=str(schema_result),
        name="get_schema_context",
        tool_call_id="schema_preload",
    )
    return [schema_msg]

def stream_question_agent(question: str, model: Optional[str] = None) -> Generator[str, None, None]:
    """
    Generator that runs the ReAct agent and yields Server-Sent Events (SSE).

    Event types emitted:
      {"type": "step_call",   "tool": <name>}
      {"type": "step_result", "tool": <name>, "snippet": <str>}
      {"type": "done",        ...response payload...}
      {"type": "error",       "message": ..., "reasons": [...]}
    """
    def sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    try:
        # Get schema once before the graph (ensures single call)
        schema_messages = _get_schema_and_messages(question)
        graph = build_graph(model or DEFAULT_MODEL)
        initial_state = {"query": question, "final_answer": "", "messages": schema_messages}
    except Exception as e:
        yield sse({"type": "error", "status": "error", "message": "Agent setup failed.", "reasons": [str(e)]})
        return

    all_messages: List[Any] = list(schema_messages)

    try:
        # Emit schema step (called outside graph)
        yield sse({"type": "step_call", "tool": "get_schema_context"})
        schema_content = str(schema_messages[0].content)
        yield sse({"type": "step_result", "tool": "get_schema_context", "snippet": schema_content[:200] + "…" if len(schema_content) > 200 else schema_content})

        for chunk in graph.stream(
            initial_state,
            stream_mode="updates",
            config={"recursion_limit": 50},
        ):
            for _node, node_output in chunk.items():
                new_msgs = node_output.get("messages", [])
                all_messages.extend(new_msgs)

                for msg in new_msgs:
                    msg_type = type(msg).__name__
                    if msg_type == "AIMessage" and getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            yield sse({"type": "step_call", "tool": tc["name"]})
                    elif msg_type == "ToolMessage":
                        tool_name = getattr(msg, "name", "unknown")
                        content = str(msg.content)
                        snippet = content[:200] + "…" if len(content) > 200 else content
                        yield sse({"type": "step_result", "tool": tool_name, "snippet": snippet})
    except Exception as e:
        yield sse({"type": "error", "status": "error", "message": "Agent execution failed.", "reasons": [str(e)]})
        return

    final = _build_response(all_messages)
    final["type"] = "done"
    yield sse(final)
