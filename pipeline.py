import json
import os
from typing import Any, Dict, Generator, List, Optional, Tuple

from SQLvalidator import (
    connect_duckdb,
    load_parquet_views,
    PARQUETS,
    validate_sql as validate_sql_checker,
)

from langchain_core.messages import ToolMessage

from retrieval.graph.node.workflow import build_graph
from retrieval.graph.outputParser import parse_data_json, extract_final_text, extract_data_json
from retrieval.graph.tool.vectorRag import get_schema_context
from retrieval.llm import DEFAULT_MODEL

DEBUG = os.getenv("DEBUG", "false").lower() == "true"


def _execute_duckdb(con, sql: str, max_rows: int = 200) -> Tuple[List[str], List[List[Any]]]:
    """
    Execute SQL and return (columns, rows). Hard caps rows to avoid huge outputs.
    """
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchmany(max_rows)
    return cols, [list(r) for r in rows]

#Handle question not needed implemented using stream question agent instead
def handle_question(question: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    NL -> SQL (LLM) -> validate -> execute -> return results
    Returns results only (never SQL) to satisfy your user story.
    """
    from SQLgenerator import generate_sql_from_nl

    # 1) Connect to DuckDB + load encrypted parquet views
    con = connect_duckdb(use_db_file=False)
    load_parquet_views(con, PARQUETS)

    allow_tables = set(PARQUETS.keys())
    restricted_tables = {"condition_occurrence", "drug_exposure_cancerdrugs", "measurement_mutation"}

    # 2) Generate SQL in backend
    result = generate_sql_from_nl(question, model=model) #if model is None it will use a default model
    sql = result['sql']

    # backend-only logging
    if DEBUG:
        print("[DEBUG] Generated SQL:", sql)

    # 3) Validate SQL
    validation = validate_sql_checker(
        con=con,
        sql=sql,
        expected_result=None,
        allow_tables=allow_tables,
        allow_columns=None,
        require_where_for_tables=restricted_tables,
        require_limit=True,
        block_select_star=True,
    )

    if not validation.is_safe:
        return {
            "status": "error",
            "message": "Blocked by safety rules.",
            "reasons": validation.safety_reasons,
        }

    if not validation.is_performant:
        return {
            "status": "error",
            "message": "Blocked due to performance risk.",
            "reasons": validation.performance_reasons,
        }

    # 4) Execute and return table only (no SQL)
    try:
        columns, rows = _execute_duckdb(con, sql, max_rows=200)
        # if it's a single scalar result (e.g., COUNT(*)), return a user-friendly value 
        is_scalar = (len(columns) == 1 and len(rows) == 1 and isinstance(rows[0], list) and len(rows[0]) == 1)
        if is_scalar:
            return {
                "status": "ok",
                "message": "Query executed successfully.",
                "metric": columns[0],     
                "value": rows[0][0],      
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
            }

        # otherwise, return normal table results
        return {
            "status": "ok",
            "message": "Query executed successfully.",
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
        }

    except Exception:
        return {
            "status": "error",
            "message": "Execution failed. Please refine your question.",
            "reasons": ["EXECUTION_FAILED"],
        }
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


def _collect_steps(messages: list) -> List[Dict[str, Any]]:
    """Extract tool call / tool result steps from agent messages for UI display."""
    steps = []
    for msg in messages:
        msg_type = type(msg).__name__
        if msg_type == "AIMessage" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                steps.append({"kind": "call", "tool": tc["name"]})
        elif msg_type == "ToolMessage":
            tool_name = getattr(msg, "name", "unknown")
            content = str(msg.content)
            snippet = content[:200] + "…" if len(content) > 200 else content
            # Schema is preloaded (no AIMessage); add synthetic call for display
            if tool_name == "get_schema_context" and (
                not steps or steps[-1].get("tool") != "get_schema_context" or steps[-1].get("kind") != "call"
            ):
                steps.append({"kind": "call", "tool": tool_name})
            steps.append({"kind": "result", "tool": tool_name, "snippet": snippet})
    return steps

def _get_schema_and_messages(question: str) -> list:
    """Call get_schema_context once and return messages with schema for the graph."""
    schema_result = get_schema_context.invoke({"query": question})
    schema_msg = ToolMessage(
        content=str(schema_result),
        name="get_schema_context",
        tool_call_id="schema_preload",
    )
    return [schema_msg]


#Handle question agent using stream question agent instead
def handle_question_agent(question: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Route a natural-language question through the LangGraph ReAct agent (blocking).
    Schema is fetched once before the graph; graph only does validate + get_data.
    """
    try:
        schema_messages = _get_schema_and_messages(question)
        graph = build_graph(model or DEFAULT_MODEL)
        initial_state = {"query": question, "final_answer": "", "messages": schema_messages}
        result = graph.invoke(initial_state)
        messages = result["messages"]
    except Exception as e:
        return {
            "status": "error",
            "message": "Agent pipeline failed.",
            "reasons": [str(e)],
            "steps": [],
        }

    steps = _collect_steps(messages)
    response = _build_response(messages)
    response["steps"] = steps
    return response


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

        for chunk in graph.stream(initial_state, stream_mode="updates"):
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
