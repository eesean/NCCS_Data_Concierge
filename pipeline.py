import json
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

from langchain_core.messages import ToolMessage,SystemMessage,HumanMessage

from retrieval.graph.node.workflow import build_graph
from retrieval.graph.outputParser import parse_data_json, extract_final_text, extract_data_json
from retrieval.graph.tool.vectorRag import get_schema_context, get_sql_template
from retrieval.graph.tool.evaluate_update import evaluate_live_query
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
    """Provides schema and SQL prompt templates as System Context."""
    schema_data = get_schema_context(question)
    sql_template_data = get_sql_template(question)

    # Use a SystemMessage for context that should not be 'called' as a tool
    system_content = (
        f"You have access to the following database schema:\n{schema_data}\n\n"
        f"Relevant SQL examples (few-shot) for similar questions:\n{sql_template_data}"
    )
    system_msg = SystemMessage(content=system_content)
    
    # Include the user's question so the graph knows what to process immediately
    user_msg = HumanMessage(content=question)
    
    return [system_msg, user_msg] # Need to return system and user message for Open AI to work. 

def stream_question_agent(question: str, model: Optional[str] = None, history: Optional[List[Dict]] = None) -> Generator[str, None, None]:
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
    start_time = time.perf_counter() ## Timer starts for latency
    try:
        # Get schema once before the graph (ensures single call)
        schema_messages = _get_schema_and_messages(question)

        # Add conversation history
        if history:
            cleaned_history = []

            for msg in history[:-1][-6:]:  
                role = msg.get("role")
                content = msg.get("content")

                # User messages
                if role == "user" and isinstance(content, str):
                    cleaned_history.append(HumanMessage(content=content))
    
                # Assistant messages (only pass previous SQL, answers not included)
                elif role == "assistant":
                    if isinstance(content, dict):
                        sql = content.get("final_sql")

                        if sql:
                            assistant_text = f"Previous SQL query:\n{sql}"
                            cleaned_history.append(SystemMessage(content=assistant_text))

            # Insert history before the current question
            schema_messages = (
                [schema_messages[0]]
                + cleaned_history
                + [schema_messages[-1]]
            )

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
            config={"recursion_limit": 20},
        ):  
            for _node, node_output in chunk.items():
                new_msgs = node_output.get("messages", [])
                all_messages.extend(new_msgs)

                for msg in new_msgs:
                    msg_type = type(msg).__name__
                    if msg_type == "AIMessage" and getattr(msg, "tool_calls", None):
                        #print("Tools: " + json.dumps(msg.tool_calls, indent=2, default=str))
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
    
    # --- Token & Cost Tracking Logic --- HQ added for metrics extractions
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    upstream_cost = 0.0
    upstream_prompt_cost = 0.0
    upstream_completion_cost = 0.0

    for msg in all_messages:
        # 1. Standard Token Usage
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            meta = msg.usage_metadata
            input_tokens += meta.get("input_tokens", 0)
            output_tokens += meta.get("output_tokens", 0)
            total_tokens += meta.get("total_tokens", 0)

        # 2. Custom Cost Details (Provider-specific metadata)
        if hasattr(msg, "response_metadata") and msg.response_metadata:
            token_usage = msg.response_metadata.get("token_usage", {})
            cost_details = token_usage.get("cost_details", {})
            
            upstream_cost += float(token_usage.get("cost", 0.0))
            upstream_prompt_cost += float(cost_details.get("upstream_inference_prompt_cost", 0.0))
            upstream_completion_cost += float(cost_details.get("upstream_inference_completions_cost", 0.0))

    # --- SQL and Final Response Build ---
    final_sql = get_latest_sql(all_messages)
    final = _build_response(all_messages)
    
    # Inject aggregated metadata
    final["type"] = "done"
    final["final_sql"] = final_sql
    final["input_tokens"] = input_tokens
    final["output_tokens"] = output_tokens
    final["total_tokens"] = total_tokens
    final["cost"] = upstream_cost
    final["prompt_cost"] = upstream_prompt_cost
    final["completion_cost"] = upstream_completion_cost

    latency = time.perf_counter() - start_time ## Timer Ends

    ## Run evaluation based off the query
    ## It will updates the live scoreboard
    try:
        evaluate_live_query(
            prompt=question,
            model=model or DEFAULT_MODEL,
            generated_sql=final_sql,
            latency=latency,
            metrics=final
        )
    except Exception as log_err:
        print(f"Failed to log live query: {log_err}")
    yield sse(final)

import re

def get_latest_sql(all_messages: list) -> str: ## For latest SQL extraction
    """
    1. Prioritizes structured tool_calls (Index 4 style).
    2. Falls back to Regex extraction from content (Index 6 style).
    """
    # 1. Search for structured tool calls (Most reliable)
    for msg in reversed(all_messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                # Check for any tool that executes SQL
                if tc.get("name") in ["get_data", "sql_db_query", "query_sql_db", "execute_sql"]:
                    # Return the exact SQL string from 'args'
                    return str(tc.get("args", {}).get("sql", ""))

    # 2. Fallback to Regex for Index [6] style
    sql_pattern = re.compile(r"SQL:\s*(.*?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
    for msg in reversed(all_messages):
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            match = sql_pattern.search(content)
            if match:
                return match.group(1).strip()
    
    return "No SQL query found in history."