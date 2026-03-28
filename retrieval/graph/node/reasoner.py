"""
Ollama native tool-calling agent loop.

Defines the tool schemas in Ollama's JSON format and a `run_agent()` function
that runs the full tool-calling loop, returning the complete message history.
"""
from retrieval.llm import ollama_chat
from retrieval.graph.tool.vectorRag import get_cancer_info, get_schema_context, get_sql_template
from retrieval.graph.tool.SQLvalidator import validate_sql_query
from retrieval.graph.tool.get_data import get_data

# ---------------------------------------------------------------------------
# Tool schemas — Ollama JSON format
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_cancer_info",
            "description": (
                "Retrieve ICD-10-AM cancer reference text for a cancer topic. "
                "MUST be called first for any query about a specific cancer type. "
                "Returns SQL_FILTER (ready ICD10 LIKE predicates) and ICD10_CODES. "
                "Copy SQL_FILTER and ICD10_CODES verbatim into the WHERE clause — "
                "do NOT invent codes such as ICDO3, C50%, or similar."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short cancer term, e.g. 'colorectal cancer', 'lung cancer'."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_schema_context",
            "description": (
                "Retrieve relevant database schema (tables and columns) for a natural language query. "
                "Use this when you need to generate SQL — it returns the schema context needed to write accurate queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's question or what they want to query."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sql_template",
            "description": (
                "Retrieve relevant SQL few-shot examples/templates based on the user's question. "
                "Use when generating SQL — returns similar past queries to guide structure and joins."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question or query intent."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_sql_query",
            "description": (
                "Validate a DuckDB SQL query for safety and correctness. "
                "Call this after generating SQL to confirm it is safe before executing it. "
                "Returns a summary of issues found, or confirms the query is valid."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL query to validate."
                    }
                },
                "required": ["sql"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_data",
            "description": (
                "Execute a validated DuckDB SQL query against the parquet dataset and return the results. "
                "Only call this after validate_sql_query has confirmed the SQL is valid. "
                "Returns the query results as a JSON array of records."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The validated SQL query to execute."
                    }
                },
                "required": ["sql"]
            }
        }
    },
]

# ---------------------------------------------------------------------------
# Tool function map
# ---------------------------------------------------------------------------

TOOL_FN_MAP = {
    "get_cancer_info": get_cancer_info,
    "get_schema_context": get_schema_context,
    "get_sql_template": get_sql_template,
    "validate_sql_query": validate_sql_query,
    "get_data": get_data,
}

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(messages: list, max_iterations: int = 10) -> list:
    """Run the Ollama native tool-calling loop; return the full message history."""
    for iteration in range(max_iterations):
        response = ollama_chat(messages, tools=TOOLS)
        assistant_msg = response["message"]

        # Ollama returns a Message object — convert to plain dict for uniform handling
        if hasattr(assistant_msg, "__dict__"):
            assistant_msg = dict(assistant_msg)
        # Convert nested tool_calls if they are objects
        if assistant_msg.get("tool_calls"):
            converted = []
            for tc in assistant_msg["tool_calls"]:
                if hasattr(tc, "__dict__"):
                    tc = dict(tc)
                if "function" in tc and hasattr(tc["function"], "__dict__"):
                    tc = dict(tc)
                    tc["function"] = dict(tc["function"])
                converted.append(tc)
            assistant_msg["tool_calls"] = converted

        messages.append(assistant_msg)

        tool_calls = assistant_msg.get("tool_calls")
        if not tool_calls:
            # No more tool calls — final answer reached
            break

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            # arguments may already be a dict or a JSON string
            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            print(f"[tool] calling {fn_name}({args})")

            fn = TOOL_FN_MAP.get(fn_name)
            if fn is None:
                result = f"ERROR: unknown tool '{fn_name}'"
            else:
                try:
                    result = fn(**args)
                except Exception as exc:
                    result = f"ERROR: {exc}"

            print(f"[tool] {fn_name} → {str(result)[:200]}")

            messages.append({
                "role": "tool",
                "name": fn_name,
                "content": str(result),
            })

    return messages
