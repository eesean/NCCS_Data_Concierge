"""
Output parser for the ReAct graph.

The LLM's final AIMessage often wraps the data in commentary text.
These helpers extract the raw JSON produced by get_data directly from
the ToolMessage in the graph state — bypassing any LLM text wrapping.
"""
import json
from typing import Optional, List


def extract_data_json(messages: list) -> Optional[str]:
    """
    Return the raw JSON string from the get_data ToolMessage, or None
    if get_data was not called.
    """
    for msg in messages:
        if type(msg).__name__ == "ToolMessage" and getattr(msg, "name", None) == "get_data":
            return msg.content
    return None


def parse_data_json(messages: list) -> Optional[List[dict]]:
    """
    Return the get_data result parsed into a Python list of dicts, or None
    if get_data was not called or the content is not valid JSON.
    """
    raw = extract_data_json(messages)
    if raw is None or raw.startswith("EXECUTION_ERROR") or raw.startswith("Query executed"):
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def extract_final_text(messages: list) -> Optional[str]:
    """
    Return the last AIMessage text that is not a tool call, or None.
    This is the LLM's natural-language summary of the results — useful as
    a caption alongside the data table in a UI.
    """
    for msg in reversed(messages):
        if (
            type(msg).__name__ == "AIMessage"
            and not getattr(msg, "tool_calls", None)
            and getattr(msg, "content", "")
        ):
            return msg.content
    return None
