"""
Output parser for the Ollama native tool-calling agent.

Messages are plain dicts with 'role' keys rather than LangChain message objects.
These helpers extract the raw JSON from get_data tool results and the final
assistant text from the message history returned by run_agent().
"""
import json
from typing import Optional, List


def extract_data_json(messages: list) -> Optional[str]:
    """
    Return the raw JSON string from the LAST get_data tool result, or None
    if get_data was not called.
    """
    found = None
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "tool" and msg.get("name") == "get_data":
            found = msg.get("content")
    return found


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
    Return the last assistant message that is not a tool call, or None.
    This is the LLM's natural-language summary of the results.
    """
    for msg in reversed(messages):
        if (
            isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and not msg.get("tool_calls")
            and msg.get("content")
        ):
            return msg["content"]
    return None
