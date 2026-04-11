import json
import re
from dotenv import load_dotenv

from ContextConfiguration import (
    EXPLAIN_SQL_SYSTEM,
    EXPLAIN_SQL_USER_TEMPLATE,
    GENERATE_SQL_FROM_NL_SYSTEM,
)
from retrieval.llm import ollama_chat

load_dotenv()


class SQLGenError(RuntimeError):
    pass


def _strip_code_fences(text: str) -> str:
    """Remove Markdown code fences like ```json ... ``` or ``` ... ```."""
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def generate_sql_from_nl(question: str, model: str = None) -> dict:
    """
    Generate SQL from natural language using the local Ollama model.
    Returns a dict with 'sql', 'explanation', 'usage', and 'cost' keys.
    Signatures are kept identical to the OpenRouter version so evaluation_update.py
    continues to work without modification.
    """
    messages = [
        {"role": "system", "content": GENERATE_SQL_FROM_NL_SYSTEM},
        {"role": "user", "content": question},
    ]

    response = ollama_chat(messages)
    content = _strip_code_fences((response["message"].get("content") or "").strip())

    if not content:
        raise SQLGenError("LLM returned empty output")

    try:
        content = content.replace("\n", " ").replace("\r", " ")
        parsed = json.loads(content)
        return {
            "sql": parsed.get("sql", "").strip(),
            "explanation": parsed.get("explanation", "").strip(),
            "usage": {
                "input_tokens": response.get("prompt_eval_count"),
                "output_tokens": response.get("eval_count"),
                "total_tokens": (
                    (response.get("prompt_eval_count") or 0)
                    + (response.get("eval_count") or 0)
                ),
            },
            "cost": {
                "total_cost": 0.0,
                "prompt_cost": 0.0,
                "completion_cost": 0.0,
            },
        }
    except Exception as e:
        raise SQLGenError(f"Failed to parse LLM response. Raw: {content}") from e


def explain_sql(sql: str, model: str = None) -> str:
    """
    Translate a SQL query into a concise clinical interrogative sentence.
    Used for adversarial validation in evaluation_update.py.
    """
    messages = [
        {"role": "system", "content": EXPLAIN_SQL_SYSTEM},
        {"role": "user", "content": EXPLAIN_SQL_USER_TEMPLATE.format(sql=sql)},
    ]

    response = ollama_chat(messages, model=model)
    content = response["message"].get("content") or ""
    return content.strip()
