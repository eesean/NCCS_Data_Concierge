import json
import re
from dotenv import load_dotenv

from retrieval.llm import ollama_chat, OLLAMA_MODEL

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
    system = """Return ONLY valid JSON in exactly this format:
{"sql": "<single SELECT statement>",
 "explanation": "<a brief, sentence explanation of what the query does>"
}

Rules:
- Generate exactly ONE SQL statement.
- Only SELECT queries are allowed. Never use INSERT/UPDATE/DELETE/DROP/CREATE/ALTER.
- Use only the available tables and join keys present in the schema.
- Include LIMIT when necessary. For COUNT-only queries, use LIMIT 1 when necessary.
- Use DuckDB-compatible functions only.
- Do NOT reference any table not explicitly listed in the schema.
- Analyse the user prompt carefully; capture all parameters and requirements.
"""

    messages = [
        {"role": "system", "content": system},
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
    system_message = (
        "Role: You are a specialized SQL-Medical Auditor.\n\n"
        "Task: Translate DuckDB SQL queries into a single, concise interrogative sentence "
        "directed at a physician.\n\n"
        "Requirements:\n"
        "- Format: exactly one sentence.\n"
        "- Content: explicitly state the population being retrieved, including all specific "
        "medical codes, date ranges, and logical filters.\n"
        "- Constraint: no SQL jargon (JOIN, WHERE, FLOAT). No introductory text.\n"
        "- Numbers: keep numeric form (e.g. 5 not 'five', ICD10 not 'I C D 10').\n"
        "- Tone: professional, precise, clinical."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"SQL: {sql}"},
    ]

    response = ollama_chat(messages, model=model)
    content = response["message"].get("content") or ""
    return content.strip()
