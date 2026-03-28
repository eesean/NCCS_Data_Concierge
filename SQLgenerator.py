import os
import json
import re
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:7b")


def _get_llm(model_name: str = None):
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=model_name or OLLAMA_DEFAULT_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


class SQLGenError(RuntimeError):
    pass


def generate_sql_from_nl(question: str, model: str = None) -> dict:
    """Generate SQL from a natural-language question using the local Ollama model."""
    llm = _get_llm(model)

    system = """Return ONLY valid JSON in exactly this format:
        {"sql": "<single SELECT statement>",
        "explanation": "<a brief sentence explanation of what the query does>"
        }

        Rules:
        - Generate exactly ONE SQL statement.
        - Only SELECT queries are allowed. Never use INSERT/UPDATE/DELETE/DROP/CREATE/ALTER.
        - Use DuckDB-compatible functions only.
        - Include LIMIT when necessary.
        - Analyze the user prompt carefully.
        """

    resp = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=question),
    ])

    content = _strip_code_fences((resp.content or "").strip())
    usage = resp.usage_metadata or {}

    if not content:
        raise SQLGenError("LLM returned empty output")

    try:
        content = content.replace('\n', ' ').replace('\r', ' ')
        parsed = json.loads(content)
        return {
            "sql": parsed.get("sql", "").strip(),
            "explanation": parsed.get("explanation", "").strip(),
            "usage": {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            "cost": {
                "total_cost": 0.0,
                "prompt_cost": 0.0,
                "completion_cost": 0.0,
            },
        }
    except Exception as e:
        raise SQLGenError(f"Failed to parse LLM response. Raw: {content}") from e


def explain_sql(sql: str, model_name: str = None) -> str:
    """
    Translate a DuckDB SQL query into a single plain-English interrogative sentence.
    Used for semantic similarity scoring in live evaluation.
    """
    llm = _get_llm(model_name)

    system_message = (
        "Role: You are a specialized SQL-Medical Auditor.\n\n"
        "Task: Translate the DuckDB SQL query into a single, concise interrogative sentence "
        "directed at a physician.\n\n"
        "Requirements:\n"
        "- The output must be exactly one sentence.\n"
        "- Explicitly state the population, medical codes, date ranges, and logical filters.\n"
        "- Do not include SQL jargon (JOIN, WHERE, FLOAT, etc.).\n"
        "- Do not add introductory text — only the question itself.\n"
        "- Leave numbers as digits; do not spell them out.\n"
        "- Tone: professional, precise, clinical."
    )

    resp = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"SQL: {sql}"),
    ])

    return resp.content.strip()


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text
