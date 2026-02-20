import os
import json
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from .SQLvalidator import PARQUETS

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-4o-mini"
ALLOWED_TABLES = ", ".join(PARQUETS.keys())

_llm = ChatOpenAI(
    model=MODEL,
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0,
)

class SQLGenError(RuntimeError):
    pass

def generate_sql_from_nl(question: str) -> str:
    """
    Generate SQL from natural language. Returns SQL string only.
    Designed to be called by pipeline/backend later.
    """
    if not OPENROUTER_API_KEY:
        raise SQLGenError("Missing OPENROUTER_API_KEY in .env")

    # Force structured output so downstream validation/execution is reliable
    system = f"""Return ONLY valid JSON in exactly this format:
        {{"sql": "<single SELECT statement>"}}

        Rules:
        - Generate exactly ONE SQL statement.
        - Do NOT assume any other OMOP tables exist.
        - Only SELECT queries are allowed. Never use INSERT/UPDATE/DELETE/DROP/CREATE/ALTER.
        - Use only the available OMOP-style tables and join keys.
        - Always include LIMIT. For COUNT-only queries, use LIMIT 1.
        - Use DuckDB-compatible functions only.
        - Do NOT reference any table not explicitly listed below.
        - You may ONLY use these tables (exact spelling, lowercase):
        {ALLOWED_TABLES}
        """

    resp = _llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=question)
    ])

    content = _strip_code_fences((resp.content or "").strip())
    if not content:
        raise SQLGenError("LLM returned empty output")

    try:
        parsed = json.loads(content)
        sql = parsed["sql"].strip()
    except Exception as e:
        raise SQLGenError(f"LLM did not return valid JSON. Got: {content}") from e

    return sql

def _strip_code_fences(text: str) -> str:
    """
    Removes Markdown code fences like ```json ... ``` or ``` ... ```
    """
    text = text.strip()
    # Matches ```json\n...\n``` or ```\n...\n```
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text
