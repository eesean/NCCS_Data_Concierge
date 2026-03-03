import os
import json
import re
import duckdb
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from SQLvalidator import PARQUETS

load_dotenv()

def get_duckdb_schema(db_path: str, table_names: list) -> str:
    """
    Connects to the DuckDB file and returns a formatted string of table schemas.
    """
    conn = duckdb.connect(str(db_path))
    schema_parts = []
    
    for table in table_names:
        # Get column names using DuckDB's PRAGMA
        columns_info = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        # columns_info returns (id, name, type, notnull, dflt_value, pk)
        col_names = [col[1] for col in columns_info]
        schema_parts.append(f"- {table}: {', '.join(col_names)}")
    
    conn.close()
    return "\n".join(schema_parts)

# Generate the schema string
SCHEMA_DESCRIPTION = get_duckdb_schema("data/nccs_cap26.db", list(PARQUETS.keys()))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "deepseek/deepseek-v3.2"
## deepseek/deepseek-v3.2 
## openai/gpt-4o-mini



_llm = ChatOpenAI(
    model=MODEL,
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0,
)

class SQLGenError(RuntimeError):
    pass

def generate_sql_from_nl(question: str, model: str = None) -> dict:
    """
    Generate SQL from natural language. Returns SQL string only.
    Designed to be called by pipeline/backend later.
    """
    if not OPENROUTER_API_KEY:
        raise SQLGenError("Missing OPENROUTER_API_KEY in .env")

    # Choose the model: Use the parameter if provided, else the default
    target_model = model or MODEL
    
    # Bind the model to the existing LLM object
    llm_with_model = _llm.bind(model=target_model)

    # Force structured output so downstream validation/execution is reliable
    system = f"""Return ONLY valid JSON in exactly this format:
        {{"sql": "<single SELECT statement>",
        "explanation": "<a brief, sentence explanation of what the query does>"
        }}

        Rules:
        - Generate exactly ONE SQL statement.
        - Do NOT assume any other OMOP tables exist.
        - Only SELECT queries are allowed. Never use INSERT/UPDATE/DELETE/DROP/CREATE/ALTER.
        - Use only the available OMOP-style tables and join keys.
        - Include LIMIT when necessary. For COUNT-only queries, use LIMIT 1 when necessary.
        - Use DuckDB-compatible functions only.
        - Do NOT reference any table not explicitly listed below.
        - Analyze the user prompt carefully, make sure all of the parameters and requirements are taken note of
        - You may ONLY use these tables (exact spelling, lowercase):
        {SCHEMA_DESCRIPTION}
        """

    resp = llm_with_model.invoke([
        SystemMessage(content=system),
        HumanMessage(content=question)
    ])

    content = _strip_code_fences((resp.content or "").strip())
    usage = resp.usage_metadata or {}
    meta = resp.response_metadata or {}
    cost_details = meta.get("token_usage", {}).get("cost_details", {})
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
                "total_tokens": usage.get("total_tokens")
            },
            "cost": {
                "total_cost": cost_details.get("upstream_inference_cost"),
                "prompt_cost": cost_details.get("upstream_inference_prompt_cost"),
                "completion_cost": cost_details.get("upstream_inference_completions_cost")
            }
        }

    except Exception as e:
        raise SQLGenError(f"Failed to parse LLM response. Raw: {content}") from e

def explain_sql(sql: str, model_name: str = None) -> str:
    """
    Independently explains a SQL query without knowing the original prompt.
    Used for adversarial validation.
    """
    target_model = model_name or MODEL
    # Use a cheap, fast model for this audit step
    llm_explainer = _llm.bind(model=target_model)
    
    system_message = (
        """Role: You are a specialized SQL-Medical Auditor.

            Task: Your goal is to translate DuckDB SQL queries into a single, concise interrogative sentence directed at a physician.

            Requirements:

            Format: The output must be exactly one sentence.

            Content: Explicitly state the population being retrieved, including all specific medical codes , date ranges, and logical filters.

            Constraint: Do not include technical SQL jargon (like "JOIN," "WHERE," or "FLOAT"). Do not include any introductory text or "extra info"—only the question itself.

            Tone: Professional, precise, and clinical."""
    )
    
    resp = llm_explainer.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"SQL: {sql}")
    ])
    
    return resp.content.strip()

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

