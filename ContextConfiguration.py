"""
Central definitions for all text fed to LLMs (system prompts, user templates, tool schemas).
No functions — import constants where needed.
"""

# -----------------------------------------------------------------------------
# CANCER_INFO_TOOL
# Used in: Agent.py → stream_question_agent()
# For: Ollama tool-calling during the context-gathering step. The model may invoke
#      get_cancer_info so ICD-10 / SQL_FILTER snippets are injected before SQL generation.
# -----------------------------------------------------------------------------
CANCER_INFO_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_cancer_info",
            "description": (
                "Retrieve ICD-10-AM cancer reference text for a specific cancer type. "
                "Call this when the user's question involves a particular cancer. "
                "Returns SQL_FILTER (ready ICD10 LIKE predicates) and ICD10_CODES. "
                "Copy SQL_FILTER verbatim into the WHERE clause — never invent codes like ICDO3."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short cancer term, e.g. 'colorectal cancer', 'lung cancer'.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# -----------------------------------------------------------------------------
# CONTEXT_SYSTEM
# Used in: Agent.py → stream_question_agent()
# For: System message for the first LLM turn: schema + few-shot examples are appended
#      in code; instructs when to call get_cancer_info and explicitly forbids SQL yet.
# -----------------------------------------------------------------------------
CONTEXT_SYSTEM = """/no_think
You are an expert assistant for the NCCS (National Cancer Centre Singapore) dataset.

You have been given the database schema and SQL examples.
If the user's question involves a specific cancer type, call get_cancer_info to retrieve the correct ICD-10 codes.
Otherwise do nothing — just acknowledge you are ready.

Do NOT generate SQL yet.
"""

# -----------------------------------------------------------------------------
# SQL_GEN_PROMPT
# Used in: Agent.py → stream_question_agent()
# For: User message appended after context/tool turns to ask for the final DuckDB SELECT
#      (raw SQL only, no markdown).
# -----------------------------------------------------------------------------
SQL_GEN_PROMPT = (
    "Now generate the DuckDB SQL query to answer the question above.\n"
    "Rules:\n"
    "- Return ONLY the raw SQL statement — no explanation, no markdown, no code fences.\n"
    "- Use ONLY columns and tables from the schema.\n"
    "- For cancer queries, copy the SQL_FILTER from get_cancer_info verbatim — never use ICDO3.\n"
    "- Use DuckDB-compatible syntax only.\n"
)

# -----------------------------------------------------------------------------
# SUMMARY_SYSTEM
# Used in: Agent.py → stream_question_agent()
# For: System message for the post-query summarisation LLM call (plain-language answer
#      for the user from executed result rows).
# -----------------------------------------------------------------------------
SUMMARY_SYSTEM = """/no_think
You are a data analyst. Summarise the query results in 1-2 clear sentences for a non-technical user.
"""

# -----------------------------------------------------------------------------
# SUMMARY_USER_TEMPLATE
# Used in: Agent.py → stream_question_agent()
# For: User message for summarisation; placeholders: {question}, {data} (truncated result text).
# -----------------------------------------------------------------------------
SUMMARY_USER_TEMPLATE = "Question: {question}\n\nData:\n{data}"

# -----------------------------------------------------------------------------
# SQL_VALIDATION_FIX_USER_TEMPLATE
# Used in: Agent.py → stream_question_agent()
# For: User message when validate_sql_query fails; placeholders: {validation_text}, {sql}.
#      Asks the model to return only corrected SQL (validation retry loop).
# -----------------------------------------------------------------------------
SQL_VALIDATION_FIX_USER_TEMPLATE = (
    "The SQL failed validation:\n{validation_text}\n\n"
    "Current SQL:\n{sql}\n\n"
    "Fix the SQL so it passes validation. "
    "Return ONLY the corrected SQL statement, no explanation."
)

# -----------------------------------------------------------------------------
# GENERATE_SQL_FROM_NL_SYSTEM
# Used in: SQLgenerator.py → generate_sql_from_nl()
# For: Standalone Ollama path: system prompt enforcing JSON with sql + explanation fields
#      and DuckDB SELECT-only rules (not used by the main API Agent flow).
# -----------------------------------------------------------------------------
GENERATE_SQL_FROM_NL_SYSTEM = """Return ONLY valid JSON in exactly this format:
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

# -----------------------------------------------------------------------------
# EXPLAIN_SQL_SYSTEM
# Used in: SQLgenerator.py → explain_sql()
# For: System prompt to rewrite SQL as one clinical interrogative sentence (for semantic
#      scoring vs the user question; also used by live evaluation logging).
# -----------------------------------------------------------------------------
EXPLAIN_SQL_SYSTEM = (
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

# -----------------------------------------------------------------------------
# EXPLAIN_SQL_USER_TEMPLATE
# Used in: SQLgenerator.py → explain_sql()
# For: User message wrapping the SQL to explain; placeholder: {sql}.
# -----------------------------------------------------------------------------
EXPLAIN_SQL_USER_TEMPLATE = "SQL: {sql}"
