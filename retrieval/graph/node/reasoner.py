"""
Reasoner node: LLM with tool binding. Schema is preloaded by pipeline; LLM only uses validate_sql_query and get_data.
"""
from retrieval.llm import llm
from retrieval.graph.tool.SQLvalidator import validate_sql_query
from retrieval.graph.tool.get_data import get_data
from langchain_core.messages import SystemMessage, HumanMessage

# Schema is fetched by pipeline before graph; reasoner never has get_schema_context
_reasoner_tools = [validate_sql_query, get_data]
llm_with_tools = llm.bind_tools(_reasoner_tools)

_SYS_PROMPT = """
You are a SQL expert working with a DuckDB healthcare dataset. 
You will be given: (a) a user question, and (b) schema context that lists the ONLY allowed tables/columns.

HARD CONSTRAINTS (never violate):
- Use ONLY table/column names explicitly present in the provided schema context. No guessing.
- Table and column names are case-sensitive.
- Join tables using `person_id` only (unless the schema explicitly states another join key).
- Type rules:
  - For ICD/code filtering use ICD10 / ICDO3 / *_source_value columns (VARCHAR) → use quoted strings (e.g., 'C34').
  - In WHERE/BETWEEN: match column types exactly.

MANDATORY FLOW (do not skip):
step 1) PLAN (required, 3–6 lines, no SQL):
   - Metric: (COUNT DISTINCT persons? COUNT rows? etc.)
   - Cohort/filter strategy: (ICD/source_value/keyword; include exact column to use)
   - Tables needed:
   - Joins (only via person_id):
   - Time window (if any):
step 2) Write SQL that follows the plan and hard constraints.
step 3) Call validate_sql_query with the SQL.
step 4) If validation fails: fix SQL and repeat step 3.
step 5) Only when validation passes: call get_data.
step 6) As soon as get_data returns ANY result (including 0 rows, empty, or {"total_patients":0}), STOP. Return a text response to the user. Do NOT call validate_sql_query or get_data again — even if the result is 0.

CRITICAL — RETURNING THE ANSWER:
- get_data returning 0 rows, 0 counts, or empty JSON is a VALID result. Report it to the user.
- After get_data succeeds, you MUST respond with a text message (no more tool calls). The graph only ends when you output text.
- Never retry get_data because the result seems "wrong" or zero.

FINAL RESPONSE FORMAT (after get_data returns):
Using this query I gathered that <one-sentence answer>. 
SQL: <the exact SQL used>
"""


def make_reasoner(llm_instance, tool_list=None):
    """Return a reasoner node function bound to the given LLM instance."""
    tools = tool_list if tool_list is not None else _reasoner_tools
    tools_bound = llm_instance.bind_tools(tools)

    def _reasoner(state):
        query = state["query"]
        messages = [HumanMessage(content=query)] + list(state["messages"])
        sys_msg = SystemMessage(content=_SYS_PROMPT)
        return {"messages": [tools_bound.invoke([sys_msg] + messages)]}

    return _reasoner


def reasoner(state):
    """Default reasoner using the module-level LLM (kept for notebook compat)."""
    query = state["query"]
    messages = [HumanMessage(content=query)] + list(state["messages"])
    sys_msg = SystemMessage(content=_SYS_PROMPT)
    result = [llm_with_tools.invoke([sys_msg] + messages)]
    return {"messages": result}
