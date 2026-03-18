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
- Join tables using `person_id` only (person.person_id = <other_table>.person_id).
- No SELECT * — always list explicit columns (e.g., COUNT(*), person_id, condition_source_value).
- Single statement only — no semicolons or multiple queries.
- Type rules:
  - *_concept_id columns are INTEGER → use integer literals (e.g., 123), never quoted strings.
  - ICD10, ICDO3, *_source_value columns are VARCHAR → use quoted strings (e.g., 'C34', 'C34%', '%cancer%').
  - Dates: use date literals or CAST; match column types exactly in WHERE.

COUNTING SEMANTICS (critical for correct answers):
- "How many patients/people?" → COUNT(DISTINCT person_id)
- "How many records/occurrences/conditions/drugs?" → COUNT(*)
- Never add LIMIT to count queries. For breakdowns (GROUP BY), include LIMIT if returning many rows.

MANDATORY FLOW (do not skip):
1) PLAN (3–6 lines): Metric, cohort filter (exact column), tables, joins, time window.
2) Write SQL. Pass raw SQL only to validate_sql_query — no markdown, no ```sql```.
3) Call validate_sql_query. If it fails, fix and repeat.
4) When validation passes, call get_data.
5) As soon as get_data returns ANY result (including 0 rows or {"total_patients":0}), STOP. Return a text response. Do NOT retry.

CRITICAL — RETURNING THE ANSWER:
- 0 rows and 0 counts are valid. Report them. The graph ends only when you output text (no more tool calls).

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
