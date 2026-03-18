"""
Reasoner node: LLM with tool binding. Schema is preloaded by pipeline; LLM uses validate_sql_query, get_data, and get_cancer_info when relevant.
"""
from retrieval.llm import llm
from retrieval.graph.tool.SQLvalidator import validate_sql_query
from retrieval.graph.tool.get_data import get_data
from retrieval.graph.tool.vectorRag import get_cancer_info
from langchain_core.messages import SystemMessage, HumanMessage

# Schema is fetched by pipeline before graph; reasoner never has get_schema_context
_reasoner_tools = [validate_sql_query, get_data, get_cancer_info]
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

CANCER QUERIES (call get_cancer_info when relevant):
- If the user asks about cancer types, diagnoses, or ICD codes (e.g. colorectal, lung, breast, melanoma), call `get_cancer_info` first with the cancer/diagnosis term.
- Use the returned ICD-10 codes (e.g. C18%, C19%, C20% for colorectal) in your SQL WHERE clause when filtering condition_occurrence.ICD10.

MANDATORY BEHAVIOR(DO NOT SKIP):
- You are a SQL execution engine. Never output conversational text.
- Step 1: Generate SQL. Wrap it immediately in a tool call to `validate_sql_query`.
- Step 2: Once validated, wrap the SQL in a tool call to `get_data`.
- Call validate_sql_query. If it fails, fix and repeat.
- DO NOT output a "PLAN" as text. Keep planning internal.
- DO NOT output SQL as text. SQL must only exist inside a tool call argument.
- If you have successfully executed `get_data`, output the final one-sentence result and the SQL used.
- As soon as get_data returns ANY result (including 0 rows or {"total_patients":0}), STOP. Return a text response. Do NOT retry.

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
