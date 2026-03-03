"""
Reasoner node: LLM with tool binding. Decides when to call get_schema_context.
Trinity (arcee-ai/trinity-large-preview:free) supports tool calling via OpenRouter.
"""
from retrieval.llm import llm
from retrieval.graph.tool.vectorRag import get_schema_context
from retrieval.graph.tool.SQLvalidator import validate_sql_query
from retrieval.graph.tool.get_data import get_data
from langchain_core.messages import SystemMessage, HumanMessage

tool_list = [get_schema_context, validate_sql_query, get_data]
llm_with_tools = llm.bind_tools(tool_list)

_SYS_PROMPT = """You are a SQL expert working with a DuckDB healthcare dataset.
                Follow these steps exactly, in order, and do not repeat any step:
                1. Call get_schema_context once to retrieve the relevant schema.
                2. Read the schema carefully. Copy the EXACT table name and EXACT column names as they appear — do NOT rename, abbreviate, or invent column names. If no column matches what you need, use the closest one that exists.
                3. Call validate_sql_query once with your SQL. If it reports safety issues (DISALLOWED_TABLES or DISALLOWED_COLUMNS), look at the schema again and fix only the table/column names. If it says the SQL is valid or to proceed, move on immediately — do not call it again.
                4. Call get_data once with the validated SQL to retrieve the results.
                5. Present the results to the user and stop. Do not call any more tools after get_data.
                Table and column names are case-sensitive. Join tables using person_id."""


def make_reasoner(llm_instance):
    """Return a reasoner node function bound to the given LLM instance."""
    tools_bound = llm_instance.bind_tools(tool_list)

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
