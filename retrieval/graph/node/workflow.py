"""
ReAct agent: LLM decides when to call get_schema_context, then generates SQL.
Requires a model with tool calling (e.g. arcee-ai/trinity-large-preview:free on OpenRouter).
"""
from functools import lru_cache

from retrieval.graph.tool.vectorRag import get_schema_context
from retrieval.graph.tool.SQLvalidator import validate_sql_query
from retrieval.graph.tool.get_data import get_data
from retrieval.graph.node.reasoner import reasoner, make_reasoner
from retrieval.llm import get_llm, DEFAULT_MODEL
from typing import Annotated, TypedDict
import operator
from langchain_core.messages import AnyMessage

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode


class GraphState(TypedDict):
    """State of the graph."""
    query: str
    finance: str
    final_answer: str
    messages: Annotated[list[AnyMessage], operator.add]


tool_list = [get_schema_context, validate_sql_query, get_data]

# Module-level graph kept for notebook / backward-compat imports
workflow = StateGraph(GraphState)
workflow.add_node("reasoner", reasoner)
workflow.add_node("tools", ToolNode(tool_list))
workflow.add_edge(START, "reasoner")
workflow.add_conditional_edges("reasoner", tools_condition)
workflow.add_edge("tools", "reasoner")
react_graph = workflow.compile()


@lru_cache(maxsize=8)
def build_graph(model_name: str = DEFAULT_MODEL):
    """Build and cache a ReAct graph for the given model name."""
    llm_instance = get_llm(model_name)
    reasoner_fn = make_reasoner(llm_instance)

    wf = StateGraph(GraphState)
    wf.add_node("reasoner", reasoner_fn)
    wf.add_node("tools", ToolNode(tool_list))
    wf.add_edge(START, "reasoner")
    wf.add_conditional_edges("reasoner", tools_condition)
    wf.add_edge("tools", "reasoner")
    return wf.compile()
