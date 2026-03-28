"""
ReAct agent — local Ollama inference only.

Schema is passed in via initial messages (fetched by pipeline before the graph).
Simulated tool calling is used: tool schemas are injected into the system
prompt and the model's plain-text output is parsed to reconstruct tool_calls,
so ToolNode and tools_condition work without modification.
"""
from functools import lru_cache
from typing import Annotated, TypedDict
import operator

from langchain_core.messages import AnyMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from retrieval.graph.tool.SQLvalidator import validate_sql_query
from retrieval.graph.tool.get_data import get_data
from retrieval.graph.tool.vectorRag import get_cancer_info
from retrieval.graph.node.reasoner import make_ollama_reasoner
from retrieval.llm import get_ollama_llm, OLLAMA_DEFAULT_MODEL


class GraphState(TypedDict):
    """State of the graph."""
    query: str
    finance: str
    final_answer: str
    messages: Annotated[list[AnyMessage], operator.add]


reasoner_tool_list = [validate_sql_query, get_data, get_cancer_info]


@lru_cache(maxsize=16)
def build_graph(model_name: str = OLLAMA_DEFAULT_MODEL):
    """Build and cache a ReAct graph for the given local Ollama model tag."""
    llm_instance = get_ollama_llm(model_name)
    reasoner_fn = make_ollama_reasoner(llm_instance, tool_list=reasoner_tool_list)

    wf = StateGraph(GraphState)
    wf.add_node("reasoner", reasoner_fn)
    wf.add_node("tools", ToolNode(reasoner_tool_list))
    wf.add_edge(START, "reasoner")
    wf.add_conditional_edges("reasoner", tools_condition)
    wf.add_edge("tools", "reasoner")
    return wf.compile()


# Module-level graph for notebook / backward-compat imports
react_graph = build_graph(OLLAMA_DEFAULT_MODEL)
