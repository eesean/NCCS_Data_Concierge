"""
ReAct agent: Schema is passed in via initial messages (fetched by pipeline before graph).
LLM uses validate_sql_query, get_data, and get_cancer_info (when user asks about cancer).

build_graph() accepts an optional ollama_base_url parameter:
- None (default) → OpenRouter path using ChatOpenAI + native bind_tools
- URL string      → Ollama path using ChatOllama + simulated tool calling
"""
from functools import lru_cache
from typing import Annotated, Optional, TypedDict
import operator

from langchain_core.messages import AnyMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from retrieval.graph.tool.SQLvalidator import validate_sql_query
from retrieval.graph.tool.get_data import get_data
from retrieval.graph.tool.vectorRag import get_cancer_info
from retrieval.graph.node.reasoner import make_reasoner, make_ollama_reasoner
from retrieval.llm import get_llm, get_ollama_llm, DEFAULT_MODEL


class GraphState(TypedDict):
    """State of the graph."""
    query: str
    finance: str
    final_answer: str
    messages: Annotated[list[AnyMessage], operator.add]


reasoner_tool_list = [validate_sql_query, get_data, get_cancer_info]


# Module-level graph kept for notebook / backward-compat imports
workflow = StateGraph(GraphState)
workflow.add_node("reasoner", make_reasoner(get_llm(), tool_list=reasoner_tool_list))
workflow.add_node("tools", ToolNode(reasoner_tool_list))
workflow.add_edge(START, "reasoner")
workflow.add_conditional_edges("reasoner", tools_condition)
workflow.add_edge("tools", "reasoner")
react_graph = workflow.compile()


@lru_cache(maxsize=16)
def build_graph(model_name: str = DEFAULT_MODEL, ollama_base_url: Optional[str] = None):
    """
    Build and cache a ReAct graph for the given model.

    Args:
        model_name:      Model identifier. For Ollama this is the local model tag
                         (e.g. 'llama3.1', 'qwen2.5'); for OpenRouter use the
                         provider/model slug.
        ollama_base_url: When provided, routes to the Ollama path (simulated tool
                         calling). Leave as None to use the OpenRouter / native
                         tool-calling path.
    """
    if ollama_base_url:
        llm_instance = get_ollama_llm(model_name, ollama_base_url)
        reasoner_fn = make_ollama_reasoner(llm_instance, tool_list=reasoner_tool_list)
    else:
        llm_instance = get_llm(model_name)
        reasoner_fn = make_reasoner(llm_instance, tool_list=reasoner_tool_list)

    wf = StateGraph(GraphState)
    wf.add_node("reasoner", reasoner_fn)
    wf.add_node("tools", ToolNode(reasoner_tool_list))
    wf.add_edge(START, "reasoner")
    wf.add_conditional_edges("reasoner", tools_condition)
    wf.add_edge("tools", "reasoner")
    return wf.compile()
