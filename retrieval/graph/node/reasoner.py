"""
Reasoner node: LLM with tool binding. Schema is preloaded by pipeline; LLM uses validate_sql_query, get_data, and get_cancer_info when relevant.

For Ollama models without native tool calling, use make_ollama_reasoner() instead of
make_reasoner(). It injects tool schemas into the system prompt and parses the model's
text output to reconstruct a proper AIMessage with tool_calls, allowing the unchanged
LangGraph ToolNode / tools_condition to continue routing correctly.
"""
import json
import re
import uuid
from typing import Optional
from retrieval.llm import llm
from retrieval.graph.tool.SQLvalidator import validate_sql_query
from retrieval.graph.tool.get_data import get_data
from retrieval.graph.tool.vectorRag import get_cancer_info
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages.tool import ToolCall

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


# ---------------------------------------------------------------------------
# Ollama / non-native-tool-calling path
# ---------------------------------------------------------------------------

_OLLAMA_TOOL_INSTRUCTION = """
TOOL CALLING FORMAT (MANDATORY — read carefully):
You do NOT have native function calling. Instead, when you want to call a tool, output
ONLY the following JSON block on its own line — no surrounding text, no markdown fences:

{"tool_call": {"name": "<tool_name>", "args": {<argument_key>: <argument_value>}}}

Rules:
- One tool call per response. After the JSON block, output nothing else.
- Valid tool names: {tool_names}
- When you have already received data from get_data and are ready to give the final answer,
  output your plain-text response (no JSON). That signals the end of the loop.
- Never mix tool-call JSON with explanatory text in the same response.
"""


def _format_tools_for_prompt(tools: list) -> str:
    """Serialize tool name, description, and parameter schema into a plain-text block."""
    sections = []
    for t in tools:
        schema = {}
        if hasattr(t, "args_schema") and t.args_schema is not None:
            try:
                raw = t.args_schema.schema()
                schema = raw.get("properties", {})
            except Exception:
                pass
        params_str = json.dumps(schema, indent=2) if schema else "{}"
        sections.append(
            f"Tool name : {t.name}\n"
            f"Description: {t.description}\n"
            f"Parameters : {params_str}"
        )
    return "\n\n".join(sections)


def _parse_tool_call(text: str) -> Optional[dict]:
    """
    Detect and parse a tool-call JSON block from raw LLM text.

    Accepts both fenced (```json ... ```) and bare JSON containing
    {"tool_call": {"name": ..., "args": {...}}}.
    Returns a dict with keys 'name' and 'args', or None if not found.
    """
    # Strip optional markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", " ", text)

    # Find the outermost {...} that contains "tool_call"
    for match in re.finditer(r'\{', cleaned):
        start = match.start()
        depth = 0
        for i, ch in enumerate(cleaned[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start:i + 1]
                    try:
                        data = json.loads(candidate)
                        tc = data.get("tool_call", {})
                        if isinstance(tc, dict) and "name" in tc:
                            return {"name": tc["name"], "args": tc.get("args", {})}
                    except (json.JSONDecodeError, AttributeError):
                        pass
                    break
    return None


def make_ollama_reasoner(llm_instance, tool_list=None):
    """
    Return a reasoner node compatible with Ollama models that lack native tool calling.

    The node injects tool schemas and a structured output instruction into the system
    prompt, invokes the LLM, then parses the response:
    - Tool-call JSON detected  → reconstructs AIMessage(tool_calls=[...]) so that
      tools_condition routes to the ToolNode as normal.
    - No tool-call JSON found  → returns the AIMessage as-is (final answer → END).
    """
    tools = tool_list if tool_list is not None else _reasoner_tools
    tool_names = ", ".join(t.name for t in tools)
    tools_schema_str = _format_tools_for_prompt(tools)
    tool_instruction = _OLLAMA_TOOL_INSTRUCTION.format(tool_names=tool_names)

    ollama_sys_prompt = (
        _SYS_PROMPT
        + "\n\n"
        + tool_instruction
        + "\n\nAVAILABLE TOOLS:\n"
        + tools_schema_str
    )

    def _ollama_reasoner(state):
        query = state["query"]
        messages = [HumanMessage(content=query)] + list(state["messages"])
        sys_msg = SystemMessage(content=ollama_sys_prompt)

        response = llm_instance.invoke([sys_msg] + messages)
        raw_text = response.content if hasattr(response, "content") else str(response)

        tool_call_data = _parse_tool_call(raw_text)
        if tool_call_data:
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name=tool_call_data["name"],
                        args=tool_call_data["args"],
                        id=str(uuid.uuid4())[:8],
                    )
                ],
            )
            return {"messages": [ai_msg]}

        # No tool call — treat as final answer; graph routes to END
        return {"messages": [response]}

    return _ollama_reasoner
