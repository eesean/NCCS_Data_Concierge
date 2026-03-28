"""
Reasoner node — Ollama simulated tool-calling.

Schema is preloaded by pipeline. The LLM does not use native function calling.
Instead, tool schemas are injected into the system prompt and the model's
plain-text output is parsed each turn to reconstruct a proper AIMessage with
tool_calls so the LangGraph ToolNode / tools_condition works without changes.
"""
import json
import re
import uuid
from typing import Optional

from retrieval.graph.tool.SQLvalidator import validate_sql_query
from retrieval.graph.tool.get_data import get_data
from retrieval.graph.tool.vectorRag import get_cancer_info
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages.tool import ToolCall

_reasoner_tools = [validate_sql_query, get_data, get_cancer_info]

_SYS_PROMPT = """
You are a SQL expert working with a DuckDB healthcare dataset.
You will be given: (a) a user question, and (b) schema context that lists the ONLY allowed tables/columns.

HARD CONSTRAINTS (never violate):
- Use ONLY table/column names explicitly present in the provided schema context. No guessing.
- Table and column names are case-sensitive.
- Join tables using `person_id` only (person.person_id = <other_table>.person_id).
- No SELECT * — always list explicit columns (e.g., COUNT(*), person_id, condition_source_value).
- Single statement only — no semicolons or multiple queries.

CANCER / ONCOLOGY QUERIES — MANDATORY get_cancer_info FIRST, THEN STRICTLY COPY ITS OUTPUT:
- If the user's question mentions cancer, tumour/tumor, oncology, a cancer site (e.g. colorectal, bowel, lung, breast, melanoma, lymphoma), or ICD codes for cancer, your VERY FIRST tool call MUST be get_cancer_info with a short query term.

AFTER get_cancer_info RETURNS (read the latest ToolMessage whose name is get_cancer_info):
- That message is the ONLY source for cancer ICD filters. Treat it as read-only text.
- Find the line starting with `SQL_FILTER:` in that message. Copy the boolean expression it gives after "use:" (the part with ICD10 LIKE ...) into your WHERE clause, character-for-character for the predicates. You may qualify the column as `condition_occurrence.ICD10` if the filter says `ICD10`.
- If there is no SQL_FILTER line, build the filter ONLY from the `ICD10_CODES:` line in that SAME message (comma-separated codes like C18, C19, C20 → use ICD10 LIKE 'C18%' OR ... exactly those prefixes). Do not add any C-code that is not listed on that ICD10_CODES line.
- Do NOT add codes from your own knowledge. Do NOT add OR-clauses for cancers not in that ToolMessage.
- FORBIDDEN unless they literally appear in both the schema context AND that ToolMessage: ICDO3, ICDO, ICD-O, OMOP, or any column name not shown in the get_cancer_info text. This dataset uses `ICD10` on `condition_occurrence` for these filters — never ICDO3.

- After SQL is built from schema + that ToolMessage only, then validate_sql_query → get_data as usual.

NON-CANCER QUERIES:
- Skip get_cancer_info. Go straight to validate_sql_query → get_data.

MANDATORY BEHAVIOR (DO NOT SKIP):
- You are a SQL execution engine. Never output conversational text except the one-sentence final answer after get_data.
- Step A (cancer only): get_cancer_info first.
- Step 1: Generate SQL. Wrap it immediately in a tool call to `validate_sql_query`.
- Step 2: Once validated, wrap the SQL in a tool call to `get_data`.
- Call validate_sql_query. If it fails, fix and repeat.
- DO NOT output a "PLAN" as text. Keep planning internal.
- DO NOT output SQL as text. SQL must only exist inside a tool call argument.
- If you have successfully executed `get_data`, output only the final one-sentence result (no SQL, no JSON).
- As soon as get_data returns ANY result (including 0 rows or {"total_patients":0}), STOP. Return a text response. Do NOT retry.

CRITICAL — RETURNING THE ANSWER:
- 0 rows and 0 counts are valid. Report them. The graph ends only when you output text (no more tool calls).

FINAL RESPONSE FORMAT (after get_data returns):
Write one plain-English sentence summarising the result.
Do NOT include raw SQL in the final answer.
"""

# IMPORTANT: Do NOT use str.format() on this template — the JSON example
# contains literal { } braces that would cause KeyError. Use plain string
# replacement for the tool_names placeholder only.
_OLLAMA_TOOL_INSTRUCTION_TEMPLATE = """
TOOL CALLING FORMAT — YOU MUST FOLLOW THIS EXACTLY:

When you want to call a tool, output ONLY this JSON on its own line.
No text before it. No text after it. No markdown fences.

{"tool_call": {"name": "<tool_name>", "args": {"<argument_key>": "<argument_value>"}}}

STEP-BY-STEP WORKFLOW (follow every step in order):

STEP 0 (ONLY if the user question is about cancer / tumours / oncology / ICD cancer codes):
{"tool_call": {"name": "get_cancer_info", "args": {"query": "<short cancer term e.g. colorectal cancer>"}}}

STEP 1 — Generate SQL (schema + if STEP 0 ran: ONLY the get_cancer_info ToolMessage text — copy SQL_FILTER / ICD10_CODES as instructed above; never ICDO3) and validate it:
{"tool_call": {"name": "validate_sql_query", "args": {"sql": "<your SQL here>"}}}

STEP 2 — As soon as validate_sql_query succeeds, YOU MUST IMMEDIATELY call get_data with the IDENTICAL SQL:
{"tool_call": {"name": "get_data", "args": {"sql": "<the same SQL you validated>"}}}

You may use "args" OR "arguments" as the key — both are valid. Example:
{"name": "get_data", "arguments": {"sql": "SELECT ..."}}

STEP 3 — After get_data returns results, output your plain-text final answer (no JSON).

CRITICAL RULES:
- Valid tool names: TOOL_NAMES_PLACEHOLDER
- One tool call per response — no extra text in the same response.
- NEVER skip Step 2 after a successful validation. Your next output MUST be get_data.
- If validate_sql_query fails, fix the SQL and repeat Step 1.
- Only output plain text (no JSON) when you are giving the final answer after get_data.
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


_KNOWN_TOOL_NAMES = {"validate_sql_query", "get_data", "get_cancer_info"}


def _coerce_tool_args(payload: dict) -> dict:
    """Qwen sometimes emits 'arguments' (OpenAI style) instead of 'args'."""
    args = payload.get("args")
    if args is None:
        args = payload.get("arguments")
    if isinstance(args, dict):
        return args
    return {}


def _parse_tool_call(text: str) -> Optional[dict]:
    """
    Detect and parse a tool-call JSON block from raw LLM output.

    Handles three common formats that Qwen2.5 and similar models produce:
      1. {"tool_call": {"name": "...", "args": {...}}}          (preferred)
      2. {"name": "...", "args": {...}}                          (shorthand)
      3. Markdown-fenced variants of the above (```json ... ```)

    Returns {"name": ..., "args": ...} or None.

    Also accepts "arguments" as an alias for "args" (common with Qwen).
    """
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", " ", text)

    candidates = []
    for match in re.finditer(r'\{', cleaned):
        start = match.start()
        depth = 0
        for i, ch in enumerate(cleaned[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidates.append(cleaned[start:i + 1])
                    break

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue

        # Format 1: {"tool_call": {"name": "...", "args" or "arguments": {...}}}
        tc = data.get("tool_call", {})
        if isinstance(tc, dict) and "name" in tc:
            return {"name": tc["name"], "args": _coerce_tool_args(tc)}

        # Format 2: {"name": "...", "args" or "arguments": {...}} — direct shorthand
        if isinstance(data.get("name"), str) and data["name"] in _KNOWN_TOOL_NAMES:
            args = _coerce_tool_args(data)
            if args or "args" in data or "arguments" in data:
                return {"name": data["name"], "args": args}

    # Format 3: bare "SQL: <query>" — model forgot the tool call wrapper.
    # Promote it into a validate_sql_query call so the graph continues.
    sql_line = re.search(r"(?i)^sql:\s*(.+)", text.strip(), re.MULTILINE)
    if sql_line:
        sql = sql_line.group(1).strip().rstrip(";")
        if sql:
            return {"name": "validate_sql_query", "args": {"sql": sql}}

    return None


def make_ollama_reasoner(llm_instance, tool_list=None):
    """
    Return a LangGraph reasoner node for a local Ollama model.

    Each turn:
    - Tool-call JSON detected  → AIMessage(tool_calls=[...]) → ToolNode
    - No tool-call JSON found  → AIMessage as-is             → END
    """
    tools = tool_list if tool_list is not None else _reasoner_tools
    tool_names = ", ".join(t.name for t in tools)
    tools_schema_str = _format_tools_for_prompt(tools)
    tool_instruction = _OLLAMA_TOOL_INSTRUCTION_TEMPLATE.replace(
        "TOOL_NAMES_PLACEHOLDER", tool_names
    )

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

        print(f"\n[REASONER RAW OUTPUT]\n{raw_text}\n[END REASONER OUTPUT]\n")
        tool_call_data = _parse_tool_call(raw_text)
        print(f"[REASONER PARSED TOOL CALL] {tool_call_data}\n")
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

        return {"messages": [response]}

    return _ollama_reasoner
