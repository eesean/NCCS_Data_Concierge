import json
import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="NCCS Data Concierge", layout="wide")

## session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False

## sidebar (AI model selection)
# Only models with tool-calling support work. See: https://openrouter.ai/collections/tool-calling-models
MODEL_OPTIONS = [
    "arcee-ai/trinity-large-preview:free",
    "stepfun/step-3.5-flash:free",
    "z-ai/glm-4.5-air:free",
    "nvidia/nemotron-3-nano-30b-a3b:free"
]

MODEL_DESCRIPTIONS = {
    "arcee-ai/trinity-large-preview:free": (
        "Good for relatively complex queries as it has stronger reasoning performance and is more reliable at multi-step SQL planning."
    ),
    "stepfun/step-3.5-flash:free": (
        "Good for schema-heavy queries where there might be more tables and columns as it is optimized for speed and long-context."
    ),
    "z-ai/glm-4.5-air:free": (
        "Good for multi-step workflows as it is designed for agentic use and typically handles tool usage and structured steps more consistently."
    ),
    "nvidia/nemotron-3-nano-30b-a3b:free": (
        "Good for relatively straightforward queries like counts or filters as it is an efficient model that has good latency."
    ),
}
# Build display labels (what users see)
MODEL_LABELS = {
    m: f"{m} — {MODEL_DESCRIPTIONS.get(m, '')}"
    for m in MODEL_OPTIONS
}

with st.sidebar:
    st.header("Settings")
    # model selection
    selected_model = st.selectbox("AI model", MODEL_OPTIONS, index=0)

    st.caption(
        "Only models with tool-calling support work. Add more from openrouter.ai/collections/tool-calling-models"
    )

    # model guide with descriptions visible 
    st.markdown("### Model guide")
    for m in MODEL_OPTIONS:
        desc = MODEL_DESCRIPTIONS.get(m, "")
        st.markdown(
            f"""
            <div style="margin: 6px 0 10px 0;">
              <div style="font-weight: 600;">{m}</div>
              <div style="color: #999; font-size: 0.85em; line-height: 1.25;">
                {desc}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.title("NCCS Data Concierge")

_TOOL_LABELS = {
    "get_schema_context": "Fetching schema context",
    "validate_sql_query": "Validating SQL",
    "get_data": "Executing query",
}

def _render_steps(steps: list):
    if not steps:
        return
    for step in steps:
        if step["kind"] == "call":
            label = _TOOL_LABELS.get(step["tool"], step["tool"])
            st.markdown(
                f'<div style="color:#999;font-size:0.82em;padding:2px 0 2px 10px;'
                f'border-left:2px solid #ddd;margin:2px 0">⚙ {label}…</div>',
                unsafe_allow_html=True,
            )
        elif step["kind"] == "result":
            snippet = step["snippet"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", " ")
            st.markdown(
                f'<div style="color:#bbb;font-size:0.78em;padding:2px 0 2px 22px;'
                f'border-left:2px solid #eee;margin:2px 0;font-family:monospace">{snippet}</div>',
                unsafe_allow_html=True,
            )


def render_assistant_payload(resp: dict):
    _render_steps(resp.get("steps", []))

    if resp.get("status") != "ok":
        st.error("**Error**")
        st.write(resp.get("message", "An error occurred."))
        if resp.get("reasons"):
            st.code("\n".join(resp["reasons"]), language="text")
        st.info("💡 **Try again:** Use a different model or simplify your question.")
        return

    st.write(resp.get("message", ""))

    # if scalar response (for COUNT)
    if "metric" in resp and "value" in resp:
        metric = str(resp["metric"])
        value = resp["value"]

        # if it is just long text 
        if isinstance(value, str) and len(value) > 60:
            st.write(value)
        else:
            st.metric(metric, value)

        return  

    # if table response
    if resp.get("columns") and resp.get("rows") is not None:
        df = pd.DataFrame(resp["rows"], columns=resp["columns"])
        st.dataframe(df, use_container_width=True, hide_index=True)

# for chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant" and isinstance(m.get("content"), dict):
            render_assistant_payload(m["content"])
        else:
            st.write(m["content"])

# user input (/ask/stream)
prompt = st.chat_input("Ask a question about the data…", disabled=st.session_state.processing)
if prompt and not st.session_state.processing:
    st.session_state.processing = True

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.write("Thinking…")

        accumulated_steps = []
        final_data = None

        try:
            r = requests.post(
                f"{API_URL}/ask/stream",
                json={"question": prompt, "model": selected_model},
                stream=True,
                timeout=300,
            )
            r.raise_for_status()

            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue

                event = json.loads(line[6:])
                evt_type = event.get("type")

                if evt_type == "step_call":
                    accumulated_steps.append({"kind": "call", "tool": event["tool"]})
                    with placeholder.container():
                        _render_steps(accumulated_steps)

                elif evt_type == "step_result":
                    accumulated_steps.append({"kind": "result", "tool": event["tool"], "snippet": event["snippet"]})
                    with placeholder.container():
                        _render_steps(accumulated_steps)

                elif evt_type in ("done", "error"):
                    event.pop("type")
                    event["steps"] = accumulated_steps
                    final_data = event
                    with placeholder.container():
                        render_assistant_payload(final_data)
                    break

        except Exception as e:
            final_data = {"status": "error", "message": "Backend not reachable.", "reasons": [str(e)], "steps": accumulated_steps}
            with placeholder.container():
                render_assistant_payload(final_data)

        finally:
            st.session_state.processing = False

        if final_data:
            st.session_state.messages.append({"role": "assistant", "content": final_data})

    st.rerun()