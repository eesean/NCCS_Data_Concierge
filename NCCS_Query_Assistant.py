import html
import json
import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="NCCS Data Concierge", layout="wide")

st.markdown(
    """
    <style>
    button.sample-q-copy, .sample-q-copy {
        color: #1f2937 !important;
        background: #e8eaed !important;
        border: 1px solid #9aa0a6 !important;
        -webkit-text-fill-color: #1f2937 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

## session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False

SAMPLE_TEMPLATES = [
    "What is the total number of {entity} diagnosed with {disease}{optional_filters}?",
    "What is the total number of {entity} diagnosed with {disease}, grouped by {dimension}{optional_filters}?",
    "What is the {aggregation_function} of {measure} for patients diagnosed with {disease}{optional_filters}?",
]

## sidebar
_MODEL_OPTIONS = [
    "qwen2.5:7b",
    "qwen3:8b",
    "deepseek-v3",
    "llama3.1",
]

_MODEL_DESCRIPTIONS = {
    "qwen2.5:7b": "Lightweight and fast. Good for straightforward queries like counts and filters.",
    "qwen3:8b": "Strong native tool-calling support. Good for multi-step and cancer queries.",
    "deepseek-v3": "Strong reasoning and SQL generation. Good for complex analytical queries.",
    "llama3.1": "General-purpose model. Good baseline for standard queries.",
}

with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox(
        "Ollama model",
        options=_MODEL_OPTIONS,
        index=0,
        help="Select a model you have pulled via `ollama pull <tag>`. All models must support tool calling.",
    )
    st.caption(
        f"Pull this model if not yet installed: `ollama pull {selected_model}`"
    )
    desc = _MODEL_DESCRIPTIONS.get(selected_model, "")
    if desc:
        st.info(desc)
    with st.expander("Add a custom model"):
        custom_model = st.text_input(
            "Custom Ollama tag",
            placeholder="e.g. qwen3:14b",
            help="Enter any Ollama model tag you have pulled locally.",
        )
        if custom_model.strip():
            selected_model = custom_model.strip()
            st.caption(f"Using custom model: `{selected_model}`")

st.title("NCCS Data Concierge")

_TOOL_LABELS = {
    "get_schema_context": "Fetching schema context",
    "get_sql_template": "Fetching SQL templates",
    "get_cancer_info": "Looking up cancer ICD-10 codes",
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
        err_msg = resp.get("message", "An error occurred.")
        reasons = resp.get("reasons", [])
        st.error(f"**Error:** {err_msg}")
        if reasons:
            st.code("\n".join(str(r) for r in reasons), language="text")
        st.info("💡 **Try again:** Rephrase your question or check that Ollama is running.")
        return

    st.write(resp.get("message", ""))

    if "metric" in resp and "value" in resp:
        metric = str(resp["metric"])
        value = resp["value"]
        if isinstance(value, str) and len(value) > 60:
            st.write(value)
        else:
            st.metric(metric, value)
        final_sql = resp.get("final_sql", "")
        if final_sql and final_sql != "No SQL found":
            with st.expander("SQL executed", expanded=False):
                st.code(final_sql, language="sql")
        return

    if resp.get("columns") and resp.get("rows") is not None:
        df = pd.DataFrame(resp["rows"], columns=resp["columns"])
        st.dataframe(df, use_container_width=True, hide_index=True)

    final_sql = resp.get("final_sql", "")
    if final_sql and final_sql != "No SQL found":
        with st.expander("SQL executed", expanded=False):
            st.code(final_sql, language="sql")

# for chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant" and isinstance(m.get("content"), dict):
            render_assistant_payload(m["content"])
        else:
            st.write(m["content"])

# sample questions — expander sits naturally above chat input, always aligned
with st.expander("Sample questions", expanded=False):
    buttons_html = ""
    for t in SAMPLE_TEMPLATES:
        data_attr = html.escape(t).replace('"', "&quot;")
        display_label = html.escape(t).replace("{", "&#123;").replace("}", "&#125;")
        buttons_html += (
            f'<button type="button" data-text="{data_attr}"'
            f' onclick="navigator.clipboard.writeText(this.getAttribute(\'data-text\'));'
            f'const L=this.textContent;this.textContent=\'Copied!\';setTimeout(()=>this.textContent=L,1500)"'
            f' style="margin:4px 8px 4px 0;padding:6px 12px;border-radius:4px;border:1px solid #888;'
            f'background:#c4c8cc;color:#111;cursor:pointer;font-size:0.9em;font-family:inherit;">'
            f"{display_label}</button>"
        )
    iframe_html = (
        f'<!DOCTYPE html><html><head><meta charset="utf-8"></head>'
        f'<body style="margin:0;padding:8px;background:transparent;color:#111;">'
        f"{buttons_html}</body></html>"
    )
    st.components.v1.html(iframe_html, height=85, scrolling=False)

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
                json={"question": prompt, "model": selected_model, "history": st.session_state.messages},
                stream=True,
                timeout=180,
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
            final_data = {
                "status": "error",
                "message": "Backend not reachable.",
                "reasons": [str(e)],
                "steps": accumulated_steps,
            }
            with placeholder.container():
                render_assistant_payload(final_data)

        finally:
            st.session_state.processing = False

        if final_data:
            st.session_state.messages.append({"role": "assistant", "content": final_data})

    st.rerun()