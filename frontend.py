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
MODEL_OPTIONS = [
    "openai/gpt-4o-mini",
    "deepseek/deepseek-v3.2",
]

with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("AI model", MODEL_OPTIONS, index=0)
    st.caption("This controls which AI model generates the SQL. The validation and execution are unchanged.")

st.title("NCCS Data Concierge")

def render_assistant_payload(resp: dict):
    st.write(resp.get("message", ""))

    if resp.get("status") != "ok":
        st.error(resp.get("message", "Error"))
        reasons = resp.get("reasons", [])
        if reasons:
            st.code("\n".join(map(str, reasons)))
        return

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

# user input (/ask)
prompt = st.chat_input("Ask a question about the data…", disabled=st.session_state.processing)
if prompt and not st.session_state.processing:
    st.session_state.processing = True
    
    # user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.write("Thinking...")
        
        try:
            r = requests.post(
                f"{API_URL}/ask",
                json={"question": prompt, "model": selected_model},
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()

            message_placeholder.empty()
            render_assistant_payload(data)

            st.session_state.messages.append({"role": "assistant", "content": data})
            
        except Exception as e:
            message_placeholder.empty()
            error_data = {"status": "error", "message": "Backend not reachable.", "reasons": [str(e)]}
            render_assistant_payload(error_data)
            st.session_state.messages.append({"role": "assistant", "content": error_data})
        
        finally:
            st.session_state.processing = False
            
    st.rerun()