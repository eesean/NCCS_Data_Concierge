# NCCS Data Concierge — Ollama / Simulated Tool-Calling Edition

This document covers what changed from the original version and how to set up and run the application under the new Ollama-compatible inference path. Read `README.md` first for the baseline setup.

---

## What is different in this version

### Problem with the original version

The original graph uses LangChain's `bind_tools()` to attach tool schemas to the LLM and relies on the model returning a structured `tool_calls` field in its response. This works for OpenRouter-hosted models that support native function calling (OpenAI-format). Local **Ollama** models that do not advertise function-calling support produce plain text responses — `tool_calls` is always empty, so the LangGraph `tools_condition` gate always routes to END, breaking the entire ReAct loop after the first step.

### Solution — simulated tool-calling wrapper

A new inference path was added that does not rely on native function calling at all. The key components are:

| Component | Location | Role |
|-----------|----------|------|
| `get_ollama_llm()` | `retrieval/llm.py` | Factory that returns either a real `ChatOllama` instance or an OpenRouter `ChatOpenAI` in simulation mode (no `extra_body` / `require_parameters`) |
| `_format_tools_for_prompt()` | `retrieval/graph/node/reasoner.py` | Serializes each tool's name, description, and JSON parameter schema into plain text for injection into the system prompt |
| `_parse_tool_call()` | `retrieval/graph/node/reasoner.py` | Brace-matching + JSON parser that extracts `{"tool_call": {"name": "...", "args": {...}}}` from raw LLM text output |
| `make_ollama_reasoner()` | `retrieval/graph/node/reasoner.py` | Builds a reasoner graph node: injects tool schemas into the system prompt, calls the LLM, parses the text response, and reconstructs a proper `AIMessage` with `tool_calls` so the downstream `ToolNode` and `tools_condition` continue to work unchanged |
| `build_graph()` — extended | `retrieval/graph/node/workflow.py` | Accepts an optional `ollama_base_url` parameter; routes to the simulated path when set, OpenRouter native path when `None` |
| `stream_question_agent()` — extended | `pipeline.py` | Reads `OLLAMA_BASE_URL` from the environment and threads it through to `build_graph()` |

### What is unchanged

- `ToolNode`, `tools_condition`, `GraphState`, and all graph edges in `workflow.py`
- All three tool implementations (`validate_sql_query`, `get_data`, `get_cancer_info`)
- The output parser (`outputParser.py`)
- All SSE streaming logic in `pipeline.py`
- The FastAPI backend (`api.py`)
- The Streamlit frontend (`NCCS_Query_Assistant.py`)
- The original `make_reasoner()` / `get_llm()` path for OpenRouter models with native tool calling

### How the simulated loop works

```
User question
      │
      ▼
pipeline.py  reads OLLAMA_BASE_URL → passes ollama_base_url to build_graph()
      │
      ▼
workflow.py  build_graph(model, ollama_base_url="simulate")
             → get_ollama_llm()   returns ChatOpenAI (OpenRouter, no extra_body)
             → make_ollama_reasoner()  builds the reasoner node
      │
      ▼
[LangGraph ReAct loop]

  reasoner node (make_ollama_reasoner):
    1. Injects tool schemas + JSON output format into system prompt
    2. Calls LLM — model returns plain text (no native tool_calls)
    3. _parse_tool_call() extracts {"tool_call": {"name": ..., "args": ...}}
    4. Reconstructs AIMessage(tool_calls=[ToolCall(...)])
       → tools_condition routes to ToolNode ✓
    5. If no tool call detected → final answer → tools_condition routes to END ✓

  ToolNode executes the tool (DuckDB / vector RAG / SQL validator)
  Result is appended as ToolMessage → back to reasoner node → repeat
```

---

## Prerequisites

Same as `README.md`, plus:

- An **OpenRouter API key** (for both native and simulation modes)
- **Ollama** (only required for the real local model path — not needed for simulation mode)

---

## Environment setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate
```

### 2. Install dependencies

`langchain-ollama` was added to `requirements.txt` in this version:

```bash
pip install -r requirements.txt
```

### 3. Configure the `.env` file

Create a `.env` file in the project root. Choose one of the two modes below.

#### Mode A — Simulation (OpenRouter backend, no Ollama needed)

Use this if you do not have Ollama installed locally. The simulated tool-calling wrapper runs with an OpenRouter model as the backend.

```
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Triggers the simulated tool-calling path
OLLAMA_BASE_URL=simulate

# Any OpenRouter model slug — does NOT need to support native tool calling
OLLAMA_DEFAULT_MODEL=arcee-ai/trinity-large-preview:free

PARQUET_KEY_NAME=key128
PARQUET_KEY_VALUE=MachuPicchu&K2BC
```

#### Mode B — Real local Ollama

Use this once Ollama is installed and a model is downloaded locally.

```
OPENROUTER_API_KEY=your_openrouter_api_key_here

# URL of the local Ollama server
OLLAMA_BASE_URL=http://localhost:11434

# Tag of the locally-pulled model (e.g. llama3.1, qwen2.5, mistral)
OLLAMA_DEFAULT_MODEL=llama3.1

PARQUET_KEY_NAME=key128
PARQUET_KEY_VALUE=MachuPicchu&K2BC
```

Pull a model before starting the server:

```bash
ollama pull llama3.1
```

Start the Ollama server (runs in the background):

```bash
ollama serve
```

#### Mode C — Original OpenRouter native tool-calling (unchanged)

Leave `OLLAMA_BASE_URL` unset (or remove it entirely). The application behaves identically to the original version.

```
OPENROUTER_API_KEY=your_openrouter_api_key_here
PARQUET_KEY_NAME=key128
PARQUET_KEY_VALUE=MachuPicchu&K2BC
```

---

## Running the application

Use two separate terminal windows. Activate the virtual environment in each.

### Terminal 1 — Backend (FastAPI)

```bash
source venv/bin/activate        # Windows: .\venv\Scripts\activate
uvicorn api:app
```

The API will be available at `http://127.0.0.1:8000`.

To verify it is running:

```bash
curl http://127.0.0.1:8000/docs
```

### Terminal 2 — Frontend (Streamlit)

```bash
source venv/bin/activate        # Windows: .\venv\Scripts\activate
streamlit run NCCS_Query_Assistant.py
```

The frontend will open automatically in your browser at `http://localhost:8501`.

> Start the backend first. The frontend connects to the backend on every query and will show a "Backend not reachable" error if the API is not up.

---

## Choosing a model at runtime

The sidebar model selector in the frontend lists OpenRouter models. In simulation mode (`OLLAMA_BASE_URL=simulate`) any model selected in the sidebar is passed as-is to `get_ollama_llm()`, which routes it to OpenRouter **without** native tool-calling parameters. The simulated tool-calling wrapper handles everything in the prompt.

In real Ollama mode the sidebar selection is overridden by `OLLAMA_DEFAULT_MODEL` in `.env` because the locally-served model tag must match what was pulled with `ollama pull`.

---

## Switching modes without restarting

The mode is read from the environment at startup. To switch:

1. Edit `.env` (change `OLLAMA_BASE_URL`).
2. Restart the backend (`Ctrl+C` in Terminal 1, then `uvicorn api:app` again).
3. The frontend does not need to be restarted.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Agent returns an answer immediately with no data | Simulated tool call JSON was not parsed — model did not follow the output format | Try a more instruction-following model; check that `OLLAMA_BASE_URL=simulate` is set |
| `langchain_ollama` import error | Package not installed | `pip install langchain-ollama` |
| Connection refused on port 11434 | Ollama server not running | `ollama serve` in a separate terminal |
| `OPENROUTER_API_KEY` not found | `.env` not loaded or key missing | Confirm `.env` is in the project root and the key is correct |
| Frontend shows "Backend not reachable" | FastAPI server not started | Start `uvicorn api:app` first |
