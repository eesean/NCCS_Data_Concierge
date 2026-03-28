# NCCS Data Concierge

A natural language interface for querying healthcare data via SQL generation and validation.

## Prerequisites

- Python 3.9+
- Virtual environment (recommended)
- [Ollama](https://ollama.com) with at least one model pulled (default configuration uses local inference)

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install [Ollama](https://ollama.com/download), start it, and pull a model (example):
   ```bash
   ollama pull qwen2.5:7b
   ```

4. Create a `.env` file in the project root. **Inference defaults to local Ollama** at `http://localhost:11434` (no extra variable needed). Set the model tag to match `ollama list`:
   ```
   OLLAMA_DEFAULT_MODEL=qwen2.5:7b
   PARQUET_KEY_NAME=key128
   PARQUET_KEY_VALUE=MachuPicchu&K2BC
   ```

   Optional — use OpenRouter instead of local Ollama:
   ```
   OLLAMA_BASE_URL=openrouter
   OPENROUTER_API_KEY=your_api_key_here
   ```

   Optional — `OPENROUTER_API_KEY` is only required for OpenRouter mode and for auxiliary features that call `SQLgenerator.py` (e.g. live evaluation explanations). You can omit it for a pure local setup; the API will still start.

## Running the Application

Run both the backend and frontend. Use **two separate terminal windows**.

### 1. Start the backend (API)

```bash
source venv/bin/activate   # On Windows: .\venv\Scripts\activate
uvicorn api:app
```

The API will be available at `http://127.0.0.1:8000`.

### 2. Start the frontend

```bash
source venv/bin/activate   # On Windows: .\venv\Scripts\activate
streamlit run NCCS_Query_Assistant.py
```

The frontend will open in your browser (typically `http://localhost:8501`).

**Note:** Start the backend first, since the frontend connects to it.
