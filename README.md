# NCCS Data Concierge

A natural language interface for querying healthcare data via SQL generation and validation.

## Prerequisites

- Python 3.9+
- Virtual environment (recommended)

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

3. Create a `.env` file in the project root with:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   PARQUET_KEY_NAME = "key128"
   PARQUET_KEY_VALUE = "MachuPicchu&K2BC"
   ```

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
streamlit run frontend.py
```

The frontend will open in your browser (typically `http://localhost:8501`).

**Note:** Start the backend first, since the frontend connects to it.
