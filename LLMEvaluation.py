from SQLgenerator import generate_sql_from_nl, SQLGenError, ALLOWED_TABLES
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from SQLEvaluator import SQLComplexityEvaluator
from dotenv import load_dotenv
import time

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-4o-mini"

chatopenai = ChatOpenAI(
    model=MODEL,
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0,
)

def generate_llm_response(llm, question: str) -> str:
    """
    Generate SQL from natural language. Returns SQL string only.
    Designed to be called by pipeline/backend later.
    """
    if not OPENROUTER_API_KEY:
        raise SQLGenError("Missing OPENROUTER_API_KEY in .env")

    # Force structured output so downstream validation/execution is reliable
    system = f"""Return ONLY valid JSON in exactly this format:
        {{"sql": "<single SELECT statement>"}}

        Rules:
        - Generate exactly ONE SQL statement.
        - Do NOT assume any other OMOP tables exist.
        - Only SELECT queries are allowed. Never use INSERT/UPDATE/DELETE/DROP/CREATE/ALTER.
        - Use only the available OMOP-style tables and join keys.
        - Always include LIMIT. For COUNT-only queries, use LIMIT 1.
        - Use DuckDB-compatible functions only.
        - Do NOT reference any table not explicitly listed below.
        - You may ONLY use these tables (exact spelling, lowercase):
        {ALLOWED_TABLES}
        """

    resp = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=question)
    ])

    return resp

def evaluate_llm_performance(nl_query: str) -> str:
    # measure end-to-end call to the LLM
    response = generate_llm_response(chatopenai, nl_query)

    tokens = response.response_metadata['token_usage']['total_tokens']
    cost = response.response_metadata['token_usage']['cost']
    print(f"Tokens used: {tokens}")
    print(f"Cost: ${cost}")

    start_time = time.perf_counter()
    sql_query = generate_sql_from_nl(nl_query)
    sql_generation_latency = time.perf_counter() - start_time
    print(f"Generated SQL:\n{sql_query}")
    print(f"SQL generation latency: {sql_generation_latency:.3f} seconds")

    evaluator = SQLComplexityEvaluator()
    sql_query_complexity = evaluator.rate_query(sql_query)
    print(f"Evaluated SQL Complexity: {sql_query_complexity}")

test_prompt = "How many deaths occurred in 2020?"
evaluate_llm_performance(test_prompt)