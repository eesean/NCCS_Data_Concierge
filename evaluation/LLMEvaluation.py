from NCCS_Data_Concierge.SQLgenerator import generate_sql_from_nl
from NCCS_Data_Concierge.evaluation.SQLEvaluator import SQLComplexityEvaluator
import time

BASE_MODEL = "openai/gpt-4o-mini"
models = [
    "openai/gpt-4o-mini",
    "deepseek/deepseek-v3.2",
    "qwen/qwen-2.5",
    "anthropic/claude-3-5-sonnet-20241022"
]

def evaluate_llm_performance(nl_query: str) -> str:
    # measure end-to-end call to the LLM
    start_time = time.perf_counter()
    response = generate_sql_from_nl(nl_query, model_name=BASE_MODEL)
    sql_generation_latency = time.perf_counter() - start_time
    sql_query = response.get("sql", "")
    tokens = response['usage']
    cost = response['cost']

    print(f"Tokens used: {tokens}")
    print(f"Cost: {cost}")
    print(f"Generated SQL:\n{sql_query}")
    print(f"SQL generation latency: {sql_generation_latency:.3f} seconds")

    evaluator = SQLComplexityEvaluator()
    sql_query_complexity = evaluator.rate_query(sql_query)
    print(f"Evaluated SQL Complexity: {sql_query_complexity}")

test_prompt = "How many deaths occurred in 2020?"
evaluate_llm_performance(test_prompt)