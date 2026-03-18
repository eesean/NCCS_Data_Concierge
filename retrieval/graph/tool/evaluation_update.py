import time
import pandas as pd
from datetime import datetime
import os
from SQLgenerator import explain_sql
print("Importing SQLEvaluator")
from evaluation.SQLEvaluator import SQLComplexityEvaluator
print("Importing SematicScoring" )
from SematicScoring import calculate_similarity

def evaluate_live_query(prompt: str, model: str, generated_sql: str, latency: float, metrics: dict):
    """
    Evaluates a single live query on the fly and logs it to a CSV.
    """
    evaluator = SQLComplexityEvaluator() # Assuming this is globally available or imported
    
    # 1. Set up the base logging dictionary
    row_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": model,
        "Prompt": prompt,
        "Generated SQL": generated_sql if generated_sql else "ERROR",
        "Latency (s)": round(latency, 3),
        "Input Tokens": metrics.get("input_tokens", 0),
        "Output Tokens": metrics.get("output_tokens", 0),
        "Total Tokens": metrics.get("total_tokens", 0),
        "Prompt Cost": metrics.get("prompt_cost", 0.0),
        "Completion Cost": metrics.get("completion_cost", 0.0),
        "Total Cost": metrics.get("cost", 0.0),
        "Complexity Score": None,
        "Complexity Level": "N/A",
        "Generated Explanation": "N/A",
        "Semantic Score": 0.0,
        "Error Message": metrics.get("error_message", "")
    }

    # 2. Run Reference-Free Evaluations (Only if SQL was actually generated)
    if generated_sql and generated_sql != "ERROR":
        try:
            # Complexity
            complexity = evaluator.rate_query(generated_sql)
            row_data["Complexity Score"] = complexity.get('score')
            row_data["Complexity Level"] = complexity.get('level')
            
            # Explanation & Semantic Score
            generated_explanation = explain_sql(generated_sql, model)
            row_data["Generated Explanation"] = generated_explanation
            row_data["Semantic Score"] = calculate_similarity(prompt, generated_explanation)
            
        except Exception as e:
            print(f"Dynamic Eval Error: {str(e)}")
            row_data["Error Message"] += f" | Eval Error: {str(e)}"

    # 3. Append to CSV (Create if it doesn't exist)
    csv_path = "live_query_logs.csv"
    df = pd.DataFrame([row_data])
    
    # If file doesn't exist, write headers. Otherwise, append without headers.
    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

    return row_data