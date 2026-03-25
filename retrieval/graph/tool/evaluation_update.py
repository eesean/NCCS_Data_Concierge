import math
import time
import pandas as pd
from datetime import datetime
import os
from SQLgenerator import explain_sql
print("Importing SQLEvaluator")
from evaluation.SQLEvaluator import SQLComplexityEvaluator
print("Importing SematicScoring" )
from SematicScoring import calculate_similarity

PENALTY_SCORE = 0.2 # value can be adjusted

def evaluate_live_query(prompt: str, model: str, generated_sql: str, latency: float, metrics: dict):
    """
    Evaluates a single live query on the fly and logs it to a CSV.
    """
    evaluator = SQLComplexityEvaluator() # Assuming this is globally available or imported
    validation_tries = metrics.get("validation_tries", 0)
    
    # 1. Set up the base logging dictionary
    row_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": model,
        "Prompt": prompt,
        "Generated SQL": generated_sql if generated_sql else "ERROR",
        "Latency (s)": round(latency * (1 + ((validation_tries - 1) * PENALTY_SCORE)), 3),
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
    df["Log Transformed Tokens"] = df["Total Tokens"].apply(lambda x: round(math.log(x + 1), 3)) # Log transform for better scaling
    columns_to_normalize = ["Latency (s)", "Complexity Score", "Log Transformed Tokens", "Semantic Score"]
    for column in columns_to_normalize:
        df[f"Normalized {column}"] = normalize_score(column, df)
    df["Efficiency Score"] = 0.6 * df["Normalized Semantic Score"] + 0.1 * df["Normalized Latency (s)"] + 0.1 * df["Normalized Log Transformed Tokens"] + 0.2 * df["Normalized Complexity Score"]

    # If file doesn't exist, write headers. Otherwise, append without headers.
    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

    return row_data

def normalize_score(column, df):
    filtered_df = df[df["Generated SQL"] != "ERROR"] # Only consider successful generations for normalization
    if filtered_df.empty:
        return df[column].apply(lambda x: 0.5) # If no successful generations, assign 0.5 to all
    min_score = filtered_df[column].min()
    max_score = filtered_df[column].max()
    if max_score - min_score == 0:
        return df[column].apply(lambda x: 0.5) # If all scores are the same, assign 0.5
    elif column == "Semantic Score":
        return df[column].apply(lambda x: (x - min_score) / (max_score - min_score))
    else: # reverse normalization applied (lower is better)
        return df[column].apply(lambda x: (max_score - x) / (max_score - min_score))