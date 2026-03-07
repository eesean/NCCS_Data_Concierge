import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import spacy
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import torch

from SQLgenerator import generate_sql_from_nl, explain_sql

load_dotenv()

# ============== Using CrossEncoder ============
## Need to make sure to load the model only once at the end product
# Cross encoder : a Cross-Encoder processes both sentences simultaneously to determine if 
# Sentence B (the explanation) logically follows or supports Sentence A (the prompt). 
# It is much more sensitive to "swapped" parameters or wrong logic.
_local_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') #ms-macro specifically trained on Q&A tasks

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_similarity(user_prompt: str, sql_explanation: str) -> float:
    # 1. Semantic Foundation (The "Vibe")
    # We get the raw score from the Cross-Encoder first
    raw_score = _local_model.predict([user_prompt.lower(), sql_explanation.lower()])
    semantic_base = sigmoid(raw_score)

    # 2. The Entity Scaler (The "Anchor")
    prompt_nums = set(re.findall(r'\d+', user_prompt))
    expl_nums = set(re.findall(r'\d+', sql_explanation))
    
    # Calculate overlap percentage for a smoother penalty
    if not prompt_nums:
        scaler = 1.0  # No numbers to compare
    else:
        # Intersection over Union (Jaccard) for numbers
        matches = prompt_nums.intersection(expl_nums)
        union = prompt_nums.union(expl_nums)
        scaler = len(matches) / len(union)
        print("scaler : " + str(scaler))

    # 3. Final Weighted Score
    # gets penalized if the numbers are wrong.
    final_score = semantic_base * scaler
    #print("semantic_base : " + str(semantic_base))
    
    return round(float(final_score), 4)

if __name__ == "__main__":
    # Test cases
    test_prompt = "How many male patients had conditions or diagnosis concerning the rectum?"
    result = generate_sql_from_nl(test_prompt)
    test_sql = result['sql']
    test_explanation = explain_sql(test_sql)
    test_explanation = "On average how many female patients had conditions or diagnosis concerning the hand?"
    score = calculate_similarity(test_prompt, test_explanation)
    print("SQL : " + test_sql)
    print("Prompt :" + (test_prompt))
    print("Explanation :" + (test_explanation))
    print(f"Match Score: {score}")


    ## Need to think of how to extract the keyword