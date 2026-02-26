import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import spacy
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

from SQLgenerator import generate_sql_from_nl

load_dotenv()

# ============== Using OpenAI for embedding ============
"""client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def get_embedding(text: str):
    # Note: OpenRouter requires the full "provider/model" slug
    # Using 'text-embedding-3-small'
    model_slug = "openai/text-embedding-3-small"
    
    response = client.embeddings.create(
        input=[text.replace("\n", " ")],
        model=model_slug
    )
    return response.data[0].embedding"""

# ============== Using SentenceTransformer for embedding ============
## Need to make sure to load the model only once at the end product
# Diff models : BAAI/bge-small-en-v1.5 / all-MiniLM-L6-v2 / multi-qa-mpnet-base-dot-v1
_local_model = SentenceTransformer('BAAI/bge-small-en-v1.5')

def get_local_embedding(text: str):
    """
    Generates embeddings locally using Sentence-Transformers.
    """
    # Clean the text to ensure consistency
    clean_text = text.replace("\n", " ").strip()
    
    # Generate the embedding
    # convert_to_numpy=True ensures it plays nice with existing cosine_similarity logic
    embedding = _local_model.encode(clean_text, convert_to_numpy=True)
    
    return embedding.tolist()

def calculate_similarity(user_prompt: str, sql_explanation: str) -> float:
    try:
        # 1. Preprocess: Lemmatize, remove stopwords, and strip AI prefixes
        processed_prompt = clean_for_similarity(user_prompt)
        processed_explanation = clean_for_similarity(sql_explanation)
        
        # Debug print to see what the vectors are actually comparing
        # print(f"Comparing: '{processed_prompt}' vs '{processed_explanation}'")

        # 2. Get embeddings for the cleaned text
        vec1 = np.array(get_local_embedding(processed_prompt)).reshape(1, -1)
        vec2 = np.array(get_local_embedding(processed_explanation)).reshape(1, -1)
        
        # 3. Calculate Cosine Similarity
        score = cosine_similarity(vec1, vec2)[0][0]
        
        return round(float(score), 4)
    except Exception as e:
        print(f"Error: {e}")
        return 0.0

nlp = spacy.load("en_core_web_sm")

def clean_for_similarity(text: str) -> str:
    """
    Standardizes text for better Cosine Similarity scoring.
    """
    # 1. Lowercase and strip whitespace
    text = text.lower().strip()

    # 2. Remove AI-generated prefixes common in LLM explanations
    prefixes = [
        r"this query returns", r"this query counts", r"this query finds",
        r"returns the", r"counts the", r"finds the", r"show me"
    ]
    for p in prefixes:
        text = re.sub(p, "", text)

    # 3. Spacy Pipeline: Tokenize, Remove Stopwords, and Lemmatize
    doc = nlp(text)
    
    # We keep words that are clinically relevant but might be flagged as 
    # generic stopwords by default (like 'above', 'under', 'between')
    cleaned_tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and not token.is_space
    ]

    return " ".join(cleaned_tokens)

if __name__ == "__main__":
    # Test cases
    test_prompt = "How many procedures in 2020 lasted for longer than a week?"
    sql,test_explanation = generate_sql_from_nl(test_prompt)
    #test_explanation = "How many procedures in 2021 lasted for longer than two week?"
    score = calculate_similarity(test_prompt, test_explanation)
    print("Clean Prompt :" + clean_for_similarity(test_prompt))
    print("Clean Explanation :" + clean_for_similarity(test_explanation))
    print(f"Match Score: {score}")


    ## Need to think of how to extract the keyword