import os
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

# Path to chroma_db (relative to vectorRag.py in retrieval/graph/tool/)
# chroma_db was created in retrieval/, so from tool/ we need two levels up
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")

# Use the same embedding model used when creating the DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load existing vector store (no need to re-embed)
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
    collection_name="schema_columns",
)

def get_schema_context(query: str, k: int = 1) -> str:
    """
    Retrieve relevant database schema (tables and columns) for a natural language query.
    Use this when you need to generate SQL - it returns the schema context needed to write accurate queries.
    Input: The user's question or what they want to query (e.g. "patient deaths", "drug prescriptions").
    """
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
