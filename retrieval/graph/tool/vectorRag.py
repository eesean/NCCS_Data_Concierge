import os
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

# Path to chroma_db (relative to vectorRag.py in retrieval/graph/tool/)
# chroma_db was created in retrieval/, so from tool/ we need two levels up
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")

# Use the same embedding model used when creating the DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load existing vector stores (no need to re-embed)
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
    collection_name="schema_columns",
)
cancer_vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
    collection_name="icd10_cancer_reference",
)
sql_template_vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
    collection_name="sql_templates",
)


def get_schema_context(query: str, k: int = 4) -> str:
    """
    Retrieve relevant database schema (tables and columns) for a natural language query.
    Use this when you need to generate SQL - it returns the schema context needed to write accurate queries.
    Input: The user's question or what they want to query (e.g. "patient deaths", "drug prescriptions").
    """
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def _get_cancer_info_impl(query: str, k: int = 3) -> str:
    """Internal: retrieve ICD-10-AM cancer codes for a query."""
    docs = cancer_vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


@tool
def get_cancer_info(query: str) -> str:
    """
    Retrieve relevant ICD-10-AM cancer codes and descriptions for mapping cancer types to codes.
    Call this when the user asks about cancer types, diagnoses, or ICD codes (e.g. colorectal, lung, breast).
    Returns matching ICD-10 codes (e.g. C18–C20 for colorectal) and descriptions to use in WHERE clauses.
    Input: Cancer type or diagnosis term (e.g. "colorectal cancer", "lip cancer", "oesophagus").
    """
    return _get_cancer_info_impl(query, k=3)


def get_sql_template(query: str, k: int = 3) -> str:
    """
    Retrieve relevant SQL few-shot examples/templates based on the user's question.
    Use when generating SQL — returns similar past queries (HEADER + SQL) to guide the correct structure,
    joins, and filters (e.g. colorectal counts, staging, age groups).
    Input: Natural language question or query intent (e.g. "number of colorectal cancer by age group").
    """
    docs = sql_template_vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
