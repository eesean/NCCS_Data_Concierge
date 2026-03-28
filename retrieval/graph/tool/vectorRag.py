import os
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

# Path to chroma_db (relative to vectorRag.py in retrieval/graph/tool/)
# chroma_db was created in retrieval/, so from tool/ we need two levels up
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")

# uncomment this if you want to use the new chroma_db
# CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "new_chroma_db")

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
    Retrieve ICD-10-AM cancer reference text for ONE cancer topic (RAG over icd10_cancer_reference).
    Each chunk includes SQL_FILTER (ready ICD10 LIKE predicates) and ICD10_CODES (allowed C-prefixes only).

    After this tool returns, copy ONLY the SQL_FILTER / ICD10_CODES from that ToolMessage into your SQL.
    Use condition_occurrence.ICD10 — never invent columns such as ICDO3 or add C-codes not listed in that message.
    Input: short cancer term (e.g. "colorectal cancer", "lung cancer").
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
