import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")

# Lazily initialised — populated on first use to avoid slow startup
_embeddings = None
_vectorstore = None
_cancer_vectorstore = None
_sql_template_vectorstore = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return _embeddings


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=_get_embeddings(),
            collection_name="schema_columns",
        )
    return _vectorstore


def _get_cancer_vectorstore():
    global _cancer_vectorstore
    if _cancer_vectorstore is None:
        _cancer_vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=_get_embeddings(),
            collection_name="icd10_cancer_reference",
        )
    return _cancer_vectorstore


def _get_sql_template_vectorstore():
    global _sql_template_vectorstore
    if _sql_template_vectorstore is None:
        _sql_template_vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=_get_embeddings(),
            collection_name="sql_templates",
        )
    return _sql_template_vectorstore


def get_schema_context(query: str, k: int = 4) -> str:
    """
    Retrieve relevant database schema (tables and columns) for a natural language query.
    Use this when you need to generate SQL - it returns the schema context needed to write accurate queries.
    Input: The user's question or what they want to query (e.g. "patient deaths", "drug prescriptions").
    """
    docs = _get_vectorstore().similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def get_cancer_info(query: str) -> str:
    """
    Retrieve ICD-10-AM cancer reference text for a cancer topic.
    Each chunk includes SQL_FILTER (ready ICD10 LIKE predicates) and ICD10_CODES.
    Input: short cancer term (e.g. "colorectal cancer", "lung cancer").
    """
    docs = _get_cancer_vectorstore().similarity_search(query, k=3)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def get_sql_template(query: str, k: int = 3) -> str:
    """
    Retrieve relevant SQL few-shot examples/templates based on the user's question.
    Use when generating SQL — returns similar past queries (HEADER + SQL) to guide the correct structure,
    joins, and filters (e.g. colorectal counts, staging, age groups).
    Input: Natural language question or query intent (e.g. "number of colorectal cancer by age group").
    """
    docs = _get_sql_template_vectorstore().similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
