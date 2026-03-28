import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_ollama_llm(model_name: str = OLLAMA_DEFAULT_MODEL, base_url: str = OLLAMA_BASE_URL):
    """Return a ChatOllama instance for the local Ollama server."""
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model_name, base_url=base_url)


# Module-level singleton for backward-compat imports (notebooks etc.)
llm = get_ollama_llm()
