import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_MODEL = "arcee-ai/trinity-large-preview:free"
OLLAMA_DEFAULT_MODEL = "llama3.1"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_llm(model_name: str = DEFAULT_MODEL) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        extra_body={
            "provider": {"require_parameters": True}
        },
    )


def get_ollama_llm(model_name: str = OLLAMA_DEFAULT_MODEL, base_url: str = OLLAMA_BASE_URL):
    """
    Return a non-tool-calling LLM for use with make_ollama_reasoner.

    Two modes, selected by the base_url value:

    • base_url == "simulate"  →  OpenRouter via ChatOpenAI **without** the
      require_parameters extra_body.  Use this when Ollama is not installed
      locally (set OLLAMA_BASE_URL=simulate in your .env).  The simulated
      tool-calling wrapper in make_ollama_reasoner is exercised identically;
      only the backend inference provider differs.

    • any other URL           →  Real local Ollama server via ChatOllama.
      Requires `langchain-ollama` installed and `ollama serve` running.
    """
    if base_url == "simulate":
        return ChatOpenAI(
            model=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            # No extra_body / require_parameters — intentionally omitted to
            # replicate a model that has no native function-calling support.
        )
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model_name, base_url=base_url)


# Module-level singleton kept for notebook / backward-compat imports
llm = get_llm()