import os
from dotenv import load_dotenv
from ollama import chat as _ollama_chat

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def ollama_chat(messages: list, tools: list = None, model: str = None) -> dict:
    """Thin wrapper around ollama.chat() for consistent call signature.

    think=False disables qwen3's extended chain-of-thought mode, which
    otherwise runs for several minutes per call before producing a response.
    """
    kwargs = {
        "model": model ,
        "messages": messages,
        "think": False,
    }
    if tools:
        kwargs["tools"] = tools
    return _ollama_chat(**kwargs)
