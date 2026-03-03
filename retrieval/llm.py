import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_MODEL = "arcee-ai/trinity-large-preview:free"


def get_llm(model_name: str = DEFAULT_MODEL) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        extra_body={
            "provider": {"require_parameters": True}
        },
    )


# Module-level singleton kept for notebook / backward-compat imports
llm = get_llm()