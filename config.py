"""Settings and model factories. Everything is driven by the .env file."""
import os
from dotenv import load_dotenv
# Load environment variables from the .env file.
load_dotenv()

# General configuration settings:
# LLM provider or openai by default:
PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
# Embedding provider or openai by default:
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
# Maximum number of research rounds:
MAX_RESEARCH_ROUNDS = int(os.getenv("MAX_RESEARCH_ROUNDS", "2"))
# Maximum number of pitch rounds:
MAX_PITCH_ROUNDS = int(os.getenv("MAX_PITCH_ROUNDS", "2"))
# Azure OpenAI API version:
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")


def get_llm():
    """Return a chat model for the provider chosen in .env."""
    kwargs = {}
    if os.getenv("LLM_TEMPERATURE"):  # leave unset for models that reject it
        kwargs["temperature"] = float(os.environ["LLM_TEMPERATURE"])

    if PROVIDER == "azure":
        from langchain_openai import AzureChatOpenAI

        # Reads AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT from the environment.
        return AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            api_version=AZURE_API_VERSION,
            **kwargs,
        )

    from langchain_openai import ChatOpenAI

    # Reads OPENAI_API_KEY from the environment.
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), **kwargs)


def get_embeddings():
    """Return an embeddings model for vector memory, or None to disable it."""
    if EMBEDDING_PROVIDER == "azure":
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            api_version=AZURE_API_VERSION,
        )
    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
    return None
