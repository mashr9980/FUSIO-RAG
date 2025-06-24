import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "9100"))
    WORKERS = int(os.getenv("WORKERS", "2"))

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
    TOP_P = float(os.getenv("TOP_P", "0.95"))
    REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.15"))
    OPENAI_API_KEY = os.getenv("OPENAI_API")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    SPLIT_CHUNK_SIZE = int(os.getenv("SPLIT_CHUNK_SIZE", "500"))
    SPLIT_OVERLAP = int(os.getenv("SPLIT_OVERLAP", "50"))
    EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-base-en-v1.5")
    SIMILAR_DOCS_COUNT = int(os.getenv("SIMILAR_DOCS_COUNT", "6"))

    OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "./rag-vectordb")

config = Config()