import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "9100"))
    WORKERS = int(os.getenv("WORKERS", "2"))

    # Model settings
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
    TOP_P = float(os.getenv("TOP_P", "0.95"))
    REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.15"))
    OPENAI_API_KEY = os.getenv("OPENAI_API", "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")

    # Data settings
    SPLIT_CHUNK_SIZE = int(os.getenv("SPLIT_CHUNK_SIZE", "500"))
    SPLIT_OVERLAP = int(os.getenv("SPLIT_OVERLAP", "50"))
    EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-base-en-v1.5")
    SIMILAR_DOCS_COUNT = int(os.getenv("SIMILAR_DOCS_COUNT", "6"))

    OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "./rag-vectordb")

config = Config()