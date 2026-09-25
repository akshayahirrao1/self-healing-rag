import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
    TOP_K = int(os.getenv("TOP_K", "4"))
    RETRIEVAL_FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", "15"))
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
