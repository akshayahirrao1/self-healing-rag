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
