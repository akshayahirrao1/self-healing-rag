import os
import sys

# Add the parent directory to sys.path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def get_embedding_model():
    """Initialize and return the HuggingFace embedding model."""
    print(f"Initializing embedding model: {Config.EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)

def get_vector_store():
    """Initialize and return the Chroma vector store."""
    embeddings = get_embedding_model()
    return Chroma(
        collection_name="rag_collection",
        embedding_function=embeddings,
        persist_directory=Config.CHROMA_PATH
    )
