import os
import sys
import warnings

# Suppress deprecation warnings from langchain_community
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add the parent directory to sys.path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.vector_store import get_vector_store

def load_documents(data_dir: str):
    print(f"Loading documents from {data_dir}...")
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents, chunk_size=200, chunk_overlap=50):
    print(f"Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def store_in_chroma(chunks):
    print("\n--- Phase 5 & 6: Embeddings & Vector Database ---")
    vector_store = get_vector_store()
    print("Storing chunks in Chroma database...")
    # This automatically uses the HuggingFace embeddings to convert text to vectors and save them
    vector_store.add_documents(chunks)
    print("Successfully saved chunks to the database!")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    # Phase 3: Document Loading
    print("--- Phase 3: Document Loading ---")
    documents = load_documents(data_dir)
    if not documents:
        print("No documents found!")
        return
        
    # Phase 4: Chunking
    print("\n--- Phase 4: Chunking ---")
    chunks = split_documents(documents)
    
    # Phase 5 & 6: Embeddings and Storage
    store_in_chroma(chunks)
        
if __name__ == "__main__":
    main()
