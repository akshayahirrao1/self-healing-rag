import sys
import os
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store import get_vector_store

def main():
    print("--- Testing Phase 6: Vector Database Retrieval ---")
    
    # Initialize connection to the database
    vector_store = get_vector_store()
    
    # We define a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    question = "What is the policy for sick leave?"
    print(f"\nQuestion: '{question}'")
    print("Searching vector database for relevant chunks...")
    
    # Perform retrieval
    results = retriever.invoke(question)
    
    print(f"\nFound {len(results)} relevant chunks:")
    for idx, doc in enumerate(results):
        print(f"\nResult {idx + 1}:")
        print(f"Content: '{doc.page_content}'")
        print(f"Source: {doc.metadata.get('source')}")

if __name__ == "__main__":
    main()
