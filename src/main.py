import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import run_self_healing_rag

def main():
    print("Self-Healing RAG Pipeline")
    print("=" * 60)
    print("Type 'quit' to exit.\n")
    
    while True:
        question = input("Ask a question: ").strip()
        
        if not question:
            print("Please enter a question.\n")
            continue
            
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        
        run_self_healing_rag(question)
        print()

if __name__ == "__main__":
    main()
