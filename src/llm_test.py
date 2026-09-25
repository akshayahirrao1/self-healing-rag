import sys
import os

# Add the parent directory to sys.path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

def main():
    print(f"Initializing LLM: {Config.MODEL_NAME}")
    
    try:
        # Initialize the model
        llm = ChatOllama(
            model=Config.MODEL_NAME,
            temperature=0.0
        )
        
        # Define a simple system prompt and a user question
        messages = [
            SystemMessage(content="You are a helpful assistant. Keep your answer brief."),
            HumanMessage(content="What is a vector database?")
        ]
        
        print("\nSending question to LLM: 'What is a vector database?'")
        print("Waiting for response...\n")
        
        # Invoke the model
        response = llm.invoke(messages)
        
        print("--- Response ---")
        print(response.content)
        print("----------------")
        print("\nLLM Test Successful!")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("\nTroubleshooting tips:")
        print("1. Is Ollama running? (Try running 'ollama list' in a terminal)")
        print(f"2. Is the '{Config.MODEL_NAME}' model pulled? (Try running 'ollama pull {Config.MODEL_NAME}')")

if __name__ == "__main__":
    main()
