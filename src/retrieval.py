import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.vector_store import get_vector_store
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# The generation prompt — instructs the LLM to answer ONLY from the provided context
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the user's question based ONLY on the provided context.
If the context does not contain enough information to answer the question, say so clearly.
Do NOT make up facts or use knowledge outside the provided context."""),
    ("human", """Context:
{context}

Question: {question}

Answer:""")
])

def get_llm():
    """Initialize the local LLM."""
    return ChatOllama(model=Config.MODEL_NAME, temperature=0.0)

def format_docs(docs):
    """Combine retrieved document chunks into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def run_normal_rag(question: str):
    """
    A simple RAG pipeline: Retrieve → Generate → Answer.
    No critic, no retry — just baseline RAG.
    """
    print(f"\nQuestion: '{question}'")
    
    # Step 1: Retrieve
    print("Retrieving relevant documents...")
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": Config.TOP_K})
    docs = retriever.invoke(question)
    print(f"Retrieved {len(docs)} chunks.")
    
    # Step 2: Format context
    context = format_docs(docs)
    
    # Step 3: Generate answer
    print("Generating answer...")
    llm = get_llm()
    chain = RAG_PROMPT | llm
    response = chain.invoke({"context": context, "question": question})
    
    print(f"\n--- Answer ---")
    print(response.content)
    print("--------------")
    
    return response.content, docs

def main():
    print("=== Phase 7: Normal RAG Pipeline ===")
    
    # Test with a question that HAS an answer in the documents
    run_normal_rag("How many sick days do employees get per year?")
    
    print("\n" + "="*50 + "\n")
    
    # Test with a question that does NOT have an answer in the documents
    run_normal_rag("What is the company's policy on stock options?")

if __name__ == "__main__":
    main()
