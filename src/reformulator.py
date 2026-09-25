import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

REFORMULATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query reformulation specialist. Your job is to improve a search query so that it retrieves better, more relevant documents.

Rules:
- Use the critic's feedback to understand what information was missing.
- Make the query more specific and targeted.
- Do NOT answer the question — only produce a better search query.
- Respond with ONLY the improved query text, nothing else."""),
    ("human", """Original question: {question}
Current search query: {current_query}
Critic feedback: {critique}

Write an improved search query:""")
])

def get_llm():
    return ChatOllama(model=Config.MODEL_NAME, temperature=0.0)

def reformulate_query(question: str, current_query: str, critique: str) -> str:
    """
    Given the original question, the current query that failed,
    and the critic's feedback, produce a better retrieval query.
    """
    llm = get_llm()
    chain = REFORMULATE_PROMPT | llm
    response = chain.invoke({
        "question": question,
        "current_query": current_query,
        "critique": critique
    })
    return response.content.strip().strip('"').strip("'")

def main():
    print("=== Phase 9: Query Reformulation Test ===\n")
    
    # Test 1
    print("--- Test 1 ---")
    new_query = reformulate_query(
        question="What is the leave policy?",
        current_query="What is the leave policy?",
        critique="The retrieved context does not contain enough detail about sick leave."
    )
    print(f"Original query: 'What is the leave policy?'")
    print(f"Reformulated:   '{new_query}'")
    
    print()
    
    # Test 2
    print("--- Test 2 ---")
    new_query2 = reformulate_query(
        question="Can I work from home?",
        current_query="Can I work from home?",
        critique="The context mentions remote work but does not specify eligibility requirements."
    )
    print(f"Original query: 'Can I work from home?'")
    print(f"Reformulated:   '{new_query2}'")

if __name__ == "__main__":
    main()
