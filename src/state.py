from typing import TypedDict

class RAGState(TypedDict):
    """
    The shared state that flows through every node in the LangGraph workflow.
    
    Fields:
    - question:      The original user question (never changes)
    - current_query: The query currently used for retrieval (changes on reformulation)
    - documents:     The retrieved document chunks
    - answer:        The current generated answer
    - critique:      The critic's explanation
    - decision:      'accept' or 'reject'
    - retry_count:   How many retrieve→generate→critic cycles have occurred
    """
    question: str
    current_query: str
    documents: list
    answer: str
    critique: str
    decision: str
    retry_count: int
