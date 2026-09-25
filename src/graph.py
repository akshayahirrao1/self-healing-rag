import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END

from src.state import RAGState
from src.config import Config
from src.vector_store import get_vector_store
from src.retrieval import get_llm, RAG_PROMPT, format_docs
from src.critic import run_critic, CriticDecision
from src.reformulator import reformulate_query

FALLBACK_RESPONSE = "I don't have enough information in the available documents to answer this reliably."

# ─── Node 1: Retrieve ─────────────────────────────────────────────
def retrieve(state: RAGState) -> dict:
    """Query the vector store and return relevant document chunks."""
    query = state["current_query"]
    attempt = state["retry_count"] + 1
    
    print(f"\n{'='*50}")
    print(f"Attempt {attempt}")
    print(f"{'='*50}")
    print(f"Query: '{query}'")
    
    vector_store = get_vector_store()
    # Fetch a larger pool of documents initially
    fetch_k = getattr(Config, "RETRIEVAL_FETCH_K", 15)
    retriever = vector_store.as_retriever(search_kwargs={"k": fetch_k})
    initial_docs = retriever.invoke(query)
    
    print(f"Retrieved initially: {len(initial_docs)} chunks")
    
    # Advanced RAG: Reranking
    if len(initial_docs) > 0:
        from sentence_transformers import CrossEncoder
        print("Reranking documents...")
        reranker_model = getattr(Config, "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker = CrossEncoder(reranker_model)
        
        # Create (query, doc_text) pairs
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = reranker.predict(pairs)
        
        # Sort docs by score (highest first)
        scored_docs = list(zip(initial_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only the top k
        top_k = Config.TOP_K
        docs = [doc for doc, score in scored_docs[:top_k]]
        print(f"Reranked and kept top {len(docs)} chunks")
    else:
        docs = initial_docs
        
    return {"documents": docs}

# ─── Node 2: Generate ─────────────────────────────────────────────
def generate(state: RAGState) -> dict:
    """Generate a draft answer using the LLM and retrieved context."""
    question = state["question"]
    docs = state["documents"]
    context = format_docs(docs)
    
    llm = get_llm()
    chain = RAG_PROMPT | llm
    response = chain.invoke({"context": context, "question": question})
    answer = response.content
    
    print(f"Generated answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
    
    return {"answer": answer}

# ─── Node 3: Critic ───────────────────────────────────────────────
def critic(state: RAGState) -> dict:
    """Evaluate whether the answer is grounded in the retrieved context."""
    question = state["question"]
    docs = state["documents"]
    answer = state["answer"]
    context = format_docs(docs)
    
    result: CriticDecision = run_critic(question, context, answer)
    
    print(f"Critic: {result.decision.upper()}")
    print(f"Reason: {result.reason}")
    
    return {
        "decision": result.decision,
        "critique": result.reason,
        "retry_count": state["retry_count"] + 1
    }

# ─── Node 4: Reformulate ──────────────────────────────────────────
def reformulate(state: RAGState) -> dict:
    """Produce an improved search query based on the critic's feedback."""
    new_query = reformulate_query(
        question=state["question"],
        current_query=state["current_query"],
        critique=state["critique"]
    )
    
    print(f"\nReformulated query: '{new_query}'")
    
    return {"current_query": new_query}

# ─── Node 5: Fallback ─────────────────────────────────────────────
def fallback(state: RAGState) -> dict:
    """Return a safe response when retries are exhausted."""
    print(f"\nMax retries reached. Returning fallback response.")
    return {"answer": FALLBACK_RESPONSE, "decision": "fallback"}

# ─── Conditional Routing ──────────────────────────────────────────
def route_after_critic(state: RAGState) -> str:
    """
    After the critic evaluates, decide the next step:
    - accept  → finish (END)
    - reject + retries left → reformulate
    - reject + no retries  → fallback
    """
    if state["decision"] == "accept":
        return "finish"
    
    if state["retry_count"] < Config.MAX_RETRIES:
        return "reformulate"
    
    return "fallback"

# ─── Build the Graph ──────────────────────────────────────────────
def build_graph():
    """
    Constructs the self-healing RAG LangGraph workflow:
    
    START → retrieve → generate → critic
                                    ↓
                        ┌── accept → END
                        │
                        └── reject
                              ↓
                        retry available?
                          yes → reformulate → retrieve (CYCLE)
                          no  → fallback → END
    """
    graph = StateGraph(RAGState)
    
    # Add nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("critic", critic)
    graph.add_node("reformulate", reformulate)
    graph.add_node("fallback", fallback)
    
    # Set entry point
    graph.set_entry_point("retrieve")
    
    # Add edges (the linear path)
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critic")
    
    # Add conditional routing after the critic
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "finish": END,
            "reformulate": "reformulate",
            "fallback": "fallback"
        }
    )
    
    # Reformulate loops back to retrieve (this creates the CYCLE)
    graph.add_edge("reformulate", "retrieve")
    
    # Fallback goes to END
    graph.add_edge("fallback", END)
    
    return graph.compile()

# ─── Run the pipeline ─────────────────────────────────────────────
def run_self_healing_rag(question: str) -> str:
    """Run the full self-healing RAG pipeline for a given question."""
    print(f"\n{'#'*60}")
    print(f"  Self-Healing RAG Pipeline")
    print(f"  Question: '{question}'")
    print(f"{'#'*60}")
    
    app = build_graph()
    
    # Initialize state
    initial_state: RAGState = {
        "question": question,
        "current_query": question,
        "documents": [],
        "answer": "",
        "critique": "",
        "decision": "",
        "retry_count": 0
    }
    
    # Run the graph
    final_state = app.invoke(initial_state)
    
    print(f"\n{'='*50}")
    print(f"FINAL ANSWER:")
    print(f"{'='*50}")
    print(final_state["answer"])
    print(f"\nTotal attempts: {final_state['retry_count']}")
    print(f"Final decision: {final_state['decision']}")
    
    return final_state["answer"]
