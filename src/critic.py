import os
import sys
import warnings
import json

warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Structured output for the critic's decision
class CriticDecision(BaseModel):
    decision: str = Field(description="Either 'accept' or 'reject'")
    reason: str = Field(description="Explanation for the decision")

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a grounding critic. Your job is to determine whether a generated answer is supported by the retrieved context.

Rules:
- If the answer explicitly states that the information is missing, not mentioned, or that it cannot answer the question (e.g., "There is no information...", "I don't know", "is not mentioned"), you MUST respond with "reject". This is the MOST IMPORTANT rule.
- If the answer contains claims NOT found in the context, or contradicts the context, respond with "reject".
- Only if the answer provides actual facts that are fully or mostly supported by the context, respond with "accept".
- Do NOT judge whether the answer is "good" — only judge whether it provides supported facts.

You MUST respond with valid JSON in this exact format:
{{"decision": "accept" or "reject", "reason": "your explanation"}}"""),
    ("human", """Retrieved Context:
{context}

Question: {question}

Generated Answer: {answer}

Is this answer supported by the retrieved context? Respond with JSON only.""")
])

def get_llm():
    return ChatOllama(model=Config.MODEL_NAME, base_url=Config.OLLAMA_BASE_URL, temperature=0.0)

def run_critic(question: str, context: str, answer: str) -> CriticDecision:
    """
    Evaluate whether the generated answer is grounded in the retrieved context.
    Returns a CriticDecision with 'accept'/'reject' and a reason.
    """
    llm = get_llm()
    chain = CRITIC_PROMPT | llm
    response = chain.invoke({
        "context": context,
        "question": question,
        "answer": answer
    })
    
    # Parse the JSON response
    try:
        raw = response.content.strip()
        # Handle cases where the LLM wraps JSON in markdown code blocks
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        return CriticDecision(
            decision=data.get("decision", "reject").lower(),
            reason=data.get("reason", "Could not parse reason.")
        )
    except (json.JSONDecodeError, Exception) as e:
        # If the LLM produces malformed output, default to reject (safe)
        return CriticDecision(
            decision="reject",
            reason=f"Critic output could not be parsed: {response.content}"
        )

def main():
    print("=== Phase 8: Standalone Critic Test ===\n")
    
    # Example A — Supported answer (should ACCEPT)
    print("--- Test A: Supported answer ---")
    result_a = run_critic(
        question="How many vacation days do employees get?",
        context="Employees receive 20 annual vacation days per year. Time off must be requested at least 2 weeks in advance.",
        answer="Employees receive 20 annual vacation days per year."
    )
    print(f"Decision: {result_a.decision.upper()}")
    print(f"Reason: {result_a.reason}")
    
    print()
    
    # Example B — Unsupported answer (should REJECT)
    print("--- Test B: Unsupported answer ---")
    result_b = run_critic(
        question="How many vacation days do employees get?",
        context="Employees receive 20 annual vacation days per year.",
        answer="Employees receive 40 annual vacation days per year."
    )
    print(f"Decision: {result_b.decision.upper()}")
    print(f"Reason: {result_b.reason}")
    
    print()
    
    # Example C — Partial support / hallucination (should REJECT)
    print("--- Test C: Partially supported answer ---")
    result_c = run_critic(
        question="Can employees work remotely?",
        context="Remote work is available to eligible employees up to 2 days per week.",
        answer="Every employee can work remotely permanently."
    )
    print(f"Decision: {result_c.decision.upper()}")
    print(f"Reason: {result_c.reason}")

if __name__ == "__main__":
    main()
