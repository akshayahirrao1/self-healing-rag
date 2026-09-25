import os
import sys
import json

# Suppress some logging for cleaner output if needed, but keeping it simple.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import build_graph, FALLBACK_RESPONSE
from src.retrieval import run_normal_rag, format_docs
from src.critic import run_critic

def main():
    questions_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "questions.json")
    with open(questions_path, "r") as f:
        questions = json.load(f)

    print("\n" + "=" * 80)
    print(" EVALUATING BASELINE (NORMAL RAG) VS SELF-HEALING RAG")
    print("=" * 80)
    
    app = build_graph()
    
    metrics = {
        "Total questions": len(questions),
        "Accepted on first attempt": 0,
        "Accepted after retry": 0,
        "Fallback responses": 0,
        "Critic rejection rate": 0.0,
        "Total retries": 0,
        "Total critic evaluations": 0
    }
    
    results = []

    # Let's temporarily disable heavy logging from modules
    # by suppressing stdout temporarily, or just let it print if we want to trace it.
    
    for idx, q_data in enumerate(questions):
        q = q_data["question"]
        expected = q_data["expected_behavior"]
        
        print(f"\n\n>>> [{idx+1}/{len(questions)}] QUESTION: {q}")
        
        # 1. Baseline
        print("\n--- Running Baseline ---")
        baseline_ans, baseline_docs = run_normal_rag(q)
        baseline_critic = run_critic(q, format_docs(baseline_docs), baseline_ans)
        
        # 2. Self-Healing
        print("\n--- Running Self-Healing ---")
        initial_state = {
            "question": q,
            "current_query": q,
            "documents": [],
            "answer": "",
            "critique": "",
            "decision": "",
            "retry_count": 0
        }
        final_state = app.invoke(initial_state)
        
        retry_count = final_state["retry_count"]
        decision = final_state["decision"]
        sh_ans = final_state["answer"]
        
        # Track metrics
        attempts = retry_count + 1 if decision != "fallback" else retry_count
        # Wait, if fallback is triggered, critic might have run `MAX_RETRIES` times and failed, or `MAX_RETRIES + 1` times.
        # `retry_count` in graph increments in the critic.
        # Initial is 0, critic makes it 1.
        metrics["Total critic evaluations"] += retry_count
        
        if decision == "accept":
            if retry_count == 1: # meaning it accepted on the first time through critic (which incremented retry_count from 0 to 1)
                metrics["Accepted on first attempt"] += 1
            else:
                metrics["Accepted after retry"] += 1
        elif decision == "fallback":
            metrics["Fallback responses"] += 1
            
        results.append({
            "Question": q,
            "Baseline Answer": baseline_ans.replace("\n", " ")[:60] + "...",
            "Baseline Critic": baseline_critic.decision.upper(),
            "Self-Healing Answer": sh_ans.replace("\n", " ")[:60] + "...",
            "Self-Healing Critic": decision.upper(),
            "Attempts": retry_count
        })

    # Calculate rejection rate
    total_accepts = metrics["Accepted on first attempt"] + metrics["Accepted after retry"]
    rejections = metrics["Total critic evaluations"] - total_accepts
    if metrics["Total critic evaluations"] > 0:
        metrics["Critic rejection rate"] = round(rejections / metrics["Total critic evaluations"], 2)
        metrics["Average attempts per question"] = round(metrics["Total critic evaluations"] / metrics["Total questions"], 2)
    
    print("\n\n" + "=" * 80)
    print(" COMPARISON RESULTS SUMMARY")
    print("=" * 80)
    
    # Simple table formatting
    header = f"{'Question':<35} | {'Baseline (Critic)':<30} | {'Self-Healing (Critic)':<30} | {'SH Attempts'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        q_trunc = (r['Question'][:32] + '...') if len(r['Question']) > 32 else r['Question']
        base_col = f"({r['Baseline Critic']})"
        sh_col = f"({r['Self-Healing Critic']})"
        print(f"{q_trunc:<35} | {base_col:<30} | {sh_col:<30} | {r['Attempts']}")
    print("-" * len(header))
    
    print("\n\n" + "=" * 80)
    print(" METRICS")
    print("=" * 80)
    for k, v in metrics.items():
        if k not in ["Total retries", "Total critic evaluations"]:
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
