import json
import re
import os
import sys
import time
from src.utils.llm_client import query_llm

# PATHS
DATA_PATH = os.path.join("data", "processed", "turkish_benchmark_with_letters.json")
RESULTS_DIR = os.path.join("results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "direct_solver_results.json")

def load_existing_results():
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_results(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

def construct_prompt(item):
    return f"""
    You are an expert logic puzzle solver.
    
    CONTEXT: {item['context']}
    QUESTION: {item['question']}
    CHOICES: {item['choices']}
    
    INSTRUCTIONS:
    1. Briefly analyze constraints.
    2. Eliminate impossible choices.
    3. State the correct Answer.
    4. END with "Final Answer: X" (where X is A, B, C, D, or E).
    """

def parse_answer(output_str):
    if not output_str: return None
    match = re.search(r"Final Answer:\s*([A-E])", output_str, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"Answer:\s*([A-E])", output_str, re.IGNORECASE)
    return match.group(1).upper() if match else None

def run_direct_pipeline():
    print("--- 🧠 DIRECT REASONING PIPELINE (TRACKING MODELS) ---")
    
    if not os.path.exists(DATA_PATH):
        print("❌ NO DATA FOUND.")
        return

    full_dataset = json.load(open(DATA_PATH, 'r', encoding='utf-8'))
    results = load_existing_results()
    solved_ids = {r['id'] for r in results}
    
    print(f"📊 Total: {len(full_dataset)} | Solved: {len(solved_ids)} | Remaining: {len(full_dataset) - len(solved_ids)}")

    try:
        for item in full_dataset:
            if item['id'] in solved_ids: continue

            print(f"\n[QUERY {item['id']}] Thinking", end='', flush=True)
            
            start_ts = time.time()
            messages = [{"role": "user", "content": construct_prompt(item)}]
            
            # UNPACK TUPLE HERE
            response, model_used = query_llm(messages, timeout=45.0)
            
            duration = time.time() - start_ts

            if response is None:
                 print(f" -> 🚨 API FAILURE (Skipping)")
                 continue 

            prediction = parse_answer(response)
            if not prediction: prediction = "PARSE_FAIL"

            truth = item['answer'].strip()
            is_correct = (prediction == truth)
            symbol = "✅" if is_correct else "❌"
            
            # Print which model was used
            print(f" ({model_used} | {duration:.1f}s) -> Pred: {prediction} | True: {truth} | {symbol}")

            results.append({
                "id": item['id'],
                "correct": is_correct,
                "prediction": prediction,
                "truth": truth,
                "model_used": model_used, # SAVING MODEL NAME
                "raw_response": response
            })
            save_results(results)

    except KeyboardInterrupt:
        print("\n🛑 Saved & Exiting.")
        save_results(results)
        sys.exit(0)

if __name__ == "__main__":
    run_direct_pipeline()