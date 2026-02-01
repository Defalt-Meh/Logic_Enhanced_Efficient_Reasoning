import json
import re
import os
import time
import sys
from src.utils.llm_client import query_llm
from src.utils.executor import execute_model_code

# --- PATHS ---
DATA_PATH = os.path.join("data", "processed", "turkish_benchmark_with_letters.json")
RESULTS_DIR = os.path.join("results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "logic_solver_results.json")

# --- CONSTANTS ---
MAX_RETRIES = 3 

# --- HELPER FUNCTIONS FOR SAFETY ---
def load_existing_results():
    """Loads previous results so we don't re-run solved questions."""
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: {RESULTS_FILE} was corrupted. Starting fresh.")
            return []
    return []

def save_results(results):
    """Saves immediately. If the script crashes, data is safe."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

def construct_system_prompt():
    return (
        "You are an expert Python programmer and logic puzzle solver. "
        "Your job is to write a script that BRUTE FORCES the solution to a constraint satisfaction problem. "
        "Never guess. Always calculate. "
        "Output ONLY valid Python code. No markdown blocks, no explanations."
    )

def construct_user_prompt(item, error_trace=None, previous_code=None):
    base_prompt = f"""
    # PUZZLE CONTEXT
    {item['context']}

    # QUESTION
    {item['question']}

    # CHOICES
    {item['choices']}

    # INSTRUCTIONS
    1. Define the variables and constraints strictly.
    2. Use 'itertools' to iterate ALL possibilities.
    3. Filter scenarios that satisfy the context.
    4. Check which Choice (A-E) is correct based on valid scenarios.
    5. PRINT the single uppercase letter of the correct answer (e.g., print("C")).
    """

    if error_trace and previous_code:
        return f"""
        {base_prompt}

        !!! PREVIOUS ATTEMPT FAILED !!!
        Here is the code you wrote:
        {previous_code}

        Here is the error it produced:
        {error_trace}

        TASK: Fix the code. Ensure it runs without errors and prints a single letter.
        """
    return base_prompt

def parse_output(output_str):
    if not output_str: return None
    lines = output_str.strip().split('\n')
    # Prioritize last line
    match = re.search(r'\b([A-E])\b', lines[-1].upper())
    if match: return match.group(1)
    # Fallback to full text
    match = re.search(r'\b([A-E])\b', output_str.upper())
    return match.group(1) if match else None

def solve_single_question(item):
    error_trace = None
    previous_code = None
    
    for attempt in range(MAX_RETRIES + 1):
        user_prompt = construct_user_prompt(item, error_trace, previous_code)
        messages = [
            {"role": "system", "content": construct_system_prompt()},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call LLM
        code = query_llm(messages)
        if not code:
            # If query_llm returns None, it means ALL retries failed or Rate Limit is fatal.
            return "API_FAIL", None

        # Execute
        exec_result = execute_model_code(code)
        
        if exec_result['success']:
            parsed_answer = parse_output(exec_result['output'])
            if parsed_answer:
                return parsed_answer, code # SUCCESS
            else:
                error_trace = "Code ran but did not print a single letter (A-E) output."
        else:
            error_trace = exec_result['error']
        
        previous_code = code

    return "MAX_RETRIES", previous_code

def run_logic_pipeline():
    print("--- 🛡️  ROBUST LOGIC PIPELINE (AUTOSAVE ENABLED) ---")
    
    if not os.path.exists(DATA_PATH):
        print("❌ NO DATA FOUND.")
        return

    # 1. Load Data & Previous Progress
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        full_dataset = json.load(f)
    
    results = load_existing_results()
    solved_ids = {r['id'] for r in results}
    
    print(f"📊 Total: {len(full_dataset)} | Solved: {len(solved_ids)} | Remaining: {len(full_dataset) - len(solved_ids)}")

    try:
        for item in full_dataset:
            # SKIP IF ALREADY DONE
            if item['id'] in solved_ids:
                continue

            print(f"\n[QUERY {item['id']}] Thinking...", end='', flush=True)

            # SOLVE
            prediction, final_code = solve_single_question(item)

            # CHECK FOR FATAL CRASH
            if prediction == "API_FAIL":
                print(f"\n\n🚨 FATAL ERROR: API limit reached or network down.")
                print("💾 Saving progress and exiting gracefully...")
                save_results(results)
                sys.exit(1) # Exit script safely

            # EVALUATE
            truth = item['answer'].strip()
            is_correct = (prediction == truth)
            symbol = "✅" if is_correct else "❌"
            
            print(f"\r[QUERY {item['id']}] Pred: {prediction} | True: {truth} | {symbol}")

            # APPEND & SAVE IMMEDIATELY
            log_entry = {
                "id": item['id'],
                "correct": is_correct,
                "prediction": prediction,
                "truth": truth,
                "generated_code": final_code
            }
            results.append(log_entry)
            save_results(results) # <--- THIS SAVES YOU FROM DATA LOSS

    except KeyboardInterrupt:
        print("\n\n🛑 USER INTERRUPTED (Ctrl+C). Saving current progress...")
        save_results(results)
        print("✅ Progress saved. Exiting.")
        sys.exit(0)
    
    # FINAL SUMMARY
    score = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = (score / total * 100) if total > 0 else 0
    
    print("\n" + "="*40)
    print(f"BENCHMARK COMPLETE")
    print(f"ACCURACY: {accuracy:.2f}% ({score}/{total})")
    print(f"LOGS:     {RESULTS_FILE}")
    print("="*40)

if __name__ == "__main__":
    run_logic_pipeline()