import sys
import os
from src.pipelines.logic_solver import run_logic_pipeline
from src.pipelines.direct_solver import run_direct_pipeline

def menu():
    print("\n--- LOGIC BENCHMARK ---")
    print("1. PBLISSFUL IGNORANCE (No Logic Solver)")
    print("2. SUMMON INTELLIGENCE (Run Logic Solver)")
    print("3. EXIT (Return to the darkness)")
    
    choice = input("\nCHOOSE (1-3): ")
    
    if choice == '1':
        run_direct_pipeline()
    elif choice == '2':
        run_logic_pipeline()
    elif choice == '3':
        print("SHUTTING DOWN.")
        sys.exit()
    else:
        print("INVALID INPUT. ARE YOU A FED?")
        menu()

if __name__ == "__main__":
    # CHECK ENVIRONMENT FIRST
    if not os.path.exists(".env"):
        print("!!! WARNING: .env FILE MISSING.")
        print("!!! THE API KEY IS REQUIRED FOR STEP 2.")
        
    menu()