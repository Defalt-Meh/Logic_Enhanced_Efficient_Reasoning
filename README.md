# Turkish Logic Benchmark

This repository evaluates LLMs on Turkish logic puzzles (ALES/DGS style).

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Add your Groq API key to `.env`
3. Run generation: `python src/generate_dataset.py`
4. Run benchmark: `python src/run_benchmark.py`