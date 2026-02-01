import pandas as pd
import json
import re

# ==========================================
# 1. SETUP & LOADING
# ==========================================
df1 = pd.read_csv('data/raw/file1.csv', encoding='utf-8')
df2 = pd.read_csv('data/raw/file2.csv', encoding='utf-8')

df = pd.concat([df1, df2], ignore_index=True)
df.columns = df.columns.str.lower().str.strip()

# ==========================================
# 2. DATA PROCESSING
# ==========================================
df['paragraf'] = df['paragraf'].ffill()
df['yıl'] = df['yıl'].ffill()

def clean_text(text):
    """
    Flattens multiline text into a single line.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def parse_options(text):
    """
    Extracts options into a dictionary first.
    """
    if not isinstance(text, str):
        return {} # Return empty dict if missing

    clean_raw_text = clean_text(text)
    
    # Regex to capture Letter and Text
    pattern = r'([A-E])\s*\)\s*(.*?)(?=\s*[A-E]\s*\)|$)'
    matches = re.findall(pattern, clean_raw_text)
    
    # Returns {'A': 'Text...', 'B': 'Text...'}
    return {k: v.strip() for k, v in matches}

# ==========================================
# 3. BUILD JSON
# ==========================================
benchmark_data = []

for index, row in df.iterrows():
    context_clean = clean_text(str(row['paragraf']))
    question_clean = clean_text(str(row['soru']))
    
    # 1. Parse into a dictionary first
    options_dict = parse_options(row['şıklar'])
    
    # 2. Rebuild the list WITH the letters included
    formatted_choices = []
    for letter in ['A', 'B', 'C', 'D', 'E']:
        # Get text for the letter, default to empty string if missing
        val = options_dict.get(letter, "")
        
        # Only add the "A) " prefix if there is actual text
        if val:
            formatted_choices.append(f"{letter}) {val}")
        else:
            formatted_choices.append("") 

    entry = {
        "id": f"{row['yıl']}_{index}",
        "context": context_clean,
        "question": question_clean,
        "choices": formatted_choices, # Now contains "A) Text"
        "answer": str(row['cevap']).strip()
    }
    
    benchmark_data.append(entry)

# ==========================================
# 4. EXPORT
# ==========================================
output_file = 'data/processed/turkish_benchmark_with_letters.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(benchmark_data, f, ensure_ascii=False, indent=4)

print(f"Done! Saved to {output_file}")