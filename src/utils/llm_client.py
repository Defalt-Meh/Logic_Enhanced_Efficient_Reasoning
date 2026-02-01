import os
import time
import random
from groq import Groq, RateLimitError, BadRequestError, APIConnectionError, APITimeoutError
from dotenv import load_dotenv

load_dotenv()

API_KEYS = [val for key, val in os.environ.items() if key.startswith("GROQ_API_KEY")]
if not API_KEYS:
    raise ValueError("❌ FATAL: No GROQ_API_KEY found in .env file.")

MODEL_CASCADE = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192"
]

class RobustClient:
    def __init__(self):
        self.key_index = 0
        self.clients = [Groq(api_key=k) for k in API_KEYS]
        self.current_client = self.clients[0]
        self.blacklisted_models = set()

    def rotate_key(self):
        if len(self.clients) > 1:
            self.key_index = (self.key_index + 1) % len(self.clients)
            self.current_client = self.clients[self.key_index]
            return True
        return False

    def query(self, messages, temperature=0.1, timeout=None):
        for model in MODEL_CASCADE:
            if model in self.blacklisted_models: continue

            for attempt in range(3):
                try:
                    response = self.current_client.chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        timeout=timeout
                    )
                    # RETURN TUPLE: (CONTENT, MODEL_USED)
                    return response.choices[0].message.content, model

                except APITimeoutError:
                    print(f"⏱️  TIMEOUT on {model}. Retrying...")
                    continue 

                except RateLimitError as e:
                    error_msg = str(e).lower()
                    if "tokens per day" in error_msg:
                        print(f"⚠️  DAILY LIMIT on {model}. Blacklisting.")
                        self.blacklisted_models.add(model)
                        if self.rotate_key(): continue 
                        break
                    
                    if "request too large" in error_msg:
                        print(f"⚠️  PROMPT TOO BIG for {model}. Skipping.")
                        break

                    wait_time = 5 * (attempt + 1) + random.uniform(1, 2)
                    print(f"⏳ Rate Limit on {model}. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    if self.rotate_key(): continue

                except BadRequestError:
                    print(f"❌ MODEL DEAD: {model}. Blacklisting.")
                    self.blacklisted_models.add(model)
                    break

                except Exception as e:
                    print(f"⚠️  Error: {e}")
                    break 

        return None, None # Return Tuple

llm_manager = RobustClient()

def query_llm(messages, temperature=0, timeout=None):
    return llm_manager.query(messages, temperature, timeout)