import httpx
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")

r = httpx.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    json={
        "model": "qwen/qwen3.6-27b",
        "messages": [{"role": "user", "content": "What is a corporation? Answer in one sentence."}],
        "max_tokens": 100,
        "temperature": 0.1
    },
    timeout=30
)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    print("Answer:", r.json()["choices"][0]["message"]["content"])
else:
    print("Error:", r.text)
