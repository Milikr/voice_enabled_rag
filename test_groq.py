import os
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()

async def test():
    client = httpx.AsyncClient()
    headers = {'Authorization': f"Bearer {os.environ.get('GROQ_API_KEY')}"}
    payload = {
        'model': 'llama-3.1-8b-instant',
        'messages': [{'role': 'user', 'content': 'hi'}]
    }
    resp = await client.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=payload)
    print(resp.status_code)
    print(resp.text)

if __name__ == "__main__":
    asyncio.run(test())
