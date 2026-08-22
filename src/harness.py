import asyncio
import time
import os
import httpx
from typing import Dict, Any, Tuple
from .guardrails import FastGuardrails

class RAGHarness:
    def __init__(self, vector_db, chunker, embedding_model=None):
        self.vector_db = vector_db
        self.chunker = chunker
        self.guardrails = FastGuardrails()
        
        # We will use Groq for ultra-fast inference to hit the <200ms target
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "")
        self.sarvam_api_key = os.environ.get("SARVAM_API_KEY", "")
        
        # A fast local embedding model function (mocked or loaded)
        self.embedding_model = embedding_model 
        
    async def stt_sarvam(self, audio_bytes: bytes) -> str:
        """Calls Sarvam STT asynchronously."""
        if not self.sarvam_api_key:
            await asyncio.sleep(0.05) # mock 50ms latency
            return "What is the capital of India?"
            
        async with httpx.AsyncClient() as client:
            headers = {"api-subscription-key": self.sarvam_api_key}
            files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
            data = {"prompt": ""}
            try:
                response = await client.post("https://api.sarvam.ai/speech-to-text-translate", headers=headers, files=files, data=data)
                response.raise_for_status()
                return response.json().get("transcript", "Failed to parse transcript")
            except Exception as e:
                return f"STT Error: {str(e)}"

    async def generate_answer(self, query: str, context: str) -> str:
        """Calls Groq API for ultra-fast generation."""
        import re
        if not self.groq_api_key:
            await asyncio.sleep(0.05)
            return "Based on the context, the capital of India is New Delhi."
            
        system_prompt = (
            "You are a helpful, extremely fast voice assistant. "
            "Use ONLY the following context to answer the user's question. "
            "If the context does not contain the answer, reply exactly with: "
            "'I cannot answer this based on the provided context.'\n\n"
            f"Context:\n{context}"
        )
        
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "qwen/qwen3.6-27b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                raw = data["choices"][0]["message"]["content"]
                # Strip <think>...</think> reasoning blocks from qwen model output
                clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                return clean if clean else raw.strip()
            except httpx.HTTPStatusError as e:
                print(f"Groq HTTP Error: {e.response.text}")
                return f"Groq API Error: {e.response.status_code} - {e.response.text}"
            except Exception as e:
                print(f"Groq General Error: {str(e)}")
                return f"Groq Error: {str(e)}"

    async def process_voice_query(self, audio_bytes: bytes) -> Tuple[str, Dict[str, float]]:
        """
        The main end-to-end pipeline function.
        Returns the answer and latency metrics.
        """
        metrics = {}
        t0 = time.perf_counter()
        
        # 1. Speech-to-Text
        transcribed_text = await self.stt_sarvam(audio_bytes)
        metrics["stt_latency_ms"] = (time.perf_counter() - t0) * 1000
        metrics["transcription"] = transcribed_text
        
        # 2. Input Guardrails
        t_guard = time.perf_counter()
        if not self.guardrails.check_input(transcribed_text):
            return "I'm sorry, I cannot process that request.", metrics
        metrics["guardrail_latency_ms"] = (time.perf_counter() - t_guard) * 1000
        
        # 3. Retrieval
        t_ret = time.perf_counter()
        # Embed query (sync operation, should be very fast with local ONNX)
        # Mocking embedding generation latency for skeleton
        query_embedding = [0.1] * 384 
        if self.embedding_model:
            query_embedding = self.embedding_model.encode([transcribed_text])[0]
            
        results = self.vector_db.search(query_embedding, k=2)
        
        # Construct context using Small-to-Big retrieval (using parent context)
        context = "\n".join([res.metadata.get('parent_context', res.text) for res in results])
        metrics["retrieval_latency_ms"] = (time.perf_counter() - t_ret) * 1000
        
        # 4. LLM Generation
        t_llm = time.perf_counter()
        answer = await self.generate_answer(transcribed_text, context)
        metrics["llm_latency_ms"] = (time.perf_counter() - t_llm) * 1000
        
        # Total latency
        metrics["total_latency_ms"] = (time.perf_counter() - t0) * 1000
        
        return answer, metrics
