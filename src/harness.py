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
        # Note: In a real implementation, you'd hit https://api.sarvam.ai/speech-to-text
        # Mocking for the hackathon skeleton if API key is not present
        if not self.sarvam_api_key:
            await asyncio.sleep(0.05) # mock 50ms latency
            return "What is the capital of India?"
            
        async with httpx.AsyncClient() as client:
            # Implement real HTTP call here
            # e.g., files={'file': audio_bytes}
            pass
        return ""

    async def generate_answer(self, query: str, context: str) -> str:
        """Calls Groq API for ultra-fast generation."""
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
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.1,
            "max_tokens": 100 # Keep it short for voice
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

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
