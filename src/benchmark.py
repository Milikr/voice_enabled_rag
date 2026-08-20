import asyncio
import numpy as np
from .chunking_strategy import AdvancedChunker
from .vector_db import FastVectorDB
from .harness import RAGHarness

def calculate_percentiles(latencies):
    return {
        "P50": np.percentile(latencies, 50),
        "P70": np.percentile(latencies, 70),
        "P100": np.percentile(latencies, 100),
    }

async def run_benchmark(num_queries=100):
    print(f"Initializing Benchmark for {num_queries} queries...")
    
    # 1. Setup Data Pipeline
    chunker = AdvancedChunker()
    vector_db = FastVectorDB()
    
    # Mocking some dataset ingestion
    doc_text = "The MS MARCO dataset is a large scale machine reading comprehension dataset."
    chunks = chunker.process_document("doc_1", "Intro", doc_text)
    
    # Mocking embeddings for the chunks
    embeddings = [np.random.rand(384) for _ in chunks]
    vector_db.add_documents(embeddings, [c.model_dump() for c in chunks])
    
    # 2. Setup Harness
    harness = RAGHarness(vector_db=vector_db, chunker=chunker)
    
    total_latencies = []
    stt_latencies = []
    retrieval_latencies = []
    llm_latencies = []
    
    print("\nStarting queries...")
    for i in range(num_queries):
        # Mock audio input
        audio = b"mock_audio_bytes"
        
        answer, metrics = await harness.process_voice_query(audio)
        
        total_latencies.append(metrics["total_latency_ms"])
        stt_latencies.append(metrics["stt_latency_ms"])
        retrieval_latencies.append(metrics["retrieval_latency_ms"])
        llm_latencies.append(metrics["llm_latency_ms"])
        
    print("\n=== Latency Benchmark Results (ms) ===")
    
    print("\n[Total End-to-End Latency]")
    for p, val in calculate_percentiles(total_latencies).items():
        print(f"{p}: {val:.2f} ms")
        
    print("\n[STT Latency]")
    for p, val in calculate_percentiles(stt_latencies).items():
        print(f"{p}: {val:.2f} ms")
        
    print("\n[Retrieval Latency]")
    for p, val in calculate_percentiles(retrieval_latencies).items():
        print(f"{p}: {val:.2f} ms")
        
    print("\n[LLM Generation Latency]")
    for p, val in calculate_percentiles(llm_latencies).items():
        print(f"{p}: {val:.2f} ms")
        
if __name__ == "__main__":
    asyncio.run(run_benchmark())
