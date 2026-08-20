import os
import asyncio
import numpy as np
import gradio as gr
from dotenv import load_dotenv

from src.chunking_strategy import AdvancedChunker
from src.vector_db import FastVectorDB
from src.harness import RAGHarness

load_dotenv()


print("Initializing Voice RAG System...")

# Initialize components
chunker = AdvancedChunker()
vector_db = FastVectorDB()

from dataset import get_msmarco_passages

# Load a small sample of the MS MARCO dataset (e.g., first 5 queries' passages)
print("Loading real dataset passages...")
msmarco_passages = get_msmarco_passages(limit=5)

all_chunks = []
for i, passage_text in enumerate(msmarco_passages):
    # Process each passage into chunks
    doc_chunks = chunker.process_document(f"msmarco_doc_{i}", f"Passage {i}", passage_text)
    all_chunks.extend(doc_chunks)

# Mock embeddings (since we don't have a real dataset fully embedded yet)
# In production, use self.embedding_model.encode() on the chunks
embeddings = [np.random.rand(384) for _ in all_chunks]
vector_db.add_documents(embeddings, [c.model_dump() for c in all_chunks])
print(f"Indexed {len(all_chunks)} chunks into FastVectorDB!")

# Initialize Harness
harness = RAGHarness(vector_db=vector_db, chunker=chunker)

async def process_audio(audio_path):
    if audio_path is None:
        return "Please record some audio.", "N/A", "N/A"
        
    # Read the audio file bytes to send to Sarvam
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
    except Exception as e:
        return f"Error reading audio: {e}", "N/A", "N/A"
        
    # Run through the pipeline
    answer, metrics = await harness.process_voice_query(audio_bytes)
    
    latency_str = (
        f"STT: {metrics['stt_latency_ms']:.2f} ms\n"
        f"Retrieval: {metrics['retrieval_latency_ms']:.2f} ms\n"
        f"LLM: {metrics['llm_latency_ms']:.2f} ms\n"
        f"Total: {metrics['total_latency_ms']:.2f} ms"
    )
    
    # We will get the real transcription from the harness now
    transcription = metrics.get('transcription', "Failed to transcribe")
    
    return transcription, answer, latency_str

def sync_process(audio):
    """Gradio runs sync functions natively, so we wrap the async call."""
    return asyncio.run(process_audio(audio))

# Build Gradio Interface
with gr.Blocks(title="Voice-Enabled RAG") as demo:
    gr.Markdown("# 🎙️ Voice-Enabled RAG System (Sub-200ms Latency)")
    gr.Markdown("Speak into your microphone to query the knowledge base.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record Question")
            submit_btn = gr.Button("Submit Query", variant="primary")
            
        with gr.Column():
            transcription_box = gr.Textbox(label="Transcription (STT)")
            answer_box = gr.Textbox(label="AI Answer", lines=4)
            latency_box = gr.Textbox(label="Latency Analytics")
            
    submit_btn.click(
        fn=sync_process,
        inputs=[audio_input],
        outputs=[transcription_box, answer_box, latency_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft())
