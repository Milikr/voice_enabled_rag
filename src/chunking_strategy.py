import re
from typing import List, Dict, Any
from pydantic import BaseModel

class Chunk(BaseModel):
    text: str
    metadata: Dict[str, Any]

class AdvancedChunker:
    def __init__(self, semantic_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the chunker with a small, fast sentence transformer model for semantic boundary detection.
        """
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            self.model = SentenceTransformer(semantic_model_name)
            self.cosine_similarity = cosine_similarity
            self.np = np
        except Exception as e:
            self.model = None
            print(f"Warning: Failed to load semantic model due to {e}. Semantic chunking will fallback to recursive.")

    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple regex-based sentence splitter."""
        sentences = re.split(r'(?<=[.!?]) +', text)
        return [s.strip() for s in sentences if s.strip()]

    def semantic_chunking(self, text: str, threshold: float = 0.5, max_chunk_size: int = 500) -> List[str]:
        """
        Groups sentences together based on semantic similarity.
        If the cosine similarity between consecutive sentences drops below the threshold, a new chunk is started.
        """
        if not self.model:
            return self.recursive_character_chunking(text, chunk_size=max_chunk_size)

        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            sim = self.cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
            
            # If sentences are semantically dissimilar, or chunk is getting too big, break chunk
            if sim < threshold or sum(len(s) for s in current_chunk) > max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def recursive_character_chunking(self, text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
        """Fallback method for fast, fixed-size overlapping chunking."""
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap
        return chunks

    def process_document(self, doc_id: str, title: str, text: str, url: str = "") -> List[Chunk]:
        """
        Processes a full document, generating multiple types of chunks (Small-to-Big approach).
        We store the original large document text in the metadata of smaller chunks.
        This allows the retriever to find the specific small chunk, but return the larger context to the LLM.
        """
        # 1. First, create a large "Parent" chunk
        parent_chunk_text = text[:2000] # truncate for safety
        
        # 2. Break down into semantic "Child" chunks
        child_texts = self.semantic_chunking(text)
        
        chunks = []
        for i, child_text in enumerate(child_texts):
            # Metadata-aware chunking
            meta = {
                "doc_id": doc_id,
                "title": title,
                "url": url,
                "chunk_index": i,
                "type": "child",
                # The crucial part for Small-to-Big retrieval:
                "parent_context": parent_chunk_text 
            }
            chunks.append(Chunk(text=child_text, metadata=meta))
            
        return chunks

# Example usage:
if __name__ == "__main__":
    chunker = AdvancedChunker()
    doc_text = "SpaceX designs, manufactures and launches advanced rockets and spacecraft. The company was founded in 2002 to revolutionize space technology, with the ultimate goal of enabling people to live on other planets. In other news, the stock market crashed today. Investors are very worried about inflation."
    
    chunks = chunker.process_document(
        doc_id="doc_1", 
        title="SpaceX & Market News", 
        text=doc_text
    )
    for c in chunks:
        print(f"--- Chunk {c.metadata['chunk_index']} ---")
        print(c.text)
