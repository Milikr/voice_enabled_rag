import faiss
import numpy as np
from typing import List
from pydantic import BaseModel
import time

class SearchResult(BaseModel):
    text: str
    metadata: dict
    score: float
    
class FastVectorDB:
    def __init__(self, embedding_dim: int = 384): # 384 is dimension for all-MiniLM-L6-v2 / bge-small
        """
        Initializes an in-memory FAISS index using HNSW for ultra-low latency searches.
        HNSW (Hierarchical Navigable Small World) provides millisecond lookup times, essential for <200ms budgets.
        """
        # IndexHNSWFlat is very fast for retrieval
        self.index = faiss.IndexHNSWFlat(embedding_dim, 32)
        self.index.hnsw.efSearch = 64
        
        self.documents = [] # Store original chunks/metadata
        
    def add_documents(self, embeddings: np.ndarray, documents: List[dict]):
        """Adds documents to the index."""
        if len(embeddings) != len(documents):
            raise ValueError("Embeddings and documents must be the same length.")
            
        # FAISS expects float32
        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)
        self.documents.extend(documents)
        
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[SearchResult]:
        """
        Performs a fast approximate nearest neighbor search.
        """
        start_time = time.perf_counter()
        query_embedding = np.array([query_embedding]).astype('float32')
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for j, i in enumerate(indices[0]):
            if i != -1 and i < len(self.documents):
                doc = self.documents[i]
                results.append(SearchResult(
                    text=doc['text'],
                    metadata=doc['metadata'],
                    score=float(distances[0][j])
                ))
                
        latency = (time.perf_counter() - start_time) * 1000
        print(f"[VectorDB] Search completed in {latency:.2f} ms")
        return results
