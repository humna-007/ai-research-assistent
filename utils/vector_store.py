"""FAISS-backed vector store for semantic similarity search."""

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # normalized vectors -> cosine via inner product
        self.chunks: list[dict] = []

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks count mismatch.")
        if len(embeddings) == 0:
            return
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({"text": chunk["text"], "metadata": chunk["metadata"], "score": float(score)})
        return results

    def search_filtered(self, query_embedding: np.ndarray, source_names: set, top_k: int = 4) -> list[dict]:
        """Restrict retrieval to specific source documents (by filename)."""
        matches = []
        for i, chunk in enumerate(self.chunks):
            if chunk["metadata"]["source"] in source_names:
                vec = self.index.reconstruct(i)
                score = float(np.dot(vec, query_embedding))
                matches.append((score, chunk))
        matches.sort(key=lambda x: x[0], reverse=True)
        return [{"text": c["text"], "metadata": c["metadata"], "score": s} for s, c in matches[:top_k]]

    @property
    def is_empty(self) -> bool:
        return self.index.ntotal == 0