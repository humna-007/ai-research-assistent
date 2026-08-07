"""Sentence embedding generation using Sentence Transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingGenerator:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        )
        return embeddings.astype("float32")