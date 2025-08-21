"""
Thin wrapper around FAISS + metadata store.
"""
from typing import List, Dict
import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .config import settings


class Retriever:
    def __init__(self, index_path: Path = None, store_path: Path = None, model_name: str = None):
        self.index_path = index_path or settings.index_path
        self.store_path = store_path or settings.store_path
        self.model_name = model_name or settings.MODEL_NAME

        assert self.index_path.exists(), f"FAISS index not found at {self.index_path}"
        assert self.store_path.exists(), f"Store not found at {self.store_path}"

        self.index = faiss.read_index(str(self.index_path))
        with self.store_path.open("r", encoding="utf-8") as f:
            self.store = json.load(f)

        self.model = SentenceTransformer(self.model_name)

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        top_k = top_k or settings.TOP_K
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores, idxs = self.index.search(q_emb, top_k)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        corpus = self.store["corpus"]
        results = []
        for i, score in zip(idxs, scores):
            if i == -1:
                continue
            item = corpus[i]
            results.append({
                "score": float(score),
                "page": item["page"],
                "chunk_id": item["chunk_id"],
                "text": item["text"]
            })
        return results