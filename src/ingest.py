"""
Extract PDF -> chunks -> embeddings -> FAISS index (+metadata).

Why:
- Keep page numbers to cite answers.
- Chunking improves retrieval.
- FAISS gives fast local vector search.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

from .config import settings


def read_pdf_with_pages(pdf_path: Path) -> List[Dict]:
    """Read PDF and return list of dicts: {page, text}"""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.replace("\x00", "").strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping character chunks."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def build_corpus(pages: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    """Create a list of chunk dicts with page numbers."""
    corpus = []
    for p in pages:
        chunks = chunk_text(p["text"], chunk_size, overlap)
        for idx, ch in enumerate(chunks):
            corpus.append({
                "page": p["page"],
                "chunk_id": f"p{p['page']}_c{idx}",
                "text": ch.strip()
            })
    return corpus


def embed_corpus(corpus: List[Dict], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in corpus]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    return embeddings


def save_index_and_store(embeddings: np.ndarray, corpus: List[Dict], index_path: Path, store_path: Path) -> None:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    with store_path.open("w", encoding="utf-8") as f:
        json.dump({"corpus": corpus}, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF policy into FAISS index.")
    parser.add_argument("--pdf", required=True, help="Path to the policy PDF (e.g., data/policy.pdf)")
    parser.add_argument("--chunk_size", type=int, default=settings.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=settings.CHUNK_OVERLAP)
    parser.add_argument("--model", default=settings.MODEL_NAME)
    parser.add_argument("--index_dir", default=settings.INDEX_DIR)
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    assert pdf_path.exists(), f"PDF not found: {pdf_path}"

    pages = read_pdf_with_pages(pdf_path)
    if not pages:
        raise ValueError("No extractable text found in PDF.")

    corpus = build_corpus(pages, args.chunk_size, args.overlap)
    embeddings = embed_corpus(corpus, args.model)

    index_dir = Path(args.index_dir)
    save_index_and_store(embeddings, corpus, index_dir / "index.faiss", index_dir / "store.json")

    print(f"[OK] Ingested {len(corpus)} chunks from {pdf_path.name}")
    print(f"[OK] Saved index to {index_dir/'index.faiss'} and store to {index_dir/'store.json'}")


if __name__ == "__main__":
    main()