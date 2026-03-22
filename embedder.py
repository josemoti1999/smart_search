"""
embedder.py — Lightweight ONNX-based text embeddings via fastembed.
Replaces sentence-transformers + torch entirely (~1.2 GB → ~80 MB).
"""
import numpy as np
import logging
from typing import Optional, List
from fastembed.text import TextEmbedding  # avoids pulling in fastembed image/PIL deps

# all-MiniLM-L6-v2 ONNX — 384 dims, ~22 MB on disk
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIM = 384

_model: Optional[TextEmbedding] = None


def load_model() -> TextEmbedding:
    global _model
    if _model is None:
        logging.info(f'Loading ONNX embedding model: {MODEL_NAME}')
        _model = TextEmbedding(model_name=MODEL_NAME)
        logging.info('ONNX embedding model ready.')
    return _model


def encode(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of strings. Returns float32 array of shape (n, 384).
    Normalized (L2) — compatible with FAISS IndexFlatL2 and cosine similarity.
    """
    model = load_model()
    embeddings = list(model.embed(texts, batch_size=batch_size))
    return np.array(embeddings, dtype=np.float32)


def encode_single(text: str) -> np.ndarray:  # type: ignore
    """Embed one string. Returns shape (384,)."""
    return encode([text])[0]


def is_ready() -> bool:
    """True if model is loaded (fast path — no disk I/O)."""
    return _model is not None
