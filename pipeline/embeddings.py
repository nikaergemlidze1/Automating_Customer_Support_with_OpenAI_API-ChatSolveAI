"""Shared embedding utilities — wraps OpenAI Embeddings API."""

import time
import random
import numpy as np
from openai import OpenAI

from .config import EMBED_MODEL, CHUNK_SIZE, OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


# ── Core embedding call ────────────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    model: str = EMBED_MODEL,
    max_retries: int = 6,
    base_delay: float = 1.0,
) -> list[list[float]]:
    """
    Embed a list of strings in a single API call.
    Retries automatically on rate-limit (429) errors with exponential backoff.
    """
    if not texts:
        return []
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as exc:
            msg = str(exc).lower()
            is_rate = any(tok in msg for tok in ("rate limit", "429", "too many requests"))
            if attempt == max_retries - 1 or not is_rate:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"  ⚠ Rate limit – retrying in {delay:.1f}s (attempt {attempt + 1})")
            time.sleep(delay)


def embed_in_chunks(
    texts: list[str],
    chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    """Split *texts* into chunks, embed each, return a stacked float32 array."""
    all_emb: list[list[float]] = []
    for i in range(0, len(texts), chunk_size):
        all_emb.extend(embed_texts(texts[i : i + chunk_size]))
    return np.array(all_emb, dtype=np.float32)


# ── Maths helpers ──────────────────────────────────────────────────────────────

def normalize_rows(X: np.ndarray) -> np.ndarray:
    """L2-normalise each row; safe against zero vectors."""
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def cosine_to_confidence(sim: float) -> float:
    """Map cosine similarity [-1, 1] → confidence [0, 1]."""
    return float(np.clip((float(sim) + 1.0) / 2.0, 0.0, 1.0))
