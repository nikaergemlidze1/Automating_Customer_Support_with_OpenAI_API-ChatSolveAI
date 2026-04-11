"""
Cross-encoder reranker.

Uses 'cross-encoder/ms-marco-MiniLM-L-6-v2' from HuggingFace — runs locally,
no API key needed.  Falls back gracefully if sentence-transformers is absent.
"""

from __future__ import annotations

import os

from .config import RERANK_MODEL, TOP_K_RERANK

# This project uses sentence-transformers through the PyTorch path only.
# Disabling TensorFlow import avoids environment-specific Keras/TF plugin issues.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    print("⚠ sentence-transformers not installed — reranking step skipped.")


class CrossEncoderReranker:
    """
    Reranks a short candidate list using a cross-encoder relevance model.

    The cross-encoder sees (query, candidate) together, producing a much more
    precise relevance score than the bi-encoder similarity used in retrieval.
    """

    def __init__(self, model_name: str = RERANK_MODEL) -> None:
        if _ST_AVAILABLE:
            print(f"Loading cross-encoder: {model_name}  (first run downloads ~90 MB)")
            self._model = _CrossEncoder(model_name)
        else:
            self._model = None

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = TOP_K_RERANK,
    ) -> list[dict]:
        """
        Sort *candidates* by cross-encoder relevance score.

        Each candidate dict gains a ``rerank_score`` key.
        Returns the top-k highest-scoring candidates.
        """
        if not candidates:
            return []

        if self._model is None:
            # No reranker available — return candidates as-is
            return candidates[:top_k]

        pairs  = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        ranked = sorted(candidates, key=lambda c: -c["rerank_score"])
        return ranked[:top_k]
