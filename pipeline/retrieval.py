"""
Hybrid retriever — FAISS (semantic) + BM25 (lexical) fused with
Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    print("⚠ faiss-cpu not installed — falling back to NumPy cosine search.")

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    print("⚠ rank-bm25 not installed — hybrid search will use semantic only.")

from .config import HYBRID_ALPHA, TOP_K_CANDIDATES
from .embeddings import embed_in_chunks, normalize_rows, cosine_to_confidence


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def _rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Merge ranked lists with RRF.  Returns [(doc_id, score), ...] descending."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


# ── HybridRetriever ────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines FAISS semantic search and BM25 lexical search via RRF.

    Parameters
    ----------
    corpus_queries  : List of query strings (used to build the index).
    corpus_answers  : Corresponding response strings.
    top_k           : Number of candidates to return per query.
    alpha           : Unused directly (RRF handles fusion); kept for reference.
    """

    def __init__(
        self,
        corpus_queries: list[str],
        corpus_answers: list[str],
        top_k: int = TOP_K_CANDIDATES,
    ) -> None:
        self.corpus_answers = corpus_answers
        self.top_k = top_k

        # Embed corpus
        emb = embed_in_chunks(corpus_queries)
        self._norm_emb = normalize_rows(emb)  # pre-normalised for cosine via IP

        # FAISS index (inner-product on L2-normalised vectors == cosine similarity)
        if _FAISS_AVAILABLE:
            dim = self._norm_emb.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(self._norm_emb)
        else:
            self._faiss_index = None

        # BM25 index
        if _BM25_AVAILABLE:
            tokenized = [q.lower().split() for q in corpus_queries]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    # ── Internal search methods ────────────────────────────────────────────────

    def _semantic_ranking(self, query: str, top_k: int) -> tuple[list[int], np.ndarray]:
        """Return (ranked_indices, raw_cosine_scores) from FAISS or NumPy."""
        q_emb = normalize_rows(embed_in_chunks([query]))  # (1, dim)

        if self._faiss_index is not None:
            scores, idx = self._faiss_index.search(q_emb, min(top_k, len(self.corpus_answers)))
            return idx[0].tolist(), scores[0]
        else:
            # NumPy fallback
            sims = (q_emb @ self._norm_emb.T)[0]
            top_idx = np.argsort(-sims)[:top_k].tolist()
            return top_idx, sims[top_idx]

    def _lexical_ranking(self, query: str, top_k: int) -> list[int]:
        """Return ranked indices from BM25."""
        if self._bm25 is None:
            return []
        scores = np.array(self._bm25.get_scores(query.lower().split()))
        return np.argsort(-scores)[:top_k].tolist()

    # ── Public API ─────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Retrieve top-k candidates for *query*.

        Returns a list of dicts:
        ```
        {
            "index":      int,   # position in corpus_answers
            "text":       str,   # response text
            "rrf_score":  float, # fused ranking score
            "confidence": float, # cosine-based confidence [0, 1]
        }
        ```
        """
        k = top_k or self.top_k

        sem_idx, sem_scores = self._semantic_ranking(query, k)
        lex_idx             = self._lexical_ranking(query, k)

        # Build a score dict: idx → cosine similarity (for confidence reporting)
        sem_score_map = {int(i): float(s) for i, s in zip(sem_idx, sem_scores)}

        # Fuse with RRF
        if lex_idx:
            merged = _rrf([sem_idx, lex_idx])
        else:
            merged = _rrf([sem_idx])

        results = []
        for doc_id, rrf_score in merged[:k]:
            if doc_id >= len(self.corpus_answers):
                continue
            raw_cos = sem_score_map.get(doc_id, 0.0)
            results.append({
                "index":      doc_id,
                "text":       self.corpus_answers[doc_id],
                "rrf_score":  rrf_score,
                "confidence": cosine_to_confidence(raw_cos),
            })
        return results
