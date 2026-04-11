"""
Intent classifier — routes queries to one of five support categories
using embedding similarity (no extra API calls beyond embeddings).
"""

from __future__ import annotations

import numpy as np
from .config import INTENT_CATEGORIES
from .embeddings import embed_in_chunks, normalize_rows

# Intent → display label + emoji
INTENT_META: dict[str, dict] = {
    "billing":   {"label": "Billing & Payments",   "emoji": "💳", "color": "#FF9800"},
    "account":   {"label": "Account & Security",   "emoji": "🔐", "color": "#2196F3"},
    "shipping":  {"label": "Shipping & Orders",    "emoji": "📦", "color": "#4CAF50"},
    "technical": {"label": "Technical Support",    "emoji": "🛠️", "color": "#9C27B0"},
    "general":   {"label": "General Enquiry",      "emoji": "💬", "color": "#607D8B"},
}


class IntentClassifier:
    """
    Zero-shot intent classifier via embedding cosine similarity.

    Category descriptions are embedded once at construction time.
    Each call to ``classify()`` embeds the query and finds the nearest category.
    """

    def __init__(self) -> None:
        names = list(INTENT_CATEGORIES.keys())
        descs = list(INTENT_CATEGORIES.values())

        cat_emb          = embed_in_chunks(descs)
        self._cat_emb    = normalize_rows(cat_emb)   # (n_categories, dim)
        self._cat_names  = names

    def classify(self, query: str) -> tuple[str, float]:
        """
        Return ``(intent_name, confidence)`` for *query*.

        confidence is the cosine similarity between the query embedding
        and the winning category description (not mapped to [0,1]).
        """
        q_emb = normalize_rows(embed_in_chunks([query]))   # (1, dim)
        sims  = (q_emb @ self._cat_emb.T)[0]              # (n_categories,)
        best  = int(np.argmax(sims))
        return self._cat_names[best], float(sims[best])

    @staticmethod
    def meta(intent: str) -> dict:
        """Return display metadata for an intent string."""
        return INTENT_META.get(intent, INTENT_META["general"])
