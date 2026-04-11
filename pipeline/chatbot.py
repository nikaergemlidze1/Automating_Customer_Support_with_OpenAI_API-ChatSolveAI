"""
ChatSolveAI — main chatbot class.

Orchestrates: intent classification → hybrid retrieval → cross-encoder reranking
→ retrieved response OR GPT-3.5-turbo fallback (with streaming support).
"""

from __future__ import annotations

import json
import numpy as np
from datetime import datetime, timezone
from openai import OpenAI

from .config import CHAT_MODEL, SIM_THRESHOLD, OPENAI_API_KEY
from .retrieval import HybridRetriever
from .reranker import CrossEncoderReranker
from .classifier import IntentClassifier

client = OpenAI(api_key=OPENAI_API_KEY)

_SYSTEM_PROMPT = (
    "You are ChatSolveAI's customer support assistant. "
    "Be concise, accurate, and professional. "
    "If you are unsure about something, acknowledge it and offer to escalate to a human agent. "
    "Keep answers under 3 sentences unless the user explicitly asks for more detail."
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class ChatSolveAI:
    """
    Full retrieval-augmented chatbot with:
    - Intent classification
    - Hybrid FAISS + BM25 retrieval
    - Cross-encoder reranking
    - GPT-3.5-turbo fallback (streaming or blocking)
    - Multi-turn conversation history
    - Structured interaction log
    """

    def __init__(
        self,
        retriever:  HybridRetriever,
        reranker:   CrossEncoderReranker,
        classifier: IntentClassifier,
        threshold:  float = SIM_THRESHOLD,
    ) -> None:
        self.retriever  = retriever
        self.reranker   = reranker
        self.classifier = classifier
        self.threshold  = threshold

        self.messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        self.log:      list[dict] = []

        # After a respond_stream() call, metadata is stored here for the caller to read
        self.last_meta: dict = {}

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_entry(
        self,
        query:      str,
        answer:     str,
        source:     str,
        confidence: float,
        intent:     str,
        candidates: list[dict],
    ) -> dict:
        return {
            "query_text":        query,
            "retrieved_response": answer,
            "timestamp":         _now_iso(),
            "confidence_score":  round(float(np.clip(confidence, 0.0, 1.0)), 6),
            "source":            source,      # "retrieved" | "generated"
            "intent":            intent,
            "top_candidates":    [c["text"] for c in candidates],
        }

    def _generate_blocking(self) -> str:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=self.messages,
            temperature=0.2,
            max_tokens=180,
        )
        return resp.choices[0].message.content.strip()

    def _route(self, query: str) -> tuple[str, float, str, list[dict]]:
        """
        Core routing logic.
        Returns (answer, confidence, source, reranked_candidates).
        """
        intent, _          = self.classifier.classify(query)
        candidates         = self.retriever.search(query)
        reranked           = self.reranker.rerank(query, candidates)

        top        = reranked[0] if reranked else None
        confidence = top["confidence"] if top else 0.0

        if top and confidence >= self.threshold:
            return top["text"], confidence, "retrieved", reranked
        else:
            # Will be filled by caller (blocking or streaming)
            return "", max(0.50, confidence * 0.85), "generated", reranked

    # ── Public API — blocking ─────────────────────────────────────────────────

    def respond(self, query: str) -> dict:
        """Process one turn, block until response is complete. Returns log entry."""
        intent, _            = self.classifier.classify(query)
        candidates           = self.retriever.search(query)
        reranked             = self.reranker.rerank(query, candidates)
        top                  = reranked[0] if reranked else None
        confidence           = top["confidence"] if top else 0.0

        self.messages.append({"role": "user", "content": query})

        if top and confidence >= self.threshold:
            answer, source = top["text"], "retrieved"
        else:
            answer         = self._generate_blocking()
            confidence     = max(0.50, confidence * 0.85)
            source         = "generated"

        self.messages.append({"role": "assistant", "content": answer})

        entry = self._build_entry(query, answer, source, confidence, intent, reranked)
        self.log.append(entry)
        return entry

    # ── Public API — streaming ─────────────────────────────────────────────────

    def respond_stream(self, query: str):
        """
        Generator that yields text chunks suitable for ``st.write_stream()``.

        After the generator is exhausted, ``self.last_meta`` contains:
        ``{source, confidence, intent, candidates}``.
        """
        intent, _  = self.classifier.classify(query)
        candidates = self.retriever.search(query)
        reranked   = self.reranker.rerank(query, candidates)
        top        = reranked[0] if reranked else None
        confidence = top["confidence"] if top else 0.0

        self.messages.append({"role": "user", "content": query})

        if top and confidence >= self.threshold:
            answer = top["text"]
            source = "retrieved"
            yield answer   # single yield — no network call needed
        else:
            confidence = max(0.50, confidence * 0.85)
            source     = "generated"
            stream     = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=self.messages,
                temperature=0.2,
                max_tokens=180,
                stream=True,
            )
            answer = ""
            for chunk in stream:
                delta  = chunk.choices[0].delta.content or ""
                answer += delta
                yield delta

        self.messages.append({"role": "assistant", "content": answer})

        entry = self._build_entry(query, answer, source, confidence, intent, reranked)
        self.log.append(entry)

        # Store metadata for the Streamlit caller to read post-stream
        self.last_meta = {
            "source":     source,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "intent":     intent,
            "candidates": reranked,
        }

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_log(self, path: str = "sample_chatbot_responses.json") -> None:
        """Save interaction log (graded schema — excludes internal fields)."""
        graded = [
            {k: v for k, v in e.items() if k not in ("source", "intent", "top_candidates")}
            for e in self.log
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(graded, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved {len(graded)} interactions → {path}")

    def reset(self) -> None:
        """Clear conversation history (keep system prompt) and log."""
        self.messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        self.log      = []
        self.last_meta = {}
