"""Pydantic request / response models for the ChatSolveAI API."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session UUID")
    query:      str = Field(..., min_length=1, max_length=2000)


class SourceDocument(BaseModel):
    content:  str
    metadata: dict[str, Any] = {}


class ChatResponse(BaseModel):
    session_id:       str
    query:            str
    answer:           str
    source_documents: list[SourceDocument] = []
    timestamp:        datetime


# ── Session / History ─────────────────────────────────────────────────────────

class MessageRecord(BaseModel):
    role:      str   # "user" | "assistant"
    content:   str
    timestamp: datetime


class SessionHistory(BaseModel):
    session_id: str
    messages:   list[MessageRecord]
    created_at: datetime


# ── Analytics ─────────────────────────────────────────────────────────────────

class AnalyticsSummary(BaseModel):
    total_sessions:     int
    total_queries:      int
    queries_today:      int
    avg_session_length: float
    top_questions:      list[dict[str, Any]]
