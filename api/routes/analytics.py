"""
Analytics routes — read-only aggregations from MongoDB.

GET /analytics              → aggregate stats across all sessions
GET /history/{session_id}   → full message history for one session
GET /health                 → liveness check
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.models import AnalyticsSummary, SessionHistory, MessageRecord
from api import database as db

router = APIRouter(tags=["analytics"])


@router.get("/health", summary="Liveness check")
async def health():
    """Returns 200 OK when the API is running."""
    return {"status": "ok"}


@router.get(
    "/analytics",
    response_model=AnalyticsSummary,
    summary="Aggregate usage statistics",
)
async def analytics():
    """
    Return aggregate statistics computed directly from MongoDB:

    - **total_sessions** — distinct conversation sessions started
    - **total_queries**  — total messages sent across all sessions
    - **queries_today**  — messages sent since midnight UTC today
    - **avg_session_length** — mean messages per session
    - **top_questions**  — most frequent user queries (top 10)
    """
    return AnalyticsSummary(
        total_sessions=await db.total_sessions(),
        total_queries=await db.total_queries(),
        queries_today=await db.queries_today(),
        avg_session_length=await db.avg_session_length(),
        top_questions=await db.top_questions(limit=10),
    )


@router.get(
    "/history/{session_id}",
    response_model=SessionHistory,
    summary="Retrieve full conversation history for a session",
)
async def session_history(session_id: str):
    """
    Return all messages (user + assistant) for a given *session_id*.

    Raises **404** if the session does not exist.
    """
    doc = await db.get_session(session_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    return SessionHistory(
        session_id=session_id,
        created_at=doc["created_at"],
        messages=[
            MessageRecord(
                role=m["role"],
                content=m["content"],
                timestamp=m["timestamp"],
            )
            for m in doc.get("messages", [])
        ],
    )
