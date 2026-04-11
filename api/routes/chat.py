"""
Chat routes — /chat and /chat/stream.

POST /chat              → blocking JSON response
POST /chat/stream       → token-by-token SSE stream
DELETE /session/{id}    → wipe session from MongoDB
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.models import ChatRequest, ChatResponse, SourceDocument
from api import database as db

router = APIRouter(prefix="/chat", tags=["chat"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_rag(request: Request):
    """Pull the shared LangChainRAG instance from app state."""
    return request.app.state.rag


# ── Blocking chat ─────────────────────────────────────────────────────────────

@router.post("", response_model=ChatResponse, summary="Send a message (blocking)")
async def chat(payload: ChatRequest, request: Request):
    """
    Send a user message and receive a full response.

    - **session_id**: UUID identifying the conversation session
    - **query**: User's message text

    The response includes the answer, source documents used for retrieval,
    and a server-side timestamp.
    """
    rag = _get_rag(request)

    await db.ensure_session(payload.session_id)
    await db.append_message(payload.session_id, "user", payload.query)

    result = rag.chat(payload.query)

    answer  = result["answer"]
    sources = [s["content"] for s in result.get("source_documents", [])]

    await db.append_message(payload.session_id, "assistant", answer)
    await db.log_query(payload.session_id, payload.query, answer, sources)

    return ChatResponse(
        session_id=payload.session_id,
        query=payload.query,
        answer=answer,
        source_documents=[
            SourceDocument(content=s["content"], metadata=s.get("metadata", {}))
            for s in result.get("source_documents", [])
        ],
        timestamp=datetime.now(timezone.utc),
    )


# ── Streaming chat ────────────────────────────────────────────────────────────

@router.post("/stream", summary="Send a message (SSE streaming)")
async def chat_stream(payload: ChatRequest, request: Request):
    """
    Stream response tokens via Server-Sent Events (SSE).

    Each event is a JSON object:
    ```
    data: {"token": "..."}\n\n
    ```
    The final event is:
    ```
    data: [DONE]\n\n
    ```

    The Streamlit client consumes this with ``requests.post(..., stream=True)``
    and passes chunks to ``st.write_stream()``.
    """
    rag = _get_rag(request)

    await db.ensure_session(payload.session_id)
    await db.append_message(payload.session_id, "user", payload.query)

    full_answer: list[str] = []

    async def generate():
        async for token in rag.astream(payload.query):
            full_answer.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"

        # Persist after stream exhausted
        answer = "".join(full_answer)
        await db.append_message(payload.session_id, "assistant", answer)
        await db.log_query(payload.session_id, payload.query, answer, [])
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── Session reset ─────────────────────────────────────────────────────────────

@router.delete(
    "/session/{session_id}",
    summary="Delete a session and clear its conversation memory",
)
async def delete_session(session_id: str, request: Request):
    """
    Wipe session history from MongoDB and clear the LangChain memory.

    Returns the number of log entries deleted.
    """
    rag = _get_rag(request)
    rag.reset()

    deleted = await db.delete_session(session_id)
    return {"session_id": session_id, "logs_deleted": deleted}
