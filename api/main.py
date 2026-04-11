"""
ChatSolveAI FastAPI application.

Startup sequence
----------------
1. Load LangChain RAG chain (builds FAISS vectorstore from chatbot_responses.json)
2. Mount chat + analytics routers
3. Start serving on port 8000

Run locally
-----------
uvicorn api.main:app --reload --port 8000

In Docker
---------
Handled by docker-compose (see docker-compose.yml).

Interactive docs
----------------
http://localhost:8000/docs  (Swagger UI)
http://localhost:8000/redoc (ReDoc)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pipeline.config import data_path
from pipeline.rag import build_rag_chain
from api.routes.chat      import router as chat_router
from api.routes.analytics import router as analytics_router


# ── Lifespan: build RAG chain once on startup ─────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("▶ Building LangChain RAG chain…")
    app.state.rag = build_rag_chain(
        corpus_path=data_path("chatbot_responses.json"),
        predefined_path=data_path("predefined_responses.json"),
    )
    print("✓ RAG chain ready")
    yield
    # (cleanup here if needed)
    print("◼ Shutting down")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="ChatSolveAI API",
    description=(
        "Production customer-support backend powered by LangChain RAG.\n\n"
        "**Stack**: FastAPI · LangChain · FAISS · GPT-3.5-turbo · MongoDB\n\n"
        "Use `/chat` for blocking responses or `/chat/stream` for SSE streaming."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (allows Streamlit on port 8501 to call this API) ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(chat_router)
app.include_router(analytics_router)


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "ChatSolveAI API",
        "version": "2.0.0",
        "docs":    "/docs",
    }
