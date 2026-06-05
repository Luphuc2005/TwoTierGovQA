"""
FastAPI application entry point.
Wires up routers, middleware, CORS, and startup events.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import get_settings
from app.database import init_db
from app.middleware import RequestLoggingMiddleware, RateLimitMiddleware
from app.routers import auth, chat, conversation, upload
from app.rag_service import init_rag_pipeline

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    logger.info("🚀 Starting Legal Chatbot API")
    init_db()
    logger.info("✅ Database tables initialized")
    
    # Init RAG Pipeline
    try:
        init_rag_pipeline(
            backend=settings.RAG_BACKEND,
            model=settings.RAG_MODEL,
            api_key=settings.GEMINI_API_KEY,
            max_tokens=settings.RAG_MAX_TOKENS,
            temperature=settings.RAG_TEMPERATURE
        )
    except Exception as e:
        logger.error(f"❌ RAG pipeline initialization FAILED: {e}")
        import traceback
        traceback.print_exc()
    yield
    logger.info("👋 Shutting down Legal Chatbot API")


# ── Create app ────────────────────────────────────────────────

app = FastAPI(
    title="Legal Chatbot API",
    description="AI-powered chatbot backend with FastAPI, PostgreSQL, and OpenAI",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────

origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Custom Middleware ─────────────────────────────────────────

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=settings.RATE_LIMIT_PER_MINUTE)

# ── Routers ───────────────────────────────────────────────────

app.include_router(auth.router, prefix="/api")
app.include_router(conversation.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(upload.router, prefix="/api")


# ── Health check ──────────────────────────────────────────────

@app.get("/api/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "service": "legal-chatbot-api"}


@app.get("/api/rag-status", tags=["Health"])
def rag_status():
    from app import rag_service
    return {
        "_initialized": rag_service._initialized,
        "_pipeline_is_none": rag_service._pipeline is None,
        "_reranker_is_none": rag_service._reranker is None,
        "module_id": id(rag_service),
    }
