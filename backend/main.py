"""
main.py — Application Entry Point
AI Research Paper Assistant (RAG-powered backend)

Responsibilities:
  - Create FastAPI application instance
  - Register all routers with URL prefixes
  - Configure CORS middleware
  - Handle startup / shutdown lifecycle
  - Provide health check and root endpoints
  - Global exception handling
"""

import os
import time
import uuid
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import settings
from routers import upload, query, papers


# ─── Logging Setup ───────────────────────────────────────────────────────────
# Use structured logging instead of print()
# Format: timestamp | level | module | message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Lifespan: Startup + Shutdown ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs BEFORE yield   → server startup (create dirs, load resources)
    Runs AFTER  yield   → server shutdown (cleanup, close connections)
    """
    # ── Startup ──────────────────────────────────────────────────────────────
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.chroma_db_path, exist_ok=True)
    logger.info("Directories verified: uploads=%s | chroma=%s",
                settings.upload_dir, settings.chroma_db_path)
    logger.info(
        "AI Research Paper Assistant started | LLM=%s | Embedding=%s",
        settings.llm_model,
        settings.embedding_model,
    )

    yield  # ← Application runs here, serving all requests

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("AI Research Paper Assistant shutting down cleanly")


# ─── App Instance ────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Research Paper Assistant",
    description=(
        "RAG-powered Q&A system over uploaded research papers.\n\n"
        "**Upload PDFs → Ask Questions → Get Grounded Answers with Citations.**\n\n"
        "Uses ChromaDB for vector storage and GPT-4o-mini for generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
)


# ─── CORS Middleware ──────────────────────────────────────────────────────────
# Required because frontend (port 3000) and backend (port 8000) are different origins.
# Without this the browser blocks responses from a cross-origin server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,      # reads from .env
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],   # explicit, not wildcard
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)


# ─── Request ID Middleware ────────────────────────────────────────────────────
# Attaches a unique request ID to every incoming request.
# Makes tracing logs for a specific request easy in production.
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.perf_counter()

    # Attach to request state (accessible anywhere in handlers)
    request.state.request_id = request_id

    response = await call_next(request)

    # Calculate latency
    latency_ms = (time.perf_counter() - start_time) * 1000

    # Add to response headers (frontend/clients can read this)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"

    logger.info(
        "method=%s path=%s status=%d latency=%.2fms request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
        request_id,
    )
    return response


# ─── Global Exception Handler ────────────────────────────────────────────────
# Catches ANY unhandled exception so raw Python tracebacks
# are NEVER shown to the user (security + clean UX).
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "Unhandled exception | request_id=%s | path=%s | error=%s",
        request_id,
        request.url.path,
        str(exc),
        exc_info=True,   # includes full stack trace in log
    )
    return JSONResponse(
        status_code=500,
        content={
            "success":    False,
            "error":      "An internal server error occurred.",
            "request_id": request_id,   # client can report this for debugging
        },
    )


# ─── Routers ─────────────────────────────────────────────────────────────────
# Each router handles one domain of functionality.
# prefix= gives all routes in that router a URL namespace.
app.include_router(upload.router, prefix="/api/upload", tags=["📄 Upload"])
app.include_router(query.router,  prefix="/api/query",  tags=["🔍 Query"])
app.include_router(papers.router, prefix="/api/papers", tags=["📚 Papers"])


# ─── Health Check ────────────────────────────────────────────────────────────
# Load balancers and Docker HEALTHCHECK poll this endpoint.
# Returns 200 when healthy, signals a problem via non-200 otherwise.
@app.get("/health", tags=["⚙️ System"], summary="Health check")
async def health_check():
    return {
        "status":           "ok",
        "version":          "1.0.0",
        "llm_model":        settings.llm_model,
        "embedding_model":  settings.embedding_model,
        "upload_dir":       settings.upload_dir,
        "vector_db":        settings.chroma_db_path,
    }


# ─── Root ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["⚙️ System"], summary="API root")
async def root():
    return {
        "message": "AI Research Paper Assistant API 🧠",
        "docs":    "/docs",
        "health":  "/health",
    }
