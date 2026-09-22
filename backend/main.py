import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from config import settings
from routers import upload, query, papers

# ─── Lifespan: runs once on startup ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create upload directory if it does not exist
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.chroma_db_path, exist_ok=True)
    print("[OK] AI Research Paper Assistant backend started")
    yield
    print("[STOP] Backend shutting down")

# ─── App Instance ───────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Research Paper Assistant",
    description="RAG-powered Q&A over research papers using GPT-4 + ChromaDB",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS Middleware ─────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ────────────────────────────────────────────────────────────────
app.include_router(upload.router, prefix="/api/upload",  tags=["Upload"])
app.include_router(query.router,  prefix="/api/query",   tags=["Query"])
app.include_router(papers.router, prefix="/api/papers",  tags=["Papers"])

# ─── Health Check ───────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "ok",
        "message": "AI Research Paper Assistant is running",
        "model": settings.llm_model,
        "embedding": settings.embedding_model,
    }

@app.get("/", tags=["Root"])
async def root():
    return {"message": "Welcome to AI Research Paper Assistant API 🧠"}
