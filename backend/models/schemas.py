"""
Pydantic Schemas
Phase 7: Request and Response models for all API endpoints
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Any


# ─── Upload Schemas ──────────────────────────────────────────────────────────
class UploadResponse(BaseModel):
    success: bool
    message: str
    paper_name: str
    file_size_mb: float
    total_chunks: Optional[int] = None
    total_pages: Optional[int] = None


# ─── Query Schemas ───────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="The question to ask")
    filter_paper: Optional[str] = Field(None, description="Limit search to a specific paper")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")


class SourceCitation(BaseModel):
    paper: str
    page: int
    score: float


class QueryResponse(BaseModel):
    success: bool
    question: str
    answer: str
    sources: List[SourceCitation]
    chunks_used: int
    model: str
    tokens_used: int


# ─── Papers Schemas ──────────────────────────────────────────────────────────
class PapersListResponse(BaseModel):
    papers: List[str]
    total: int


class PaperStatsResponse(BaseModel):
    paper_name: str
    total_chunks: int
    total_pages: int


class DeleteResponse(BaseModel):
    success: bool
    message: str
    paper_name: str


# ─── Error Schema ────────────────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[Any] = None
