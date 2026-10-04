"""
models/schemas.py — Pydantic Request & Response Schemas
AI Research Paper Assistant

PURPOSE:
  These classes define the EXACT shape of data that:
    - Enters  the API (Request models)  → validated by FastAPI automatically
    - Leaves  the API (Response models) → serialized to JSON automatically

WHY THIS FILE EXISTS:
  Without schemas, your API is a black box:
    - No documentation of what fields are expected
    - No automatic validation (wrong data = silent bugs)
    - No automatic serialization (dict → JSON)

  With schemas:
    - FastAPI reads the type hints → validates every request
    - OpenAPI/Swagger docs auto-generated
    - Response always has a guaranteed shape → frontend can trust it
    - Wrong input → 422 Unprocessable Entity BEFORE your code runs

FLOW:
  JSON request body
        ↓
  FastAPI reads schema
        ↓
  Pydantic validates every field (type, length, range)
        ↓
  If invalid → 422 Unprocessable Entity (your handler never called!)
  If valid   → Python object passed to your handler
        ↓
  Handler returns Python dict or Pydantic model
        ↓
  FastAPI serializes to JSON
        ↓
  HTTP Response to browser
"""

from datetime import datetime
from typing import List, Optional, Any

from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════════════
# UPLOAD SCHEMAS
# Used by: POST /api/upload, POST /api/upload/sync
# ══════════════════════════════════════════════════════════════════

class UploadResponse(BaseModel):
    """
    Returned after a PDF upload.

    Async upload: total_chunks and total_pages are None
    (processing happens in background — we don't know yet).

    Sync upload: total_chunks and total_pages are populated
    (we waited for processing to finish).
    """
    success:       bool
    message:       str
    paper_name:    str  = Field(description="Filename without .pdf extension")
    file_size_mb:  float = Field(description="Size of uploaded file in megabytes")
    total_chunks:  Optional[int] = Field(
        default=None,
        description="Number of text chunks created. None if processing is async."
    )
    total_pages:   Optional[int] = Field(
        default=None,
        description="Number of PDF pages extracted. None if processing is async."
    )


# ══════════════════════════════════════════════════════════════════
# QUERY SCHEMAS
# Used by: POST /api/query, POST /api/query/stream
# ══════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    """
    What the frontend sends when user asks a question.

    Example JSON body:
    {
        "question": "What is self-attention?",
        "filter_paper": "attention_is_all_you_need",
        "top_k": 5
    }
    """
    question: str = Field(
        ...,                      # ... means REQUIRED (no default)
        min_length=3,
        max_length=1000,
        description="The question to ask about the uploaded research papers.",
        examples=["What is the main contribution of this paper?"]
    )
    filter_paper: Optional[str] = Field(
        default=None,
        description=(
            "Restrict retrieval to a specific paper (paper name without .pdf). "
            "If None, searches across ALL uploaded papers."
        ),
        examples=["attention_is_all_you_need"]
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description=(
            "How many chunks to retrieve from ChromaDB. "
            "Overrides the default top_k_results from config. "
            "Range: 1–20."
        )
    )

    # Validator: strip whitespace from the question before validation
    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        """
        Strip leading/trailing whitespace before checking min_length.
        Without this: "   " (spaces) would pass min_length=3 but be useless.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("Question cannot be empty or whitespace only.")
        return stripped

    @field_validator("filter_paper")
    @classmethod
    def clean_paper_name(cls, v: Optional[str]) -> Optional[str]:
        """Normalize paper name: strip whitespace, remove .pdf if present."""
        if v is None:
            return None
        v = v.strip()
        if v.lower().endswith(".pdf"):
            v = v[:-4]   # remove .pdf suffix
        return v if v else None


class SourceCitation(BaseModel):
    """
    Represents one source chunk used to generate the answer.

    Example:
    {
        "paper": "attention_is_all_you_need",
        "page":  3,
        "score": 0.923
    }

    score: cosine similarity between query vector and chunk vector.
    Higher = more relevant. Range roughly 0.0 – 1.0.
    """
    paper: str  = Field(description="Paper name (without .pdf extension)")
    page:  int  = Field(ge=1, description="Page number in the original PDF (1-indexed)")
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Cosine similarity score. 1.0 = identical vectors, 0.0 = unrelated."
    )

    # Round score to 4 decimal places (clean JSON output)
    @field_validator("score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(v, 4)


class QueryResponse(BaseModel):
    """
    Full response returned after a RAG query.

    Example:
    {
        "success":      true,
        "question":     "What is self-attention?",
        "answer":       "Self-attention allows each token...",
        "sources":      [{"paper": "attention.pdf", "page": 3, "score": 0.92}],
        "chunks_used":  5,
        "model":        "gpt-4o-mini",
        "tokens_used":  1834,
        "latency_ms":   1230.5
    }
    """
    success:       bool
    question:      str
    answer:        str
    sources:       List[SourceCitation]   = Field(description="Chunks used to generate this answer")
    chunks_used:   int                    = Field(ge=0, description="Number of chunks passed to LLM")
    model:         str                    = Field(description="LLM model that generated the answer")
    tokens_used:   int                    = Field(ge=0, description="Total tokens consumed (input + output)")
    latency_ms:    Optional[float]        = Field(default=None, description="End-to-end query latency in ms")


# ══════════════════════════════════════════════════════════════════
# PAPERS SCHEMAS
# Used by: GET /api/papers, GET /api/papers/{name}, DELETE /api/papers/{name}
# ══════════════════════════════════════════════════════════════════

class PapersListResponse(BaseModel):
    """
    Lists all papers currently stored in ChromaDB.

    Example:
    {
        "papers": ["attention_is_all_you_need", "bert_paper"],
        "total":  2
    }
    """
    papers: List[str] = Field(description="List of paper names (without .pdf)")
    total:  int       = Field(ge=0, description="Total number of papers in the system")


class PaperStatsResponse(BaseModel):
    """
    Statistics for a single paper.

    Example:
    {
        "paper_name":   "attention_is_all_you_need",
        "total_chunks": 87,
        "total_pages":  15
    }
    """
    paper_name:   str = Field(description="Paper name (without .pdf)")
    total_chunks: int = Field(ge=0, description="Number of text chunks stored in ChromaDB")
    total_pages:  int = Field(ge=0, description="Number of pages in the original PDF")


class DeleteResponse(BaseModel):
    """
    Returned after deleting a paper's chunks from ChromaDB.

    Example:
    {
        "success":    true,
        "message":    "Paper 'attention_paper' deleted. 87 chunks removed.",
        "paper_name": "attention_paper"
    }
    """
    success:    bool
    message:    str
    paper_name: str


# ══════════════════════════════════════════════════════════════════
# ERROR SCHEMA
# ══════════════════════════════════════════════════════════════════

class ErrorResponse(BaseModel):
    """
    Standard error response shape for all API errors.

    Using a consistent error schema means frontend always knows
    what to expect when something goes wrong.

    Example:
    {
        "success":    false,
        "error":      "Paper not found",
        "detail":     "No paper named 'xyz' exists in ChromaDB",
        "request_id": "abc-123"
    }
    """
    success:    bool         = False
    error:      str          = Field(description="Short error message")
    detail:     Optional[Any] = Field(
        default=None,
        description="Additional context — debug info, field errors, etc."
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for log tracing. Report this when filing a bug."
    )
