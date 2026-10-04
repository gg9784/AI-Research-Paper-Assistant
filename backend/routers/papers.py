"""
routers/papers.py — Paper Management Router
AI Research Paper Assistant

PURPOSE:
  Provides CRUD-like management of indexed research papers.
  All operations work on the FAISS vector store (not the uploaded PDF files).

ENDPOINTS:
  GET    /api/papers               → list all papers
  GET    /api/papers/{name}/stats  → get stats for one paper
  DELETE /api/papers/{name}        → delete paper's embeddings

IMPORTANT DISTINCTION:
  These endpoints manage EMBEDDINGS in the vector store.
  They do NOT manage the original PDF files in ./uploads/.
  Deleting a paper here removes its vectors from FAISS
  but the PDF file remains on disk.

  ⚠️  CURRENT LIMITATION:
    DELETE /api/papers/{name} removes vectors but not the PDF file.
    If user re-uploads the same PDF, it can be re-processed.
    A production system should offer explicit control:
      DELETE /api/papers/{name}?delete_file=true → also deletes PDF
      DELETE /api/papers/{name}?delete_file=false → only removes vectors

CALLED BY:  FastAPI via main.py include_router()
CALLS:      vector_store.list_papers(), get_paper_stats(), delete_paper()
"""

import logging
from fastapi import APIRouter, HTTPException, Path

from models.schemas import PapersListResponse, PaperStatsResponse, DeleteResponse
from services.vector_store import list_papers, get_paper_stats, delete_paper

logger = logging.getLogger(__name__)

router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/papers — List All Papers
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/",
    response_model=PapersListResponse,
    summary="List all indexed papers",
    description=(
        "Returns all paper names currently indexed in the vector store. "
        "Only papers that have been fully processed (chunked + embedded) appear here."
    ),
)
async def list_all_papers():
    """
    Return all unique paper names from the FAISS metadata store.

    HOW IT WORKS:
      list_papers() loads metadata.json → extracts unique "source" values
      → returns sorted list of paper names (without .pdf extension)

    WHY 'total' FIELD?
      Frontend can show: "3 papers indexed" without counting the array.
      Useful for pagination in the future (know total before fetching a page).

    NOTE:
      A paper appears here ONLY after background processing completes.
      If user just uploaded a paper via async /api/upload,
      it may not appear here immediately (still embedding in background).
    """
    papers = list_papers()
    logger.debug("list_all_papers(): %d papers found", len(papers))
    return PapersListResponse(
        papers=papers,
        total=len(papers),
    )


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/papers/{paper_name}/stats — Paper Statistics
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/{paper_name}/stats",
    response_model=PaperStatsResponse,
    summary="Get stats for a specific paper",
    description="Returns total chunk count and unique page count for a given paper.",
)
async def paper_stats(
    paper_name: str = Path(
        ...,
        description="Paper name exactly as stored (without .pdf extension)",
        examples=["attention_is_all_you_need"],
    ),
):
    """
    Return statistics for one specific paper.

    FLOW:
      1. Check paper exists in vector store (prevents 404 surprise from get_paper_stats)
      2. get_paper_stats() scans metadata.json for matching chunks
      3. Returns total_chunks and total_pages (unique page numbers seen)

    WHY CHECK EXISTENCE FIRST?
      Without the existence check:
        get_paper_stats("nonexistent") returns {total_chunks: 0, total_pages: 0}
        Client sees a valid 200 response with zeros — confusing.

      With the existence check:
        → HTTP 404 "Paper 'nonexistent' not found."
        → Client knows the paper doesn't exist

    PATH PARAMETER:
      {paper_name} in the URL.
      FastAPI extracts it automatically from the URL.
      Example: GET /api/papers/attention_paper/stats
               → paper_name = "attention_paper"

    RESPONSE EXAMPLE:
      {
        "paper_name":   "attention_is_all_you_need",
        "total_chunks": 87,
        "total_pages":  15
      }
    """
    papers = list_papers()
    if paper_name not in papers:
        logger.info("paper_stats(): paper not found: %s", paper_name)
        raise HTTPException(
            status_code=404,
            detail=f"Paper '{paper_name}' not found. "
                   f"Available papers: {papers}"
        )

    stats = get_paper_stats(paper_name)
    logger.debug(
        "paper_stats(): %s | chunks=%d | pages=%d",
        paper_name, stats["total_chunks"], stats["total_pages"],
    )
    return PaperStatsResponse(**stats)


# ══════════════════════════════════════════════════════════════════════════════
# DELETE /api/papers/{paper_name} — Delete Paper Embeddings
# ══════════════════════════════════════════════════════════════════════════════

@router.delete(
    "/{paper_name}",
    response_model=DeleteResponse,
    summary="Delete a paper's embeddings",
    description=(
        "Removes ALL chunks and embeddings for the specified paper from the vector store. "
        "The original PDF file in ./uploads/ is NOT deleted. "
        "After deletion, the paper will no longer appear in search results."
    ),
)
async def delete_paper_route(
    paper_name: str = Path(
        ...,
        description="Paper name to delete (without .pdf extension)",
        examples=["attention_is_all_you_need"],
    ),
):
    """
    Delete all embeddings for a specific paper.

    WHAT HAPPENS INTERNALLY (vector_store.delete_paper):
      1. Load all metadata.json and embeddings.npy
      2. Find indices where metadata["source"] == paper_name
      3. Keep all OTHER indices
      4. Rebuild FAISS index from kept embeddings
      5. Save all three files

      Before delete:  870 chunks (10 papers × ~87 chunks each)
      After delete:   783 chunks (9 papers)

    WHAT IS NOT DELETED:
      ❌ ./uploads/paper_name.pdf (the original PDF file stays on disk)
      → Users can re-upload and re-process without re-uploading the file
      → Could be a privacy concern if disk is shared

    WHY USE HTTP DELETE METHOD?
      REST convention:
        GET    → read (safe, idempotent, no side effects)
        POST   → create
        PUT    → update/replace
        DELETE → remove (idempotent — deleting twice = same result)

      Using GET to delete is an anti-pattern (GET should be safe/cacheable).
      DELETE clearly communicates intent to API consumers and documentation.

    IDEMPOTENCY:
      DELETE is idempotent — calling it twice has the same result as once.
      First call:  removes 87 chunks → 404 on second call (paper gone)
      This is correct REST behaviour.

    RESPONSE EXAMPLE:
      {
        "success":    true,
        "message":    "Deleted 87 chunks for paper 'attention_paper'.",
        "paper_name": "attention_paper"
      }
    """
    papers = list_papers()
    if paper_name not in papers:
        logger.info("delete_paper_route(): not found: %s", paper_name)
        raise HTTPException(
            status_code=404,
            detail=f"Paper '{paper_name}' not found."
        )

    deleted = delete_paper(paper_name)

    logger.info(
        "Paper deleted: '%s' | chunks_removed=%d",
        paper_name, deleted,
    )

    return DeleteResponse(
        success=True,
        message=f"Deleted {deleted} chunks for paper '{paper_name}'. "
                f"The original PDF file was NOT deleted.",
        paper_name=paper_name,
    )
