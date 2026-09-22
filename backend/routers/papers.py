"""
Papers Router
Phase 6: List, inspect and delete uploaded papers
"""
from fastapi import APIRouter, HTTPException
from models.schemas import PapersListResponse, PaperStatsResponse, DeleteResponse
from services.vector_store import list_papers, get_paper_stats, delete_paper

router = APIRouter()


@router.get("/", response_model=PapersListResponse)
async def list_all_papers():
    """Return a list of all uploaded and indexed papers."""
    papers = list_papers()
    return PapersListResponse(papers=papers, total=len(papers))


@router.get("/{paper_name}/stats", response_model=PaperStatsResponse)
async def paper_stats(paper_name: str):
    """Return stats (chunks, pages) for a specific paper."""
    papers = list_papers()
    if paper_name not in papers:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_name}' not found.")
    stats = get_paper_stats(paper_name)
    return PaperStatsResponse(**stats)


@router.delete("/{paper_name}", response_model=DeleteResponse)
async def delete_paper_route(paper_name: str):
    """Delete all embeddings for a specific paper."""
    papers = list_papers()
    if paper_name not in papers:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_name}' not found.")
    deleted = delete_paper(paper_name)
    return DeleteResponse(
        success=True,
        message=f"Deleted {deleted} chunks for paper '{paper_name}'.",
        paper_name=paper_name,
    )
