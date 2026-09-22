"""
Query Router
Phase 6: Handle user questions and return RAG answers
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models.schemas import QueryRequest, QueryResponse
from services.llm_service import get_rag_answer, stream_rag_answer

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Ask a question about the uploaded research papers.
    Returns a grounded answer with source citations.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = get_rag_answer(
            query=request.question,
            filter_paper=request.filter_paper,
            top_k=request.top_k,
        )
        return QueryResponse(
            success=True,
            question=request.question,
            answer=result["answer"],
            sources=result["sources"],
            chunks_used=result["chunks_used"],
            model=result.get("model", ""),
            tokens_used=result.get("tokens_used", 0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@router.post("/stream")
async def stream_question(request: QueryRequest):
    """
    Streaming endpoint - returns answer token by token (Server-Sent Events).
    The React frontend consumes this for a typewriter effect.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    async def generate():
        async for token in stream_rag_answer(
            query=request.question,
            filter_paper=request.filter_paper,
            top_k=request.top_k,
        ):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
