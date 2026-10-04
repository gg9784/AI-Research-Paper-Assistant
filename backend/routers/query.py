"""
routers/query.py — Query Router
AI Research Paper Assistant

PURPOSE:
  Handles the question-answering endpoints.
  Receives a user's question → delegates to llm_service → returns answer.

TWO ENDPOINTS:
  POST /api/query         → full answer (synchronous)
  POST /api/query/stream  → token-by-token SSE stream (async)

HTTP LIFECYCLE (POST /api/query):
  Browser
    ↓  POST /api/query  {question: "...", filter_paper: null, top_k: 5}
  FastAPI + Pydantic validates QueryRequest
    ↓  question: str ✅  top_k: int ✅
  ask_question(request: QueryRequest)
    ↓
  llm_service.get_rag_answer()
    ↓  [vector search + GPT call]
  QueryResponse(success=True, answer=..., sources=[...])
    ↓  Pydantic serialises → JSON
  HTTP 200 response to browser

CALLED BY:  FastAPI via main.py include_router()
CALLS:      llm_service.get_rag_answer(), llm_service.stream_rag_answer()
"""

import logging
import time
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from models.schemas import QueryRequest, QueryResponse
from services.llm_service import get_rag_answer, stream_rag_answer

logger = logging.getLogger(__name__)

router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
# POST /api/query — Synchronous RAG Query
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/",
    response_model=QueryResponse,
    summary="Ask a question (full response)",
    description=(
        "Ask a question about the uploaded research papers. "
        "Returns a complete grounded answer with source citations. "
        "Use `/stream` for a real-time typewriter effect."
    ),
)
async def ask_question(request: QueryRequest, http_request: Request):
    """
    FULL (non-streaming) RAG query.

    FLOW:
      Pydantic already validated request fields (question, top_k, filter_paper)
      before this function is called — no manual re-validation needed here.

      ┌────────────────────────────────────────┐
      │ POST /api/query                        │
      │ {                                      │
      │   "question": "What is attention?",    │
      │   "filter_paper": null,                │
      │   "top_k": 5                           │
      │ }                                      │
      └────────────────┬───────────────────────┘
                       ↓  Pydantic validated → QueryRequest object
               ask_question(request)
                       ↓
               get_rag_answer(query, filter_paper, top_k)
                 → similarity_search()    [FAISS retrieval]
                 → build_context()        [prompt construction]
                 → GPT-4o-mini API call   [generation]
                 → build_sources_list()   [citation dedup]
                       ↓
               QueryResponse(success=True, answer=..., sources=[...])
                       ↓  FastAPI serialises with response_model
               HTTP 200 JSON

    WHY `response_model=QueryResponse`?
      FastAPI validates the return value against QueryResponse.
      If llm_service returns an unexpected field, it's stripped.
      This prevents accidental data leakage (e.g., internal IDs).
      Swagger /docs shows the exact response shape.
    """
    request_id = getattr(http_request.state, "request_id", "N/A")
    logger.info(
        "Query received | request_id=%s | question='%s...' | filter=%s | top_k=%s",
        request_id, request.question[:60], request.filter_paper, request.top_k,
    )

    try:
        result = get_rag_answer(
            query=request.question,
            filter_paper=request.filter_paper,
            top_k=request.top_k,
        )

        logger.info(
            "Query answered | request_id=%s | chunks=%d | tokens=%d | latency=%.0fms",
            request_id,
            result.get("chunks_used", 0),
            result.get("tokens_used", 0),
            result.get("latency_ms", 0),
        )

        return QueryResponse(
            success=True,
            question=request.question,
            answer=result["answer"],
            sources=result["sources"],
            chunks_used=result["chunks_used"],
            model=result.get("model", ""),
            tokens_used=result.get("tokens_used", 0),
            latency_ms=result.get("latency_ms"),
        )

    except Exception as e:
        # Log the full exception with stack trace for debugging
        logger.error(
            "Query failed | request_id=%s | error=%s",
            request_id, str(e), exc_info=True,
        )
        # Raise HTTP 500 — the global handler in main.py also catches this,
        # but we raise explicitly here for clarity.
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# POST /api/query/stream — Streaming RAG Query (SSE)
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/stream",
    summary="Ask a question (streaming SSE)",
    description=(
        "Ask a question and receive the answer token-by-token via Server-Sent Events. "
        "Use this endpoint for a real-time typewriter effect in the UI. "
        "Response Content-Type: text/event-stream."
    ),
    response_class=StreamingResponse,   # tells Swagger this returns a stream
)
async def stream_question(request: QueryRequest, http_request: Request):
    """
    STREAMING RAG query via Server-Sent Events (SSE).

    HOW SSE WORKS IN THIS ENDPOINT:
      1. Browser opens connection to POST /api/query/stream
      2. FastAPI calls stream_question()
      3. stream_question returns a StreamingResponse
      4. StreamingResponse wraps generate() — an async generator
      5. FastAPI keeps the HTTP connection OPEN
      6. Each `yield` in generate() sends one event to the browser
      7. Browser EventSource receives events in real-time
      8. When generate() exhausts, the connection closes

    SSE WIRE FORMAT:
      Each event must be: "data: <content>\n\n"
      (Two newlines mark the end of one event)

      Actual bytes sent over the wire:
        data: Self\n\n
        data: -attention\n\n
        data:  is\n\n
        data:  a\n\n
        data:  mechanism\n\n
        ...
        data: [DONE]\n\n

    RESPONSE HEADERS:
      Content-Type:      text/event-stream
        → Tells browser this is SSE, not regular JSON
      Cache-Control:     no-cache
        → Prevents proxies from buffering the stream
      X-Accel-Buffering: no
        → Disables nginx buffering (essential for streaming behind nginx!)
        → Without this: nginx collects all tokens before sending → no streaming!

    WHY NO `response_model` HERE?
      `response_model` only works with JSON responses.
      StreamingResponse is a raw HTTP stream — Pydantic can't validate it.
      The /docs page shows "No schema" for this endpoint.

    ⚠️  LIMITATION:
      The stream only sends text tokens. Metadata (sources, tokens_used)
      is NOT sent. The frontend does not receive citations for stream responses.

    🚀 PRODUCTION IMPROVEMENT:
      Send metadata as a final special SSE event before [DONE]:
        yield f'data: {json.dumps({"type": "metadata", "sources": [...] })}\n\n'
        yield "data: [DONE]\n\n"
      Frontend checks event.type → if "metadata" → parse sources, display citations.
    """
    request_id = getattr(http_request.state, "request_id", "N/A")
    logger.info(
        "Stream query | request_id=%s | question='%s...'",
        request_id, request.question[:60],
    )

    async def generate():
        """
        Async generator that wraps stream_rag_answer() in SSE format.

        Each `yield` sends one SSE event to the browser.
        SSE format: "data: <content>\n\n"

        stream_rag_answer() yields raw token strings.
        We wrap each in the SSE format before yielding to StreamingResponse.
        """
        token_count = 0
        try:
            async for token in stream_rag_answer(
                query=request.question,
                filter_paper=request.filter_paper,
                top_k=request.top_k,
            ):
                token_count += 1
                yield f"data: {token}\n\n"      # SSE event format

            yield "data: [DONE]\n\n"             # signals stream is complete

        except Exception as e:
            logger.error(
                "Stream failed mid-generation | request_id=%s | error=%s",
                request_id, str(e), exc_info=True,
            )
            # Send error as SSE event so frontend can handle it
            yield f"data: [ERROR] Generation failed: {str(e)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            logger.info(
                "Stream complete | request_id=%s | tokens_sent=%d",
                request_id, token_count,
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",        # no proxy buffering
            "X-Accel-Buffering": "no",              # no nginx buffering
            "Connection":        "keep-alive",      # explicit keep-alive
        },
    )
