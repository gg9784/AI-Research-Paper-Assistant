"""
services/llm_service.py — RAG Answer Generation Service
AI Research Paper Assistant

PURPOSE:
  The "generation" half of the RAG pipeline.
  Takes a user question, retrieves relevant chunks (via vector_store),
  constructs a grounded prompt, and calls GPT-4o-mini to generate an answer.

TWO MODES:
  get_rag_answer()    → synchronous, returns complete answer at once
  stream_rag_answer() → async generator, yields tokens as they arrive (SSE)

RAG PIPELINE (this file handles steps 3–6):
  1. [vector_store] Embed query → similarity search → top-K chunks
  2. [this file]    Build context string from chunks
  3. [this file]    Construct system prompt + user message
  4. [this file]    Call GPT-4o-mini
  5. [this file]    Extract answer + token usage
  6. [this file]    Build source citation list

CALLED BY:  routers/query.py (ask_question, stream_question)
CALLS:      vector_store.similarity_search(), OpenAI Chat API

KEY CONCEPTS:
  - System prompt:   permanent instructions to the LLM (its "rules")
  - Context window:  everything the LLM sees in one call
  - Temperature:     controls randomness (0.2 = mostly deterministic)
  - Hallucination:   model generating content not in context
  - Grounding:       anchoring answers to retrieved evidence
  - Streaming (SSE): server sends tokens as generated (typewriter effect)
"""

import logging
import time
from typing import Any, Optional, AsyncGenerator

from openai import OpenAI, AsyncOpenAI

from config import settings
from services.vector_store import similarity_search

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# OPENAI CLIENT SINGLETONS
# ══════════════════════════════════════════════════════════════════════════════

# Sync client — for get_rag_answer()
_sync_client: Optional[OpenAI] = None

# Async client — for stream_rag_answer()
_async_client: Optional[AsyncOpenAI] = None


def _get_sync_client() -> OpenAI:
    """Lazy singleton for synchronous OpenAI client."""
    global _sync_client
    if _sync_client is None:
        _sync_client = OpenAI(api_key=settings.openai_api_key)
    return _sync_client


def _get_async_client() -> AsyncOpenAI:
    """Lazy singleton for asynchronous OpenAI client."""
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _async_client


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert AI research assistant. Your job is to answer
questions about academic research papers accurately and clearly.

Rules you MUST follow:
1. Answer ONLY using the provided context from the papers.
2. If the answer is not in the context, say exactly: "This information was not found in the uploaded papers."
3. Always cite your sources by mentioning the paper name and page number.
4. Be concise but thorough. Use bullet points when listing multiple points.
5. If asked to compare papers, structure your answer clearly by paper.
6. Never make up information or use knowledge outside the provided context.
7. If the context contains conflicting information from different papers, present both perspectives clearly.
"""

# WHY THESE RULES?
# Rule 1: Forces grounding — LLM must stay within retrieved context.
# Rule 2: Defines exact fallback phrasing — predictable failure mode.
#         Frontend can check for this string to detect "no answer" case.
# Rule 3: Citations — user can verify in original paper.
# Rule 4: Format guidance — structured output is easier to read.
# Rule 5: Multi-paper queries — prevents jumbled cross-paper answers.
# Rule 6: Hallucination guard — reinforces rule 1 with extra emphasis.
# Rule 7: New — handles contradictions honestly (important for research).
#
# CAN SYSTEM PROMPTS COMPLETELY ELIMINATE HALLUCINATIONS?
#   NO. The model still has its own weights and can "leak" knowledge.
#   These rules REDUCE hallucinations but never eliminate them.
#   The primary guard against hallucination in RAG is GOOD RETRIEVAL.
#   If the right context is retrieved, the LLM rarely needs to invent.


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_context(chunks: list[dict[str, Any]]) -> str:
    """
    Convert retrieved chunks into a structured, readable context block.

    WHY FORMAT CHUNKS THIS WAY?
      The LLM reads the context as plain text.
      We need it to:
        1. Know which paper each chunk came from (for citations)
        2. Know the page number (for citations)
        3. Be able to distinguish between different sources

      Without [Source N] labels:
        LLM receives: "Attention is all you need... BERT uses bidirectional..."
        → LLM cannot distinguish which paper said what
        → Cannot cite correctly
        → Citations are guessed → potentially hallucinated

      With [Source N] labels:
        LLM receives:
          [Source 1] Paper: 'attention_paper' | Page 3
          Attention is all you need...
          ---
          [Source 2] Paper: 'bert_paper' | Page 7
          BERT uses bidirectional...

        → LLM sees structure → can reference "[Source 1]" in answer
        → Can cite "attention_paper, page 3" correctly

    WHY "---" SEPARATOR?
      Visual separation between sources in the context window.
      Helps the LLM treat each source as a distinct unit.
      Without it, chunks may "bleed" into each other semantically.

    INPUT:
      chunks = [
        {text: "Attention is...", source: "attention_paper", page_number: 3, ...},
        {text: "BERT uses...", source: "bert_paper", page_number: 7, ...},
        ...
      ]

    OUTPUT (string passed to LLM):
      [Source 1] Paper: 'attention_paper' | Page 3
      Attention is all you need...

      ---

      [Source 2] Paper: 'bert_paper' | Page 7
      BERT uses bidirectional encoders...
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[Source {i}] Paper: '{chunk['source']}' | "
            f"Page {chunk['page_number']}"
        )
        parts.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# SOURCES LIST BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_sources_list(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deduplicate chunks → clean list of unique (paper, page) citations.

    WHY DEDUPLICATE?
      Multiple chunks may come from the same paper+page.
      Example: page 3 of attention_paper has 4 overlapping chunks,
               all retrieved because the query matches that page.
      Without dedup:
        sources = [{paper: "attention_paper", page: 3, score: 0.92},
                   {paper: "attention_paper", page: 3, score: 0.89},
                   {paper: "attention_paper", page: 3, score: 0.85},
                   ...]
      Redundant and confusing in the UI.

      After dedup:
        sources = [{paper: "attention_paper", page: 3, score: 0.92}]
      Clean, one entry per unique (paper, page) pair.
      Score kept = highest score from that page (most relevant chunk).

    WHY SORT BY SCORE DESCENDING?
      Most relevant source appears first in the UI.
      Tells user: "Here's the most relevant part of the paper."
    """
    seen: set[tuple[str, int]] = set()
    sources = []

    for chunk in chunks:
        key = (chunk["source"], chunk["page_number"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "paper": chunk["source"],
                "page":  chunk["page_number"],
                "score": chunk["similarity_score"],
            })

    # Sort most relevant first
    return sorted(sources, key=lambda x: x["score"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# SYNCHRONOUS RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def get_rag_answer(
    query:        str,
    filter_paper: Optional[str] = None,
    top_k:        Optional[int] = None,
) -> dict[str, Any]:
    """
    Full synchronous RAG pipeline: question → grounded answer + citations.

    COMPLETE FLOW:
      query: "What is self-attention?"
          ↓
      similarity_search(query, k=5)
        → embed query (OpenAI API)
        → FAISS cosine similarity search
        → top-5 chunks with scores
          ↓
      build_context(chunks)
        → "[Source 1] Paper: '...' | Page 3\n<text>\n---\n..."
          ↓
      Construct user_message:
        "Context from research papers:\n{context}\nQuestion: {query}\n..."
          ↓
      OpenAI Chat Completions API:
        messages = [
          {"role": "system", "content": SYSTEM_PROMPT},
          {"role": "user",   "content": user_message}
        ]
        model       = "gpt-4o-mini"
        temperature = 0.2
        max_tokens  = 1500
          ↓
      response.choices[0].message.content → answer text
      response.usage.total_tokens         → token count
          ↓
      Return dict with answer + sources + metadata

    WHY NOT JUST SEND THE QUESTION TO GPT DIRECTLY?
      GPT has no knowledge of YOUR uploaded papers.
      GPT's training data is a frozen snapshot (knowledge cutoff).
      GPT cannot access private documents.

      Without RAG:
        "What does the attention_paper say about multi-head attention?"
        → GPT: "I don't have access to specific papers you've uploaded."
               OR worse: invents an answer based on general training.

      With RAG:
        We retrieve relevant chunks from YOUR papers.
        We inject them as context.
        GPT answers FROM the provided context.
        → Grounded, citable, accurate for YOUR documents.

    WHY temperature=0.2?
      Temperature controls randomness of token selection.
      temperature=0.0 → always picks highest probability token (deterministic)
      temperature=1.0 → samples from full distribution (creative but inconsistent)
      temperature=2.0 → very random (often incoherent)

      For factual RAG: 0.2 is standard.
      Low enough for consistent, factual answers.
      Not exactly 0.0 because slight variation sometimes helps phrasing.

      IMPORTANT: temperature=0.0 does NOT guarantee truth!
        It only makes the model more deterministic.
        If the model's weights contain wrong information,
        it will deterministically produce that wrong information.
        Grounding via retrieved context is the real hallucination guard.

    WHY max_tokens=1500?
      max_tokens limits OUTPUT tokens only (the answer).
      Does NOT affect input (context + question).

      Token budget for this call:
        SYSTEM_PROMPT:   ~100 tokens
        context (5×500chars ≈ 5×125tokens): ~625 tokens
        user question:   ~20 tokens
        Total input:     ~745 tokens

        gpt-4o-mini context window: 128,000 tokens
        We're well under the limit.

      max_tokens=1500: allows detailed answers without excessive cost.
      Each gpt-4o-mini output token ≈ $0.0000006
      1500 tokens ≈ $0.0009 per query (very cheap).

    PARAMETERS:
      query:        user's question
      filter_paper: optional — restrict retrieval to one paper
      top_k:        optional — override default chunk count

    RETURNS:
      {
        "answer":      str,              # LLM-generated answer
        "sources":     List[dict],       # [{paper, page, score}, ...]
        "chunks_used": int,              # how many chunks retrieved
        "model":       str,              # model name used
        "tokens_used": int,              # total tokens consumed
        "latency_ms":  float,            # end-to-end latency
      }
    """
    if top_k is None:
        top_k = settings.top_k_results

    t_start = time.perf_counter()

    # ── Step 1: Retrieve relevant chunks ─────────────────────────────────────
    chunks = similarity_search(
        query=query,
        k=top_k,
        filter_source=filter_paper,
    )

    if not chunks:
        logger.info("No chunks retrieved for query: '%s'", query[:60])
        return {
            "answer":      "No relevant content found in the uploaded papers. "
                           "Please upload papers first and wait for processing to complete.",
            "sources":     [],
            "chunks_used": 0,
            "model":       settings.llm_model,
            "tokens_used": 0,
            "latency_ms":  round((time.perf_counter() - t_start) * 1000, 2),
        }

    # ── Step 2: Build context from retrieved chunks ───────────────────────────
    context = build_context(chunks)

    # ── Step 3: Construct the full user message ───────────────────────────────
    # WHY THREE PARTS (system + context + question)?
    #   System prompt: LLM's standing instructions (persistent rules)
    #   Context:       the evidence from papers (dynamic, per query)
    #   Question:      what the user actually wants to know
    #
    #   Without context: LLM uses its own knowledge → no grounding
    #   Without system:  LLM has no rules → may ignore context → hallucinate
    #   Without question:LLM doesn't know what to answer → generic summary
    user_message = (
        f"Context from research papers:\n\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        f"Please answer based solely on the context above, "
        f"citing the source papers and page numbers."
    )

    # ── Step 4: Call OpenAI Chat API ─────────────────────────────────────────
    logger.info(
        "Calling %s | query='%s...' | chunks=%d | filter=%s",
        settings.llm_model, query[:50], len(chunks), filter_paper,
    )

    client   = _get_sync_client()
    response = client.chat.completions.create(
        model=settings.llm_model,                        # "gpt-4o-mini"
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=settings.llm_temperature,            # from config: 0.2
        max_tokens=settings.llm_max_tokens,              # from config: 1500
    )

    # ── Step 5: Extract answer and usage ─────────────────────────────────────
    # response.choices[0]         → first (and usually only) completion choice
    # .message.content            → the actual text generated
    # response.usage.total_tokens → prompt_tokens + completion_tokens
    answer      = response.choices[0].message.content or ""
    tokens_used = response.usage.total_tokens

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    logger.info(
        "LLM response complete | tokens=%d | latency=%.0fms | answer_len=%d chars",
        tokens_used, latency_ms, len(answer),
    )

    return {
        "answer":      answer,
        "sources":     build_sources_list(chunks),
        "chunks_used": len(chunks),
        "model":       settings.llm_model,
        "tokens_used": tokens_used,
        "latency_ms":  latency_ms,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING RAG PIPELINE (SSE)
# ══════════════════════════════════════════════════════════════════════════════

async def stream_rag_answer(
    query:        str,
    filter_paper: Optional[str] = None,
    top_k:        Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """
    Async streaming RAG pipeline — yields answer tokens as they are generated.

    USED BY: POST /api/query/stream (query.py → StreamingResponse)

    WHY STREAMING?
      Without streaming:
        User clicks "Ask"
        [nothing happens for 8 seconds]
        Full answer appears at once

      With streaming:
        User clicks "Ask"
        [after ~0.5s]
        "Self" → "-attention" → " is" → " a" → " mechanism..." (typewriter effect)
        User sees progress immediately → feels faster → better UX

      Perceived latency is dramatically lower even though total time is similar.

    WHAT IS AN ASYNC GENERATOR?
      A function with `yield` inside `async def`.
      Returns an AsyncGenerator object.
      Caller iterates with: `async for token in stream_rag_answer(...)`
      Each iteration suspends until next token arrives (non-blocking).

    HOW SERVER-SENT EVENTS (SSE) WORK:
      This function yields raw token strings.
      query.py wraps them in SSE format:
        yield f"data: {token}\n\n"
      Browser EventSource receives:
        data: Self
        data: -attention
        data: is
        data: [DONE]
      JavaScript reconstructs token by token → typewriter effect.

    SSE vs WebSocket:
      SSE: one-way (server → client), built into HTTP, automatic reconnect
      WebSocket: bidirectional, separate protocol, more complex setup
      For token streaming (server → client only): SSE is perfect and simpler.

    RETRIEVAL IS SYNC — STREAMING IS ASYNC:
      similarity_search() is synchronous (FAISS + numpy).
      We call it before starting the stream (retrieval happens once, fully).
      Only the LLM token generation is streamed.

      Why not stream retrieval too?
        Retrieval is fast (< 200ms). No benefit to streaming it.
        We need ALL chunks to build context BEFORE calling the LLM.

    FLOW:
      similarity_search() → chunks (sync, blocking, fast)
          ↓
      build_context(chunks) → context string
          ↓
      AsyncOpenAI().chat.completions.create(stream=True)
          ↓ returns an async stream object
      async for chunk in stream:
          delta = chunk.choices[0].delta.content
          → None (metadata chunk) or token string
          if delta: yield delta
          ↓
      query.py receives each yielded token
          ↓
      StreamingResponse sends: "data: {token}\n\n" to browser
          ↓
      Browser EventSource fires onmessage event for each token
          ↓
      JavaScript appends token to displayed text

    PARAMETERS / RETURNS:
      Same parameters as get_rag_answer().
      Returns AsyncGenerator[str, None] — each yield is one token string.

    ⚠️  LIMITATION:
      Streaming version does NOT return sources or token count.
      The frontend only gets the text stream, not metadata.
      Production solution: send metadata as a special final SSE event:
        yield json.dumps({"sources": [...], "tokens_used": 1500})
      Then send "data: [DONE]\n\n" signal.
    """
    if top_k is None:
        top_k = settings.top_k_results

    # ── Retrieve chunks (synchronous — must complete before streaming) ────────
    chunks = similarity_search(
        query=query,
        k=top_k,
        filter_source=filter_paper,
    )

    if not chunks:
        yield "No relevant content found in the uploaded papers. Please upload papers first."
        return

    context = build_context(chunks)
    user_message = (
        f"Context from research papers:\n\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        f"Please answer based solely on the context above, "
        f"citing the source papers and page numbers."
    )

    # ── Stream from OpenAI ────────────────────────────────────────────────────
    client = _get_async_client()
    stream = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        stream=True,    # ← enables streaming mode
    )

    # ── Yield tokens as they arrive ───────────────────────────────────────────
    # OpenAI stream yields "chunks" (not our text chunks — confusingly named)
    # Each stream chunk = one small piece of the response
    # chunk.choices[0].delta.content = token string (or None for metadata chunks)
    token_count = 0
    async for stream_chunk in stream:
        delta = stream_chunk.choices[0].delta.content
        if delta:               # skip None (metadata) chunks
            token_count += 1
            yield delta         # send token to query.py → SSE → browser

    logger.info(
        "Stream complete | query='%s...' | tokens_yielded=%d",
        query[:50], token_count,
    )
