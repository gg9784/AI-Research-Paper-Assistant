"""
LLM Service
Phase 5: RAG query pipeline - retrieve context, call GPT-4, return answer
"""
from typing import List, Dict, Any, Optional, AsyncGenerator
from openai import OpenAI, AsyncOpenAI
from config import settings
from services.vector_store import similarity_search

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert AI research assistant. Your job is to answer
questions about academic research papers accurately and clearly.

Rules you MUST follow:
1. Answer ONLY using the provided context from the papers.
2. If the answer is not in the context, say: "This information was not found in the uploaded papers."
3. Always cite your sources by mentioning the paper name and page number.
4. Be concise but thorough. Use bullet points when listing multiple points.
5. If asked to compare papers, structure your answer clearly by paper.
6. Never make up information or use knowledge outside the provided context.
"""


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a readable context block."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}] Paper: '{chunk['source']}' | Page {chunk['page_number']}\n"
            f"{chunk['text']}\n"
        )
    return "\n---\n".join(context_parts)


def build_sources_list(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate and format source citations."""
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["source"], chunk["page_number"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "paper": chunk["source"],
                "page": chunk["page_number"],
                "score": chunk["similarity_score"],
            })
    return sorted(sources, key=lambda x: x["score"], reverse=True)


def get_rag_answer(
    query: str,
    filter_paper: Optional[str] = None,
    top_k: int = None,
) -> Dict[str, Any]:
    """
    Synchronous RAG pipeline:
    1. Embed query
    2. Retrieve top-K similar chunks
    3. Build context
    4. Call GPT-4
    5. Return answer + sources
    """
    if top_k is None:
        top_k = settings.top_k_results

    # Step 1: Retrieve relevant chunks
    chunks = similarity_search(query=query, k=top_k, filter_source=filter_paper)

    if not chunks:
        return {
            "answer": "No relevant content found in the uploaded papers. Please upload papers first.",
            "sources": [],
            "chunks_used": 0,
        }

    # Step 2: Build context
    context = build_context(chunks)

    # Step 3: Build full prompt
    user_message = f"""Context from research papers:
{context}

Question: {query}

Please answer based solely on the context above, citing the source papers."""

    # Step 4: Call LLM
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1500,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": build_sources_list(chunks),
        "chunks_used": len(chunks),
        "model": settings.llm_model,
        "tokens_used": response.usage.total_tokens,
    }


async def stream_rag_answer(
    query: str,
    filter_paper: Optional[str] = None,
    top_k: int = None,
) -> AsyncGenerator[str, None]:
    """
    Streaming RAG pipeline - yields answer tokens as they arrive.
    Use with FastAPI StreamingResponse for real-time UI updates.
    """
    if top_k is None:
        top_k = settings.top_k_results

    chunks = similarity_search(query=query, k=top_k, filter_source=filter_paper)

    if not chunks:
        yield "No relevant content found in the uploaded papers."
        return

    context = build_context(chunks)
    user_message = f"""Context from research papers:
{context}

Question: {query}

Please answer based solely on the context above, citing the source papers."""

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    stream = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1500,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
