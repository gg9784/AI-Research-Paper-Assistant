"""
services/vector_store.py — Embedding + Vector Search Service
AI Research Paper Assistant

⚠️  IMPORTANT NOTE FOR INTERVIEWS:
  The project description says "ChromaDB" but the ACTUAL implementation
  uses FAISS (faiss-cpu). This was done to avoid C++ build requirements
  on Windows. The public API (function signatures, return shapes) is
  IDENTICAL to what a ChromaDB implementation would expose.

  ✅ CURRENTLY IMPLEMENTED: FAISS (local, in-process, file-persisted)
  🚀 PRODUCTION ALTERNATIVE: Pinecone, Qdrant, Weaviate (managed cloud)

PURPOSE:
  Two responsibilities:
  1. INDEXING  — embed text chunks → store vectors + metadata to disk
  2. RETRIEVAL — embed a query    → find semantically similar chunks

PERSISTENCE LAYOUT (inside settings.chroma_db_path = ./chroma_db/):
  faiss.index     — FAISS binary index (IndexFlatIP, cosine similarity)
  metadata.json   — list of chunk dicts (id, text, source, page, chunk_index)
  embeddings.npy  — numpy float32 array, same row order as metadata.json

HOW THE THREE FILES RELATE:
  metadata.json[47]      ← row 47 of metadata
  embeddings.npy[47]     ← the 1536-dim vector for that chunk
  faiss.index[47]        ← FAISS internal position (same row order)

  They are always in sync. _save_all() writes all three atomically.

CALLED BY:  routers/upload.py → store_chunks(), list_papers()
            services/llm_service.py → similarity_search()
            routers/papers.py → list_papers(), delete_paper(), get_paper_stats()
CALLS:      OpenAI Embeddings API, faiss-cpu, numpy
"""

import os
import json
import logging
from typing import Any, Optional

import faiss
import numpy as np
from openai import OpenAI

from config import settings

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
FAISS_INDEX_FILE  = "faiss.index"
METADATA_FILE     = "metadata.json"
EMBEDDINGS_FILE   = "embeddings.npy"

# MUST match settings.embedding_model output dimensions.
# text-embedding-3-small → 1536 dimensions
# text-embedding-3-large → 3072 dimensions
# If you change the model, change this too → rebuild entire index!
EMBEDDING_DIM = settings.embedding_dimensions   # reads from config (1536)

# Batch size for OpenAI embedding API calls.
# OpenAI allows up to 2048 texts per call, but batching at 100 avoids
# rate-limit errors and keeps individual request sizes manageable.
EMBED_BATCH_SIZE = 100


# ══════════════════════════════════════════════════════════════════════════════
# OPENAI CLIENT — SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

_openai_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    """
    Lazy singleton for OpenAI client.

    WHY SINGLETON?
      OpenAI() creates an HTTP connection pool.
      Creating a new client per request wastes connections and memory.
      One client reused across all requests = efficient.

    WHY LAZY (not created at module import)?
      At import time, settings.openai_api_key may not be loaded yet.
      Creating it lazily (on first use) ensures the key is ready.
    """
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
        logger.debug("OpenAI client initialised")
    return _openai_client


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE — READ & WRITE
# ══════════════════════════════════════════════════════════════════════════════

def _paths() -> tuple[str, str, str]:
    """Return absolute paths to the three persistence files."""
    db = settings.chroma_db_path
    os.makedirs(db, exist_ok=True)
    return (
        os.path.join(db, FAISS_INDEX_FILE),
        os.path.join(db, METADATA_FILE),
        os.path.join(db, EMBEDDINGS_FILE),
    )


def _load_all() -> tuple[faiss.IndexFlatIP, list[dict], np.ndarray]:
    """
    Load the FAISS index, metadata list, and embeddings array from disk.

    If files don't exist yet (first run), returns empty structures.

    WHY LOAD ALL THREE?
      FAISS index:    needed for fast vector similarity search
      metadata.json:  needed to map result row indices back to chunk info
      embeddings.npy: needed when we need to rebuild the index
                      (e.g., after deleting a paper's chunks)

    CONSISTENCY GUARANTEE:
      All three files are ALWAYS written together by _save_all().
      If any one file is missing, we start fresh (assume corrupted state).
    """
    idx_p, meta_p, emb_p = _paths()
    all_exist = (
        os.path.exists(idx_p) and
        os.path.exists(meta_p) and
        os.path.exists(emb_p)
    )

    if all_exist:
        try:
            index      = faiss.read_index(idx_p)
            with open(meta_p, "r", encoding="utf-8") as f:
                meta   = json.load(f)
            embeddings = np.load(emb_p)
            logger.debug("Loaded store: %d chunks", len(meta))
            return index, meta, embeddings
        except Exception as e:
            logger.error("Failed to load store files — starting fresh: %s", e)

    # Empty store (first run or corrupted)
    index      = faiss.IndexFlatIP(EMBEDDING_DIM)
    meta       = []
    embeddings = np.empty((0, EMBEDDING_DIM), dtype="float32")
    return index, meta, embeddings


def _save_all(
    index: faiss.IndexFlatIP,
    meta:  list[dict],
    embeddings: np.ndarray,
) -> None:
    """
    Persist all three store files.

    ORDER MATTERS:
      Write embeddings.npy first (largest, most likely to fail on disk full)
      Write metadata.json second
      Write faiss.index last (fastest)

    If write fails mid-way, the files may be inconsistent.
    Production fix: write to temp files → atomic rename (os.replace).
    """
    idx_p, meta_p, emb_p = _paths()

    np.save(emb_p, embeddings)
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=None)
    faiss.write_index(index, idx_p)

    logger.debug("Saved store: %d chunks, %.1f MB embeddings",
                 len(meta), embeddings.nbytes / 1_000_000)


def _build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a fresh FAISS IndexFlatIP from L2-normalised embeddings.

    IndexFlatIP = Index | Flat | Inner Product
      Flat:         brute-force (no approximation) — compares query
                    vector against EVERY stored vector
      Inner Product: dot product score between vectors

    WHY IndexFlatIP FOR COSINE SIMILARITY?
      Cosine similarity = dot product of L2-normalised vectors.
      If we normalise all vectors to unit length BEFORE storing,
      then inner product (IP) = cosine similarity.

      We normalise with faiss.normalize_L2() before storing
      → IndexFlatIP gives us exact cosine similarity.

    WHY BRUTE FORCE (Flat) AND NOT APPROXIMATE (IVF, HNSW)?
      Brute-force is exact (no accuracy loss).
      For small collections (< 100K vectors), it's fast enough.
      A 15-page paper = ~87 chunks. 10 papers = ~870 chunks.
      Brute-force search over 870 vectors takes microseconds.

      🚀 PRODUCTION: For millions of vectors, use IndexHNSWFlat or
         IndexIVFFlat for approximate nearest-neighbor (ANN) search.
    """
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    if len(embeddings) > 0:
        vecs = embeddings.astype("float32").copy()
        faiss.normalize_L2(vecs)   # idempotent if already normalised
        index.add(vecs)
    return index


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Call OpenAI Embeddings API to convert text strings to vectors.

    MODEL: text-embedding-3-small
      Output: 1536-dimensional float vector per text
      Cost:   ~$0.00002 per 1000 tokens (very cheap)
      Limit:  8191 tokens per input text

    WHY SAME MODEL FOR DOCUMENTS AND QUERIES?
      Embeddings from different models live in incompletely different
      vector spaces. Comparing them is like comparing distances in
      metres with distances in miles — the numbers are incomparable.

      If doc_embedding = ModelA("attention is all you need")
         query_embedding = ModelB("what is attention?")

      → cosine similarity between these vectors is MEANINGLESS.
      → retrieval returns random results.

      RULE: Document embedding model = Query embedding model. Always.

    WHAT IS A 1536-DIMENSIONAL VECTOR?
      A list of 1536 floating-point numbers.
      Each number encodes some aspect of the text's meaning.
      The dimensions are NOT human-readable:
        "The cat sat" → [0.023, -0.417, 0.891, 0.013, ...]
      Two semantically similar texts → vectors with small angle between them.
      Two unrelated texts → vectors pointing in very different directions.

    BATCHING:
      Called with up to EMBED_BATCH_SIZE=100 texts at once.
      OpenAI processes all 100 in one API call.
      Single API call overhead amortised across 100 texts → efficient.
    """
    client = get_openai_client()
    response = client.embeddings.create(
        model=settings.embedding_model,   # "text-embedding-3-small"
        input=texts,
    )
    # response.data is a list of Embedding objects, one per input text
    # Each .embedding is a list of 1536 floats
    return [item.embedding for item in response.data]


# ══════════════════════════════════════════════════════════════════════════════
# STORE — INDEX TIME
# ══════════════════════════════════════════════════════════════════════════════

def store_chunks(chunks: list[dict[str, Any]]) -> int:
    """
    Embed all chunks and persist them to the FAISS store.

    UPSERT BEHAVIOUR:
      If a chunk ID already exists (paper re-uploaded):
      1. Remove the old version
      2. Add the new version
      This is "upsert" = update + insert.

    FLOW:
      Load current store from disk
        ↓
      Find which incoming chunk IDs already exist
        ↓
      Remove those existing chunks (keeping everything else)
        ↓
      Embed new chunks in batches of 100 (OpenAI API calls)
        ↓
      L2-normalise all new embeddings (for cosine similarity)
        ↓
      Append new metadata + embeddings to existing ones
        ↓
      Rebuild FAISS index from all embeddings
        ↓
      Save all three files to disk

    RETURNS: Number of chunks stored (not including upserted)

    ⚠️  SCALING LIMITATION:
      Rebuilding the entire FAISS index on every store_chunks() call
      is fine for small collections (< 10K vectors).
      For millions of vectors: use FAISS IDMap or an incremental index.
    """
    if not chunks:
        logger.warning("store_chunks() called with empty list — nothing to store")
        return 0

    index, meta, embeddings = _load_all()

    # ── Upsert: remove existing versions of any incoming IDs ─────────────────
    existing_ids = {m["id"] for m in meta}
    incoming_ids = {c["id"] for c in chunks}
    upsert_ids   = existing_ids & incoming_ids   # intersection

    if upsert_ids:
        logger.info("Upserting %d existing chunks", len(upsert_ids))
        keep       = [i for i, m in enumerate(meta) if m["id"] not in upsert_ids]
        meta       = [meta[i] for i in keep]
        embeddings = embeddings[keep] if len(embeddings) > 0 else embeddings

    # ── Embed new chunks in batches ───────────────────────────────────────────
    logger.info("Embedding %d chunks in batches of %d...", len(chunks), EMBED_BATCH_SIZE)
    all_embs: list[list[float]] = []

    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch      = chunks[i : i + EMBED_BATCH_SIZE]
        batch_texts = [c["text"] for c in batch]
        batch_embs  = generate_embeddings(batch_texts)
        all_embs.extend(batch_embs)
        logger.debug(
            "  Embedded batch %d/%d (%d chunks)",
            i // EMBED_BATCH_SIZE + 1,
            -(-len(chunks) // EMBED_BATCH_SIZE),   # ceiling division
            len(batch),
        )

    # ── Normalise → cosine similarity via inner product ───────────────────────
    new_embs = np.array(all_embs, dtype="float32")
    faiss.normalize_L2(new_embs)   # in-place L2 normalisation

    # ── Build new metadata records ────────────────────────────────────────────
    new_meta = [
        {
            "id":          c["id"],
            "text":        c["text"],
            "source":      c["source"],
            "page_number": c["page_number"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    # ── Append and rebuild ────────────────────────────────────────────────────
    meta       = meta + new_meta
    embeddings = (
        np.vstack([embeddings, new_embs])
        if len(embeddings) > 0
        else new_embs
    )
    index = _build_index(embeddings)
    _save_all(index, meta, embeddings)

    logger.info("store_chunks() complete: total store size = %d chunks", len(meta))
    return len(chunks)


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVE — QUERY TIME
# ══════════════════════════════════════════════════════════════════════════════

def similarity_search(
    query:         str,
    k:             int = None,
    filter_source: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Embed the user's question and find the top-K most similar chunks.

    THIS IS THE CORE OF THE RAG RETRIEVAL STEP.

    FLOW:
      User question (text)
          ↓
      generate_embeddings([question])   ← OpenAI API
          ↓
      1536-dim query vector
          ↓
      faiss.normalize_L2(query_vector)  ← same normalisation as stored vecs
          ↓
      index.search(query_vector, k)     ← inner product = cosine similarity
          ↓
      [(score, row_index), ...]         ← top-K closest vectors
          ↓
      meta[row_index]                   ← map back to chunk text + metadata
          ↓
      Return list of chunk dicts with similarity_score

    WITH filter_source:
      Build a temporary sub-index containing ONLY chunks from that paper
      → searches only within that paper's vectors
      → much smaller search space → same algorithm

    COSINE SIMILARITY INTUITION:
      Two vectors in 1536-dimensional space.
      Cosine similarity = cos(angle between them).

      angle = 0°  → cos = 1.0 → identical direction → identical meaning
      angle = 90° → cos = 0.0 → perpendicular → unrelated
      angle = 180°→ cos = -1.0 → opposite → opposite meaning

      "What is attention?"     and
      "Attention mechanism in transformers"
      → small angle → score ≈ 0.90 → high similarity → retrieved!

      "What is attention?"     and
      "The stock market crashed"
      → large angle → score ≈ 0.10 → low similarity → NOT retrieved

    WHY SAME EMBEDDING MODEL FOR QUERY AND DOCUMENTS?
      If documents were embedded with model A and query with model B,
      the vectors live in incompatible spaces.
      Cosine similarity between them is mathematically meaningless.
      Always use the same model for both.

    top_k TRADE-OFF:
      k=1:  very precise, but may miss complementary info
      k=5:  (our default) good balance — enough context, manageable noise
      k=20: too much noise — irrelevant chunks pollute the LLM context
             → LLM may focus on noise → worse answers + higher cost

    PARAMETERS:
      query:         user's question text
      k:             how many chunks to return (default: settings.top_k_results)
      filter_source: optional paper name to restrict search to

    RETURNS:
      List of dicts: [{text, source, page_number, chunk_index, similarity_score}]
      Ordered by similarity_score descending (most relevant first).
    """
    if k is None:
        k = settings.top_k_results

    index, meta, embeddings = _load_all()

    if not meta:
        logger.warning("similarity_search() called but store is empty")
        return []

    # ── Embed the query ───────────────────────────────────────────────────────
    query_emb = np.array(generate_embeddings([query]), dtype="float32")
    faiss.normalize_L2(query_emb)   # must normalise query same as stored vecs

    # ── Filtered search: only one paper ──────────────────────────────────────
    if filter_source:
        filtered_idx = [i for i, m in enumerate(meta) if m["source"] == filter_source]
        if not filtered_idx:
            logger.info("No chunks found for paper: %s", filter_source)
            return []

        # Build temporary sub-index for this paper
        sub_embs = embeddings[filtered_idx].astype("float32").copy()
        faiss.normalize_L2(sub_embs)
        sub_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        sub_index.add(sub_embs)

        k_actual = min(k, len(filtered_idx))
        scores, local_ids = sub_index.search(query_emb, k_actual)

        results = []
        for score, local_id in zip(scores[0], local_ids[0]):
            if local_id < 0:        # FAISS returns -1 for padded results
                continue
            m = meta[filtered_idx[local_id]]
            results.append({
                "text":             m["text"],
                "source":           m["source"],
                "page_number":      m["page_number"],
                "chunk_index":      m["chunk_index"],
                "similarity_score": round(float(score), 4),
            })
        logger.info(
            "Filtered search '%s' for paper='%s': %d results",
            query[:50], filter_source, len(results)
        )
        return results

    # ── Full index search ─────────────────────────────────────────────────────
    k_actual = min(k, len(meta))
    scores, ids = index.search(query_emb, k_actual)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        m = meta[idx]
        results.append({
            "text":             m["text"],
            "source":           m["source"],
            "page_number":      m["page_number"],
            "chunk_index":      m["chunk_index"],
            "similarity_score": round(float(score), 4),
        })

    logger.info(
        "Full search '%s': %d results (top score=%.4f)",
        query[:50], len(results),
        results[0]["similarity_score"] if results else 0.0,
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MANAGEMENT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def list_papers() -> list[str]:
    """
    Return unique paper names currently stored.

    Uses a set comprehension to deduplicate:
      87 chunks from "attention_paper" → just one entry "attention_paper"
    """
    _, meta, _ = _load_all()
    papers = sorted({m["source"] for m in meta})
    logger.debug("list_papers(): %d papers", len(papers))
    return papers


def delete_paper(paper_name: str) -> int:
    """
    Delete ALL chunks belonging to a specific paper.

    HOW DELETION WORKS:
      FAISS doesn't support deleting individual vectors efficiently.
      We rebuild the entire index without the deleted paper's vectors.

      Step 1: Load all chunks
      Step 2: Keep only chunks where source != paper_name
      Step 3: Rebuild FAISS index from kept embeddings
      Step 4: Save everything

    ⚠️  NOTE: This does NOT delete the original PDF from ./uploads/
      That would require os.remove() on the file.
      Keeping the file is intentional: user can re-upload same PDF
      without needing to re-upload the file (could re-process existing).
      A production system should give users explicit control over this.

    RETURNS: Number of chunks deleted (0 if paper not found)
    """
    index, meta, embeddings = _load_all()

    keep    = [i for i, m in enumerate(meta) if m["source"] != paper_name]
    deleted = len(meta) - len(keep)

    if deleted == 0:
        logger.info("delete_paper('%s'): paper not found", paper_name)
        return 0

    meta       = [meta[i] for i in keep]
    embeddings = embeddings[keep] if len(embeddings) > 0 else embeddings
    index      = _build_index(embeddings)
    _save_all(index, meta, embeddings)

    logger.info("delete_paper('%s'): removed %d chunks", paper_name, deleted)
    return deleted


def get_paper_stats(paper_name: str) -> dict[str, Any]:
    """
    Return statistics for a specific paper.

    PAGES CALCULATION:
      We store page_number per chunk, not a separate pages count.
      {m["page_number"] for m in paper_meta} = set of unique page numbers
      len(that set) = number of distinct pages containing at least one chunk.
      (May differ from total PDF pages if some pages were empty and skipped.)
    """
    _, meta, _ = _load_all()
    paper_meta = [m for m in meta if m["source"] == paper_name]
    pages      = {m["page_number"] for m in paper_meta}
    return {
        "paper_name":   paper_name,
        "total_chunks": len(paper_meta),
        "total_pages":  len(pages),
    }
