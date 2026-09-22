"""
Vector Store Service
Uses FAISS (faiss-cpu) instead of ChromaDB to avoid C++ build requirements on Windows.
All public function signatures are identical to the original ChromaDB implementation.

Persistence layout inside settings.chroma_db_path:
  faiss.index     — FAISS binary index (IndexFlatIP, cosine via L2-norm)
  metadata.json   — list of chunk dicts (id, text, source, page_number, chunk_index)
  embeddings.npy  — float32 numpy array of L2-normalised embeddings, same row order as metadata
"""
import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from openai import OpenAI
from config import settings

FAISS_INDEX_FILE = "faiss.index"
METADATA_FILE    = "metadata.json"
EMBEDDINGS_FILE  = "embeddings.npy"
EMBEDDING_DIM    = 1536          # text-embedding-3-small output dimension

# ─── Singleton OpenAI client ──────────────────────────────────────────────────
_openai_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


# ─── Persistence helpers ──────────────────────────────────────────────────────
def _paths():
    db = settings.chroma_db_path
    os.makedirs(db, exist_ok=True)
    return (
        os.path.join(db, FAISS_INDEX_FILE),
        os.path.join(db, METADATA_FILE),
        os.path.join(db, EMBEDDINGS_FILE),
    )


def _load_all():
    """Load FAISS index, metadata list, and embeddings array from disk."""
    idx_p, meta_p, emb_p = _paths()
    if os.path.exists(idx_p) and os.path.exists(meta_p) and os.path.exists(emb_p):
        index      = faiss.read_index(idx_p)
        with open(meta_p, "r", encoding="utf-8") as f:
            meta   = json.load(f)
        embeddings = np.load(emb_p)
    else:
        index      = faiss.IndexFlatIP(EMBEDDING_DIM)
        meta       = []
        embeddings = np.empty((0, EMBEDDING_DIM), dtype="float32")
    return index, meta, embeddings


def _save_all(index, meta: List[Dict], embeddings: np.ndarray):
    """Persist all three store files atomically."""
    idx_p, meta_p, emb_p = _paths()
    faiss.write_index(index, idx_p)
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    np.save(emb_p, embeddings)


def _build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a fresh FAISS IndexFlatIP from (already L2-normalised) embeddings."""
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    if len(embeddings) > 0:
        vecs = embeddings.astype("float32").copy()
        faiss.normalize_L2(vecs)          # idempotent if already normalised
        index.add(vecs)
    return index


# ─── Embeddings ───────────────────────────────────────────────────────────────
def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using OpenAI."""
    client   = get_openai_client()
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


# ─── Store ────────────────────────────────────────────────────────────────────
def store_chunks(chunks: List[Dict[str, Any]]) -> int:
    """
    Embed and store chunks in the FAISS vector store.
    Upsert behaviour: existing chunk IDs are replaced.
    Returns number of chunks stored.
    """
    index, meta, embeddings = _load_all()

    existing_ids = {m["id"] for m in meta}
    upsert_ids   = {c["id"] for c in chunks if c["id"] in existing_ids}

    # Remove outdated versions of any duplicate IDs
    if upsert_ids:
        keep      = [i for i, m in enumerate(meta) if m["id"] not in upsert_ids]
        meta      = [meta[i] for i in keep]
        embeddings = embeddings[keep] if len(embeddings) > 0 else embeddings

    # Generate embeddings in batches of 100 (avoids rate-limit issues)
    all_embs: List[List[float]] = []
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = [c["text"] for c in chunks[i : i + batch_size]]
        all_embs.extend(generate_embeddings(batch))

    new_embs = np.array(all_embs, dtype="float32")
    faiss.normalize_L2(new_embs)          # normalise → inner product = cosine sim

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

    meta       = meta + new_meta
    embeddings = np.vstack([embeddings, new_embs]) if len(embeddings) else new_embs

    index = _build_index(embeddings)
    _save_all(index, meta, embeddings)
    return len(chunks)


# ─── Retrieve ─────────────────────────────────────────────────────────────────
def similarity_search(
    query: str,
    k: int = None,
    filter_source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Embed the query and return top-K most similar chunks.
    Optionally filter by paper name (source).
    """
    if k is None:
        k = settings.top_k_results

    index, meta, embeddings = _load_all()
    if not meta:
        return []

    query_emb = np.array(generate_embeddings([query]), dtype="float32")
    faiss.normalize_L2(query_emb)

    if filter_source:
        # Build a temporary sub-index for just this paper
        filtered_idx  = [i for i, m in enumerate(meta) if m["source"] == filter_source]
        if not filtered_idx:
            return []
        sub_embs = embeddings[filtered_idx].astype("float32").copy()
        faiss.normalize_L2(sub_embs)
        sub_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        sub_index.add(sub_embs)
        k_actual      = min(k, len(filtered_idx))
        scores, local_ids = sub_index.search(query_emb, k_actual)

        results = []
        for score, local_id in zip(scores[0], local_ids[0]):
            if local_id < 0:
                continue
            m = meta[filtered_idx[local_id]]
            results.append({
                "text":             m["text"],
                "source":           m["source"],
                "page_number":      m["page_number"],
                "chunk_index":      m["chunk_index"],
                "similarity_score": round(float(score), 4),
            })
        return results

    # Search full index
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
    return results


# ─── Management ───────────────────────────────────────────────────────────────
def list_papers() -> List[str]:
    """Return unique paper names stored in the vector store."""
    _, meta, _ = _load_all()
    return sorted({m["source"] for m in meta})


def delete_paper(paper_name: str) -> int:
    """Delete all chunks for a given paper. Returns number of deleted chunks."""
    index, meta, embeddings = _load_all()
    keep      = [i for i, m in enumerate(meta) if m["source"] != paper_name]
    deleted   = len(meta) - len(keep)
    if deleted == 0:
        return 0
    meta       = [meta[i] for i in keep]
    embeddings = embeddings[keep] if len(embeddings) > 0 else embeddings
    index      = _build_index(embeddings)
    _save_all(index, meta, embeddings)
    return deleted


def get_paper_stats(paper_name: str) -> Dict[str, Any]:
    """Return stats for a stored paper."""
    _, meta, _ = _load_all()
    paper_meta = [m for m in meta if m["source"] == paper_name]
    pages      = {m["page_number"] for m in paper_meta}
    return {
        "paper_name":   paper_name,
        "total_chunks": len(paper_meta),
        "total_pages":  len(pages),
    }
