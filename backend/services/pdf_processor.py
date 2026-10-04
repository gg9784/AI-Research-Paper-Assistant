"""
services/pdf_processor.py — PDF Text Extraction & Chunking Service
AI Research Paper Assistant

PURPOSE:
  Converts an uploaded PDF into a list of text chunks,
  each carrying metadata (source, page, index, id).
  These chunks are then embedded and stored in ChromaDB.

PIPELINE:
  PDF file on disk
      ↓  fitz.open() — PyMuPDF
  Raw text per page
      ↓  clean_text() — regex normalization
  Cleaned text per page
      ↓  RecursiveCharacterTextSplitter
  Overlapping chunks with metadata
      ↓  returned to vector_store.store_chunks()
  ChromaDB

CALLED BY:  routers/upload.py (_process_and_store background task)
CALLS:      PyMuPDF (fitz), LangChain RecursiveCharacterTextSplitter
RETURNS:    List of chunk dicts with text + metadata
"""

import re
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF — pip install pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import settings

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Normalize raw PDF-extracted text to reduce noise before chunking.

    WHY CLEANING MATTERS:
      Raw PDF text extracted by PyMuPDF often contains:
        - 3+ consecutive blank lines (from section breaks, headers, footers)
        - Multiple spaces/tabs (from column formatting)
        - Non-printable characters (control chars, ligatures, special Unicode)
        - Hyphenated line breaks ("transfor-\nmer" from column wrapping)

      Noisy text → noisy embeddings → noisy retrieval → bad answers.
      Clean text → consistent embeddings → better semantic search.

    WHAT EACH STEP DOES:
      Step 1: Fix hyphenated words broken across lines
              "transfor-\nmer attention" → "transformer attention"

      Step 2: Collapse 3+ blank lines into exactly 2
              Preserves paragraph structure without excessive whitespace

      Step 3: Collapse multiple spaces/tabs into a single space
              "attention   mechanism" → "attention mechanism"

      Step 4: Remove non-printable ASCII characters
              Keeps: 0x20 (space) to 0x7E (~), plus newlines
              Removes: control chars, null bytes, exotic Unicode symbols

    WARNING — OVER-CLEANING RISK:
      Aggressive regex can accidentally remove meaningful content:
        - Greek letters (α, β, σ) used in math → removed by step 4!
        - Subscripts/superscripts in formulas
        - Special notation in research papers

      ⚠️  CURRENT LIMITATION:
        step 4 uses [^\x20-\x7E\n] → removes ALL non-ASCII including
        Greek letters and math symbols commonly found in research papers.

      🚀 PRODUCTION IMPROVEMENT:
        Use Unicode-aware cleaning instead of ASCII-only:
          re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]\{\}]', '', text)
        Or use a dedicated PDF parser like pdfminer.six that handles
        Unicode encoding more gracefully.
    """
    # Step 1: Rejoin hyphenated line breaks (common in multi-column papers)
    text = re.sub(r'-\n(\w)', r'\1', text)

    # Step 2: Collapse 3+ consecutive newlines into exactly 2 (preserve paragraphs)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Step 3: Collapse multiple spaces/tabs into a single space
    text = re.sub(r'[ \t]+', ' ', text)

    # Step 4: Remove non-printable characters (ASCII control chars + exotic Unicode)
    # \x20 = space, \x7E = ~   → keeps standard printable ASCII + newlines
    text = re.sub(r'[^\x20-\x7E\n]', '', text)

    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# PDF TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: str) -> list[dict[str, Any]]:
    """
    Open a PDF and extract clean text from each page using PyMuPDF (fitz).

    WHY PAGE-BY-PAGE (not whole document at once)?
      - Preserves page_number metadata → used for citations
      - Limits memory — process one page at a time
      - Allows skipping empty pages (image-only, blank pages)

    WHAT PyMuPDF DOES:
      fitz.open()        → loads PDF into memory (not entire file — lazy loading)
      page.get_text()    → extracts text layer from the PDF's internal structure
      doc.close()        → releases memory and file handle

    KNOWN LIMITATIONS OF PDF TEXT EXTRACTION:
      ✅ Works well for:
          Text-based PDFs (most academic papers from arXiv, journals)
          PDFs with embedded text fonts

      ❌ Fails or degrades for:
          Scanned PDFs (images of pages — no text layer at all)
          → Returns empty string → chunk is skipped → paper not indexed!
          → Fix: OCR with Tesseract (pytesseract) or AWS Textract

          Multi-column layouts (Nature, IEEE format)
          → Text extracted in wrong reading order (column 1 bottom, then column 2 top)
          → Chunks contain incoherent sentence fragments
          → Fix: pdfplumber with bounding-box extraction

          Mathematical formulas (LaTeX-rendered)
          → Symbols extracted as garbage characters or omitted
          → Fix: MathPix API, LaTeX parser, or accept this limitation

          Tables
          → Extracted as flat text without structure
          → "Column A | Column B" becomes "Column A Column B" → context lost
          → Fix: pdfplumber.extract_tables(), Camelot, Tabula

          Headers and footers
          → Appear on every page → pollute chunks with repeated noise
          → Example: "Journal of ML Research | Vol 12 | Page 47"
          → Fix: detect by position (top/bottom 10% of page) and strip

    RETURNS:
      List of dicts: [{page_number: int, text: str}, ...]
      Pages with no extractable text are SKIPPED.
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

    pages = []
    doc = None

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        logger.info("Opened PDF: %s | Total pages: %d", pdf_path, total_pages)

        for page_num in range(total_pages):
            page = doc[page_num]

            # "text" mode: extracts text in reading order
            # Alternatives: "blocks", "words", "html", "dict"
            raw_text = page.get_text("text")
            cleaned  = clean_text(raw_text)

            if not cleaned:
                # Skip truly empty pages (scanned, blank, image-only)
                logger.debug("Page %d is empty after cleaning — skipped", page_num + 1)
                continue

            pages.append({
                "page_number": page_num + 1,   # 1-indexed (matches PDF page numbers)
                "text":        cleaned,
                "char_count":  len(cleaned),
            })

    except fitz.FileDataError as e:
        raise ValueError(f"PDF is corrupted or unreadable: {e}") from e
    finally:
        if doc:
            doc.close()   # Always release file handle even on error

    logger.info(
        "Extracted text from %d/%d pages (skipped %d empty)",
        len(pages), total_pages, total_pages - len(pages)
    )
    return pages


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def chunk_pages(
    pages: list[dict[str, Any]],
    paper_name: str,
) -> list[dict[str, Any]]:
    """
    Split page texts into overlapping fixed-size chunks using
    LangChain's RecursiveCharacterTextSplitter.

    WHY CHUNKING IS NECESSARY:
      Embedding models have token limits:
        text-embedding-3-small: max 8191 tokens per input
        A 15-page paper: ~15,000 tokens → EXCEEDS the limit

      Even if it fit: embedding an entire paper as ONE vector loses all
      fine-grained information. A query about "self-attention" would have
      to match a vector representing ALL topics in the paper.

      Chunking: split into small pieces → embed each piece separately
      → query finds the SPECIFIC chunk about self-attention → precise retrieval.

    HOW RecursiveCharacterTextSplitter WORKS:
      chunk_size=500    → target maximum characters per chunk
      chunk_overlap=50  → overlap characters between consecutive chunks
      separators=["\n\n", "\n", ". ", " ", ""]

      RECURSIVE BEHAVIOR (tries separators in order until chunk fits):

      Given text of 800 chars (too large for chunk_size=500):
        Try "\n\n" (paragraph break) → splits into 2 chunks? ✅ done
        If chunk still > 500:
          Try "\n" (line break)
          If still > 500:
            Try ". " (sentence end)
            If still > 500:
              Try " " (word boundary)
              If still > 500:
                Force split at exactly 500 chars (no separator)

      Result: Splits prefer natural text boundaries.
              Worst case: splits mid-word only if absolutely necessary.

    WHY OVERLAP?
      Text:  "...transformer uses self-attention. | Self-attention allows..."
                                                  ↑ chunk boundary

      Without overlap:
        Chunk A: "...transformer uses self-attention."
        Chunk B: "Self-attention allows each token..."
        Query: "what does self-attention allow transformers to do?"
        → The connection between "transformer" and "allows" is split!
        → Neither chunk alone has the full answer.

      With overlap=50:
        Chunk A: "...transformer uses self-attention. Self-attention allow"
        Chunk B: "uses self-attention. Self-attention allows each token..."
        → Both chunks contain the connection!
        → Query retrieves either → full context available.

    ⚠️  CHARACTER vs TOKEN chunking — CURRENT LIMITATION:
      chunk_size=500 means 500 CHARACTERS, not 500 tokens.
      Tokens ≠ Characters:
        "hello" = 1 token, 5 chars
        "GPT-4o-mini" = 4 tokens, 11 chars
        "transformer" = 2 tokens, 11 chars
        Average: ~4 chars per token (English text)

      500 chars ÷ 4 = ~125 tokens per chunk
      text-embedding-3-small limit: 8191 tokens
      So we're well under the limit. This is NOT a problem for correctness.

      The issue is inconsistency:
        Dense technical text: "BERT, GPT, XLNet, RoBERTa, ALBERT, T5" = 12 tokens
        Sparse prose:         "The model is very good." = 5 tokens
        Same character count → very different token counts → inconsistent context.

      🚀 PRODUCTION IMPROVEMENT:
        Use tiktoken to count actual tokens:
          from langchain.text_splitter import TokenTextSplitter
          splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=32)
        Now chunks have consistent SEMANTIC density.

    METADATA ON EACH CHUNK:
      Every chunk carries:
        text:         the actual chunk text (stored in ChromaDB as document)
        source:       paper name (used for filtering and citations)
        page_number:  which PDF page this came from (used for citations)
        chunk_index:  global sequence number (used for ordering and dedup)
        id:           unique identifier (used for ChromaDB add/delete)

    RETURNS:
      List of chunk dicts ready to be passed to vector_store.store_chunks()
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,         # from config: 500
        chunk_overlap=settings.chunk_overlap,   # from config: 50
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,                    # count characters (not tokens)
    )

    all_chunks = []
    global_idx = 0   # monotonically increasing index across ALL pages

    for page in pages:
        # Split this page's text into chunks
        page_chunks = splitter.split_text(page["text"])

        for chunk_text in page_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue   # skip if cleaning produced empty chunk

            # Warn about very short chunks (they rarely contain useful info)
            if len(chunk_text) < 50:
                logger.debug(
                    "Very short chunk (%d chars) from page %d: '%s...'",
                    len(chunk_text), page["page_number"], chunk_text[:30]
                )

            all_chunks.append({
                # ── Core content ──────────────────────────────────────────────
                "text":         chunk_text,

                # ── Metadata (used for citations, filtering, deletion) ────────
                "source":       paper_name,           # "attention_is_all_you_need"
                "page_number":  page["page_number"],  # 3 (1-indexed)
                "chunk_index":  global_idx,           # 47 (global, across all pages)

                # ── Unique ID (required by ChromaDB) ─────────────────────────
                # Format: {paper_name}_chunk_{index}
                # Example: "attention_is_all_you_need_chunk_47"
                # Must be unique within the ChromaDB collection.
                "id": f"{paper_name}_chunk_{global_idx}",
            })
            global_idx += 1

    logger.info(
        "Chunked '%s': %d pages → %d chunks (size=%d, overlap=%d)",
        paper_name, len(pages), len(all_chunks),
        settings.chunk_size, settings.chunk_overlap,
    )
    return all_chunks


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def process_pdf(pdf_path: str, paper_name: str) -> dict[str, Any]:
    """
    Full ingestion pipeline: PDF file → list of chunks ready for embedding.

    FLOW:
      extract_text_from_pdf()
        → PyMuPDF opens PDF
        → Extract + clean text page by page
        → Return [{page_number, text}, ...]
            ↓
      chunk_pages()
        → RecursiveCharacterTextSplitter
        → Split each page into overlapping chunks
        → Attach metadata to each chunk
        → Return [{text, source, page_number, chunk_index, id}, ...]
            ↓
      Return summary dict (used by upload.py for response + logging)

    CALLED BY:
      routers/upload.py → _process_and_store() (background task)
      routers/upload.py → upload_pdf_sync() (synchronous upload)

    RAISES:
      FileNotFoundError: PDF not found at path
      ValueError:        PDF is corrupted or unreadable

    RETURNS:
      {
        "paper_name":   "attention_is_all_you_need",
        "total_pages":  15,
        "total_chunks": 87,
        "chunks":       [{text, source, page_number, chunk_index, id}, ...]
      }
    """
    logger.info("Starting PDF processing pipeline: %s", paper_name)

    pages  = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages, paper_name)

    if not chunks:
        logger.warning(
            "No chunks produced for '%s'. "
            "The PDF may be scanned/image-only or have no extractable text.",
            paper_name
        )

    result = {
        "paper_name":   paper_name,
        "total_pages":  len(pages),
        "total_chunks": len(chunks),
        "chunks":       chunks,
    }

    logger.info(
        "PDF processing complete: '%s' | pages=%d | chunks=%d",
        paper_name, len(pages), len(chunks),
    )
    return result
