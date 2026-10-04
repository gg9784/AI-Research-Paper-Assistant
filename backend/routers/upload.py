"""
routers/upload.py — PDF Upload Router
AI Research Paper Assistant

PURPOSE:
  Handles all PDF upload HTTP endpoints.
  Validates incoming files, saves them to disk,
  and triggers the ingestion pipeline (PDF → chunks → embeddings → ChromaDB).

ENDPOINTS:
  POST /api/upload       → async (returns immediately, processes in background)
  POST /api/upload/sync  → sync  (waits for full processing before responding)

UPLOAD PIPELINE:
  Browser
    ↓  multipart/form-data
  upload.py
    ↓  validate extension + MIME + size
  ./uploads/<filename>.pdf
    ↓  background task (async) OR immediate (sync)
  pdf_processor.py → extract → clean → chunk
    ↓
  vector_store.py → embed → store in ChromaDB

CALLED BY:  FastAPI router via main.py include_router()
CALLS:      pdf_processor.process_pdf()
            vector_store.store_chunks(), list_papers(), get_paper_stats()
"""

import logging
from pathlib import Path

import aiofiles
import magic                      # python-magic: real MIME type detection
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse

from config import settings
from models.schemas import UploadResponse
from services.pdf_processor import process_pdf
from services.vector_store import store_chunks, list_papers

logger = logging.getLogger(__name__)

router = APIRouter()

# ─── Allowed MIME types ───────────────────────────────────────────────────────
# Extension check alone is NOT secure (rename any file to .pdf to bypass it).
# MIME type check reads the actual file bytes (magic bytes).
ALLOWED_MIME_TYPES = {"application/pdf"}
ALLOWED_EXTENSIONS = {".pdf"}


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# (prefixed with _ → internal use only, not part of public API)
# ══════════════════════════════════════════════════════════════════════════════

def _sanitize_filename(filename: str) -> str:
    """
    Sanitize uploaded filename to prevent path traversal attacks.

    SECURITY ISSUE prevented:
      Without sanitization, a malicious user could send:
        filename = "../../etc/passwd"
        → file saved to /etc/passwd → overwrites system file!

      With Path().name:
        "../../etc/passwd" → "passwd"
        "/etc/passwd"      → "passwd"
        Only the base filename is kept, directory components stripped.
    """
    # Path().name extracts only the filename, ignoring any directory component
    safe = Path(filename).name
    # Remove characters that are invalid in filenames on Windows/Linux
    safe = "".join(c for c in safe if c.isalnum() or c in "._- ")
    safe = safe.strip()
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return safe


def _validate_extension(filename: str) -> None:
    """Check file extension is .pdf (case-insensitive)."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Only PDF files are accepted.",
        )


async def _validate_mime_type(file: UploadFile) -> None:
    """
    Read the first 2KB of the file and check its MIME type.
    This catches renamed files (e.g., image.jpg renamed to paper.pdf).

    WHY THIS MATTERS:
      Extension check:   easy to bypass (rename any file to .pdf)
      MIME type check:   reads actual file bytes (magic numbers)
      PDF magic bytes:   first 4 bytes must be "%PDF" = 0x25 0x50 0x44 0x46

    Requires: pip install python-magic
              (and libmagic system library)
    """
    await file.seek(0)                    # reset to start
    header = await file.read(2048)        # read first 2KB
    await file.seek(0)                    # reset again for later reading

    detected = magic.from_buffer(header, mime=True)
    if detected not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File content is '{detected}', not a PDF. "
                   f"Renaming a non-PDF to '.pdf' is not allowed.",
        )


async def _save_upload_async(file: UploadFile, safe_filename: str) -> tuple[str, float]:
    """
    Save uploaded file to disk using async streaming I/O.

    WHY ASYNC + STREAMING:
      - `aiofiles` does NOT block the event loop during disk writes
      - Reading in 64KB chunks avoids loading entire file into RAM
      - Size check happens DURING streaming → stops early if too large
        (don't read a 2GB file fully before rejecting it!)

    FLOW:
      Open destination file
        ↓
      Read 64KB chunk from upload stream
        ↓
      Check running total against max_bytes
        ↓
      If too large → delete partial file + raise 413
        ↓
      Write chunk to disk
        ↓
      Repeat until EOF
        ↓
      Return (path, size_in_mb)

    Returns:
      (str) absolute path to saved file
      (float) file size in megabytes
    """
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    dest_path = upload_dir / safe_filename
    total_bytes = 0
    max_bytes   = settings.max_file_size_bytes   # uses new property from config

    try:
        async with aiofiles.open(dest_path, "wb") as out_file:
            await file.seek(0)
            while True:
                chunk = await file.read(65536)  # 64 KB per read
                if not chunk:
                    break                        # EOF
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    # Stop IMMEDIATELY — don't read the rest
                    raise ValueError("File exceeds size limit")
                await out_file.write(chunk)

    except ValueError:
        # Clean up the partially written file
        if dest_path.exists():
            dest_path.unlink()
            logger.warning("Deleted partial upload: %s", dest_path)
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {settings.max_file_size_mb} MB.",
        )

    size_mb = total_bytes / (1024 * 1024)
    logger.info("Saved upload: %s (%.2f MB)", dest_path, size_mb)
    return str(dest_path), size_mb


def _process_and_store(pdf_path: str, paper_name: str) -> None:
    """
    Background task: full PDF ingestion pipeline.

    Called by FastAPI BackgroundTasks AFTER the HTTP response is sent.
    Runs in the SAME process/thread as the main app but AFTER the response.

    IMPORTANT LIMITATION:
      If the server crashes mid-processing, this task is lost.
      No retry, no persistence.
      For production, use Celery + Redis or a message queue.

    FLOW:
      pdf_processor.process_pdf()
        → extract text page by page (PyMuPDF)
        → clean text (regex)
        → chunk text (RecursiveCharacterTextSplitter)
        → returns {"chunks": [...], "total_pages": N}
      vector_store.store_chunks()
        → embed each chunk (text-embedding-3-small)
        → store in ChromaDB (vector + text + metadata)
    """
    logger.info("Background ingestion started: %s", paper_name)
    try:
        result = process_pdf(pdf_path, paper_name)
        stored = store_chunks(result["chunks"])
        logger.info(
            "Background ingestion complete: %s | pages=%d | chunks=%d | stored=%d",
            paper_name, result["total_pages"], result["total_chunks"], stored,
        )
    except Exception as e:
        logger.error(
            "Background ingestion FAILED for '%s': %s",
            paper_name, str(e), exc_info=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/",
    response_model=UploadResponse,
    summary="Upload PDF (async)",
    description=(
        "Upload a research paper PDF. "
        "File is saved immediately and processed in the background. "
        "Response returns before chunking/embedding is complete."
    ),
)
async def upload_pdf(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload"),
):
    """
    ASYNC upload: saves file and returns immediately.
    Processing happens after response is sent.

    HTTP lifecycle:
      Browser → POST /api/upload (multipart/form-data)
        ↓ FastAPI parses UploadFile
      Validate extension
        ↓
      Validate MIME type (actual file bytes)
        ↓
      Save file to ./uploads/ (async streaming, 64KB chunks)
        ↓
      Check if paper already indexed in ChromaDB
        ↓
      Return HTTP 200 immediately
        ↓  (response sent to browser here)
      BackgroundTask runs: process_pdf → store_chunks
    """
    safe_name = _sanitize_filename(file.filename)
    _validate_extension(safe_name)
    await _validate_mime_type(file)

    pdf_path, size_mb = await _save_upload_async(file, safe_name)
    paper_name = Path(safe_name).stem           # "attention_paper.pdf" → "attention_paper"

    # ── Duplicate check ───────────────────────────────────────────────────────
    existing_papers = list_papers()
    if paper_name in existing_papers:
        logger.info("Duplicate upload detected for: %s — re-processing", paper_name)
        # Still re-process — user may have uploaded a corrected version
        background_tasks.add_task(_process_and_store, pdf_path, paper_name)
        return UploadResponse(
            success=True,
            message=f"Paper '{paper_name}' already exists and is being re-processed.",
            paper_name=paper_name,
            file_size_mb=round(size_mb, 2),
        )

    # ── Schedule background ingestion ─────────────────────────────────────────
    # add_task() registers the function — it runs AFTER this response is sent
    background_tasks.add_task(_process_and_store, pdf_path, paper_name)

    logger.info(
        "Upload accepted: %s (%.2f MB) | request_id=%s",
        paper_name, size_mb,
        getattr(request.state, "request_id", "N/A"),
    )

    return UploadResponse(
        success=True,
        message=(
            f"'{file.filename}' uploaded successfully. "
            f"Processing in background — ask questions in ~30 seconds."
        ),
        paper_name=paper_name,
        file_size_mb=round(size_mb, 2),
    )


@router.post(
    "/sync",
    response_model=UploadResponse,
    summary="Upload PDF (synchronous)",
    description=(
        "Upload a research paper PDF and WAIT for full processing. "
        "Response includes chunk and page counts. "
        "Use for smaller files or when you need immediate confirmation."
    ),
)
async def upload_pdf_sync(
    file: UploadFile = File(..., description="PDF file to upload"),
):
    """
    SYNC upload: saves AND processes before responding.

    Use when:
      - Testing/debugging (want to know chunks created)
      - Small files (< 5MB) where processing is fast
      - Frontend needs chunk count before allowing questions

    Trade-off:
      - Slower response (waits for embedding API calls)
      - Blocks the response until ALL chunks are embedded
      - For 50MB paper → can take 30-60 seconds → browser may timeout
    """
    safe_name = _sanitize_filename(file.filename)
    _validate_extension(safe_name)
    await _validate_mime_type(file)

    pdf_path, size_mb = await _save_upload_async(file, safe_name)
    paper_name = Path(safe_name).stem

    # Process immediately (blocking — response waits for this)
    result = process_pdf(pdf_path, paper_name)
    stored = store_chunks(result["chunks"])

    logger.info(
        "Sync upload complete: %s | pages=%d | chunks=%d | stored=%d",
        paper_name, result["total_pages"], result["total_chunks"], stored,
    )

    return UploadResponse(
        success=True,
        message=(
            f"Successfully processed '{file.filename}': "
            f"{stored} chunks from {result['total_pages']} pages."
        ),
        paper_name=paper_name,
        file_size_mb=round(size_mb, 2),
        total_chunks=result["total_chunks"],
        total_pages=result["total_pages"],
    )
