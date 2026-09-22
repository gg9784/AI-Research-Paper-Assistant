"""
Upload Router
Phase 6: Handle PDF file uploads, process and store in ChromaDB
"""
import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pathlib import Path
from config import settings
from services.pdf_processor import process_pdf
from services.vector_store import store_chunks, get_paper_stats, list_papers
from models.schemas import UploadResponse, ErrorResponse

router = APIRouter()


def _save_upload(file: UploadFile) -> str:
    """Save the uploaded file and return its path."""
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(exist_ok=True)
    dest = upload_dir / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return str(dest)


def _process_and_store(pdf_path: str, paper_name: str):
    """Background task: process PDF and store embeddings."""
    result = process_pdf(pdf_path, paper_name)
    store_chunks(result["chunks"])


@router.post("/", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a research paper PDF.
    - Validates file type and size
    - Saves file to disk
    - Triggers background processing (extract → chunk → embed → store)
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Validate file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_file_size_mb}MB",
        )
    await file.seek(0)  # reset after reading

    # Save file
    pdf_path = _save_upload(file)
    paper_name = Path(file.filename).stem  # filename without .pdf

    # Check if already uploaded
    existing_papers = list_papers()
    if paper_name in existing_papers:
        return UploadResponse(
            success=True,
            message=f"Paper '{paper_name}' already exists. Re-processing...",
            paper_name=paper_name,
            file_size_mb=round(size_mb, 2),
        )

    # Process in background (non-blocking)
    background_tasks.add_task(_process_and_store, pdf_path, paper_name)

    return UploadResponse(
        success=True,
        message=f"PDF '{file.filename}' uploaded successfully. Processing in background...",
        paper_name=paper_name,
        file_size_mb=round(size_mb, 2),
    )


@router.post("/sync", response_model=UploadResponse)
async def upload_pdf_sync(file: UploadFile = File(...)):
    """
    Synchronous upload — waits for processing to complete before responding.
    Use for smaller files or when you need immediate confirmation.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(status_code=413, detail=f"File too large. Max: {settings.max_file_size_mb}MB")
    await file.seek(0)

    pdf_path = _save_upload(file)
    paper_name = Path(file.filename).stem

    result = process_pdf(pdf_path, paper_name)
    stored = store_chunks(result["chunks"])

    return UploadResponse(
        success=True,
        message=f"Processed and stored {stored} chunks from {result['total_pages']} pages.",
        paper_name=paper_name,
        file_size_mb=round(size_mb, 2),
        total_chunks=result["total_chunks"],
        total_pages=result["total_pages"],
    )
