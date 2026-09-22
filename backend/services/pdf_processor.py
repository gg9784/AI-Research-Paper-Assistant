"""
PDF Processing Service
Phase 3: Extract text from PDFs and split into chunks
"""
import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import settings


def clean_text(text: str) -> str:
    """Remove noise: extra whitespace, weird chars, repeated newlines."""
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text page-by-page from a PDF.
    Returns a list of dicts: {page_number, text}
    """
    pages = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text("text")
        cleaned = clean_text(raw_text)
        if cleaned:  # skip empty pages
            pages.append({
                "page_number": page_num + 1,
                "text": cleaned,
            })
    doc.close()
    return pages


def chunk_pages(pages: List[Dict], paper_name: str) -> List[Dict[str, Any]]:
    """
    Split page texts into overlapping chunks.
    Each chunk carries metadata: source, page_number, chunk_index.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks = []
    global_idx = 0

    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk_text in page_chunks:
            if chunk_text.strip():
                all_chunks.append({
                    "text": chunk_text.strip(),
                    "source": paper_name,
                    "page_number": page["page_number"],
                    "chunk_index": global_idx,
                    "id": f"{paper_name}_chunk_{global_idx}",
                })
                global_idx += 1

    return all_chunks


def process_pdf(pdf_path: str, paper_name: str) -> Dict[str, Any]:
    """
    Full pipeline: extract → clean → chunk.
    Returns summary + all chunks.
    """
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages, paper_name)

    return {
        "paper_name": paper_name,
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "chunks": chunks,
    }
