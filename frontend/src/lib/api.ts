// src/lib/api.ts
// API client for the FastAPI backend

import axios from "axios";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,
  headers: { "Content-Type": "application/json" },
});

// ─── Types ────────────────────────────────────────────────────────────────────
export interface SourceCitation {
  paper: string;
  page:  number;
  score: number;
}

export interface QueryResponse {
  success:     boolean;
  question:    string;
  answer:      string;
  sources:     SourceCitation[];
  chunks_used: number;
  model:       string;
  tokens_used: number;
}

export interface UploadResponse {
  success:      boolean;
  message:      string;
  paper_name:   string;
  file_size_mb: number;
  total_chunks?: number;
  total_pages?:  number;
}

export interface Paper {
  name: string;
}

export interface PapersListResponse {
  papers: string[];
  total:  number;
}

// ─── API Functions ────────────────────────────────────────────────────────────

/** Upload a PDF research paper */
export async function uploadPaper(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await api.post<UploadResponse>("/api/upload/sync", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

/** Ask a question about uploaded papers */
export async function askQuestion(
  question: string,
  filterPaper?: string,
  topK?: number
): Promise<QueryResponse> {
  const res = await api.post<QueryResponse>("/api/query/", {
    question,
    filter_paper: filterPaper || null,
    top_k:        topK || null,
  });
  return res.data;
}

/** List all uploaded papers */
export async function listPapers(): Promise<PapersListResponse> {
  const res = await api.get<PapersListResponse>("/api/papers/");
  return res.data;
}

/** Delete a paper by name */
export async function deletePaper(paperName: string): Promise<void> {
  await api.delete(`/api/papers/${encodeURIComponent(paperName)}`);
}

/** Streaming query — returns a ReadableStream of tokens */
export function streamQuestion(question: string, filterPaper?: string): EventSource {
  // We use POST streaming via fetch, not EventSource (which only supports GET)
  // Frontend handles this with ReadableStream
  throw new Error("Use fetchStreamAnswer instead");
}

export async function* fetchStreamAnswer(
  question: string,
  filterPaper?: string
): AsyncGenerator<string> {
  const response = await fetch(`${BASE_URL}/api/query/stream`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ question, filter_paper: filterPaper || null }),
  });

  const reader    = response.body!.getReader();
  const decoder   = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    const lines = text.split("\n");
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const token = line.slice(6);
        if (token === "[DONE]") return;
        if (token) yield token;
      }
    }
  }
}
