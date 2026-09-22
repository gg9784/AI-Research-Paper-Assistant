# 🧠 AI Research Paper Assistant

> **RAG-powered Q&A over research papers using GPT-4 + ChromaDB**

Upload any research paper PDFs and chat with them in natural language. The system retrieves the most relevant sections using semantic search and grounds answers with source citations.

---

## ✨ Features

- 📄 **Multi-PDF Upload** — drag & drop multiple papers at once
- 🔍 **Semantic Search** — cosine similarity over 1536-dim embeddings
- 🤖 **GPT-4o Streaming** — real-time typewriter answer display
- 📌 **Source Citations** — every answer shows paper name + page number
- 🔎 **Paper Filtering** — ask questions scoped to a specific paper
- 🗑️ **Paper Management** — list, filter, delete indexed papers

---

## 🛠️ Tech Stack

| Layer      | Technology                        |
|------------|-----------------------------------|
| Frontend   | Next.js 14, TypeScript, Tailwind  |
| Backend    | FastAPI, Python 3.11, Uvicorn     |
| LLM        | OpenAI GPT-4o-mini                |
| Embeddings | text-embedding-3-small (1536-dim) |
| Vector DB  | ChromaDB (persistent)             |
| PDF Parse  | PyMuPDF (fitz)                    |
| Framework  | LangChain (chunking)              |
| DevOps     | Docker, Docker Compose            |

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone <your-repo>
cd "AI Research Paper Assistant"
```

### 2. Configure Backend
```bash
cd backend
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn main:app --reload
# → http://localhost:8000
```

### 4. Run Frontend
```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

### 5. (Optional) Docker
```bash
# Add OPENAI_API_KEY to .env in root
docker-compose up --build
```

---

## 📡 API Endpoints

| Method | Endpoint                  | Description              |
|--------|---------------------------|--------------------------|
| GET    | /health                   | Health check             |
| POST   | /api/upload/sync          | Upload + process PDF     |
| POST   | /api/query/               | Ask a question (sync)    |
| POST   | /api/query/stream         | Ask a question (stream)  |
| GET    | /api/papers/              | List all papers          |
| GET    | /api/papers/{name}/stats  | Paper stats              |
| DELETE | /api/papers/{name}        | Delete a paper           |

---

## 📁 Project Structure

```
AI Research Paper Assistant/
├── backend/
│   ├── main.py              # FastAPI app + CORS + routers
│   ├── config.py            # Settings from .env
│   ├── routers/
│   │   ├── upload.py        # PDF upload endpoints
│   │   ├── query.py         # Q&A endpoints (sync + stream)
│   │   └── papers.py        # Paper management
│   ├── services/
│   │   ├── pdf_processor.py # PDF extraction + chunking
│   │   ├── vector_store.py  # ChromaDB operations
│   │   └── llm_service.py   # GPT-4 RAG pipeline
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx     # Main application page
│   │   │   ├── layout.tsx   # Root layout + fonts
│   │   │   └── globals.css  # Design system + animations
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx   # Streaming chat UI
│   │   │   ├── PDFUploader.tsx     # Drag & drop uploader
│   │   │   └── PapersSidebar.tsx   # Papers list + filter
│   │   └── lib/
│   │       └── api.ts       # Axios client + types
│   └── package.json
├── docker-compose.yml
└── README.md
```

---

## 🎯 How RAG Works

```
PDF → Extract Text → Chunk (500 tokens) → Embed (1536-dim)
                                              ↓
User Query → Embed → Cosine Similarity Search → Top-5 Chunks
                                              ↓
             GPT-4 receives: [Context] + [Question]
                                              ↓
                    Grounded Answer + Source Citations
```
