"""
RAG System — FastAPI Backend (Groq + Database Edition)
=======================================================
Connects the HTML chat UI to the RAG pipeline powered by Groq.
All uploaded files, chunks, and chat history are stored in SQLite.

Run:
    $env:GROQ_API_KEY="gsk_..."
    uvicorn rag_api:app --reload --port 8000

Get your free Groq API key at: https://console.groq.com
"""

import os
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag_ingestion import ingest
from rag_vectorstore import get_embeddings, build_vectorstore, load_vectorstore, add_chunks
from rag_chain import RAGChatSession, get_groq_llm
from database import (
    init_db,
    insert_document, mark_document_indexed, mark_document_failed,
    get_all_documents, get_document_by_id, get_document_by_filename,
    delete_document, get_document_stats,
    insert_chunks, get_chunks_for_document,
    save_chat_message, get_chat_history, clear_chat_history,
)


# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

app = FastAPI(title="RAG API — Groq + DB", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("./uploads")
CHROMA_DIR = "./chroma_db"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
GROQ_MODEL = "llama-3.1-8b-instant"

embeddings  = get_embeddings()
vectorstore = None
session     = None


# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────

@app.on_event("startup")
def startup():
    init_db()
    _get_vectorstore()
    print("RAG API with database is ready.")


def _get_vectorstore():
    global vectorstore
    if vectorstore is None and Path(CHROMA_DIR).exists():
        vectorstore = load_vectorstore(embeddings=embeddings, persist_dir=CHROMA_DIR)
    return vectorstore


def _get_session():
    global session
    vs = _get_vectorstore()
    if vs is None:
        return None
    if session is None:
        session = RAGChatSession(vs, llm=get_groq_llm(model=GROQ_MODEL), top_k=4)
    return session


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    chunks_used: int


# ─────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────

@app.get("/status")
def status():
    stats = get_document_stats()
    return {
        "status":           "ok",
        "model":            GROQ_MODEL,
        "vectorstore_ready": _get_vectorstore() is not None,
        "total_docs":       stats.get("total_docs") or 0,
        "total_chunks":     stats.get("total_chunks") or 0,
        "total_size_bytes": stats.get("total_size_bytes") or 0,
        "indexed_count":    stats.get("indexed_count") or 0,
        "failed_count":     stats.get("failed_count") or 0,
    }


# ─────────────────────────────────────────────
# Documents
# ─────────────────────────────────────────────

@app.get("/documents")
def list_documents():
    """List all uploaded documents stored in the database."""
    docs = get_all_documents()
    return {"documents": docs, "total": len(docs)}


@app.get("/documents/{document_id}")
def get_document(document_id: int):
    doc = get_document_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@app.get("/documents/{document_id}/chunks")
def get_chunks(document_id: int):
    """Return all text chunks stored for a document."""
    if not get_document_by_id(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    chunks = get_chunks_for_document(document_id)
    return {"document_id": document_id, "chunks": chunks, "total": len(chunks)}


@app.get("/documents/{document_id}/download")
def download_document(document_id: int):
    """Download the original uploaded file from disk."""
    doc = get_document_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    if not Path(doc["file_path"]).exists():
        raise HTTPException(status_code=404, detail="File missing from disk.")
    return FileResponse(
        path=doc["file_path"],
        filename=doc["original_name"],
        media_type="application/octet-stream",
    )


@app.delete("/documents/{document_id}")
def remove_document(document_id: int):
    """Delete a document from the database and disk."""
    if not delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"status": "deleted", "document_id": document_id}


# ─────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload files → save to disk → record in DB → chunk → embed → index.
    Every step is tracked in SQLite.
    """
    global vectorstore, session

    results  = []
    rejected = []

    for file in files:
        ext = Path(file.filename).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            rejected.append(file.filename)
            continue

        content   = await file.read()
        file_size = len(content)
        dest      = UPLOAD_DIR / file.filename
        dest.write_bytes(content)

        # Skip if already indexed
        existing = get_document_by_filename(file.filename)
        if existing:
            results.append({
                "filename":    file.filename,
                "status":      "skipped",
                "reason":      "already indexed",
                "document_id": existing["id"],
            })
            continue

        # Save to database as 'processing'
        doc_id = insert_document(
            filename=file.filename,
            original_name=file.filename,
            file_path=str(dest),
            file_size_bytes=file_size,
            file_type=ext.lstrip("."),
        )

        try:
            # Ingest + chunk
            chunks = ingest([str(dest)], chunk_size=500, chunk_overlap=50)

            # Save chunks to database
            insert_chunks(doc_id, chunks)

            # Add to vector store
            if vectorstore is None:
                vectorstore = build_vectorstore(chunks, embeddings=embeddings, persist_dir=CHROMA_DIR)
            else:
                add_chunks(vectorstore, chunks)

            # Mark as indexed
            mark_document_indexed(doc_id, len(chunks))
            session = None   # reset so new content is picked up

            results.append({
                "filename":    file.filename,
                "status":      "indexed",
                "document_id": doc_id,
                "chunks":      len(chunks),
                "size_bytes":  file_size,
            })

        except Exception as e:
            mark_document_failed(doc_id, str(e))
            results.append({
                "filename":    file.filename,
                "status":      "failed",
                "document_id": doc_id,
                "error":       str(e),
            })

    if not results and rejected:
        raise HTTPException(
            status_code=400,
            detail=f"No supported files. Allowed: {list(ALLOWED_EXTENSIONS)}"
        )

    stats = get_document_stats()
    return {
        "results":      results,
        "rejected":     rejected,
        "total_docs":   stats.get("total_docs") or 0,
        "total_chunks": stats.get("total_chunks") or 0,
    }


# ─────────────────────────────────────────────
# Ask
# ─────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    """Answer a question — saves Q&A to chat_history in database."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    s = _get_session()
    if s is None:
        answer = "No documents indexed yet. Please upload files first."
        save_chat_message(body.question, answer, [], 0)
        return AskResponse(answer=answer, sources=[], chunks_used=0)

    result = s.chat(body.question)

    save_chat_message(
        question=body.question,
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=len(result["chunks"]),
    )

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=len(result["chunks"]),
    )


# ─────────────────────────────────────────────
# Chat history
# ─────────────────────────────────────────────

@app.get("/history")
def history(limit: int = 50):
    """Return recent chat history from the database."""
    messages = get_chat_history(limit=limit)
    return {"messages": messages, "total": len(messages)}


@app.post("/clear")
def clear_session():
    """Reset in-memory conversation context (keeps DB history)."""
    global session
    if session:
        session.clear_history()
    return {"status": "cleared"}


@app.delete("/history")
def delete_history():
    """Permanently delete all chat history from the database."""
    clear_chat_history()
    return {"status": "chat history deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=True)
