"""
RAG System — Database Layer
============================
SQLite database to track uploaded files, their chunks, and chat history.
No extra install needed — SQLite is built into Python.

Tables:
    documents   — every uploaded file with metadata
    chunks      — every chunk created from each file
    chat_history — every question and answer
"""

import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

DB_PATH = "./rag_database.db"


# ─────────────────────────────────────────────
# Connection
# ─────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """Open a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL")   # safer for concurrent access
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ─────────────────────────────────────────────
# Setup — create tables if they don't exist
# ─────────────────────────────────────────────

def init_db() -> None:
    """
    Create all tables on first run.
    Safe to call every startup — uses IF NOT EXISTS.
    """
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                filename        TEXT    NOT NULL,
                original_name   TEXT    NOT NULL,
                file_path       TEXT    NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                file_type       TEXT    NOT NULL,
                chunk_count     INTEGER DEFAULT 0,
                status          TEXT    DEFAULT 'processing',
                uploaded_at     TEXT    NOT NULL,
                indexed_at      TEXT
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index     INTEGER NOT NULL,
                content         TEXT    NOT NULL,
                content_length  INTEGER NOT NULL,
                created_at      TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                question        TEXT    NOT NULL,
                answer          TEXT    NOT NULL,
                sources         TEXT,
                chunks_used     INTEGER DEFAULT 0,
                asked_at        TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_documents_status   ON documents(status);
            CREATE INDEX IF NOT EXISTS idx_chat_asked_at      ON chat_history(asked_at);
        """)
        conn.commit()
        print(f"Database ready: {DB_PATH}")
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Documents table
# ─────────────────────────────────────────────

def insert_document(
    filename: str,
    original_name: str,
    file_path: str,
    file_size_bytes: int,
    file_type: str,
) -> int:
    """
    Insert a new document record. Returns the new document ID.
    Status starts as 'processing' until chunks are saved.
    """
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO documents
                (filename, original_name, file_path, file_size_bytes, file_type, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (filename, original_name, file_path, file_size_bytes,
             file_type, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def mark_document_indexed(document_id: int, chunk_count: int) -> None:
    """Mark a document as fully indexed after chunking is done."""
    conn = get_connection()
    try:
        conn.execute(
            """
            UPDATE documents
            SET status='indexed', chunk_count=?, indexed_at=?
            WHERE id=?
            """,
            (chunk_count, datetime.utcnow().isoformat(), document_id),
        )
        conn.commit()
    finally:
        conn.close()


def mark_document_failed(document_id: int, reason: str = "") -> None:
    """Mark a document as failed if ingestion or embedding broke."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE documents SET status='failed' WHERE id=?",
            (document_id,),
        )
        conn.commit()
    finally:
        conn.close()


def get_all_documents() -> List[dict]:
    """Return all documents as a list of dicts."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY uploaded_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_document_by_id(document_id: int) -> Optional[dict]:
    """Return a single document by ID."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE id=?", (document_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_document_by_filename(filename: str) -> Optional[dict]:
    """Return a document by filename (used to check for duplicates)."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE filename=?", (filename,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_document(document_id: int) -> bool:
    """
    Delete a document and all its chunks from the database.
    Also deletes the file from disk.
    Returns True if deleted, False if not found.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT file_path FROM documents WHERE id=?", (document_id,)
        ).fetchone()
        if not row:
            return False

        file_path = row["file_path"]
        if os.path.exists(file_path):
            os.remove(file_path)

        conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
        conn.commit()
        return True
    finally:
        conn.close()


def get_document_stats() -> dict:
    """Return summary stats for the status endpoint."""
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*)                                    AS total_docs,
                SUM(chunk_count)                           AS total_chunks,
                SUM(file_size_bytes)                       AS total_size_bytes,
                COUNT(CASE WHEN status='indexed' THEN 1 END) AS indexed_count,
                COUNT(CASE WHEN status='failed'  THEN 1 END) AS failed_count
            FROM documents
            """
        ).fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Chunks table
# ─────────────────────────────────────────────

def insert_chunks(document_id: int, chunks: list) -> None:
    """
    Save all chunks for a document in one batch.
    chunks: list of LangChain Document objects.
    """
    conn = get_connection()
    now = datetime.utcnow().isoformat()
    try:
        conn.executemany(
            """
            INSERT INTO chunks (document_id, chunk_index, content, content_length, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (document_id, i, chunk.page_content,
                 len(chunk.page_content), now)
                for i, chunk in enumerate(chunks)
            ],
        )
        conn.commit()
    finally:
        conn.close()


def get_chunks_for_document(document_id: int) -> List[dict]:
    """Return all chunks for a given document."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE document_id=? ORDER BY chunk_index",
            (document_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Chat history table
# ─────────────────────────────────────────────

def save_chat_message(
    question: str,
    answer: str,
    sources: List[str],
    chunks_used: int,
) -> int:
    """Save a question+answer pair to chat history. Returns the new ID."""
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO chat_history (question, answer, sources, chunks_used, asked_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (question, answer, ",".join(sources), chunks_used,
             datetime.utcnow().isoformat()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_chat_history(limit: int = 50) -> List[dict]:
    """Return the most recent chat messages."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM chat_history ORDER BY asked_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["sources"] = d["sources"].split(",") if d["sources"] else []
            result.append(d)
        return result
    finally:
        conn.close()


def clear_chat_history() -> None:
    """Delete all chat history."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM chat_history")
        conn.commit()
    finally:
        conn.close()
