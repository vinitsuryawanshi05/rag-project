"""
RAG System — Embedding & Vector Store
======================================
Embeds chunks using a FREE local model (no API key needed for embeddings).
Stores vectors in ChromaDB on disk.

Groq does NOT provide an embedding API, so we use the free local
sentence-transformers model for embeddings — this is standard practice
when using Groq as the LLM.

Install:
    pip install langchain-huggingface chromadb sentence-transformers
"""

from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "./chroma_db"


def get_embeddings():
    """
    Free local embedding model — no API key needed.
    Downloads once (~90 MB), then runs offline forever.
    This is the standard approach when using Groq as LLM.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(
    chunks: List[Document],
    embeddings=None,
    collection_name: str = "rag_collection",
    persist_dir: str = CHROMA_DIR,
) -> Chroma:
    """
    Embed all chunks and save into ChromaDB.
    Call this once when you first add documents.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    print(f"\nEmbedding {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    print(f"Saved {len(chunks)} vectors to '{persist_dir}'")
    return vectorstore


def load_vectorstore(
    embeddings=None,
    collection_name: str = "rag_collection",
    persist_dir: str = CHROMA_DIR,
) -> Chroma:
    """
    Load existing vectorstore from disk.
    Use this on every restart instead of re-embedding everything.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    count = vectorstore._collection.count()
    print(f"Loaded vectorstore '{collection_name}' — {count} vectors")
    return vectorstore


def add_chunks(vectorstore: Chroma, new_chunks: List[Document]) -> None:
    """Add new chunks to an existing vectorstore without rebuilding."""
    vectorstore.add_documents(new_chunks)
    print(f"Added {len(new_chunks)} new chunks to vectorstore")


def search(
    vectorstore: Chroma,
    query: str,
    top_k: int = 4,
) -> List[Document]:
    """Find the most relevant chunks for a query."""
    return vectorstore.similarity_search(query, k=top_k)


def search_with_scores(
    vectorstore: Chroma,
    query: str,
    top_k: int = 4,
) -> List[tuple]:
    """Same as search() but returns (Document, score) tuples. Lower score = more relevant."""
    return vectorstore.similarity_search_with_score(query, k=top_k)


def format_context(docs: List[Document]) -> str:
    """Join retrieved chunks into a single context string for the LLM prompt."""
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source {i+1}: {source}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)
