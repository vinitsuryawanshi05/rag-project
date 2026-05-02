"""
RAG System — Document Ingestion & Chunking
==========================================
Supports: PDF, DOCX, TXT, Markdown, CSV, Web URLs

Install:
    pip install langchain langchain-community langchain-text-splitters
    pip install pypdf docx2txt unstructured beautifulsoup4
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    WebBaseLoader,
)

SUPPORTED_EXTENSIONS = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
    ".md":   UnstructuredMarkdownLoader,
    ".csv":  CSVLoader,
}


def load_file(file_path: str) -> List[Document]:
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {list(SUPPORTED_EXTENSIONS.keys())}"
        )
    loader = SUPPORTED_EXTENSIONS[ext](str(path))
    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", path.name)
    print(f"  Loaded  {path.name}  ->  {len(docs)} page(s)")
    return docs


def load_url(url: str) -> List[Document]:
    loader = WebBaseLoader(url)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = url
    print(f"  Fetched  {url}  ->  {len(docs)} page(s)")
    return docs


def load_directory(directory: str, recursive: bool = True) -> List[Document]:
    all_docs: List[Document] = []
    root = Path(directory)
    pattern = "**/*" if recursive else "*"
    for file_path in root.glob(pattern):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                all_docs.extend(load_file(str(file_path)))
            except Exception as e:
                print(f"  Skipped {file_path.name}: {e}")
    print(f"\n  Total loaded from '{directory}': {len(all_docs)}")
    return all_docs


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)
    print(f"  Chunked {len(documents)} doc(s)  ->  {len(chunks)} chunk(s)")
    return chunks


def ingest(
    sources: List[str],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Main entry point. Pass a mixed list of file paths, folders, or URLs.
    Returns ready-to-embed chunks.

    Example:
        chunks = ingest(["report.pdf", "notes.txt", "my_docs/"])
    """
    print("=" * 50)
    print("RAG Ingestion Pipeline")
    print("=" * 50)
    all_docs: List[Document] = []
    for source in sources:
        print(f"\nProcessing: {source}")
        path = Path(source)
        if source.startswith("http://") or source.startswith("https://"):
            all_docs.extend(load_url(source))
        elif path.is_dir():
            all_docs.extend(load_directory(source))
        elif path.is_file():
            all_docs.extend(load_file(source))
        else:
            print(f"  Not found: {source}")
    chunks = chunk_documents(all_docs, chunk_size, chunk_overlap)
    print(f"\nTotal chunks ready for embedding: {len(chunks)}")
    print("=" * 50)
    return chunks


def chunk_stats(chunks: List[Document]) -> dict:
    lengths = [len(c.page_content) for c in chunks]
    return {
        "total_chunks": len(chunks),
        "avg_length":   round(sum(lengths) / len(lengths)) if lengths else 0,
        "min_length":   min(lengths) if lengths else 0,
        "max_length":   max(lengths) if lengths else 0,
        "sources":      list({c.metadata.get("source", "?") for c in chunks}),
    }
