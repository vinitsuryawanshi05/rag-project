"""
RAG System — Query & Answering with Groq LLM
=============================================
Uses Groq's fast inference API as the language model.
Retrieves relevant chunks from ChromaDB and generates grounded answers.

Install:
    pip install langchain-groq

Get your free Groq API key at: https://console.groq.com
Set it in PowerShell:
    $env:GROQ_API_KEY="gsk_..."
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma


# ─────────────────────────────────────────────
# Groq LLM
# ─────────────────────────────────────────────

def get_groq_llm(model: str = "llama-3.1-8b-instant"):
    """
    Groq LLM — fast, free tier available.
    
    Good model options:
        llama-3.1-8b-instant   — fastest, good quality (default)
        llama-3.3-70b-versatile — best quality, slightly slower
        mixtral-8x7b-32768     — good for long documents
        gemma2-9b-it           — alternative option

    Get your free API key at: https://console.groq.com
    Set it: $env:GROQ_API_KEY="gsk_..."
    """
    from langchain_groq import ChatGroq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set.\n"
            "Get a free key at https://console.groq.com\n"
            "Then run in PowerShell: $env:GROQ_API_KEY='gsk_...'"
        )

    return ChatGroq(
        model=model,
        temperature=0,        # 0 = factual, deterministic answers
        api_key=api_key,
    )


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY
the context provided below. If the answer is not clearly in the context, say
"I don't have enough information to answer that from the provided documents."

Context:
{context}

Question: {question}

Answer:"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE,
)


# ─────────────────────────────────────────────
# RAG Chain — single question answering
# ─────────────────────────────────────────────

class RAGChain:
    """
    Single question answering using Groq + ChromaDB.

    Usage:
        rag = RAGChain(vectorstore)
        result = rag.ask("What does the document say about pricing?")
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(
        self,
        vectorstore: Chroma,
        llm=None,
        top_k: int = 4,
        score_threshold: float = 1.2,
        model: str = "llama-3.1-8b-instant",
    ):
        self.vectorstore = vectorstore
        self.llm = llm or get_groq_llm(model)
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, question: str) -> List[Document]:
        results = self.vectorstore.similarity_search_with_score(
            question, k=self.top_k
        )
        return [doc for doc, score in results if score <= self.score_threshold]

    def build_prompt(self, question: str, docs: List[Document]) -> str:
        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(
                f"[Source {i+1}: {source}]\n{doc.page_content.strip()}"
            )
        context = "\n\n---\n\n".join(context_parts)
        return RAG_PROMPT.format(context=context, question=question)

    def ask(self, question: str) -> dict:
        """
        Full pipeline: retrieve -> build prompt -> get Groq answer.

        Returns:
            answer   : the answer string
            sources  : list of source filenames used
            chunks   : raw retrieved Document objects
        """
        docs = self.retrieve(question)

        if not docs:
            return {
                "answer": "I don't have enough information to answer that from the provided documents.",
                "sources": [],
                "chunks": [],
            }

        prompt = self.build_prompt(question, docs)
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        sources = list({doc.metadata.get("source", "unknown") for doc in docs})

        return {
            "answer": answer.strip(),
            "sources": sources,
            "chunks": docs,
        }


# ─────────────────────────────────────────────
# RAG Chat Session — multi-turn with memory
# ─────────────────────────────────────────────

class RAGChatSession:
    """
    Multi-turn chat with memory. Handles follow-up questions like
    "tell me more about that" by keeping recent conversation history.

    Usage:
        session = RAGChatSession(vectorstore)
        session.chat("What is the main topic?")
        session.chat("Can you expand on point 2?")
    """

    CHAT_PROMPT = """You are a helpful assistant. Use the context below to answer the question.
If the answer is not in the context, say so clearly.
You may refer to the conversation history for follow-up questions.

Context:
{context}

Conversation history:
{history}

Question: {question}

Answer:"""

    def __init__(
        self,
        vectorstore: Chroma,
        llm=None,
        top_k: int = 4,
        model: str = "llama-3.1-8b-instant",
    ):
        self.rag = RAGChain(vectorstore, llm, top_k, model=model)
        self.history: List[dict] = []

    def _format_history(self) -> str:
        if not self.history:
            return "None"
        lines = []
        for turn in self.history[-6:]:    # keep last 3 exchanges
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)

    def chat(self, question: str) -> dict:
        """Send a message, get an answer. History is managed automatically."""
        docs = self.rag.retrieve(question)

        if not docs:
            answer = "I don't have enough information to answer that from the provided documents."
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": answer})
            return {"answer": answer, "sources": [], "chunks": []}

        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(f"[Source {i+1}: {source}]\n{doc.page_content.strip()}")
        context = "\n\n---\n\n".join(context_parts)

        prompt = self.CHAT_PROMPT.format(
            context=context,
            history=self._format_history(),
            question=question,
        )

        response = self.rag.llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        answer = answer.strip()

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        sources = list({doc.metadata.get("source", "?") for doc in docs})
        return {"answer": answer, "sources": sources, "chunks": docs}

    def clear_history(self):
        self.history = []
        print("Conversation history cleared.")
