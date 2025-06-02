from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

from dotenv import load_dotenv
from langchain.globals import set_verbose

# LangChain community & Ollama wrappers
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

from llama_index.llms.ollama import Ollama
# from langchain_ollama import OllamaEmbeddings      # Embedding wrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
DATA_DIR: Path = Path("data/pdfs")
VECTOR_DIR: Path = Path("data/vectorstore")
PROMPT_DIR: Path = Path("app/prompts")

load_dotenv()
set_verbose(True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_documents(path: Union[str, Path] = DATA_DIR) -> List:
    """Load every PDF in *path* into LangChain Documents list."""
    path = Path(path)
    docs = []
    for pdf in path.glob("*.pdf"):
        docs.extend(PyPDFLoader(str(pdf)).load())
    return docs

def _get_embedder() -> OllamaEmbeddings:
    """Return an OllamaEmbeddings instance targeting nomic‑embed‑text."""
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

# ---------------------------------------------------------------------------
# Vector store build / load
# ---------------------------------------------------------------------------

def save_vectorstore(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    persist_path: Union[str, Path] = VECTOR_DIR,
) -> None:
    """Ingest PDFs → chunks → embeddings → FAISS → disk."""
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    embeddings = _get_embedder()
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    vectordb.save_local(str(persist_path))


def load_vectorstore_from_disk(
    persist_path: Union[str, Path] = VECTOR_DIR,
) -> FAISS:
    """Load FAISS index with the same embedder used to create it."""
    embeddings = _get_embedder()
    return FAISS.load_local(
        str(persist_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )

# ---------------------------------------------------------------------------
# RAG chain
# ---------------------------------------------------------------------------

def build_chain(
    vectordb: FAISS,
    prompt_version: str = "v3_asistente_pdi",
):
    """Return a ConversationalRetrievalChain wired to Ollama and FAISS."""
    prompt_file = PROMPT_DIR / f"{prompt_version}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_file}")

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_file.read_text(encoding="utf-8"),
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatOllama(
        model="llama3:8b",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
        request_timeout=300,
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False,
    )

# ---------------------------------------------------------------------------
# CLI utility
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG pipeline helpers")
    parser.add_argument(
        "--rebuild_index", action="store_true", help="Re‑create FAISS index from PDFs"
    )
    args = parser.parse_args()

    if args.rebuild_index:
        print("[INFO] Rebuilding FAISS vector store from PDFs…")
        save_vectorstore()
        print(f"[INFO] Vector store saved to {VECTOR_DIR.resolve()}")

# # To run this script:
# # python -m app.rag_pipeline --rebuild_index