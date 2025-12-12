"""
Ingestion script for the Databricks-usage RAG domain.

- Reads structured usage data from the SQLite DB (usage.db)
- Builds high-level "RAG documents" (OUs, jobs, queries, etc.)
- Embeds them using the configured provider (OpenAI / Gemini / etc.)
- Stores them in a FAISS index under ./indexes/usage_faiss

Usage (from repo root):

    python -m src.Databricks_Usage.ingest_embed_index
    # or:
    python src/Databricks_Usage/ingest_embed_index.py

You can override paths with CLI flags if you want:

    python -m src.Databricks_Usage.ingest_embed_index \
        --db-path data/databricks_usage/usage.db \
        --index-dir indexes \
        --index-name usage_faiss
"""

import argparse
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from .config import (
    PROJECT_ROOT,
    USAGE_DB_PATH,
    LLM_PROVIDER,
    get_embed_model_name,
)
from .ingest_usage_domain import build_usage_rag_docs, RagDoc


# ---------------------------------------------------------------------------
# Embedding provider routing
# ---------------------------------------------------------------------------

def get_embeddings():
    """
    Return a LangChain Embeddings object based on LLM_PROVIDER and model name.

    Supported:
      - openai  -> langchain_openai.OpenAIEmbeddings
      - gemini  -> langchain_google_genai.GoogleGenerativeAIEmbeddings

    For 'grok', this currently raises NotImplementedError – plug in your
    preferred xAI/Grok embeddings here when ready.
    """
    provider = LLM_PROVIDER.lower()
    model_name = get_embed_model_name()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model_name)

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=model_name)

    if provider == "grok":
        # TODO: replace this with a real Grok embeddings class once you decide
        # which client / integration you want to use.
        raise NotImplementedError(
            "Grok embeddings are not wired up yet. "
            "Update ingest_embed_index.get_embeddings() with your Grok client."
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


# ---------------------------------------------------------------------------
# RAG document → LangChain Document conversion
# ---------------------------------------------------------------------------

def rag_docs_to_documents(rag_docs: List[RagDoc]) -> List[Document]:
    """
    Convert our simple RagDoc dataclass objects into LangChain Documents.
    """
    documents: List[Document] = []
    for d in rag_docs:
        metadata = {
            **d.metadata,
            "doc_id": d.doc_id,
            "source": "databricks_usage",
        }
        documents.append(Document(page_content=d.text, metadata=metadata))
    return documents


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

def build_faiss_index(
    db_path: Path,
    index_path: Path,
) -> None:
    """
    End-to-end:
      1. Build RAG docs from SQLite usage DB
      2. Convert to LangChain Documents
      3. Embed and store in FAISS
      4. Save to disk
    """
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found at {db_path}")

    index_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ingest] Using DB: {db_path}")
    print(f"[ingest] Index will be saved to: {index_path}")
    print(f"[ingest] Provider: {LLM_PROVIDER} | Embed model: {get_embed_model_name()}")

    # 1) Build logical RAG docs from the domain adapter
    rag_docs = build_usage_rag_docs()
    print(f"[ingest] Built {len(rag_docs)} RAG docs from usage domain")

    # 2) Convert to LangChain Document objects
    documents = rag_docs_to_documents(rag_docs)
    print(f"[ingest] Converted to {len(documents)} LangChain Documents")

    if not documents:
        raise RuntimeError("No documents produced from usage DB. "
                           "Check seed data and ingest_usage_domain.py")

    # 3) Get embeddings and build FAISS index
    embeddings = get_embeddings()
    print("[ingest] Creating FAISS index (this may take a moment)...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 4) Save index to disk
    vectorstore.save_local(str(index_path))
    print("[ingest] ✅ FAISS index saved successfully.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest Databricks usage data into a FAISS vector index."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(USAGE_DB_PATH),
        help="Path to the SQLite usage DB (default: from config.USAGE_DB_PATH)",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=str(PROJECT_ROOT / "indexes"),
        help="Directory where the FAISS index will be saved (default: ./indexes)",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="usage_faiss",
        help="Name of the FAISS index directory (default: usage_faiss)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    db_path = Path(args.db_path).resolve()
    index_dir = Path(args.index_dir).resolve()
    index_path = index_dir / args.index_name

    build_faiss_index(db_path=db_path, index_path=index_path)


if __name__ == "__main__":
    main()
