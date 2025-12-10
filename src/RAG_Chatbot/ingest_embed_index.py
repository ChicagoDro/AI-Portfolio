# ====================================================================================
# INGESTION SCRIPT: Creates the FAISS Vector Index with Metadata Tags (OpenAI version)
# ====================================================================================

import os
import json
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
load_dotenv()

# --- LangChain Core Imports ---
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- LangChain Community / OpenAI Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


# ----------------------------- CONFIG ------------------------------------------------
# ----------------------------- PATH CONFIG ------------------------------------------------

# Dynamically resolve repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input metadata describing PDFs
METADATA_FILE_PATH = PROJECT_ROOT / "data" / "document-metadata.json"

# Directory that stores PDF files
DATA_DIR = PROJECT_ROOT / "data" / "Chitown_Custom_Choppers"

# Directory where FAISS index is saved/loaded
INDEX_DIR = PROJECT_ROOT / "indices" / "faiss_chitowncustomchoppers_index"

# Create dir if needed
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------- HELPERS ----------------------------------------------

def load_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
    """Load document metadata JSON."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    docs = data.get("documents", [])
    if not isinstance(docs, list):
        raise ValueError("Expected 'documents' to be a list in metadata JSON.")

    print(f"Loaded {len(docs)} metadata entries from {metadata_path}")
    return docs


def load_documents_with_metadata(metadata_entries: List[Dict[str, Any]]) -> List[Document]:
    """Load PDFs and attach metadata from JSON entries."""
    all_docs: List[Document] = []

    for meta in metadata_entries:
        file_path = meta.get("file_path")
        file_name = meta.get("file_name")

        if not file_path:
            print(f"Skipping entry with missing file_path: {meta}")
            continue

        # e.g. "data/Chitown_Custom_Choppers/HR_Employee_Handbook.pdf"
        full_path = PROJECT_ROOT / file_path

        if not full_path.exists():
            print(f"WARNING: File not found: {full_path} (from {file_name})")
            continue

        print(f"Loading PDF: {full_path}")
        loader = PyPDFLoader(str(full_path))
        raw_docs = loader.load()

        for d in raw_docs:
            d.metadata.update(
                {
                    "file_name": file_name,
                    "file_path": str(full_path),
                    "document_type": meta.get("document_type", "Unknown"),
                    "category": meta.get("category", "Unknown"),
                    "description": meta.get("description", ""),
                    "sensitivity": meta.get("sensitivity", "Unknown"),
                }
            )
            all_docs.append(d)

    print(f"Loaded {len(all_docs)} raw documents (pages).")
    return all_docs


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


# ----------------------------- MAIN -------------------------------------------------

def main():
    # 0. Check API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Please add it to your .env or environment."
        )

    # 1. Load metadata
    metadata_entries = load_metadata(METADATA_FILE_PATH)

    # 2. Load PDFs with metadata
    raw_documents = load_documents_with_metadata(metadata_entries)

    if not raw_documents:
        print("No documents loaded. Exiting without creating an index.")
        return

    # 3. Chunk documents
    document_chunks = chunk_documents(raw_documents)

    # 4. Create vector store with OpenAI embeddings
    print("\n--- 4. Creating Vector Store (Embedding with OpenAI) ---")
    try:
        embedding_model = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        )
        vectorstore = FAISS.from_documents(document_chunks, embedding_model)

        vectorstore.save_local(INDEX_DIR)
        print(f"Successfully created and saved FAISS index: '{INDEX_DIR}'")
    except Exception as e:
        print(f"An error occurred during embedding or indexing: {e}")


if __name__ == "__main__":
    main()
