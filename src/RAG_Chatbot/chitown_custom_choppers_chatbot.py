import os
from typing import Dict, Any, List, Tuple
from operator import itemgetter
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

# --- LangChain Core ---
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from pydantic import BaseModel, Field

# --- Vector Store & LLM ---
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# ----------------------------- CONFIG ------------------------------------------------

# Script lives at: src/RAG_Chatbot/chitown_custom_choppers_chatbot.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directory where FAISS index was saved by ingest_embed_index.py
INDEX_DIR = PROJECT_ROOT / "indices" / "faiss_chitowncustomchoppers_index"


# ----------------------------- LLM FACTORY ----------------------------------------


def get_llm():
    """
    Return a ChatOpenAI LLM configured for either:
      - OpenAI (default)
      - Grok (OpenAI-compatible endpoint, if configured).
    Controlled via env var: LLM_PROVIDER in {"openai", "grok"}.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "grok":
        # Requires:
        #   GROK_API_KEY
        #   GROK_MODEL (e.g. "grok-2-latest")
        #   GROK_API_BASE (if needed)
        return ChatOpenAI(
            api_key=os.environ["GROK_API_KEY"],
            model=os.getenv("GROK_MODEL", "grok-2-latest"),
            base_url=os.getenv("GROK_API_BASE", None),
            temperature=0.2,
            timeout=60.0,
        )
    else:
        # Default: OpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            temperature=0.2,
            timeout=60.0,
        )


# ----------------------------- CLASSIFICATION MODEL ------------------------------


class DocumentCategory(BaseModel):
    category: str = Field(
        description=(
            "The most relevant document_type category for answering this query. "
            "Must be one of: 'HR Policy', 'Financial Data', 'HR/Org Structure, 'Customer Policy', 'Customer Service', "
            "'Operations Manual', 'Marketing', or 'Other'."
        )
    )


# ----------------------------- RAG CHAIN SETUP (HYBRID) --------------------------


@st.cache_resource(show_spinner="Loading vector index, BM25, and LLM…")
def setup_rag_chain():
    # 1. Load vector store with OpenAI embeddings
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Please add it to your .env or environment."
        )

    embedding_model = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"Vector index directory '{INDEX_DIR}' not found. "
            f"Run ingest_embed_index.py first."
        )

    loaded_vectorstore: FAISS = FAISS.load_local(
        INDEX_DIR,
        embedding_model,
        allow_dangerous_deserialization=True,  # OK for local/dev
    )

    # Build BM25 retriever from the documents stored in the FAISS docstore
    # (this avoids re-reading PDFs).
    docstore_dict = getattr(loaded_vectorstore.docstore, "_dict", {})
    base_docs: List[Document] = list(docstore_dict.values())

    if not base_docs:
        raise RuntimeError("No documents found in FAISS docstore for BM25 construction.")

    bm25_retriever = BM25Retriever.from_documents(base_docs)
    # Optionally tweak how many docs BM25 returns
    bm25_retriever.k = 8

    llm = get_llm()

    # ---------------- Classification prompt & chain ----------------

    format_instructions = JsonOutputParser(
        pydantic_object=DocumentCategory
    ).get_format_instructions()

    CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a routing assistant for Chitown Custom Choppers, a custom chopper "
                    "bicycle shop in Rogers Park, Chicago.\n\n"
                    "Your job is to select the single best internal document category for a query.\n"
                    "Choose from:\n"
                    "  - HR Policy\n"
                    "  - Customer Policy\n"
                    "  - Customer Service\n"
                    "  - Operations Manual\n"
                    "  - Marketing\n"
                    "  - Financial Data\n"
                    "  - HR/Org Structure\n"
                    "  - Other\n\n"
                    "Return ONLY a JSON object following these instructions:\n"
                    "{format_instructions}"
                ),
            ),
            ("human", "User query: {query}"),
        ]
    ).partial(format_instructions=format_instructions)

    parser = JsonOutputParser(pydantic_object=DocumentCategory)
    classification_chain = CLASSIFICATION_PROMPT | llm | parser

    valid_categories = {
        "HR Policy",
        "Customer Policy",
        "Customer Service",
        "Operations Manual",
        "Financial Data",
        "HR/Org Structure",
        "Marketing",
    }

    # ---------------- Helper: build filter from classification ----------------

    def build_metadata_filter(classifier_output: Any) -> Dict[str, Any] | None:
        """Convert classifier output into a FAISS/BM25 metadata filter."""
        if isinstance(classifier_output, DocumentCategory):
            category = classifier_output.category
        elif isinstance(classifier_output, dict):
            category = classifier_output.get("category")
        else:
            category = None

        print(f"\n[DEBUG] Classification raw output: {classifier_output}")
        print(f"[DEBUG] Chosen category: {category}")

        if category in valid_categories:
            return {"document_type": category}
        else:
            print("[DEBUG] Unknown or missing category; using unfiltered search.")
            return None

    # ---------------- Helper: filter docs by metadata -------------------------

    def matches_filter(doc: Document, metadata_filter: Dict[str, Any] | None) -> bool:
        if not metadata_filter:
            return True
        for k, v in metadata_filter.items():
            if doc.metadata.get(k) != v:
                return False
        return True

    # ---------------- Helper: HYBRID RETRIEVAL (FAISS + BM25) -----------------

    def hybrid_retrieve_with_scores(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hybrid retrieval step:
        - Dense retrieval via FAISS similarity_search_with_score.
        - Sparse retrieval via BM25Retriever.
        - Metadata filter applied to both channels.
        - Scores normalized and combined into a hybrid score.
        - Returns:
            - query
            - context (string for LLM)
            - sources (rich provenance, including dense/sparse/hybrid scores).
        """
        query = inputs["query"]
        classification = inputs["classification"]

        metadata_filter = build_metadata_filter(classification)

        # --- Dense retrieval (FAISS) ---
        dense_kwargs: Dict[str, Any] = {"k": 5}
        if metadata_filter:
            dense_kwargs["filter"] = metadata_filter

        dense_docs_and_scores: List[Tuple[Document, float]] = (
            loaded_vectorstore.similarity_search_with_score(query, **dense_kwargs)
        )

        print(
            f"[DEBUG][DENSE] Retrieved {len(dense_docs_and_scores)} documents for query '{query}'."
        )

        # --- Sparse retrieval (BM25) ---
        # BM25Retriever in your setup behaves like a Runnable, so we use .invoke()
        try:
            bm25_candidates: List[Document] = bm25_retriever.get_relevant_documents(query)
        except AttributeError:
            # Fallback for retrievers implemented as LCEL runnables
            bm25_candidates: List[Document] = bm25_retriever.invoke(query)

        # Apply metadata filter post-hoc to BM25 results
        sparse_docs: List[Document] = [
            d for d in bm25_candidates if matches_filter(d, metadata_filter)
        ]


        print(
            f"[DEBUG][SPARSE] Retrieved {len(sparse_docs)} BM25 documents before/after filtering."
        )

        # --- Build lookup maps per doc identity (file_name + page + source) ---

        def doc_key(doc: Document) -> Tuple[Any, Any, Any]:
            return (
                doc.metadata.get("file_name"),
                doc.metadata.get("page"),
                doc.metadata.get("source"),
            )

        dense_map: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
        for doc, distance in dense_docs_and_scores:
            key = doc_key(doc)
            dense_map[key] = {
                "doc": doc,
                "dense_distance": float(distance),
            }

        sparse_map: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
        # Use rank-based pseudo-scores for BM25 (higher rank → smaller score).
        for rank, doc in enumerate(sparse_docs):
            key = doc_key(doc)
            # Simple decreasing score: 1.0, 0.5, 0.33, ...
            sparse_score = 1.0 / (rank + 1)
            sparse_map[key] = {
                "doc": doc,
                "sparse_score": sparse_score,
            }

        # --- Combine keys and compute hybrid scores ---------------------------

        all_keys = set(dense_map.keys()) | set(sparse_map.keys())
        combined: List[Dict[str, Any]] = []

        for key in all_keys:
            dense_entry = dense_map.get(key)
            sparse_entry = sparse_map.get(key)

            if dense_entry:
                dense_distance = dense_entry["dense_distance"]
                dense_conf = 1.0 / (1.0 + dense_distance)  # lower distance => higher confidence
            else:
                dense_distance = None
                dense_conf = 0.0

            sparse_score = sparse_entry["sparse_score"] if sparse_entry else 0.0

            # Simple normalization: both already ~[0,1]
            dense_norm = dense_conf
            sparse_norm = sparse_score

            # Hybrid weighting (you can tune these)
            alpha = 0.6  # dense
            beta = 0.4   # sparse

            hybrid_score = alpha * dense_norm + beta * sparse_norm

            # Prefer doc object from dense; fallback to sparse
            doc = (
                dense_entry["doc"]
                if dense_entry is not None
                else sparse_entry["doc"]
            )

            combined.append(
                {
                    "doc": doc,
                    "dense_distance": dense_distance,
                    "dense_conf": dense_conf,
                    "sparse_score": sparse_score,
                    "hybrid_score": hybrid_score,
                }
            )

        # Sort by hybrid_score descending (higher = better)
        combined_sorted = sorted(
            combined, key=lambda x: x["hybrid_score"], reverse=True
        )

        # Limit final number of chunks
        top_k_final = 5
        combined_sorted = combined_sorted[:top_k_final]

        # --- Build context string + source metadata ---------------------------

        context_chunks: List[str] = []
        sources: List[Dict[str, Any]] = []

        for i, entry in enumerate(combined_sorted):
            doc = entry["doc"]
            file_name = doc.metadata.get("file_name", "Unknown")
            doc_type = doc.metadata.get("document_type", "Unknown")
            page = doc.metadata.get("page", "Unknown")

            dense_distance = entry["dense_distance"]
            dense_conf = entry["dense_conf"]
            sparse_score = entry["sparse_score"]
            hybrid_score = entry["hybrid_score"]

            # Human-readable context header
            context_chunks.append(
                f"[Source {i+1} | {file_name} | type={doc_type} | page={page} | "
                f"dense_distance={dense_distance if dense_distance is not None else 'NA'} | "
                f"dense_conf≈{dense_conf:.3f} | sparse≈{sparse_score:.3f} | "
                f"hybrid≈{hybrid_score:.3f}]\n"
                f"{doc.page_content}"
            )

            sources.append(
                {
                    "id": i + 1,
                    "file_name": file_name,
                    "document_type": doc_type,
                    "page": page,
                    "dense_distance": dense_distance,
                    "dense_conf": dense_conf,
                    "sparse_score": sparse_score,
                    "hybrid_score": hybrid_score,
                }
            )

        context = (
            "\n\n---\n\n".join(context_chunks)
            if context_chunks
            else "No relevant documents found."
        )

        return {
            "query": query,
            "context": context,
            "sources": sources,
        }

    # ---------------- Answer prompt (RAG with context) -----------------------

    ANSWER_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are the in-house assistant for Chitown Custom Choppers, a custom "
                    "chopper bicycle shop in Rogers Park, Chicago.\n\n"
                    "Answer ONLY using the information in the 'Context' below. "
                    "If the answer is not clearly supported by the context, say that you "
                    "cannot answer from the shop documents.\n\n"
                    "Keep your tone friendly, clear, and grounded in the provided text."
                ),
            ),
            (
                "human",
                "Context:\n{context}\n\n"
                "Customer Question: {query}\n\n"
                "Answer in a helpful, concise way."
            ),
        ]
    )

    # ---------------- Final LCEL chain (answer + sources) --------------------

    rag_chain = (
        {
            "query": RunnablePassthrough(),
            "classification": {"query": RunnablePassthrough()} | classification_chain,
        }
        | RunnableLambda(hybrid_retrieve_with_scores)
        | {
            "answer": ANSWER_PROMPT | llm | StrOutputParser(),
            "sources": itemgetter("sources"),
        }
    )

    return rag_chain


# ----------------------------- STREAMLIT UI ----------------------------------------


def main():
    st.set_page_config(page_title="Chitown Custom Choppers RAG Bot", page_icon="🛠️")
    st.title("🛠️ Chitown Custom Choppers – Shop Knowledge Assistant")

    try:
        rag_chain = setup_rag_chain()
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    prompt_input = st.chat_input(
        "Ask about HR policies, returns, builds, store operations, or promotions…"
    )
    if not prompt_input:
        return

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Checking the Chitown Custom Choppers docs…"):
            try:
                result = rag_chain.invoke(prompt_input)
            except Exception as e:
                st.error(f"Error generating response: {e}")
                return

            answer = result.get("answer", "")
            sources = result.get("sources", [])

            st.markdown(answer)

            # Show provenance block
            if sources:
                st.markdown("---")
                st.markdown("**Sources used:**")
                for s in sources:
                    dense_part = (
                        f"dense_distance={s['dense_distance']:.4f}, "
                        f"dense_conf≈{s['dense_conf']:.3f}"
                        if s["dense_distance"] is not None
                        else "dense_distance=NA, dense_conf≈0.000"
                    )
                    st.markdown(
                        f"- Source {s['id']}: `{s['file_name']}` "
                        f"(type: {s['document_type']}, page: {s['page']}, "
                        f"{dense_part}, sparse≈{s['sparse_score']:.3f}, "
                        f"hybrid≈{s['hybrid_score']:.3f})"
                    )

    # Store assistant answer in history (just the text, not the sources block)
    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
