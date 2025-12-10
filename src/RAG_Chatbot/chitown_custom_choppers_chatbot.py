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

# --- Vector Store & Embeddings ---
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# ----------------------------- CONFIG ---------------------------------------------

# Directory where the FAISS index was saved by ingest_embed_index.py
# Dynamically resolve repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directory where FAISS index is saved/loaded
INDEX_DIR = PROJECT_ROOT / "indices" / "faiss_chitowncustomchoppers_index"


# ----------------------------- LLM FACTORY ----------------------------------------


def get_llm():
    """
    Return a ChatOpenAI LLM configured for either:
      - OpenAI (default)
      - Grok (OpenAI-compatible endpoint, if you configure it)
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
            "Must be one of: 'HR Policy', 'Customer Policy', 'Customer Service', "
            "'Operations Manual', 'Marketing', or 'Other'."
        )
    )


# ----------------------------- RAG CHAIN SETUP -----------------------------------


@st.cache_resource(show_spinner="Loading vector index and LLM…")
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

    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError(
            f"Vector index directory '{INDEX_DIR}' not found. "
            f"Run ingest_embed_index.py first."
        )

    loaded_vectorstore = FAISS.load_local(
        INDEX_DIR,
        embedding_model,
        allow_dangerous_deserialization=True,  # ok for local/dev
    )

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
        "Marketing",
    }

    # ---------------- Helper: build filter from classification ----------------

    def build_metadata_filter(classifier_output: Any) -> Dict[str, Any] | None:
        """Convert classifier output into a FAISS metadata filter."""
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

    # ---------------- Helper: retrieve docs with scores + build context --------

    def retrieve_with_scores(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upgraded retrieval step:
        - Uses similarity_search_with_score (so we keep scores)
        - Applies optional metadata filter from classification
        - Returns:
            - query
            - context (string to feed to LLM)
            - sources (list of provenance dicts for UI)
        """
        query = inputs["query"]
        classification = inputs["classification"]

        metadata_filter = build_metadata_filter(classification)

        search_kwargs: Dict[str, Any] = {"k": 3}
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter

        print(f"[DEBUG] search_kwargs: {search_kwargs}")

        # Directly use similarity_search_with_score to get scores
        docs_and_scores: List[Tuple[Document, float]] = (
            loaded_vectorstore.similarity_search_with_score(
                query, **search_kwargs
            )
        )

        print(
            f"[DEBUG] Retrieved {len(docs_and_scores)} documents for query '{query}'."
        )

        # Build context string + source metadata
        context_chunks: List[str] = []
        sources: List[Dict[str, Any]] = []

        for i, (doc, score) in enumerate(docs_and_scores):
            file_name = doc.metadata.get("file_name", "Unknown")
            doc_type = doc.metadata.get("document_type", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            # FAISS scores are distances; lower = closer. Convert to a confidence-ish score:
            # quick heuristic: confidence = 1 / (1 + distance)
            distance = float(score)
            confidence = 1.0 / (1.0 + distance)

            context_chunks.append(
                f"[Source {i+1} | {file_name} | type={doc_type} | page={page} | "
                f"distance={distance:.4f} | conf≈{confidence:.3f}]\n"
                f"{doc.page_content}"
            )

            sources.append(
                {
                    "id": i + 1,
                    "file_name": file_name,
                    "document_type": doc_type,
                    "page": page,
                    "distance": distance,
                    "confidence": confidence,
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

    # ---------------- Answer prompt (now RAG with context) ---------------------

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

    # ---------------- Final LCEL chain (answer + sources) ----------------------

    rag_chain = (
        {
            "query": RunnablePassthrough(),
            "classification": {"query": RunnablePassthrough()} | classification_chain,
        }
        | RunnableLambda(retrieve_with_scores)
        | {
            # Run LLM on {query, context}
            "answer": ANSWER_PROMPT | llm | StrOutputParser(),
            # Pass through sources untouched for UI
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
                    st.markdown(
                        f"- Source {s['id']}: `{s['file_name']}` "
                        f"(type: {s['document_type']}, page: {s['page']}, "
                        f"distance: {s['distance']:.4f}, conf≈{s['confidence']:.3f})"
                    )

    # Store assistant answer in history (just the text, not the sources block)
    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
