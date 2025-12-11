import os
from typing import Dict, Any, List, Tuple
from operator import itemgetter

import sys
from pathlib import Path

import re

# Ensure project root is on sys.path so `src` can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


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
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)



# --- GraphRAG utilities ---
from src.RAG_Chatbot.graph_retrieval import (
    load_graph,
    list_people,
    format_org_summary,
    format_person_sales_summary,
    get_q3_2024_total_sales,
)



load_dotenv()

# ----------------------------- CONFIG ---------------------------------------------

# Dynamically resolve repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directory where FAISS index is saved/loaded
PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
INDEX_DIR = PROJECT_ROOT / "indices" / f"faiss_chitowncustomchoppers_index_{PROVIDER}"


# ----------------------------- LLM FACTORY ----------------------------------------


def get_llm():
    """
    Return an LLM configured for one of the supported providers:
      - OpenAI   (default)
      - Grok     (OpenAI-compatible endpoint)
      - Gemini   (Google Generative AI)

    Controlled via env var: LLM_PROVIDER in {"openai", "grok", "gemini"}.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    # Env vars come in as strings — cast them
    temperature = float(os.getenv("LLM_TEMP", "0.2"))
    timeout = float(os.getenv("LLM_TIMEOUT", "60.0"))

    # ---------------------------
    # Gemini (Google GenAI)
    # ---------------------------
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            google_api_key=os.environ["GEMINI_API_KEY"],
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            temperature=temperature,
            timeout=timeout,
        )

    # ---------------------------
    # Grok (OpenAI-compatible)
    # ---------------------------
    if provider == "grok":
        return ChatOpenAI(
            api_key=os.environ["GROK_API_KEY"],
            model=os.getenv("GROK_MODEL", "grok-2-latest"),
            base_url=os.getenv("GROK_API_BASE", None),
            temperature=temperature,
            timeout=timeout,
        )

    # ---------------------------
    # OpenAI (default)
    # ---------------------------
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=temperature,
        timeout=timeout,
    )

def get_embedding_model():
    """
    Return an embeddings model compatible with the current provider.

    By default, we tie embeddings to LLM_PROVIDER:
      - LLM_PROVIDER=gemini  -> GoogleGenerativeAIEmbeddings
      - LLM_PROVIDER=openai  -> OpenAIEmbeddings
      - LLM_PROVIDER=grok    -> OpenAIEmbeddings (OpenAI-compatible)

    You can override with EMBEDDING_PROVIDER if needed.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", os.getenv("LLM_PROVIDER", "openai")).lower()

    # ---------------------------
    # Gemini embeddings
    # ---------------------------
    if provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            google_api_key=os.environ["GEMINI_API_KEY"],
            model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"),
        )

    # ---------------------------
    # Default: OpenAI embeddings
    # ---------------------------
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Either:\n"
            "- Set LLM_PROVIDER=gemini (and GEMINI_API_KEY) so embeddings use Gemini, or\n"
            "- Provide OPENAI_API_KEY for OpenAI/Grok embeddings."
        )

    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )


# ----------------------------- CLASSIFICATION MODEL ------------------------------


class DocumentCategory(BaseModel):
    category: str = Field(
        description=(
            "The most relevant document_type category for answering this query. "
            "Must be one of: 'HR Policy', 'Customer Policy', 'Customer Service', "
            " 'Operations Manual', 'Marketing', or 'Other'."
        )
    )


# ----------------------------- RAG CHAIN SETUP -----------------------------------


@st.cache_resource(show_spinner="Loading vector index and LLM…")
def setup_rag_chain():
    # 1. Load vector store with LLM specific embeddings
    embedding_model = get_embedding_model()

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

    # ---------------- Answer prompt (RAG with context) ---------------------

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


# ----------------------------- GRAPH RAG SETUP ------------------------------------


@st.cache_resource(show_spinner="Loading knowledge graph…")
def load_knowledge_graph():
    """Load the knowledge graph from data/graph/graph_output.json."""
    return load_graph()


GRAPH_KEYWORDS = [
    "org chart",
    "organization",
    "organisational structure",
    "org structure",
    "who works there",
    "who works at",
    "who reports to",
    "reports to",
    "manager",
    "direct reports",
    "team",
    "department",
    "headcount",
    "q3 sales",
    "quarter 3 sales",
    "monthly sales",
    "sales by employee",
    "sales by person",
]


def _normalize_for_routing(text: str) -> str:
    """
    Lowercase, remove apostrophes/punctuation, collapse spaces.
    This makes 'who's the CEO' and 'who is the ceo?' look similar.
    """
    text = text.lower()
    # remove apostrophes like who's → whos
    text = re.sub(r"[’']", "", text)
    # replace non-alphanumeric with space
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


ORG_TOKENS = {
    "org", "organization", "structure", "department", "team", "orgchart",
}
PEOPLE_TOKENS = {
    "who", "employee", "employees", "people", "staff", "manager", "managers",
    "report", "reports", "reporting", "headcount", "ceo", "founder", "owner",
}
SALES_TOKENS = {
    "sales", "revenue", "q3", "quarter", "monthly",
}


def is_graph_query(user_input: str) -> bool:
    """
    Fuzzy-ish router for graph questions.

    Heuristics:
    - If the question is about org structure / people / roles / headcount → graph
    - If the question is about Q3 / monthly sales by person or overall → graph
    """
    raw = user_input
    text = _normalize_for_routing(user_input)
    tokens = set(text.split())

    # Debug so you can see what the router is doing
    print(f"[DEBUG][ROUTER] normalized='{text}', tokens={tokens}")

    # 1) Direct “org-ish” intent: structure / departments / teams / headcount
    if ("org" in tokens and "chart" in tokens) or ("structure" in tokens and "org" in tokens):
        print(f"[DEBUG][ROUTER] GraphRAG (org+chart/structure) for: {raw!r}")
        return True

    if tokens & ORG_TOKENS and tokens & PEOPLE_TOKENS:
        # e.g. "organization structure", "team structure", "department staff"
        print(f"[DEBUG][ROUTER] GraphRAG (org+people tokens) for: {raw!r}")
        return True

    # 2) “Who” questions about people or roles
    if "who" in tokens and (tokens & {"works", "work", "reports", "manages", "manager", "managers", "ceo", "founder", "owner"}):
        # catches: "who is the ceo", "who's the ceo", "who works here", "who reports to rosa"
        print(f"[DEBUG][ROUTER] GraphRAG (who+people) for: {raw!r}")
        return True

    if "ceo" in tokens or "founder" in tokens or "owner" in tokens:
        print(f"[DEBUG][ROUTER] GraphRAG (role=ceo/founder/owner) for: {raw!r}")
        return True

    # 3) Headcount / how many people / employees
    if "how" in tokens and ("many" in tokens or "total" in tokens) and (tokens & {"people", "employees", "staff", "headcount"}):
        print(f"[DEBUG][ROUTER] GraphRAG (headcount) for: {raw!r}")
        return True

    # 4) Sales questions (Q3 / monthly / by employee)
    if tokens & SALES_TOKENS and "sales" in tokens:
        # e.g. "q3 sales", "monthly sales", "sales by employee"
        print(f"[DEBUG][ROUTER] GraphRAG (sales) for: {raw!r}")
        return True

    print(f"[DEBUG][ROUTER] NOT routing to GraphRAG for: {raw!r}")
    return False



def answer_with_graphrag(user_query: str, G) -> Dict[str, Any]:
    """
    Handle org / people / sales questions using the knowledge graph,
    then let the LLM turn the graph facts into a natural language answer.

    Returns:
        {
          "answer": str,
          "sources": []   # kept empty for now, so UI won't show the provenance block
        }
    """
    q = user_query.lower()

    # 1) Person-specific sales / info if we spot a known name
    person_id = None
    for person in list_people(G):
        name_lower = person["name"].lower()
        if name_lower in q:
            person_id = person["id"]
            break

    if person_id:
        graph_context = format_person_sales_summary(G, person_id)
    # 2) Org structure questions
    elif (
        "org chart" in q
        or "org structure" in q
        or "organization" in q
        or "who works there" in q
        or "headcount" in q
        or "who works at" in q
    ):
        graph_context = format_org_summary(G)
    # 3) Q3 sales / total sales questions
    elif "q3" in q and "sales" in q or "total sales" in q:
        total = get_q3_2024_total_sales(G)
        graph_context = (
            f"Total Q3 2024 sales for Chitown Custom Choppers were "
            f"${total:,.2f} based on all SalesMetric nodes in the graph."
        )
    else:
        # Fallback: org summary is a safe default
        graph_context = format_org_summary(G)

    # Use a small prompt tailored for graph-based answers
    graph_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an internal assistant for Chitown Custom Choppers. "
                    "You are given structured information from the company's "
                    "knowledge graph (org chart, departments, Q3 2024 sales).\n\n"
                    "Use ONLY this information when answering. If the graph context "
                    "does not clearly answer the question, say that the information "
                    "is not available in the current company graph."
                ),
            ),
            (
                "human",
                "User question:\n{query}\n\n"
                "Graph context:\n{graph_context}\n\n"
                "Provide a clear, concise answer grounded only in the graph context."
            ),
        ]
    )

    llm = get_llm()
    chain = graph_prompt | llm | StrOutputParser()

    answer_text = chain.invoke(
        {
            "query": user_query,
            "graph_context": graph_context,
        }
    )

    # For now, we don't show a 'Sources' block for graph answers
    # You could later add explicit graph provenance here.
    return {
        "answer": answer_text,
        "sources": [],
    }


# ----------------------------- STREAMLIT UI ----------------------------------------


def main():
    st.set_page_config(page_title="Chitown Custom Choppers RAG Bot", page_icon="🛠️")
    st.title("🛠️ Chitown Custom Choppers – Shop Knowledge Assistant")

    # Initialize RAG and GraphRAG resources
    try:
        rag_chain = setup_rag_chain()
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        st.stop()

    try:
        knowledge_graph = load_knowledge_graph()
    except Exception as e:
        st.error(f"Error loading knowledge graph: {e}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    prompt_input = st.chat_input(
        "Ask about HR policies, returns, builds, store operations, org structure, or Q3 sales…"
    )
    if not prompt_input:
        return

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Consulting Chitown Custom Choppers knowledge…"):
            try:
                if is_graph_query(prompt_input):
                    result = answer_with_graphrag(prompt_input, knowledge_graph)
                else:
                    result = rag_chain.invoke(prompt_input)
            except Exception as e:
                st.error(f"Error generating response: {e}")
                return

            answer = result.get("answer", "")
            sources = result.get("sources", [])

            st.markdown(answer)

            # Show provenance block for vector-RAG answers
            if sources:
                st.markdown("---")
                st.markdown("**Sources used:**")
                for s in sources:
                    st.markdown(
                        f"- Source {s['id']}: `{s['file_name']}` "
                        f"(type: {s['document_type']}, page: {s['page']}, "
                        f"distance={s['distance']:.4f}, conf≈{s['confidence']:.3f})"
                    )

    # Store assistant answer in history (just the text, not the sources block)
    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
