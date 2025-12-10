import os
from typing import Dict, Any, List
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

# ------------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "indices" / "faiss_chitowncustomchoppers_index"

# To avoid potential issues on some systems (e.g., Mac M1/M2)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


load_dotenv()


# ----------------------------- LLM FACTORY -----------------------------------------

def get_llm():
    """
    Return a ChatOpenAI LLM configured for either:
      - OpenAI (default)
      - Grok (OpenAI-compatible endpoint, if you configure it)
    Controlled via env var: LLM_PROVIDER in {"openai", "grok"}.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "grok":
        # You need to set these based on Grok/xAI docs:
        #   GROK_API_KEY
        #   GROK_API_BASE (if needed)
        #   GROK_MODEL (e.g. "grok-2-latest")
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


# ----------------------------- CLASSIFICATION MODEL --------------------------------

class DocumentCategory(BaseModel):
    category: str = Field(
        description=(
            "The most relevant document_type category for answering this query. "
            "Must be one of: 'HR Policy', 'Customer Policy', 'Customer Service', "
            "'Operations Manual', 'Marketing', or 'Other'."
        )
    )


# ------------------------------------------------------------------------------------
# RAG CHAIN SETUP
# ------------------------------------------------------------------------------------

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
        allow_dangerous_deserialization=True,  # fine for local/dev use
    )

    llm = get_llm()

    # 2. Classification prompt & chain
    format_instructions = JsonOutputParser(pydantic_object=DocumentCategory).get_format_instructions()

    CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a routing assistant that decides which internal document "
                    "category best matches the user's query.\n\n"
                    "Choose the single best category from:\n"
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

    # 3. Dynamic retriever

    valid_categories = {
        "HR Policy",
        "Customer Policy",
        "Customer Service",
        "Operations Manual",
        "Marketing",
    }

    def get_filtered_retriever(query: str, classifier_output: Any):
        """Build a retriever with optional metadata filter based on classification."""
        # Handle DocumentCategory (Pydantic) vs dict vs anything else
        if isinstance(classifier_output, DocumentCategory):
            category = classifier_output.category
        elif isinstance(classifier_output, dict):
            category = classifier_output.get("category")
        else:
            category = None

        print(f"\n[DEBUG] Classification raw output: {classifier_output}")
        print(f"[DEBUG] Chosen category: {category}")

        metadata_filter = None
        if category in valid_categories:
            metadata_filter = {"document_type": category}
        else:
            print("[DEBUG] Unknown or missing category; using unfiltered retriever.")

        search_kwargs: Dict[str, Any] = {"k": 3}
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter

        retriever_filtered = loaded_vectorstore.as_retriever(search_kwargs=search_kwargs)
        return retriever_filtered

    def retrieve_and_format(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        LCEL step:
          - Uses classification to get a filtered retriever
          - Retrieves docs
          - Formats them into a 'context' string for the final prompt
        """
        query = inputs["query"]
        classification = inputs["classification"]

        retriever = get_filtered_retriever(query, classification)
        docs: List[Document] = retriever.invoke(query)

        # Build a context string, but keep some metadata for potential debugging
        context_chunks = []
        for i, d in enumerate(docs):
            source = d.metadata.get("file_name", "Unknown source")
            doc_type = d.metadata.get("document_type", "Unknown type")
            context_chunks.append(
                f"[Source {i+1} | {source} | {doc_type}]\n{d.page_content}"
            )

        context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant documents found."

        print(f"\n[DEBUG] Retrieved {len(docs)} documents for query '{query}'.")

        return {
            "context": context,
            "query": query,
        }

    # 4. Final answer prompt
    ANSWER_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are Chi-Town Custom Choppers, a helpful assistant that answers questions "
                    "using the provided company documents for a small bookstore / café.\n\n"
                    "Use ONLY the information in the 'Context' below when answering. "
                    "If the answer is not in the context, say you do not know based on "
                    "the available documents.\n\n"
                    "Keep your tone warm, clear, and concise."
                ),
            ),
            (
                "human",
                "Context:\n{context}\n\nUser Question: {query}\n\nAnswer in a friendly tone.",
            ),
        ]
    )

    def debug_prompt_inputs(x: Dict[str, Any]) -> Dict[str, Any]:
        print("\n[DEBUG] Inputs to ANSWER_PROMPT:")
        for k, v in x.items():
            # v might be long; slice if it's a string
            if isinstance(v, str):
                print(f"{k.upper()}:\n{v[:1000]}")
            else:
                print(f"{k.upper()}:\n{v}")
        return x

    rag_chain = (
        {
            "query": RunnablePassthrough(),
            "classification": {"query": RunnablePassthrough()} | classification_chain,
        }
        | RunnableLambda(retrieve_and_format)   # returns {"context": ..., "query": ...}
        | RunnableLambda(debug_prompt_inputs)   # <-- debug here
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )


    return rag_chain


# ------------------------------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Chi-Town Custom Choppers Chatbot", page_icon="☕")
    st.title("Chi-Town Custom Choppers Knowledge Assistant")

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
    prompt_input = st.chat_input("Ask me about policies, returns, operations, or promotions…")
    if not prompt_input:
        return

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching the Chi-Town Custom Choppers knowledge base…"):
            try:
                response = rag_chain.invoke(prompt_input)
            except Exception as e:
                st.error(f"Error generating response: {e}")
                return
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
