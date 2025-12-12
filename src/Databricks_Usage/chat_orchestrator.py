# src/Databricks_Usage/chat_orchestrator.py

"""
Main chat orchestration for the Databricks Usage domain.

- Uses GraphRAGRetriever for graph-aware retrieval.
- Uses your configured LLM provider (OpenAI / Gemini / Grok stub).
- Returns both:
    * The LLM answer.
    * A graph explanation string that your UI can display or log.

This is the "brain" of your Databricks Usage assistant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .config import (
    LLM_PROVIDER,
    get_chat_model_name,
    DEFAULT_TEMPERATURE,
)
from .graph_retriever import GraphRAGRetriever


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def get_llm():
    """
    Return a LangChain ChatModel based on LLM_PROVIDER + model name.

    Supported:
      - openai  -> langchain_openai.ChatOpenAI
      - gemini  -> langchain_google_genai.ChatGoogleGenerativeAI
      - grok    -> (placeholder) raise for now or plug in your client later.
    """
    provider = LLM_PROVIDER.lower()
    model_name = get_chat_model_name()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=DEFAULT_TEMPERATURE,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=DEFAULT_TEMPERATURE,
        )

    if provider == "grok":
        raise NotImplementedError(
            "Grok chat model not wired up yet. "
            "Update chat_orchestrator.get_llm() with your Grok client."
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


# ---------------------------------------------------------------------------
# Prompt for Databricks Usage assistant
# ---------------------------------------------------------------------------

ASSISTANT_SYSTEM_PROMPT = """You are an enterprise Databricks usage copilot.

You are answering questions for:
- Platform teams,
- FinOps teams,
- Data engineering leads,
- Analytics / ML practitioners.

Your knowledge comes from:
- Databricks jobs and their metadata,
- Job runs and cluster configs,
- Compute usage (DBUs, cost, instance types),
- Ad-hoc SQL queries,
- Events and spot evictions,
- Org units, users, and their departments.

Guidelines:
- Always ground your answers in the provided CONTEXT snippets.
- When explaining cost or reliability, reference specific jobs, compute resources,
  or queries by name/ID when possible.
- If multiple root causes are possible, say so and explain the tradeoffs.
- If the answer is not fully supported by the context, say you are inferring
  and call that out explicitly.
"""

ASSISTANT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", ASSISTANT_SYSTEM_PROMPT),
        (
            "system",
            "CONTEXT:\n{context}\n\nUse this context to answer the user's question.",
        ),
        ("human", "{question}"),
    ]
)


# ---------------------------------------------------------------------------
# Graph explanation builder
# ---------------------------------------------------------------------------

def build_graph_explanation(
    node_ids: List[str],
    retriever: GraphRAGRetriever,
) -> str:
    """
    Build a human-readable explanation string about the subgraph that was used.

    Example:
      "Looked at 18 nodes across types: 1 org units, 3 users, 5 jobs, 4 job runs,
       3 compute_usage, 2 events."
    """
    if not node_ids:
        return "GraphRAG fallback: no anchor nodes – used plain vector search."

    nodes = retriever.adj.nodes  # type: ignore[attr-defined]
    by_type: Dict[str, int] = {}
    for nid in node_ids:
        n = nodes.get(nid)
        if not n:
            continue
        by_type[n.type] = by_type.get(n.type, 0) + 1

    parts = [f"GraphRAG used a subgraph with {len(node_ids)} nodes."]
    if by_type:
        type_chunks = [
            f"{count} {node_type}" for node_type, count in sorted(by_type.items())
        ]
        parts.append("Node types: " + ", ".join(type_chunks) + ".")
    else:
        parts.append("Node type breakdown unavailable.")

    # Optional: show a couple of example nodes to make it feel concrete
    sample_ids = node_ids[:5]
    sample_descriptions = []
    for nid in sample_ids:
        n = nodes.get(nid)
        if not n:
            continue
        label = n.properties.get("job_name") or n.properties.get("name") or n.id
        sample_descriptions.append(f"{n.id} ({n.type}, label={label})")

    if sample_descriptions:
        parts.append("Example nodes: " + "; ".join(sample_descriptions) + ".")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main orchestration API
# ---------------------------------------------------------------------------

@dataclass
class ChatResult:
    answer: str
    context_docs: List[Document]
    graph_explanation: str


class DatabricksUsageAssistant:
    """
    Main chat orchestrator for Databricks usage questions.

    Usage:

        assistant = DatabricksUsageAssistant.from_local()
        result = assistant.answer("Why is my logistics job so expensive?")
        print(result.answer)
        print(result.graph_explanation)
    """

    def __init__(
        self,
        graph_retriever: GraphRAGRetriever,
    ) -> None:
        self.graph_retriever = graph_retriever
        self.llm = get_llm()
        self.chain = ASSISTANT_PROMPT_TEMPLATE | self.llm | StrOutputParser()

    @classmethod
    def from_local(cls) -> "DatabricksUsageAssistant":
        retriever = GraphRAGRetriever.from_local_index()
        return cls(graph_retriever=retriever)

    def _render_context(self, docs: List[Document]) -> str:
        """
        Render retrieved docs into a single context string.
        Keep it compact but informative.
        """
        chunks = []
        for d in docs:
            doc_id = d.metadata.get("doc_id", "unknown")
            doc_type = d.metadata.get("type", d.metadata.get("source", "doc"))
            header = f"[{doc_type} | {doc_id}]"
            body = d.page_content
            chunks.append(f"{header}\n{body}")
        return "\n\n---\n\n".join(chunks)

    def answer(self, question: str) -> ChatResult:
        """
        Full orchestration:
          1. GraphRAGRetriever: get subgraph docs + node_ids.
          2. Build graph explanation string.
          3. Run LLM with context + question.
          4. Return answer + context + explanation.
        """
        docs, node_ids = self.graph_retriever.get_subgraph_for_query(
            query=question,
            anchor_k=4,
            max_hops=2,
            max_nodes=40,
        )

        context_str = self._render_context(docs)
        graph_explanation = build_graph_explanation(node_ids=node_ids, retriever=self.graph_retriever)

        answer = self.chain.invoke(
            {
                "context": context_str,
                "question": question,
            }
        )

        return ChatResult(
            answer=answer,
            context_docs=docs,
            graph_explanation=graph_explanation,
        )


# ---------------------------------------------------------------------------
# CLI entry point for manual testing
# ---------------------------------------------------------------------------

def _interactive_cli() -> None:
    assistant = DatabricksUsageAssistant.from_local()
    print("[usage-assistant] Interactive mode. Type a question, or 'exit' to quit.")

    while True:
        try:
            q = input("\nYou> ").strip()
        except EOFError:
            break
        if not q or q.lower() in {"exit", "quit"}:
            break

        result = assistant.answer(q)
        print("\nAssistant>\n", result.answer)
        print("\n[graph_debug]", result.graph_explanation)


if __name__ == "__main__":
    _interactive_cli()
