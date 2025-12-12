# src/Databricks_Usage/graph_retriever.py

"""
Graph-aware retriever for the Databricks Usage domain.

High-level flow:
  1. Load FAISS vector index of RAG documents.
  2. Build in-memory usage graph (nodes + edges) from SQLite.
  3. For a natural language query:
      - Use vector search to find anchor docs / nodes.
      - Expand a subgraph around those anchors (BFS over edges).
      - Collect all docs whose doc_id matches nodes in the subgraph.
  4. Return those docs as retrieval context for the LLM.

This lets you:
  - Treat Databricks usage as a graph (OU ↔ jobs ↔ runs ↔ usage ↔ events ↔ evictions ↔ queries),
  - But still run everything on top of a FAISS + RAG pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from .config import PROJECT_ROOT
from .ingest_embed_index import get_embeddings  # reuse provider routing
from .graph_model import build_usage_graph, GraphNode, GraphEdge


# ---------------------------------------------------------------------------
# Internal helper: adjacency index for the graph
# ---------------------------------------------------------------------------

@dataclass
class GraphAdjacency:
    nodes: Dict[str, GraphNode]
    edges: List[GraphEdge]
    neighbors: Dict[str, Set[str]]  # node_id -> set(node_id)


def _build_adjacency() -> GraphAdjacency:
    """
    Build an adjacency structure from the usage graph.

    - nodes: node_id -> GraphNode
    - edges: list of GraphEdge
    - neighbors: node_id -> set of neighboring node_ids (undirected for traversal)
    """
    nodes, edges = build_usage_graph()

    neighbors: Dict[str, Set[str]] = {}
    for e in edges:
        neighbors.setdefault(e.src, set()).add(e.dst)
        neighbors.setdefault(e.dst, set()).add(e.src)

    return GraphAdjacency(nodes=nodes, edges=edges, neighbors=neighbors)


# ---------------------------------------------------------------------------
# GraphRAG-style retriever
# ---------------------------------------------------------------------------

class GraphRAGRetriever:
    """
    Graph-aware retriever:
      - wraps a FAISS vectorstore of usage-domain docs
      - uses the graph to expand from anchor nodes to a subgraph
    """

    def __init__(
        self,
        vectorstore: FAISS,
        adjacency: GraphAdjacency,
        doc_id_to_doc: Dict[str, Document],
    ) -> None:
        self.vectorstore = vectorstore
        self.adj = adjacency
        self.doc_id_to_doc = doc_id_to_doc

    # ------------------------------------------------------------------
    # Factory constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_local_index(
        cls,
        index_dir: Path | str | None = None,
        index_name: str = "usage_faiss",
    ) -> "GraphRAGRetriever":
        """
        Load FAISS index from disk, build graph + adjacency, and construct retriever.

        Args:
            index_dir: directory containing the FAISS index (defaults to ./indexes)
            index_name: name of the index directory (defaults to "usage_faiss")
        """
        if index_dir is None:
            index_dir = PROJECT_ROOT / "indexes"
        index_dir = Path(index_dir).resolve()
        index_path = index_dir / index_name

        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        # Build graph adjacency
        adjacency = _build_adjacency()

        # Build doc_id -> Document mapping from the vectorstore docstore
        doc_id_to_doc: Dict[str, Document] = {}
        # In LangChain FAISS, docs are stored in docstore._dict
        if hasattr(vectorstore, "docstore") and hasattr(vectorstore.docstore, "_dict"):
            for _store_id, doc in vectorstore.docstore._dict.items():
                doc_id = doc.metadata.get("doc_id")
                if doc_id:
                    doc_id_to_doc[doc_id] = doc

        return cls(
            vectorstore=vectorstore,
            adjacency=adjacency,
            doc_id_to_doc=doc_id_to_doc,
        )

    # ------------------------------------------------------------------
    # Core retrieval API
    # ------------------------------------------------------------------

    def _get_anchor_node_ids(
        self,
        query: str,
        k: int = 4,
    ) -> List[str]:
        """
        Use vector search to find initial anchor docs/nodes for the query.
        Returns a list of node_ids (doc_ids) in relevance order.
        """
        docs = self.vectorstore.similarity_search(query, k=k)
        anchor_ids: List[str] = []
        for d in docs:
            doc_id = d.metadata.get("doc_id")
            if doc_id and doc_id in self.adj.nodes:
                anchor_ids.append(doc_id)
        return anchor_ids

    def _expand_subgraph(
        self,
        anchor_ids: Iterable[str],
        max_hops: int = 2,
        max_nodes: int = 50,
    ) -> Set[str]:
        """
        Simple BFS over the graph starting from anchor_ids.
        Returns a set of node_ids in the subgraph.

        - max_hops: how many layers out from the anchor to explore
        - max_nodes: upper bound to keep the subgraph small / efficient
        """
        visited: Set[str] = set()
        frontier: Set[str] = set(anchor_ids)

        hops = 0
        while frontier and hops <= max_hops and len(visited) < max_nodes:
            next_frontier: Set[str] = set()
            for n in frontier:
                if n in visited:
                    continue
                visited.add(n)
                for nbr in self.adj.neighbors.get(n, []):
                    if nbr not in visited and len(visited) + len(next_frontier) < max_nodes:
                        next_frontier.add(nbr)
            frontier = next_frontier
            hops += 1

        return visited

    def _docs_for_nodes(
        self,
        node_ids: Iterable[str],
    ) -> List[Document]:
        """
        Map node_ids -> Documents using doc_id_to_doc map.
        Nodes without a direct doc are skipped.
        """
        docs: List[Document] = []
        for nid in node_ids:
            doc = self.doc_id_to_doc.get(nid)
            if doc:
                docs.append(doc)
        return docs

    def get_subgraph_for_query(
        self,
        query: str,
        anchor_k: int = 4,
        max_hops: int = 2,
        max_nodes: int = 50,
    ) -> Tuple[List[Document], List[str]]:
        """
        Main GraphRAG retrieval method:

          1. Vector search to find anchor nodes (anchor_k docs).
          2. BFS over the usage graph up to `max_hops` to collect node_ids.
          3. Collect Documents corresponding to those nodes.

        Returns:
            (docs, node_ids)

            docs: list of Documents to be fed into the LLM as context
            node_ids: the set of node_ids in the subgraph (useful for debugging / explanations)
        """
        anchor_ids = self._get_anchor_node_ids(query, k=anchor_k)
        if not anchor_ids:
            # fallback: just normal RAG with top-k docs
            docs = self.vectorstore.similarity_search(query, k=anchor_k)
            node_ids: List[str] = [
                d.metadata.get("doc_id") for d in docs if d.metadata.get("doc_id")
            ]
            return docs, node_ids

        subgraph_node_ids = self._expand_subgraph(
            anchor_ids=anchor_ids,
            max_hops=max_hops,
            max_nodes=max_nodes,
        )
        docs = self._docs_for_nodes(subgraph_node_ids)
        return docs, list(subgraph_node_ids)

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def describe_node(self, node_id: str) -> str:
        """
        Pretty-print a node and its immediate neighbors (for debugging).
        """
        node = self.adj.nodes.get(node_id)
        if not node:
            return f"[graph] No node found for id: {node_id}"

        lines = [
            f"Node {node.id} ({node.type})",
            f"Properties: {node.properties}",
            "",
            "Neighbors:",
        ]
        for nbr in sorted(self.adj.neighbors.get(node_id, [])):
            n = self.adj.nodes.get(nbr)
            if n:
                lines.append(f"  - {n.id} ({n.type})")
            else:
                lines.append(f"  - {nbr} (unknown)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI test entrypoint
# ---------------------------------------------------------------------------

def _interactive_cli() -> None:
    """
    Lightweight CLI for manual testing:

      python -m src.Databricks_Usage.graph_retriever
    """
    retriever = GraphRAGRetriever.from_local_index()

    print("[graph_retriever] Interactive mode. Type a question, or 'exit' to quit.")
    while True:
        try:
            q = input("\nQuery> ").strip()
        except EOFError:
            break
        if not q or q.lower() in {"exit", "quit"}:
            break

        docs, node_ids = retriever.get_subgraph_for_query(
            query=q,
            anchor_k=4,
            max_hops=2,
            max_nodes=40,
        )
        print(f"[graph_retriever] Subgraph has {len(node_ids)} nodes, {len(docs)} docs.")
        print("Node IDs:", node_ids)

        # Show a preview of the first doc
        if docs:
            first = docs[0]
            print("\n--- First Doc Preview ---")
            print("doc_id:", first.metadata.get("doc_id"))
            print(first.page_content[:600], "...")


if __name__ == "__main__":
    _interactive_cli()
