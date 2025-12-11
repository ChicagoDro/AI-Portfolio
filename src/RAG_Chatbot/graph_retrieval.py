"""
Graph Retrieval Utilities for Chitown Custom Choppers (GraphRAG Upgrade)

This module loads the canonical knowledge graph produced by
`graph_kg_builder.py` from:

    data/graph/graph_output.json

and exposes helper functions for querying the graph. The graph is
represented as a directed property graph using NetworkX.

Typical usage in an application:

    from RAG_Chatbot.graph_retrieval import load_graph, get_direct_reports

    G = load_graph()
    reports = get_direct_reports(G, "person_rosa_martinez")

In a real enterprise environment, this layer would be analogous to
a thin query API over a graph database such as Neo4j or Neptune.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import networkx as nx

# --------------------------------------------------------------------------------------
# PATH CONFIG
# --------------------------------------------------------------------------------------

# This file lives at: src/RAG_Chatbot/graph_retrieval.py
# Project root: AI-Portfolio/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_DIR = PROJECT_ROOT / "data" / "graph"
GRAPH_OUTPUT_PATH = GRAPH_DIR / "graph_output.json"


# --------------------------------------------------------------------------------------
# LOADING THE GRAPH
# --------------------------------------------------------------------------------------


def load_graph() -> nx.DiGraph:
    """
    Load the knowledge graph from graph_output.json into a NetworkX DiGraph.

    Each node in the graph has:
        - node_id (string)
        - attributes from the original JSON node dict (type, name, etc.)

    Each edge in the graph has:
        - source
        - target
        - 'relation' attribute
        - optional 'metadata' attribute

    Returns:
        nx.DiGraph
    """
    if not GRAPH_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Graph file not found at {GRAPH_OUTPUT_PATH}. "
            f"Make sure to run graph_kg_builder.py first."
        )

    with GRAPH_OUTPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    nodes_data: List[Dict[str, Any]] = data.get("nodes", [])
    edges_data: List[Dict[str, Any]] = data.get("edges", [])

    G = nx.DiGraph()

    # Add nodes with attributes
    for node in nodes_data:
        node_id = node["id"]
        attrs = {k: v for k, v in node.items() if k != "id"}
        G.add_node(node_id, **attrs)

    # Add edges with relation + optional metadata
    for edge in edges_data:
        src = edge["source"]
        tgt = edge["target"]
        rel = edge.get("relation", "RELATED_TO")
        metadata = edge.get("metadata", {})
        G.add_edge(src, tgt, relation=rel, metadata=metadata)

    return G


# --------------------------------------------------------------------------------------
# BASIC LOOKUPS
# --------------------------------------------------------------------------------------


def get_node(G: nx.DiGraph, node_id: str) -> Optional[Dict[str, Any]]:
    """Return node attributes for a given node_id, or None if missing."""
    if node_id not in G.nodes:
        return None
    attrs = G.nodes[node_id].copy()
    attrs["id"] = node_id
    return attrs


def find_person_by_name(G: nx.DiGraph, name: str) -> Optional[str]:
    """
    Find a person node by exact name (case-insensitive).
    Returns the node_id or None if not found.
    """
    name_lower = name.lower()
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") == "Person":
            if str(attrs.get("name", "")).lower() == name_lower:
                return node_id
    return None


def list_people(G: nx.DiGraph) -> List[Dict[str, Any]]:
    """Return a list of all Person nodes (with id and attributes)."""
    result = []
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") == "Person":
            item = attrs.copy()
            item["id"] = node_id
            result.append(item)
    return result


def list_departments(G: nx.DiGraph) -> List[Dict[str, Any]]:
    """Return a list of all Department nodes."""
    result = []
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") == "Department":
            item = attrs.copy()
            item["id"] = node_id
            result.append(item)
    return result


# --------------------------------------------------------------------------------------
# ORG STRUCTURE QUERIES
# --------------------------------------------------------------------------------------


def get_direct_reports(G: nx.DiGraph, manager_id: str) -> List[Dict[str, Any]]:
    """
    Return all Person nodes that have a REPORTS_TO edge pointing to the manager.

    In our graph:
        (employee) -[REPORTS_TO]-> (manager)
    """
    reports: List[Dict[str, Any]] = []
    for src, tgt, attrs in G.in_edges(manager_id, data=True):
        if attrs.get("relation") == "REPORTS_TO":
            node_attrs = G.nodes[src].copy()
            node_attrs["id"] = src
            reports.append(node_attrs)
    return reports


def get_manager(G: nx.DiGraph, person_id: str) -> Optional[Dict[str, Any]]:
    """
    Return the manager (Person node) for the given person_id, if any.

    In our graph:
        (employee) -[REPORTS_TO]-> (manager)
    """
    for src, tgt, attrs in G.out_edges(person_id, data=True):
        if attrs.get("relation") == "REPORTS_TO":
            manager_id = tgt
            result = G.nodes[manager_id].copy()
            result["id"] = manager_id
            return result
    return None


def get_department_for_person(G: nx.DiGraph, person_id: str) -> Optional[Dict[str, Any]]:
    """
    Return the Department node a person belongs to (via BELONGS_TO_DEPARTMENT edge).
    """
    for src, tgt, attrs in G.out_edges(person_id, data=True):
        if attrs.get("relation") == "BELONGS_TO_DEPARTMENT":
            dept_id = tgt
            result = G.nodes[dept_id].copy()
            result["id"] = dept_id
            return result
    return None


def get_department_team(G: nx.DiGraph, department_id: str) -> List[Dict[str, Any]]:
    """
    Return all Person nodes that belong to the given department.
    """
    team: List[Dict[str, Any]] = []
    for src, tgt, attrs in G.in_edges(department_id, data=True):
        if attrs.get("relation") == "BELONGS_TO_DEPARTMENT":
            node_attrs = G.nodes[src].copy()
            node_attrs["id"] = src
            team.append(node_attrs)
    return team


# --------------------------------------------------------------------------------------
# SALES QUERIES
# --------------------------------------------------------------------------------------


def get_sales_metrics_for_person(
    G: nx.DiGraph, person_id: str
) -> List[Dict[str, Any]]:
    """
    Return all SalesMetric nodes for a given person, sorted by (year, month).
    """
    metrics: List[Dict[str, Any]] = []
    for src, tgt, attrs in G.out_edges(person_id, data=True):
        if attrs.get("relation") == "HAS_SALES_METRIC":
            metric_attrs = G.nodes[tgt].copy()
            metric_attrs["id"] = tgt
            metrics.append(metric_attrs)

    # Sort by (year, month)
    def sort_key(m: Dict[str, Any]) -> Tuple[int, int]:
        year = int(m.get("year", 0))
        month = int(m.get("month", 0))
        return (year, month)

    metrics.sort(key=sort_key)
    return metrics


def get_total_sales_for_person(
    G: nx.DiGraph, person_id: str
) -> float:
    """
    Sum all sales metrics for the given person.
    """
    metrics = get_sales_metrics_for_person(G, person_id)
    return float(sum(m.get("amount_usd", 0.0) for m in metrics))


def get_total_sales_for_department(
    G: nx.DiGraph, department_id: str
) -> float:
    """
    Sum all sales metrics for all people belonging to a department.
    """
    team = get_department_team(G, department_id)
    total = 0.0
    for person in team:
        total += get_total_sales_for_person(G, person["id"])
    return float(total)


def get_q3_2024_total_sales(G: nx.DiGraph) -> float:
    """
    Compute the total Q3 2024 sales across all SalesMetric nodes in the graph.

    This is slightly more generic than summing per person and matches how
    you'd aggregate in a graph database.
    """
    total = 0.0
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") == "SalesMetric":
            year = attrs.get("year")
            if year == 2024:
                # Our data only covers Q3, but this filter is easy to extend.
                total += float(attrs.get("amount_usd", 0.0))
    return float(total)


# --------------------------------------------------------------------------------------
# HIGH-LEVEL SUMMARIES (USEFUL FOR LLM PROMPTS)
# --------------------------------------------------------------------------------------


def format_org_summary(G: nx.DiGraph) -> str:
    """
    Produce a human-readable summary of the org structure suitable for
    inclusion in an LLM context window.
    """
    lines: List[str] = []

    # CEO
    ceos = [
        (nid, attrs)
        for nid, attrs in G.nodes(data=True)
        if attrs.get("type") == "Person" and attrs.get("role", "").upper() == "CEO"
    ]
    for nid, attrs in ceos:
        lines.append(f"CEO: {attrs.get('name')} (id={nid})")

        # Direct reports to CEO (directors)
        reports = get_direct_reports(G, nid)
        for r in reports:
            lines.append(f"  Director: {r['name']} — {r.get('role')} (id={r['id']})")

    # Departments and their members
    lines.append("\nDepartments:")
    for dept in list_departments(G):
        lines.append(f"- {dept['name']} (id={dept['id']})")
        team = get_department_team(G, dept["id"])
        for member in team:
            lines.append(f"    * {member['name']} — {member.get('role')} (id={member['id']})")

    return "\n".join(lines)


def format_person_sales_summary(G: nx.DiGraph, person_id: str) -> str:
    """
    Produce a text summary of an employee's Q3 2024 sales performance.
    """
    person = get_node(G, person_id)
    if not person:
        return f"No data found for person_id={person_id}"

    metrics = get_sales_metrics_for_person(G, person_id)
    if not metrics:
        return f"No sales metrics found for {person['name']}."

    lines = [f"Sales summary for {person['name']} ({person_id}):"]
    total = 0.0
    for m in metrics:
        year = m.get("year")
        month = m.get("month")
        amt = float(m.get("amount_usd", 0.0))
        total += amt
        lines.append(f"  - {year}-{month}: ${amt:,.2f}")
    lines.append(f"Total Q3 2024 sales: ${total:,.2f}")

    return "\n".join(lines)
