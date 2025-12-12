"""
Graph KG Builder for Chitown Custom Choppers (GraphRAG Upgrade, Data-Driven)

This script builds a knowledge graph from structured JSON data files:

    data/graph/departments.json
    data/graph/employees.json
    data/graph/employee_mom_sales_q3_2024.json

It produces a canonical graph representation:

    data/graph/graph_output.json

In a real enterprise setting, these JSON files would typically be
replaced by tables in a warehouse (e.g., Snowflake) or API calls,
but the transformation pattern (data -> nodes/edges -> graph store)
remains the same.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

# --------------------------------------------------------------------------------------
# PATH CONFIG
# --------------------------------------------------------------------------------------

# This file lives at: src/RAG_Chatbot/graph_kg_builder.py
# Project root: AI-Portfolio/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_DIR = PROJECT_ROOT / "data" / "Chitown_Custom_Choppers"/ "graph"
DEPARTMENTS_PATH = GRAPH_DIR / "departments.json"
EMPLOYEES_PATH = GRAPH_DIR / "employees.json"
SALES_PATH = GRAPH_DIR / "employee_mom_sales_q3_2024.json"
GRAPH_OUTPUT_PATH = GRAPH_DIR / "graph_output.json"


# --------------------------------------------------------------------------------------
# GRAPH MODEL HELPERS
# --------------------------------------------------------------------------------------


def make_department_node(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": record["id"],
        "type": "Department",
        "name": record["name"],
    }


def make_person_node(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": record["id"],
        "type": "Person",
        "name": record["name"],
        "role": record["role"],
        "department_id": record.get("department_id"),
    }


def make_sales_metric_node(metric_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": metric_id,
        "type": "SalesMetric",
        "person_id": record["person_id"],
        "month": record["month"],
        "year": record["year"],
        "amount_usd": float(record["amount_usd"]),
    }


def make_edge(
    source: str,
    target: str,
    relation: str,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    edge = {
        "source": source,
        "target": target,
        "relation": relation,
    }
    if metadata:
        edge["metadata"] = metadata
    return edge


# --------------------------------------------------------------------------------------
# BUILD GRAPH FROM DATA FILES
# --------------------------------------------------------------------------------------


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Expected data file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got: {type(data)}")
    return data


def build_graph() -> Dict[str, List[Dict[str, Any]]]:
    """
    Construct the graph nodes and edges from:
      - departments.json
      - employees.json
      - employee_mom_sales_q3_2024.json
    """

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # ---------------- Load data files ----------------
    departments_raw = load_json_array(DEPARTMENTS_PATH)
    employees_raw = load_json_array(EMPLOYEES_PATH)
    sales_raw = load_json_array(SALES_PATH)

    # ---------------- Departments ----------------
    dept_nodes = [make_department_node(d) for d in departments_raw]
    nodes.extend(dept_nodes)

    dept_ids = {d["id"] for d in departments_raw}

    # ---------------- People ----------------
    person_nodes = [make_person_node(e) for e in employees_raw]
    nodes.extend(person_nodes)

    # Index employees by id for edge construction
    employees_by_id: Dict[str, Dict[str, Any]] = {e["id"]: e for e in employees_raw}

    # ---------------- Reporting Lines & Department Membership ----------------
    for emp in employees_raw:
        emp_id = emp["id"]
        manager_id = emp.get("manager_id")
        dept_id = emp.get("department_id")

        # REPORTS_TO edge (employee -> manager)
        if manager_id:
            if manager_id not in employees_by_id:
                # In a real system you'd log this somewhere central
                print(f"[WARN] manager_id '{manager_id}' not found for employee '{emp_id}'")
            else:
                edges.append(
                    make_edge(
                        source=emp_id,
                        target=manager_id,
                        relation="REPORTS_TO",
                    )
                )

        # BELONGS_TO_DEPARTMENT edge (employee -> department)
        if dept_id:
            if dept_id not in dept_ids:
                print(f"[WARN] department_id '{dept_id}' not found for employee '{emp_id}'")
            else:
                edges.append(
                    make_edge(
                        source=emp_id,
                        target=dept_id,
                        relation="BELONGS_TO_DEPARTMENT",
                    )
                )

    # ---------------- Q3 2024 Sales Metrics ----------------
    sales_nodes: List[Dict[str, Any]] = []

    for s in sales_raw:
        person_id = s["person_id"]
        year = s["year"]
        month = s["month"]
        amount = s["amount_usd"]

        metric_id = f"sales_{person_id}_{year}-{month}"

        metric_node = make_sales_metric_node(
            metric_id,
            {
                "person_id": person_id,
                "year": year,
                "month": month,
                "amount_usd": amount,
            },
        )
        sales_nodes.append(metric_node)

        edges.append(
            make_edge(
                source=person_id,
                target=metric_id,
                relation="HAS_SALES_METRIC",
                metadata={"period": f"{year}-{month}"},
            )
        )

    nodes.extend(sales_nodes)

    return {"nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------


def main() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    graph_data = build_graph()

    with GRAPH_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

    print(f"Graph written to: {GRAPH_OUTPUT_PATH}")
    print(f"Nodes: {len(graph_data['nodes'])}, Edges: {len(graph_data['edges'])}")


if __name__ == "__main__":
    main()
