# src/Databricks_Usage/graph_model.py

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .config import USAGE_DB_PATH


# ---------------------------------------------------------------------------
# Core graph data structures
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    id: str            # e.g. "job::J-FIN-DLY"
    type: str          # e.g. "job", "user", "company_ou"
    properties: Dict[str, Any]


@dataclass
class GraphEdge:
    src: str           # node id
    dst: str           # node id
    type: str          # e.g. "OWNS", "RUN_OF", "EXECUTES_ON"
    properties: Dict[str, Any]


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(USAGE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def _add_node(nodes: Dict[str, GraphNode], node_id: str, node_type: str, props: Dict[str, Any]) -> None:
    if node_id not in nodes:
        nodes[node_id] = GraphNode(id=node_id, type=node_type, properties=props)


def _add_edge(edges: List[GraphEdge], src: str, dst: str, edge_type: str, props: Dict[str, Any] | None = None) -> None:
    edges.append(GraphEdge(src=src, dst=dst, type=edge_type, properties=props or {}))


# ---------------------------------------------------------------------------
# Build nodes & edges from each table
# ---------------------------------------------------------------------------

def _build_company_ou(nodes: Dict[str, GraphNode]) -> None:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT company_ou_id, name, cost_center_code, description
            FROM company_ou
        """).fetchall()

    for r in rows:
        nid = f"ou::{r['company_ou_id']}"
        _add_node(
            nodes,
            nid,
            "company_ou",
            {
                "company_ou_id": r["company_ou_id"],
                "name": r["name"],
                "cost_center_code": r["cost_center_code"],
                "description": r["description"],
            },
        )


def _build_users(nodes: Dict[str, GraphNode], edges: List[GraphEdge]) -> None:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT user_id, name, company_ou_id, department
            FROM users_lookup
        """).fetchall()

    for r in rows:
        user_id = f"user::{r['user_id']}"
        ou_id = f"ou::{r['company_ou_id']}"

        _add_node(
            nodes,
            user_id,
            "user",
            {
                "user_id": r["user_id"],
                "name": r["name"],
                "company_ou_id": r["company_ou_id"],
                "department": r["department"],
            },
        )
        # USER -> OU (BELONGS_TO)
        _add_edge(edges, user_id, ou_id, "BELONGS_TO")


def _build_jobs(nodes: Dict[str, GraphNode], edges: List[GraphEdge]) -> None:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT
                job_id,
                job_name,
                description,
                tags,
                company_ou_id
            FROM jobs
        """).fetchall()

    for r in rows:
        job_id = f"job::{r['job_id']}"
        ou_id = f"ou::{r['company_ou_id']}"

        _add_node(
            nodes,
            job_id,
            "job",
            {
                "job_id": r["job_id"],
                "job_name": r["job_name"],
                "description": r["description"],
                "tags": r["tags"],
                "company_ou_id": r["company_ou_id"],
            },
        )
        # JOB -> OU (OWNED_BY)
        _add_edge(edges, job_id, ou_id, "OWNED_BY")


def _build_compute_resources(nodes: Dict[str, GraphNode], edges: List[GraphEdge]) -> None:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT
                compute_id,
                compute_name,
                compute_type,
                company_ou_id
            FROM non_job_compute
        """).fetchall()

    for r in rows:
        cid = f"compute::{r['compute_id']}"
        ou_id = f"ou::{r['company_ou_id']}"

        _add_node(
            nodes,
            cid,
            "compute_resource",
            {
                "compute_id": r["compute_id"],
                "compute_name": r["compute_name"],
                "compute_type": r["compute_type"],
                "company_ou_id": r["company_ou_id"],
            },
        )
        # COMPUTE_RESOURCE -> OU (OWNED_BY)
        _add_edge(edges, cid, ou_id, "OWNED_BY")


def _build_job_runs(nodes: Dict[str, GraphNode], edges: List[GraphEdge]) -> None:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT
                job_run_id,
                job_id,
                start_time,
                end_time,
                duration_ms,
                run_status,
                error_summary,
                worker_instance_type,
                fixed_nodes,
                min_nodes,
                max_nodes,
                is_autoscaling_enabled
            FROM job_runs
        """).fetchall()

    for r in rows:
        rid = f"jobrun::{r['job_run_id']}"
        job_id = f"job::{r['job_id']}"

        _add_node(
            nodes,
            rid,
            "job_run",
            {
                "job_run_id": r["job_run_id"],
                "job_id": r["job_id"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "duration_ms": r["duration_ms"],
                "run_status": r["run_status"],
                "error_summary": r["error_summary"],
                "worker_instance_type": r["worker_instance_type"],
                "fixed_nodes": r["fixed_nodes"],
                "min_nodes": r["min_nodes"],
                "max_nodes": r["max_nodes"],
                "is_autoscaling_enabled": r["is_autoscaling_enabled"],
            },
        )
        # JOB_RUN -> JOB (RUN_OF)
        _add_edge(edges, rid, job_id, "RUN_OF")


def _build_compute_usage(
    nodes: Dict[str, GraphNode],
    edges: List[GraphEdge],
) -> None:
    """
    compute_usage.parent_id can be either:
      - job_runs.job_run_id  (JOB_RUN)
      - non_job_compute.compute_id (APC_CLUSTER, SQL_WAREHOUSE)
    """
    with _get_conn() as conn:
        # Preload job_run_ids and compute_ids to classify parents
        job_run_ids = {
            row["job_run_id"] for row in conn.execute("SELECT job_run_id FROM job_runs").fetchall()
        }
        compute_ids = {
            row["compute_id"] for row in conn.execute("SELECT compute_id FROM non_job_compute").fetchall()
        }

        rows = conn.execute("""
            SELECT
                compute_usage_id,
                parent_id,
                compute_type,
                sku,
                dbus_consumed,
                instance_id,
                instance_type,
                cost_usd,
                avg_cpu_utilization,
                max_memory_used_gb,
                disk_io_wait_time_ms,
                cloud_market_available,
                usage_date
            FROM compute_usage
        """).fetchall()

    for r in rows:
        uid = f"usage::{r['compute_usage_id']}"
        parent_key = r["parent_id"]

        _add_node(
            nodes,
            uid,
            "compute_usage",
            {
                "compute_usage_id": r["compute_usage_id"],
                "parent_id": r["parent_id"],
                "compute_type": r["compute_type"],
                "sku": r["sku"],
                "dbus_consumed": r["dbus_consumed"],
                "instance_id": r["instance_id"],
                "instance_type": r["instance_type"],
                "cost_usd": r["cost_usd"],
                "avg_cpu_utilization": r["avg_cpu_utilization"],
                "max_memory_used_gb": r["max_memory_used_gb"],
                "disk_io_wait_time_ms": r["disk_io_wait_time_ms"],
                "cloud_market_available": r["cloud_market_available"],
                "usage_date": r["usage_date"],
            },
        )

        # Edge: USAGE -> (JOB_RUN | COMPUTE_RESOURCE)
        if parent_key in job_run_ids:
            parent_node_id = f"jobrun::{parent_key}"
            _add_edge(edges, uid, parent_node_id, "USAGE_OF_JOB_RUN")
        elif parent_key in compute_ids:
            parent_node_id = f"compute::{parent_key}"
            _add_edge(edges, uid, parent_node_id, "USAGE_OF_COMPUTE")
        else:
            # Unknown; keep parent_id just as a property
            pass


def _build_events(nodes: Dict[str, GraphNode], edges: List[GraphEdge]) -> None:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT
                event_id,
                compute_usage_id,
                timestamp,
                event_type,
                user_id,
                details,
                eviction_id
            FROM events
        """).fetchall()

    for r in rows:
        eid = f"event::{r['event_id']}"
        usage_id = f"usage::{r['compute_usage_id']}"

        _add_node(
            nodes,
            eid,
            "event",
            {
                "event_id": r["event_id"],
                "compute_usage_id": r["compute_usage_id"],
                "timestamp": r["timestamp"],
                "event_type": r["event_type"],
                "user_id": r["user_id"],
                "details": r["details"],
                "eviction_id": r["eviction_id"],
            },
        )

        # EVENT -> USAGE (ON_USAGE)
        _add_edge(edges, eid, usage_id, "ON_USAGE")

        # EVENT -> USER (TRIGGERED_BY), if present
        if r["user_id"]:
            user_node_id = f"user::{r['user_id']}"
            _add_edge(edges, eid, user_node_id, "TRIGGERED_BY")

        # EVENT -> EVICTION (ASSOCIATED_WITH)
        if r["eviction_id"]:
            evict_node_id = f"evict::{r['eviction_id']}"
            _add_edge(edges, eid, evict_node_id, "ASSOCIATED_EVICTION")


def _build_evictions(nodes: Dict[str, GraphNode]) -> None:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT
                eviction_id,
                instance_id,
                eviction_time,
                cloud_provider_message,
                eviction_reason_code,
                instance_reclaim_rate,
                eviction_policy_used,
                replacement_on_demand
            FROM eviction_details
        """).fetchall()

    for r in rows:
        eid = f"evict::{r['eviction_id']}"
        _add_node(
            nodes,
            eid,
            "eviction",
            {
                "eviction_id": r["eviction_id"],
                "instance_id": r["instance_id"],
                "eviction_time": r["eviction_time"],
                "cloud_provider_message": r["cloud_provider_message"],
                "eviction_reason_code": r["eviction_reason_code"],
                "instance_reclaim_rate": r["instance_reclaim_rate"],
                "eviction_policy_used": r["eviction_policy_used"],
                "replacement_on_demand": r["replacement_on_demand"],
            },
        )


def _build_queries(nodes: Dict[str, GraphNode], edges: List[GraphEdge]) -> None:
    with _get_conn() as conn:
        # For classifying parent_id
        job_run_ids = {
            row["job_run_id"] for row in conn.execute("SELECT job_run_id FROM job_runs").fetchall()
        }
        compute_ids = {
            row["compute_id"] for row in conn.execute("SELECT compute_id FROM non_job_compute").fetchall()
        }

        rows = conn.execute("""
            SELECT
                q.query_id,
                q.parent_id,
                q.user_id,
                q.start_time,
                q.duration_ms,
                q.warehouse_sku,
                q.sql_text,
                q.error_message,
                u.company_ou_id,
                u.name AS user_name,
                u.department AS department
            FROM sql_query_history q
            LEFT JOIN users_lookup u ON q.user_id = u.user_id
        """).fetchall()

    for r in rows:
        qid = f"query::{r['query_id']}"
        parent_key = r["parent_id"]

        _add_node(
            nodes,
            qid,
            "query",
            {
                "query_id": r["query_id"],
                "parent_id": r["parent_id"],
                "user_id": r["user_id"],
                "start_time": r["start_time"],
                "duration_ms": r["duration_ms"],
                "warehouse_sku": r["warehouse_sku"],
                "sql_text": r["sql_text"],
                "error_message": r["error_message"],
                "company_ou_id": r["company_ou_id"],
                "user_name": r["user_name"],
                "department": r["department"],
            },
        )

        # QUERY -> USER
        if r["user_id"]:
            user_node_id = f"user::{r['user_id']}"
            _add_edge(edges, qid, user_node_id, "EXECUTED_BY")

        # QUERY -> (JOB_RUN | COMPUTE_RESOURCE)
        if parent_key in job_run_ids:
            parent_node_id = f"jobrun::{parent_key}"
            _add_edge(edges, qid, parent_node_id, "EXECUTES_ON_JOB_RUN")
        elif parent_key in compute_ids:
            parent_node_id = f"compute::{parent_key}"
            _add_edge(edges, qid, parent_node_id, "EXECUTES_ON_COMPUTE")


# ---------------------------------------------------------------------------
# PUBLIC ENTRYPOINT
# ---------------------------------------------------------------------------

def build_usage_graph() -> Tuple[Dict[str, GraphNode], List[GraphEdge]]:
    """
    Build an in-memory graph (nodes + edges) from the SQLite Databricks usage DB.

    Returns:
        nodes: dict[node_id, GraphNode]
        edges: list[GraphEdge]
    """
    nodes: Dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []

    _build_company_ou(nodes)
    _build_users(nodes, edges)
    _build_jobs(nodes, edges)
    _build_compute_resources(nodes, edges)
    _build_job_runs(nodes, edges)
    _build_compute_usage(nodes, edges)
    _build_evictions(nodes)
    _build_events(nodes, edges)
    _build_queries(nodes, edges)

    return nodes, edges


# Optional: quick CLI test
if __name__ == "__main__":
    nodes, edges = build_usage_graph()
    print(f"Graph built: {len(nodes)} nodes, {len(edges)} edges")
    by_type: Dict[str, int] = {}
    for n in nodes.values():
        by_type[n.type] = by_type.get(n.type, 0) + 1
    print("Node counts by type:", by_type)
