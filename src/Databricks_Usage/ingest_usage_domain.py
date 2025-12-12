# src/Databricks_Usage/ingest_usage_domain.py

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List

from .config import USAGE_DB_PATH


# ---------------------------------------------------------------------------
# Core document structure for ingestion
# ---------------------------------------------------------------------------

@dataclass
class RagDoc:
    doc_id: str
    text: str
    metadata: Dict[str, Any]


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(USAGE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# ORG UNIT DOCUMENTS
# ---------------------------------------------------------------------------

def _fetch_company_ou_docs() -> List[RagDoc]:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT company_ou_id, name, cost_center_code, description
            FROM company_ou
        """).fetchall()

    docs = []
    for r in rows:
        docs.append(RagDoc(
            doc_id=f"ou::{r['company_ou_id']}",
            text=(
                f"Org Unit: {r['name']} (ID: {r['company_ou_id']})\n"
                f"Cost Center: {r['cost_center_code']}\n\n"
                f"Description:\n{r['description']}"
            ),
            metadata={
                "type": "company_ou",
                "company_ou_id": r["company_ou_id"],
                "ou_name": r["name"],
                "cost_center_code": r["cost_center_code"],
            }
        ))
    return docs


# ---------------------------------------------------------------------------
# USER DOCUMENTS
# ---------------------------------------------------------------------------

def _fetch_user_docs() -> List[RagDoc]:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT user_id, name, department, company_ou_id
            FROM users_lookup
        """).fetchall()

    docs = []
    for r in rows:
        docs.append(RagDoc(
            doc_id=f"user::{r['user_id']}",
            text=(
                f"User: {r['name']} (ID: {r['user_id']})\n"
                f"Department: {r['department']}\n"
                f"Org Unit: {r['company_ou_id']}"
            ),
            metadata={
                "type": "user",
                "user_id": r["user_id"],
                "name": r["name"],
                "department": r["department"],
                "company_ou_id": r["company_ou_id"],
            }
        ))
    return docs


# ---------------------------------------------------------------------------
# JOB DOCUMENTS
# ---------------------------------------------------------------------------

def _fetch_job_docs() -> List[RagDoc]:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT
                j.job_id,
                j.job_name,
                j.description,
                j.tags,
                j.company_ou_id,
                ou.name AS ou_name,
                ou.cost_center_code
            FROM jobs j
            LEFT JOIN company_ou ou ON j.company_ou_id = ou.company_ou_id
        """).fetchall()

    docs = []
    for r in rows:
        docs.append(RagDoc(
            doc_id=f"job::{r['job_id']}",
            text=(
                f"Job: {r['job_name']} (ID: {r['job_id']})\n"
                f"Org Unit: {r['ou_name']} ({r['company_ou_id']})\n"
                f"Cost Center: {r['cost_center_code']}\n\n"
                f"Description:\n{r['description']}\n\n"
                f"Tags: {r['tags']}"
            ),
            metadata={
                "type": "job",
                "job_id": r["job_id"],
                "job_name": r["job_name"],
                "company_ou_id": r["company_ou_id"],
                "ou_name": r["ou_name"],
                "cost_center_code": r["cost_center_code"],
                "tags_json": r["tags"],
            }
        ))
    return docs


# ---------------------------------------------------------------------------
# COMPUTE RESOURCE DOCUMENTS (APC + SQL WAREHOUSE)
# ---------------------------------------------------------------------------

def _fetch_compute_resource_docs() -> List[RagDoc]:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT
                c.compute_id,
                c.compute_name,
                c.compute_type,
                c.company_ou_id,
                ou.name AS ou_name,
                ou.cost_center_code
            FROM non_job_compute c
            LEFT JOIN company_ou ou ON c.company_ou_id = ou.company_ou_id
        """).fetchall()

    docs = []
    for r in rows:
        docs.append(RagDoc(
            doc_id=f"compute::{r['compute_id']}",
            text=(
                f"Compute Resource: {r['compute_name']} (ID: {r['compute_id']})\n"
                f"Type: {r['compute_type']}\n"
                f"Owned by OU: {r['ou_name']} ({r['company_ou_id']})\n"
                f"Cost Center: {r['cost_center_code']}"
            ),
            metadata={
                "type": "compute_resource",
                "compute_id": r["compute_id"],
                "compute_name": r["compute_name"],
                "compute_type": r["compute_type"],
                "company_ou_id": r["company_ou_id"],
                "ou_name": r["ou_name"],
                "cost_center_code": r["cost_center_code"],
            }
        ))
    return docs


# ---------------------------------------------------------------------------
# JOB RUN DOCUMENTS (SUMMARIES)
# ---------------------------------------------------------------------------

def _fetch_job_run_docs(limit=200) -> List[RagDoc]:
    with _get_conn() as conn:
        rows = conn.execute(f"""
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
            ORDER BY start_time DESC
            LIMIT {limit}
        """).fetchall()

    docs = []
    for r in rows:
        text = (
            f"Job Run: {r['job_run_id']}\n"
            f"Job ID: {r['job_id']}\n"
            f"Status: {r['run_status']}\n"
            f"Duration: {r['duration_ms']} ms\n"
            f"Cluster: worker type {r['worker_instance_type']}, "
            f"{r['min_nodes']}-{r['max_nodes']} nodes (fixed={r['fixed_nodes']})\n"
            f"Autoscaling: {r['is_autoscaling_enabled']}\n"
        )
        if r["error_summary"]:
            text += f"\nError:\n{r['error_summary']}"

        docs.append(RagDoc(
            doc_id=f"jobrun::{r['job_run_id']}",
            text=text,
            metadata={
                "type": "job_run",
                "job_run_id": r["job_run_id"],
                "job_id": r["job_id"],
                "run_status": r["run_status"],
                "duration_ms": r["duration_ms"],
            }
        ))
    return docs


# ---------------------------------------------------------------------------
# COMPUTE USAGE DOCUMENTS (SUMMARIES)
# ---------------------------------------------------------------------------

def _fetch_compute_usage_docs(limit=300) -> List[RagDoc]:
    """
    One document per compute_usage row.

    DDL:

        compute_usage_id, parent_id, compute_type, sku,
        dbus_consumed, instance_id, instance_type,
        cost_usd, avg_cpu_utilization, max_memory_used_gb,
        disk_io_wait_time_ms, cloud_market_available, usage_date
    """
    with _get_conn() as conn:
        rows = conn.execute(f"""
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
            ORDER BY usage_date DESC
            LIMIT {limit}
        """).fetchall()

    docs: List[RagDoc] = []
    for r in rows:
        cpu = r["avg_cpu_utilization"]
        cpu_str = f"{cpu:.2%}" if cpu is not None else "n/a"

        text = (
            f"Compute Usage ID: {r['compute_usage_id']}\n"
            f"Parent ID: {r['parent_id']} ({r['compute_type']})\n"
            f"SKU: {r['sku']}\n"
            f"DBUs Consumed: {r['dbus_consumed']}\n"
            f"Cost (USD): {r['cost_usd']:.2f}\n"
            f"Instance: {r['instance_id']} ({r['instance_type']})\n"
            f"Avg CPU Utilization: {cpu_str}\n"
            f"Max Memory Used (GB): {r['max_memory_used_gb']}\n"
            f"Disk IO Wait (ms): {r['disk_io_wait_time_ms']}\n"
            f"Cloud Market Available: {r['cloud_market_available']}\n"
            f"Usage Date: {r['usage_date']}"
        )

        docs.append(RagDoc(
            doc_id=f"usage::{r['compute_usage_id']}",
            text=text,
            metadata={
                "type": "compute_usage",
                "compute_usage_id": r["compute_usage_id"],
                "parent_id": r["parent_id"],
                "compute_type": r["compute_type"],
                "sku": r["sku"],
                "usage_date": r["usage_date"],
                "cost_usd": r["cost_usd"],
                "dbus_consumed": r["dbus_consumed"],
            }
        ))
    return docs


# ---------------------------------------------------------------------------
# EVENT DOCUMENTS
# ---------------------------------------------------------------------------

def _fetch_event_docs(limit=300) -> List[RagDoc]:
    """
    One document per event, tied to compute_usage and (optionally) eviction.
    """
    with _get_conn() as conn:
        rows = conn.execute(f"""
            SELECT
                event_id,
                compute_usage_id,
                timestamp,
                event_type,
                user_id,
                details,
                eviction_id
            FROM events
            ORDER BY timestamp DESC
            LIMIT {limit}
        """).fetchall()

    docs: List[RagDoc] = []
    for r in rows:
        text = (
            f"Event: {r['event_id']}\n"
            f"Type: {r['event_type']}\n"
            f"Time: {r['timestamp']}\n"
            f"Compute Usage ID: {r['compute_usage_id']}\n"
            f"User ID: {r['user_id']}\n"
        )
        if r["eviction_id"]:
            text += f"Eviction ID: {r['eviction_id']}\n"

        text += f"\nDetails:\n{r['details']}"

        docs.append(RagDoc(
            doc_id=f"event::{r['event_id']}",
            text=text,
            metadata={
                "type": "event",
                "event_id": r["event_id"],
                "compute_usage_id": r["compute_usage_id"],
                "event_type": r["event_type"],
                "eviction_id": r["eviction_id"],
            }
        ))
    return docs


# ---------------------------------------------------------------------------
# EVICTION DOCUMENTS
# ---------------------------------------------------------------------------

def _fetch_eviction_docs(limit=200) -> List[RagDoc]:
    """
    One document per eviction_details row.
    """
    with _get_conn() as conn:
        rows = conn.execute(f"""
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
            ORDER BY eviction_time DESC
            LIMIT {limit}
        """).fetchall()

    docs: List[RagDoc] = []
    for r in rows:
        text = (
            f"Eviction ID: {r['eviction_id']}\n"
            f"Instance ID: {r['instance_id']}\n"
            f"Eviction Time: {r['eviction_time']}\n"
            f"Reason Code: {r['eviction_reason_code']}\n"
            f"Eviction Policy: {r['eviction_policy_used']}\n"
            f"Instance Reclaim Rate: {r['instance_reclaim_rate']}\n"
            f"Replacement On-Demand: {r['replacement_on_demand']}\n\n"
            f"Cloud Provider Message:\n{r['cloud_provider_message']}"
        )

        docs.append(RagDoc(
            doc_id=f"evict::{r['eviction_id']}",
            text=text,
            metadata={
                "type": "eviction",
                "eviction_id": r["eviction_id"],
                "instance_id": r["instance_id"],
                "eviction_reason_code": r["eviction_reason_code"],
                "eviction_policy_used": r["eviction_policy_used"],
            }
        ))
    return docs



# ---------------------------------------------------------------------------
# QUERY DOCUMENTS (same as before)
# ---------------------------------------------------------------------------

def _fetch_query_docs(limit=300) -> List[RagDoc]:
    with _get_conn() as conn:
        rows = conn.execute(f"""
            SELECT
                q.query_id,
                q.parent_id,
                q.user_id,
                q.start_time,
                q.duration_ms,
                q.warehouse_sku,
                q.sql_text,
                q.error_message,
                u.name AS user_name,
                u.department,
                u.company_ou_id
            FROM sql_query_history q
            LEFT JOIN users_lookup u ON q.user_id = u.user_id
            ORDER BY q.start_time DESC
            LIMIT {limit}
        """).fetchall()

    docs = []
    for r in rows:
        lines = [
            f"Query ID: {r['query_id']}",
            f"User: {r['user_name']} ({r['user_id']})",
            f"Department: {r['department']}",
            f"Org Unit: {r['company_ou_id']}",
            f"Parent Compute: {r['parent_id']} (SKU: {r['warehouse_sku']})",
            f"Start: {r['start_time']}",
            f"Duration: {r['duration_ms']} ms",
            "",
            "SQL Text:",
            r["sql_text"] or ""
        ]
        if r["error_message"]:
            lines += ["", "Error:", r["error_message"]]

        docs.append(RagDoc(
            doc_id=f"query::{r['query_id']}",
            text="\n".join(lines),
            metadata={
                "type": "query",
                "query_id": r["query_id"],
                "user_id": r["user_id"],
                "company_ou_id": r["company_ou_id"],
                "parent_id": r["parent_id"],
                "duration_ms": r["duration_ms"],
            }
        ))
    return docs


# ---------------------------------------------------------------------------
# PUBLIC ENTRYPOINT — ALL DOCS
# ---------------------------------------------------------------------------

def build_usage_rag_docs() -> List[RagDoc]:
    docs = []
    docs.extend(_fetch_company_ou_docs())
    docs.extend(_fetch_user_docs())
    docs.extend(_fetch_job_docs())
    docs.extend(_fetch_compute_resource_docs())
    docs.extend(_fetch_job_run_docs())
    docs.extend(_fetch_compute_usage_docs())
    docs.extend(_fetch_event_docs())
    docs.extend(_fetch_eviction_docs())
    docs.extend(_fetch_query_docs())
    return docs
