-- Converted DDL Script for Databricks Usage Monitoring Tables
-- Target Database: SQLite (File-based)
-- Note: 'USE usage_data;' and TBLPROPERTIES are removed.

-- --------------------------------------------------------------------------------------
-- Lookup Tables (L1, L2, L3)
-- --------------------------------------------------------------------------------------

-- Table L1: users_lookup (Needed for linking queries/events to users)
CREATE TABLE IF NOT EXISTS users_lookup (
    user_id                 TEXT PRIMARY KEY NOT NULL, -- STRING -> TEXT, PRIMARY KEY declared inline
    name                    TEXT NOT NULL,
    company_ou_id           TEXT NOT NULL, -- FOREIGN KEY linking user to the owning team.
    department              TEXT
    -- Note: Databricks style comments are kept, but no functional PRIMARY KEY constraint here
    -- Note: PRIMARY KEY constraint is declared inline above: user_id TEXT PRIMARY KEY NOT NULL
);


-- Table L2: instance_pools (Needed for linking job runs to pool configuration)
CREATE TABLE IF NOT EXISTS instance_pools (
    instance_pool_id        TEXT PRIMARY KEY NOT NULL,
    pool_name               TEXT NOT NULL,
    pool_instance_type      TEXT, -- STRING -> TEXT
    min_size                INTEGER,
    max_size                INTEGER,
    auto_termination_mins   INTEGER
);


-- Table L3: non_job_compute (For APC Clusters, SQL Warehouses, etc.)
CREATE TABLE IF NOT EXISTS non_job_compute (
    compute_id              TEXT PRIMARY KEY NOT NULL,
    compute_name            TEXT NOT NULL,
    compute_type            TEXT NOT NULL, -- APC_CLUSTER or SQL_WAREHOUSE.
    company_ou_id           TEXT -- Owning team.
);


-- --------------------------------------------------------------------------------------
-- Core Data Tables (1-7)
-- --------------------------------------------------------------------------------------

-- Table 1: company_ou (Organizational Unit / Team)
CREATE TABLE IF NOT EXISTS company_ou (
    company_ou_id           TEXT PRIMARY KEY NOT NULL,
    name                    TEXT NOT NULL,
    cost_center_code        TEXT,
    description             TEXT -- Vector Embedding Candidate.
);


-- Table 2: jobs (Job Definitions)
CREATE TABLE IF NOT EXISTS jobs (
    job_id                  TEXT PRIMARY KEY NOT NULL,
    company_ou_id           TEXT NOT NULL, -- FOREIGN KEY linking job to the owning team.
    job_name                TEXT NOT NULL,
    description             TEXT, -- Vector Embedding Candidate.
    tags                    TEXT -- ARRAY<STRING> changed to TEXT (JSON array string).
);


-- Table 3: job_runs (Job Executions and Compute Config)
CREATE TABLE IF NOT EXISTS job_runs (
    job_run_id              TEXT PRIMARY KEY NOT NULL,
    job_id                  TEXT NOT NULL, -- FOREIGN KEY linking run back to the job definition.
    start_time              DATETIME NOT NULL, -- TIMESTAMP -> DATETIME (SQLite can store as TEXT/REAL/INTEGER)
    end_time                DATETIME,
    duration_ms             INTEGER, -- LONG -> INTEGER (SQLite default is 64-bit)
    run_status              TEXT NOT NULL, -- SUCCESS, FAILED, CANCELED.
    error_summary           TEXT, -- Vector Embedding Candidate.
    driver_instance_type    TEXT,
    worker_instance_type    TEXT,
    is_fleet_cluster        INTEGER, -- BOOLEAN -> INTEGER (0 or 1)
    instance_pool_id        TEXT, -- FOREIGN KEY to instance_pools.
    min_nodes               INTEGER,
    max_nodes               INTEGER,
    fixed_nodes             INTEGER,
    is_autoscaling_enabled  INTEGER, -- BOOLEAN -> INTEGER (0 or 1)
    spot_instance_ratio     REAL -- DECIMAL(3, 2) -> REAL (floating point)
);


-- Table 4: compute_usage (Cost and Usage Metrics)
CREATE TABLE IF NOT EXISTS compute_usage (
    compute_usage_id        TEXT PRIMARY KEY NOT NULL,
    parent_id               TEXT NOT NULL, -- Links to job_run_id OR compute_id.
    compute_type            TEXT NOT NULL, -- JOB_RUN, APC_CLUSTER, SQL_WAREHOUSE.
    sku                     TEXT NOT NULL,
    dbus_consumed           REAL NOT NULL, -- DECIMAL(18, 4) -> REAL
    instance_id             TEXT, -- The actual cloud VM Instance ID used.
    instance_type           TEXT,
    cost_usd                REAL NOT NULL, -- DECIMAL(18, 4) -> REAL
    avg_cpu_utilization     REAL, -- DECIMAL(4, 3) -> REAL
    max_memory_used_gb      REAL, -- DECIMAL(18, 2) -> REAL
    disk_io_wait_time_ms    INTEGER, -- LONG -> INTEGER
    cloud_market_available  INTEGER, -- BOOLEAN -> INTEGER (0 or 1)
    usage_date              TEXT NOT NULL -- DATE -> TEXT (YYYY-MM-DD format)
);


-- Table 5: eviction_details
CREATE TABLE IF NOT EXISTS eviction_details (
    eviction_id             TEXT PRIMARY KEY NOT NULL,
    instance_id             TEXT NOT NULL,
    eviction_time           DATETIME NOT NULL, -- TIMESTAMP -> DATETIME
    cloud_provider_message  TEXT, -- Vector Embedding Candidate.
    eviction_reason_code    TEXT,
    instance_reclaim_rate   REAL, -- DECIMAL(4, 3) -> REAL
    eviction_policy_used    TEXT,
    replacement_on_demand   INTEGER -- BOOLEAN -> INTEGER (0 or 1)
);


-- Table 6: events (Diagnostic Logs and Audit Events)
CREATE TABLE IF NOT EXISTS events (
    event_id                TEXT PRIMARY KEY NOT NULL,
    compute_usage_id        TEXT NOT NULL, -- FOREIGN KEY linking event to the consuming resource.
    timestamp               DATETIME NOT NULL, -- TIMESTAMP -> DATETIME
    event_type              TEXT NOT NULL, -- CLUSTER_START, CLUSTER_FAILURE, AUDIT_LOGIN, SPOT_EVICTION.
    user_id                 TEXT, -- FOREIGN KEY to users_lookup.
    details                 TEXT, -- Full raw event log, stack trace (Vector Embedding Candidate).
    eviction_id             TEXT -- FOREIGN KEY to eviction_details if event_type is SPOT_EVICTION.
);


-- Table 7: sql_query_history (Optimization Target)
CREATE TABLE IF NOT EXISTS sql_query_history (
    query_id                TEXT PRIMARY KEY NOT NULL,
    parent_id               TEXT NOT NULL, -- Links to job_run_id or compute_id (SQL Warehouse).
    user_id                 TEXT NOT NULL, -- FOREIGN KEY to users_lookup.
    start_time              DATETIME NOT NULL, -- TIMESTAMP -> DATETIME
    duration_ms             INTEGER, -- LONG -> INTEGER
    warehouse_sku           TEXT,
    sql_text                TEXT NOT NULL, -- The complete SQL text of the query (Vector Embedding Candidate).
    error_message           TEXT -- Detailed error message if the query failed (Vector Embedding Candidate).
);