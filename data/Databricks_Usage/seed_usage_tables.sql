-- SQLite Seed Script: DML only (assuming DDL is already run)
-- Note: This script uses SQLite's date/time functions and Recursive CTEs.
-- The DDL for the 10 tables MUST be created before running this script.

-- 1. Define constants and Date Series
-- We use a single constant table to define the range and cost.
CREATE TEMP TABLE IF NOT EXISTS constants (
    start_date      TEXT, -- YYYY-MM-DD
    end_date        TEXT, -- YYYY-MM-DD
    ou_id           TEXT,
    cost_per_dbu    REAL
);

-- Insert constants
DELETE FROM constants;
INSERT INTO constants (start_date, end_date, ou_id, cost_per_dbu)
VALUES ('2025-11-28', '2025-12-11', 'OU-CHICOLA', 0.40);

-- Helper table used for generating daily dates between start_date and end_date
CREATE TEMP TABLE IF NOT EXISTS date_series (
    date TEXT PRIMARY KEY -- YYYY-MM-DD
);

DELETE FROM date_series; -- Clear existing data if table persists

WITH RECURSIVE dates (date) AS (
    SELECT start_date FROM constants
    UNION ALL
    SELECT date(date, '+1 day')
    FROM dates
    WHERE date < (SELECT end_date FROM constants)
)
INSERT INTO date_series (date)
SELECT date FROM dates;


-- --------------------------------------------------------------------------------------
-- 2. Insert Core Entities (DML)
-- --------------------------------------------------------------------------------------

-- Clear existing data (in case the script is run multiple times)
DELETE FROM company_ou;
DELETE FROM users_lookup;
DELETE FROM instance_pools;
DELETE FROM non_job_compute;
DELETE FROM jobs;

INSERT INTO company_ou
SELECT t1.ou_id, 'Chicago Cola Co.', 'CCOL-1234',
       'Leading beverage distributor, heavily focused on optimizing supply chain and finance reporting systems.'
FROM constants t1;

INSERT INTO users_lookup (user_id, name, company_ou_id, department)
SELECT 'U-ALICE',   'Alice',   t1.ou_id, 'Finance'      FROM constants t1 UNION ALL 
SELECT 'U-BOB',     'Bob',     t1.ou_id, 'Supply Chain' FROM constants t1 UNION ALL 
SELECT 'U-CHARLIE', 'Charlie', t1.ou_id, 'HR'           FROM constants t1 UNION ALL 
SELECT 'U-DAVID',   'David',   t1.ou_id, 'Logistics'    FROM constants t1 UNION ALL 
SELECT 'U-EMILY',   'Emily',   t1.ou_id, 'Data Science' FROM constants t1 UNION ALL
SELECT 'U-SYSTEM',  'System',  t1.ou_id, 'System'       FROM constants t1;

INSERT INTO instance_pools (instance_pool_id, pool_name, pool_instance_type, min_size, max_size, auto_termination_mins) VALUES
    ('POOL-DBS-L', 'Low Latency Worker Pool', 'i3.xlarge', 2, 5, 0);

INSERT INTO non_job_compute (compute_id, compute_name, compute_type, company_ou_id)
SELECT 'APC-001', 'Logistics Analytics APC', 'APC_CLUSTER',   (SELECT ou_id FROM constants)
UNION ALL SELECT 'APC-002', 'Finance Reporting APC',  'APC_CLUSTER',   (SELECT ou_id FROM constants)
UNION ALL SELECT 'WH-1',    'Finance SQL Warehouse',  'SQL_WAREHOUSE', (SELECT ou_id FROM constants)
UNION ALL SELECT 'WH-2',    'Supply Chain SQL Warehouse', 'SQL_WAREHOUSE', (SELECT ou_id FROM constants)
UNION ALL SELECT 'WH-3',    'HR SQL Warehouse',       'SQL_WAREHOUSE', (SELECT ou_id FROM constants)
UNION ALL SELECT 'WH-4',    'Logistics SQL Warehouse','SQL_WAREHOUSE', (SELECT ou_id FROM constants)
UNION ALL SELECT 'WH-5',    'ML SQL Warehouse',       'SQL_WAREHOUSE', (SELECT ou_id FROM constants);

INSERT INTO jobs (job_id, company_ou_id, job_name, description, tags)
-- Note: ARRAY('...') is replaced with json_array() for JSON text storage
SELECT 'J-FIN-DLY', (SELECT ou_id FROM constants), 'Financial Report Aggregation',
       'Daily job for calculating month-end revenue figures.', json_array('Finance', 'Critical')
UNION ALL SELECT 'J-SPLY-ETL', (SELECT ou_id FROM constants), 'Supply Chain Data ETL',
       'Extracts and transforms raw sensor data for logistics dashboard.', json_array('Logistics', 'Fleet')
UNION ALL SELECT 'J-HR-DASH', (SELECT ou_id FROM constants), 'HR Dashboard Prep',
       'Prepares aggregated employee data using a dedicated pool.', json_array('HR', 'Pooled')
UNION ALL SELECT 'J-LOGI-OPT', (SELECT ou_id FROM constants), 'Logistics Optimizer',
       'Autoscaling job for calculating optimal shipping routes.', json_array('Logistics', 'Autoscaling', 'Optimization')
UNION ALL SELECT 'J-PROD-ML', (SELECT ou_id FROM constants), 'Production ML Training',
       'Daily training of the ML product recommendation model.', json_array('DataScience', 'Critical', 'Autoscaling');


-- --------------------------------------------------------------------------------------
-- 3. Generate Job Runs, Evictions, and Compute Usage (DML)
-- --------------------------------------------------------------------------------------

DELETE FROM job_runs;
DELETE FROM eviction_details;
DELETE FROM compute_usage;
DELETE FROM events;
DELETE FROM sql_query_history;

-- Define temporary view 'run_data': calculates duration_ms and end_time
CREATE TEMP VIEW run_data AS
WITH job_config AS (
    SELECT 'J-FIN-DLY'  AS job_id, 'i3.xlarge'   AS worker_instance_type,
           6 AS fixed_nodes, 6 AS min_nodes, 6 AS max_nodes,
           0 AS is_autoscaling_enabled, 0 AS is_fleet_cluster,
           NULL AS instance_pool_id, 0.15 AS avg_cpu_base, 0.8 AS dbus_per_minute, 0 AS run_failure_day
    UNION ALL SELECT 'J-SPLY-ETL', 'r5.xlarge', 5, 5, 5,
           0, 1, NULL, 0.20, 0.7, 0 
    UNION ALL SELECT 'J-HR-DASH', 'i3.xlarge', 5, 5, 5,
           0, 0, 'POOL-DBS-L', 0.18, 0.8, 0
    UNION ALL SELECT 'J-LOGI-OPT', 'r5.2xlarge', 5, 5, 10,
           1, 0, NULL, 0.10, 1.2, 1
    UNION ALL SELECT 'J-PROD-ML', 'g4dn.xlarge', 5, 5, 10,
           1, 0, NULL, 0.75, 2.5, 1
),
run_calculations AS (
    SELECT 
        ds.date, jc.*,
        'RUN-' || jc.job_id || '-' || strftime('%Y%m%d', ds.date) AS job_run_id,
        datetime(ds.date, '+10 hours', '+' || CAST(abs(random() % 30) AS TEXT) || ' minutes') AS start_time_str,
        CASE
            WHEN (CAST(strftime('%d', ds.date) AS INTEGER) % 5) = 0
                 AND jc.run_failure_day = 1
            THEN 'FAILED'
            ELSE 'SUCCESS'
        END AS run_status,
        CASE jc.job_id 
            WHEN 'J-PROD-ML' THEN 3600000 + abs(random() % (15 * 60 * 1000))  -- 1 hr + rand(15 mins) in ms
            ELSE 1800000 + abs(random() % (10 * 60 * 1000))                   -- 30 mins + rand(10 mins) in ms
        END AS base_duration_ms
    FROM date_series ds
    CROSS JOIN job_config jc
)
SELECT 
    *,
    CASE
        WHEN run_status = 'SUCCESS' THEN base_duration_ms
        ELSE CAST(base_duration_ms * 0.5 AS INTEGER)
    END AS duration_ms,
    datetime(
        start_time_str,
        '+' || CAST(
            CASE
                WHEN run_status = 'SUCCESS' THEN base_duration_ms
                ELSE CAST(base_duration_ms * 0.5 AS INTEGER)
            END / 1000
            AS TEXT
        ) || ' seconds'
    ) AS end_time_str
FROM run_calculations;

-- Insert into job_runs
INSERT INTO job_runs
SELECT 
    job_run_id,
    job_id,
    start_time_str AS start_time,
    end_time_str   AS end_time,
    duration_ms,
    run_status,
    CASE WHEN run_status = 'FAILED'
         THEN 'Cluster node eviction caused immediate job termination.'
         ELSE NULL
    END AS error_summary,
    'm5.xlarge'        AS driver_instance_type,
    worker_instance_type AS worker_instance_type,
    is_fleet_cluster,
    instance_pool_id,
    min_nodes,
    max_nodes,
    fixed_nodes,
    is_autoscaling_enabled,
    0.80 AS avg_cpu_credits
FROM run_data;

-- Insert into eviction_details
INSERT INTO eviction_details
SELECT 
    'EVICT-' || T1.job_run_id       AS eviction_id, 
    T1.worker_instance_type || '-' || CAST(abs(random() % 1000) AS TEXT) AS cloud_instance_id, 
    datetime(T1.start_time_str, '+' || CAST(T1.duration_ms * 0.8 / 1000 AS INTEGER) || ' seconds') AS eviction_time,
    'AWS EC2 Spot Instance Termination Notice: Not enough available capacity to maintain your requested spot instances.'
        AS cloud_provider_message,
    'CAPACITY_CHANGE'  AS eviction_reason,
    0.08               AS spot_price,
    'STOP_DEALLOCATE'  AS eviction_action,
    0                  AS was_retried -- FALSE
FROM run_data T1
WHERE T1.run_status = 'FAILED';

-- Insert into compute_usage (Job Runs)
INSERT INTO compute_usage
SELECT 
    t1.job_run_id        AS compute_usage_id,
    t1.job_run_id        AS parent_id,
    'JOB_RUN'            AS parent_type,
    'STANDARD_ALL_PURPOSE_COMPUTE' AS compute_sku,
    CAST(
        t1.dbus_per_minute * (t1.duration_ms / 60000.0) * (t1.min_nodes + t1.max_nodes) / 2.0
        AS REAL
    ) AS dbus_consumed,
    'i3.xlarge-' || CAST(abs(random() % 1000) AS TEXT) AS cluster_id,
    t1.worker_instance_type AS cluster_instance_type,
    (
        t1.dbus_per_minute * (t1.duration_ms / 60000.0) * (t1.min_nodes + t1.max_nodes) / 2.0
    ) * (SELECT cost_per_dbu FROM constants) AS total_cost,
    t1.avg_cpu_base + (random() / 9223372036854775807.0 * 0.1) AS avg_cpu_utilization,
    128.0  AS avg_memory_gb,
    abs(random() % 100) AS peak_concurrent_users,
    1      AS is_production,
    t1.date AS usage_date
FROM run_data t1;

-- Insert into compute_usage (APC Clusters)
CREATE TEMP VIEW apc_usage_data AS
SELECT 
    T1.compute_id,
    T1.compute_type,
    ds.date
FROM date_series ds
CROSS JOIN non_job_compute T1
WHERE T1.compute_type = 'APC_CLUSTER';

INSERT INTO compute_usage
SELECT 
    T1.compute_id || '-' || strftime('%Y%m%d', T1.date) AS compute_usage_id,
    T1.compute_id       AS parent_id,
    T1.compute_type     AS parent_type,
    'STANDARD_ALL_PURPOSE_COMPUTE' AS compute_sku,
    CAST(abs(random() % 50) + 500 AS INTEGER) AS dbus_consumed,
    T1.compute_id       AS cluster_id,
    'i3.2xlarge'        AS cluster_instance_type,
    (CAST(abs(random() % 50) + 500 AS INTEGER)) * (SELECT cost_per_dbu FROM constants) AS total_cost,
    (0.9 + (random() / 9223372036854775807.0 * 0.1)) * 0.70 AS avg_cpu_utilization,
    256.0 AS avg_memory_gb,
    0     AS peak_concurrent_users,
    1     AS is_production,
    T1.date AS usage_date
FROM apc_usage_data T1;

-- Insert into events (START/END/EVICTION Events)
INSERT INTO events
SELECT
    'EVT-START-' || T1.job_run_id AS event_id,
    T1.job_run_id                 AS compute_usage_id,
    T1.start_time_str             AS event_time,
    'CLUSTER_START'               AS event_type,
    'U-SYSTEM'                    AS user_id,
    'Cluster starting for job: ' || T1.job_id ||
        ' with ' || T1.min_nodes || ' to ' || T1.max_nodes || ' nodes.' AS details,
    NULL                          AS eviction_id
FROM run_data T1
UNION ALL
SELECT
    'EVT-END-' || T1.job_run_id,
    T1.job_run_id,
    T1.end_time_str,
    'CLUSTER_TERMINATE',
    'U-SYSTEM',
    'Cluster terminated successfully.',
    NULL
FROM run_data T1
WHERE T1.run_status = 'SUCCESS'
UNION ALL
SELECT
    'EVT-EVICT-' || T1.job_run_id,
    T1.job_run_id,
    datetime(T1.start_time_str, '+' || CAST(T1.duration_ms * 0.8 / 1000 AS INTEGER) || ' seconds'),
    'SPOT_EVICTION',
    'U-SYSTEM',
    'Spot instance loss detected during execution. Job terminated.',
    'EVICT-' || T1.job_run_id
FROM run_data T1
WHERE T1.run_status = 'FAILED';


-- --------------------------------------------------------------------------------------
-- 4. Generate Ad-Hoc SQL Query History (DML)
-- --------------------------------------------------------------------------------------

CREATE TEMP VIEW adhoc_queries AS
SELECT 1 AS q_id,
       'SELECT count(*), date FROM raw_sales WHERE region = ''EAST'' GROUP BY date' AS sql_text,
       'U-ALICE' AS user_id
UNION ALL
SELECT 2,
       'SELECT sum(revenue), product_id FROM fin_data WHERE month = 11 GROUP BY product_id ORDER BY 1 DESC' AS sql_text,
       'U-ALICE' AS user_id
UNION ALL
SELECT 3,
       'CREATE OR REPLACE TEMP VIEW new_supply_chain AS SELECT * FROM raw_logistics_sensor_data WHERE temp > 80' AS sql_text,
       'U-BOB' AS user_id
UNION ALL
SELECT 4,
       'SELECT employee_id, salary FROM hr_data WHERE tenure > 5' AS sql_text,
       'U-CHARLIE' AS user_id;

INSERT INTO sql_query_history (
    query_id,
    parent_id,
    user_id,
    start_time,
    duration_ms,
    warehouse_sku,
    sql_text,
    error_message
)
SELECT
    'Q-ADHOC-' || q_id AS query_id,
    CASE 
        WHEN q_id IN (1, 2) THEN 'WH-1'
        WHEN q_id = 3         THEN 'WH-2'
        ELSE 'WH-3'
    END AS parent_id,
    user_id,
    datetime((SELECT start_date FROM constants), '+' || (q_id * 10) || ' minutes') AS start_time,
    500 + (q_id * 100) AS duration_ms,
    CASE 
        WHEN q_id IN (1, 2) THEN 'SQLWH-SMALL'
        WHEN q_id = 3         THEN 'SQLWH-MEDIUM'
        ELSE 'SQLWH-LARGE'
    END AS warehouse_sku,
    sql_text,
    CASE WHEN q_id = 4 THEN 'Query exceeded timeout due to large result set.' ELSE NULL END AS error_message
FROM adhoc_queries;
