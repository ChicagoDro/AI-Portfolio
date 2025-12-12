import sqlite3
import os

# --- Configuration ---
DATABASE_FILE = 'usage_rag_data.db'
DDL_SCRIPT_FILE = 'create_usage_tables.sql' # Your converted DDL from the first step
DML_SCRIPT_FILE = 'seed_usage_tables.sql'   # Your converted ANSI SQL DML (seeding)

def setup_database(db_file: str, ddl_script: str, dml_script: str):
    """
    Creates the SQLite database file and executes the DDL (schema) and DML (data) scripts.
    """
    # 1. Connect to the database file (creates it if it doesn't exist)
    # The 'with' statement ensures the connection is closed automatically.
    try:
        with sqlite3.connect(db_file) as conn:
            print(f"✅ Successfully connected to and created database: {db_file}")

            # Enable Foreign Key enforcement (recommended for data integrity)
            conn.execute("PRAGMA foreign_keys = ON;")
            
            # Create a Cursor object to execute commands
            cursor = conn.cursor()

            # 2. Execute the DDL (Schema) Script
            print(f"\n--- 1. Executing DDL from: {ddl_script} ---")
            with open(ddl_script, 'r') as f:
                ddl_script_content = f.read()
            
            # The executescript method is used to run a string containing multiple SQL statements.
            cursor.executescript(ddl_script_content)
            print("✅ DDL (Table Creation) executed successfully.")
            
            # 3. Execute the DML (Seeding) Script
            print(f"\n--- 2. Executing DML from: {dml_script} ---")
            with open(dml_script, 'r') as f:
                dml_script_content = f.read()
            
            cursor.executescript(dml_script_content)
            
            # 4. Commit and Close (Done automatically by 'with' but explicitly committed here)
            conn.commit()
            print("✅ DML (Data Seeding) executed and committed successfully.")
            
            # 5. Simple verification query
            cursor.execute("SELECT count(*) FROM company_ou")
            row_count = cursor.fetchone()[0]
            print(f"\n✨ Verification: company_ou table contains {row_count} row(s).")
            
    except sqlite3.OperationalError as e:
        print(f"❌ Database Operation Failed: {e}")
    except FileNotFoundError as e:
        print(f"❌ File Not Found: {e}. Ensure both SQL files are in the same directory.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == '__main__':
    # Place your DDL and DML files in the same directory as this Python script
    setup_database(DATABASE_FILE, DDL_SCRIPT_FILE, DML_SCRIPT_FILE)