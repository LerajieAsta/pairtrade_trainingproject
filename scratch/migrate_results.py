import sqlite3
import os

source_db = "data/sp500_Current.db"
dest_db = "results/result.db"

if not os.path.exists(source_db):
    print(f"Source database {source_db} does not exist.")
    exit(1)

# Ensure destination directory exists
os.makedirs(os.path.dirname(dest_db), exist_ok=True)

print(f"Connecting to destination database: {dest_db}")
dest_conn = sqlite3.connect(dest_db)
dest_cursor = dest_conn.cursor()

# Enable WAL mode for destination
dest_conn.execute("PRAGMA journal_mode=WAL;")
dest_conn.execute("PRAGMA synchronous=NORMAL;")

# Connect to source database to read schemas
print(f"Connecting to source: {source_db}")
src_conn = sqlite3.connect(source_db)
src_cursor = src_conn.cursor()

src_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in src_cursor.fetchall()]
print("Tables in source database:", tables)

target_tables = ["strategy_summaries", "trade_logs", "strategy_pairs"]
tables_to_migrate = [t for t in target_tables if t in tables]

if not tables_to_migrate:
    print("No backtest result tables found in source database to migrate.")
    src_conn.close()
    dest_conn.close()
    exit(0)

# 1. Create tables in destination if not exist (preserving schemas/constraints/indexes)
for table in tables_to_migrate:
    src_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,))
    create_sql = src_cursor.fetchone()[0]
    # Ensure CREATE TABLE uses IF NOT EXISTS
    create_sql_safe = create_sql.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ", 1)
    dest_cursor.execute(create_sql_safe)

src_conn.close() # Close source connection before attaching

# 2. Attach source DB to destination connection and perform native SQLite migration
print("Attaching source database for high-performance in-engine migration...")
dest_cursor.execute(f"ATTACH DATABASE ? AS src", (source_db,))

for table in tables_to_migrate:
    print(f"Migrating table {table}...")
    
    # Delete conflicting rows in destination
    if table == "strategy_summaries":
        dest_cursor.execute("DELETE FROM main.strategy_summaries WHERE _path IN (SELECT _path FROM src.strategy_summaries)")
    elif table in ("trade_logs", "strategy_pairs"):
        dest_cursor.execute(f"DELETE FROM main.{table} WHERE strategy_id IN (SELECT strategy_id FROM src.{table})")
        
    # Insert rows directly from source, matching column names
    # First, dynamically query column names in source via attached 'src' database
    dest_cursor.execute(f"PRAGMA src.table_info({table})")
    cols = [f'"{r[1]}"' for r in dest_cursor.fetchall()]
    cols_str = ", ".join(cols)
    dest_cursor.execute(f"INSERT INTO main.{table} ({cols_str}) SELECT {cols_str} FROM src.{table}")
    dest_conn.commit()
    print(f"Table {table} migration completed.")

# Detach source
dest_cursor.execute("DETACH DATABASE src")
dest_conn.commit()

# Clean up source database tables to save space and clean it up
print("Cleaning up tables from source database...")
src_conn = sqlite3.connect(source_db)
src_cursor = src_conn.cursor()
for table in tables_to_migrate:
    src_cursor.execute(f"DROP TABLE IF EXISTS {table}")
src_conn.commit()
src_conn.close()

dest_conn.close()
print("Migration completed successfully!")
