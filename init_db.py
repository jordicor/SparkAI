import sqlite3
import os

def init_db():
    db_path = 'data/Spark.db'
    schema_path = 'spark_schema.sql'

    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    try:
        with sqlite3.connect(db_path) as conn:
            with open(schema_path, 'r') as f:
                conn.executescript(f.read())
        print(f"Database {db_path} initialized successfully.")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except FileNotFoundError as e:
        print(f"Schema file not found: {e}")

if __name__ == '__main__':
    init_db()