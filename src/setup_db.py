"""
Database setup script: creates schema by executing SQL files.
Can be used both locally and in CI/CD without needing psql client.
"""

import os
import sys
from pathlib import Path

import psycopg2


def setup_database(sql_files: list[str]) -> None:
    """Execute SQL files to set up database schema."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    # Convert SQLAlchemy URL to psycopg2 format if needed
    db_url = db_url.replace("postgresql+psycopg2://", "postgresql://")

    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        cursor = conn.cursor()

        for sql_file in sql_files:
            sql_path = Path(sql_file)
            if not sql_path.exists():
                print(f"ERROR: SQL file not found: {sql_file}")
                sys.exit(1)

            print(f"Executing {sql_file}...")
            with open(sql_path) as f:
                sql_content = f.read()
                cursor.execute(sql_content)
            print(f"✓ {sql_file} executed successfully")

        cursor.close()
        conn.close()
        print("\n✓ Database setup completed successfully")

    except psycopg2.Error as e:
        print(f"ERROR: Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    sql_files = [
        "sql/01_schema.sql",
        "sql/02_ml_schema.sql",
    ]
    setup_database(sql_files)
