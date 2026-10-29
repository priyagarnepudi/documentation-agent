import os
import urllib.request

import duckdb

DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
PARQUET_FILE = "yellow_tripdata_2024-01.parquet"

con = duckdb.connect("taxi.db")


def setup_database():
    """Download the parquet file and load it into DuckDB."""
    if not os.path.exists(PARQUET_FILE):
        print(f"Downloading {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, PARQUET_FILE)

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS trips AS
        SELECT * FROM '{PARQUET_FILE}'
    """)
    count = con.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    print(f"Loaded {count} rows")
    return count


class SQLTools:
    def get_schema(self) -> str:
        """Return all column names and types in the trips table."""
        rows = con.execute("DESCRIBE trips").fetchall()
        return "\n".join(f"{row[0]}: {row[1]}" for row in rows)

    def run_sql(self, query: str) -> str:
        """Execute SQL and return column headers and at most 50 result rows."""
        cursor = con.execute(query)
        headers = [column[0] for column in cursor.description]
        rows = cursor.fetchmany(50)
        lines = ["\t".join(headers)]
        lines.extend(
            "\t".join("NULL" if value is None else str(value) for value in row)
            for row in rows
        )
        return "\n".join(lines)
