from retrieval.graph.tool.SQLvalidator import _get_connection


def get_data(sql: str) -> str:
    """Execute a validated DuckDB SQL query against the parquet dataset and return the results.
    Only call this after validate_sql_query has confirmed the SQL is valid.
    Returns the query results as a JSON array of records, or an error message if execution fails."""
    con = _get_connection()
    try:
        df = con.execute(sql).fetchdf()
        if df.empty:
            return "Query executed successfully but returned no rows."
        return df.to_json(orient="records", indent=2, date_format="iso")
    except Exception as e:
        return f"EXECUTION_ERROR: {e}"
