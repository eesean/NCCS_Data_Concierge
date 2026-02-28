import os
from typing import Any, Dict, List, Tuple

from SQLgenerator import generate_sql_from_nl

from SQLvalidator import (
    connect_duckdb,
    load_parquet_views,
    PARQUETS,
    validate_sql as validate_sql_checker,
)

DEBUG = os.getenv("DEBUG", "false").lower() == "true"


def _execute_duckdb(con, sql: str, max_rows: int = 200) -> Tuple[List[str], List[List[Any]]]:
    """
    Execute SQL and return (columns, rows). Hard caps rows to avoid huge outputs.
    """
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchmany(max_rows)
    return cols, [list(r) for r in rows]


def handle_question(question: str, model: str | None = None) -> Dict[str, Any]:
    """
    NL -> SQL (LLM) -> validate -> execute -> return results
    Returns results only (never SQL) to satisfy your user story.
    """
    # 1) Connect to DuckDB + load encrypted parquet views
    con = connect_duckdb(use_db_file=False)
    load_parquet_views(con, PARQUETS)

    allow_tables = set(PARQUETS.keys())
    restricted_tables = {"condition_occurrence", "drug_exposure_cancerdrugs", "measurement_mutation"}

    # 2) Generate SQL in backend
    result = generate_sql_from_nl(question, model=model) #if model is None it will use a default model
    sql = result['sql']

    # backend-only logging
    if DEBUG:
        print("[DEBUG] Generated SQL:", sql)

    # 3) Validate SQL
    validation = validate_sql_checker(
        con=con,
        sql=sql,
        expected_result=None,
        allow_tables=allow_tables,
        allow_columns=None,
        require_where_for_tables=restricted_tables,
        require_limit=True,
        block_select_star=True,
    )

    if not validation.is_safe:
        return {
            "status": "error",
            "message": "Blocked by safety rules.",
            "reasons": validation.safety_reasons,
        }

    if not validation.is_performant:
        return {
            "status": "error",
            "message": "Blocked due to performance risk.",
            "reasons": validation.performance_reasons,
        }

    # 4) Execute and return table only (no SQL)
    try:
        columns, rows = _execute_duckdb(con, sql, max_rows=200)
        # if it's a single scalar result (e.g., COUNT(*)), return a user-friendly value 
        is_scalar = (len(columns) == 1 and len(rows) == 1 and isinstance(rows[0], list) and len(rows[0]) == 1)
        if is_scalar:
            return {
                "status": "ok",
                "message": "Query executed successfully.",
                "metric": columns[0],     
                "value": rows[0][0],      
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
            }

        # otherwise, return normal table results
        return {
            "status": "ok",
            "message": "Query executed successfully.",
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
        }

    except Exception:
        return {
            "status": "error",
            "message": "Execution failed. Please refine your question.",
            "reasons": ["EXECUTION_FAILED"],
        }
#====== TESTING PIPELINE ============#
if __name__ == "__main__":
    print(handle_question("How many deaths occurred in 2020?"))