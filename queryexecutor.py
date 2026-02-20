from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, List, Optional, Dict
import time
import uuid

from SQLvalidator import (
    connect_duckdb,
    load_parquet_views,
    PARQUETS,
    validate_sql,
)

@dataclass
class UserQueryOutput:
    success: bool
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    truncated: bool
    latency_ms: float
    message: Optional[str]
    query_id: str

    # Optional debug info (only filled if debug=True)
    debug: Optional[Dict[str, Any]] = None


class QueryExecutor:
    """
    Runs validated SQL and returns structured table output.
      - converts validator reasons to user-friendly messages
      - enforcing output truncation
      - optionally returning debug info for developers
    """

    def __init__(self, max_rows: int = 200):
        self.con = connect_duckdb(use_db_file=False)
        load_parquet_views(self.con, PARQUETS)
        self.max_rows = max_rows

    # ----------------------------
    # Convert validator reasons -> user message
    # ----------------------------
    def _message_from_reasons(self, safety_reasons, performance_reasons, error) -> str:
        reasons = (safety_reasons or []) + (performance_reasons or [])

        # --- Safety ---
        if "MULTI_STATEMENT_NOT_ALLOWED" in reasons:
            return "Please ask one question at a time."

        if "PARSE_ERROR" in reasons:
            return "I couldn’t understand the request. Please rephrase using simpler wording."

        if "ONLY_SELECT_ALLOWED" in reasons or any(r.startswith("FORBIDDEN_STATEMENT:") for r in reasons):
            return "This system can only retrieve data (it cannot modify records). Please rephrase as a data question."

        if any(r.startswith("DISALLOWED_TABLES:") for r in reasons) or any(r.startswith("DISALLOWED_COLUMNS:") for r in reasons):
            return "That information isn’t available in the current dataset. Try asking using different terms or a more general description."

        if "SELECT_STAR_NOT_ALLOWED" in reasons:
            return "Your request is too broad. Please specify what information you want to see (e.g., a summary, specific fields, or counts)."

        if "MISSING_LIMIT" in reasons:
            return "Please ask for a smaller result (e.g., 'top 20', 'first 50', or a summary/count)."

        if "MISSING_WHERE_FOR_RESTRICTED_TABLE" in reasons:
            return "Please narrow your request by adding filters like year, diagnosis, age range, or patient group (e.g., 'in 2021', 'colon cancer', 'age > 55')."

        # --- Performance ---
        if "SEQ_SCAN_WITHOUT_FILTER" in reasons:
            return "This request may take too long. Please add more details (e.g., year, diagnosis, age range) to narrow it down."

        if "SORT_WITHOUT_LIMIT" in reasons:
            return "If you want results ordered, please also specify how many results you need (e.g., 'top 10')."

        if "TOO_MANY_JOINS" in reasons:
            return "This request is too complex. Please break it into smaller questions or ask for a summary first."

        if "CROSS_PRODUCT_DETECTED" in reasons:
            return "I’m missing a clear link between the items in your request. Please rephrase and specify how they should be related."

        if "EXPLAIN_FAILED" in reasons:
            return "Unable to evaluate this request safely. Please simplify or add filters."

        if error:
            return "Unable to retrieve results for this request. Please try rephrasing your question."

        return "Unable to retrieve results for this request. Please try rephrasing your question."



    def run(self, sql: str, debug: bool = False) -> UserQueryOutput:
        query_id = str(uuid.uuid4())

        validation = validate_sql(
            con=self.con,
            sql=sql,
            allow_tables=set(PARQUETS.keys()),
            require_where_for_tables={
                "condition_occurrence",
                "drug_exposure_cancerdrugs",
                "measurement_mutation",
            },
            require_limit=False, #false for now, testing queries
            block_select_star=True,
        )

        # If validation fails, craft message based on reason codes
        if (not validation.is_safe) or (not validation.is_performant):
            msg = self._message_from_reasons(
                safety_reasons=validation.safety_reasons,
                performance_reasons=validation.performance_reasons,
                error=validation.error,
            )

            out = UserQueryOutput(
                success=False,
                columns=[],
                rows=[],
                row_count=0,
                truncated=False,
                latency_ms=validation.latency_ms,
                message=f"Your question could not be processed. {msg}",
                query_id=query_id,
                debug=None,
            )

            if debug:
                out.debug = {
                    "safety_reasons": validation.safety_reasons,
                    "performance_reasons": validation.performance_reasons,
                    "correctness_reasons": validation.correctness_reasons,
                    "ast_features": validation.ast_features,
                    "error": validation.error,
                }
            return out

        # Otherwise execute
        return self._execute(sql, query_id=query_id, debug=debug)


    def _execute(self, sql: str, query_id: str, debug: bool = False) -> UserQueryOutput:
        start = time.time()

        try:
            cleaned = sql.strip().rstrip(";")
            wrapped_sql = f"SELECT * FROM ({cleaned}) q LIMIT {self.max_rows + 1}"

            cur = self.con.execute(wrapped_sql)
            columns = [desc[0] for desc in (cur.description or [])]
            fetched = cur.fetchall()

            truncated = len(fetched) > self.max_rows
            if truncated:
                fetched = fetched[: self.max_rows]

            rows = [list(r) for r in fetched]
            latency_ms = (time.time() - start) * 1000

            msg = None
            if truncated:
                msg = f"Results truncated to the first {self.max_rows} rows."

            return UserQueryOutput(
                success=True,
                columns=columns,
                rows=rows,
                row_count=len(rows),
                truncated=truncated,
                latency_ms=latency_ms,
                message=msg,
                query_id=query_id,
                debug=None,
            )

        except Exception as e:
            latency_ms = (time.time() - start) * 1000

            out = UserQueryOutput(
                success=False,
                columns=[],
                rows=[],
                row_count=0,
                truncated=False,
                latency_ms=latency_ms,
                message="The system encountered an issue retrieving data for this request.",
                query_id=query_id,
                debug=None,
            )

            if debug:
                out.debug = {
                    "execution_error": repr(e),
                }

            return out


# ----------------------------
# Quick demo runner
# ----------------------------
def demo():
    service = QueryExecutor(max_rows=10)

    queries = [
        # should fail: SELECT *
        "SELECT * FROM person LIMIT 5",

        # should pass
        """
        SELECT gender_concept_id, COUNT(*) AS n
        FROM person
        GROUP BY gender_concept_id
        """,

        # should fail: missing WHERE for restricted table
        """
        SELECT p.person_id, COUNT(*) AS num_conditions
        FROM person p
        JOIN condition_occurrence c ON c.person_id = p.person_id
        GROUP BY p.person_id
        """,
    ]

    for i, q in enumerate(queries, 1):
        print("\n==============================")
        print(f"Demo Query {i}")

        # debug can be set to True for debugging purposes
        result = service.run(q, debug=None)
        print(asdict(result))


if __name__ == "__main__":
    demo()
