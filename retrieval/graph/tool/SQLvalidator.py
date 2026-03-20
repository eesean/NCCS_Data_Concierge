from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import os
import time

import duckdb
import sqlglot
from sqlglot import exp
from dotenv import load_dotenv
from langchain_core.tools import tool

# SQLvalidator.py is at retrieval/graph/tool/ → parents[3] is data_concierge/
load_dotenv(Path(__file__).parents[3] / ".env")

# ----------------------------
# Config: data paths
# ----------------------------
# SQLvalidator.py lives at retrieval/graph/tool/ → parents[3] is data_concierge/
DATA_DIR = Path(__file__).parents[3] / "data"

PARQUETS: Dict[str, Path] = {
    "person": DATA_DIR / "person.parquet",
    "condition_occurrence": DATA_DIR / "condition_occurrence.parquet",
    "procedure_occurrence": DATA_DIR / "procedure_occurrence.parquet",
    "drug_exposure_cancerdrugs": DATA_DIR / "drug_exposure_cancerdrugs.parquet",
    "measurement_mutation": DATA_DIR / "measurement_mutation.parquet",
    "death": DATA_DIR / "death.parquet",
}

# ----------------------------
# Result structure
# ----------------------------
@dataclass
class ValidationResult:
    is_safe: bool
    is_performant: bool
    safety_reasons: List[str]
    performance_reasons: List[str]
    ast_features: Dict[str, Any]
    plan_text: Optional[str]
    latency_ms: float
    error: Optional[str]


# ----------------------------
# DuckDB connection
# ----------------------------
def connect_duckdb() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=":memory:")


def load_parquet_views(con, parquet_map):
    key_name = os.getenv("PARQUET_KEY_NAME", "nccs_key")
    key_value = os.getenv("PARQUET_KEY_VALUE")
    con.execute(f"PRAGMA add_parquet_key('{key_name}', '{key_value}');")

    for table, path in parquet_map.items():
        con.execute(f"""
            CREATE OR REPLACE VIEW {table} AS
            SELECT * FROM read_parquet(
                '{path.as_posix()}',
                encryption_config = {{footer_key: '{key_name}'}}
            );
        """)

def get_schema_map(con: duckdb.DuckDBPyConnection, parquet_map: Dict[str, Path]) -> Dict[str, Set[str]]:
    schema_map: Dict[str, Set[str]] = {}

    for table in parquet_map.keys():
        rows = con.execute(f"DESCRIBE {table}").fetchall()
        schema_map[table] = {row[0] for row in rows}

    return schema_map


def flatten_schema_map(schema_map: Dict[str, Set[str]]) -> Set[str]:
    all_columns: Set[str] = set()

    for cols in schema_map.values():
        all_columns.update(cols)

    return all_columns


# ----------------------------
# AST parsing helpers
# ----------------------------
FORBIDDEN_STATEMENTS = {
    "insert", "update", "delete", "drop",
    "alter", "create", "copy", "attach", "export",
}

STAR_ALLOWLIST: Set[str] = set()

# Tables that do NOT exist — when SQL references these, include a hint for the LLM
DISALLOWED_TABLE_HINTS: Dict[str, str] = {
    "concept": (
        "The concept table does NOT exist. Do NOT use it. "
        "Use drug_source_value, condition_source_value, procedure_source_value, measurement_concept_name, etc. for human-readable names."
    ),
}


def parse_sql(sql: str) -> exp.Expression:
    return sqlglot.parse_one(sql, read="duckdb")


def detect_statement_type(tree: exp.Expression) -> str:
    return tree.key.lower()


def extract_tables(tree: exp.Expression) -> List[str]:
    return sorted({t.name for t in tree.find_all(exp.Table)})


def extract_columns(tree: exp.Expression) -> List[str]:
    alias_names = set()

    # collect aliases created in SELECT expressions
    for alias in tree.find_all(exp.Alias):
        if alias.alias:
            alias_names.add(alias.alias)

    columns = set()
    for col in tree.find_all(exp.Column):
        col_name = col.name
        if col_name not in alias_names:
            columns.add(col_name)

    return sorted(columns)


def count_joins(tree: exp.Expression) -> int:
    return len(list(tree.find_all(exp.Join)))


def has_where(tree: exp.Expression) -> bool:
    return tree.args.get("where") is not None


def has_limit(tree: exp.Expression) -> bool:
    return tree.args.get("limit") is not None


def has_select_star(tree: exp.Expression) -> bool:
    for sel in tree.find_all(exp.Select):
        for proj in sel.expressions:
            if isinstance(proj, exp.Star):
                return True
            if isinstance(proj, exp.Column) and isinstance(proj.this, exp.Star):
                return True
    return False


# ----------------------------
# Safety check
# ----------------------------
def safety_check(
    sql: str,
    allow_tables: Optional[Set[str]] = None,
    allow_columns: Optional[Set[str]] = None,
    require_limit: bool = True,
    require_where_for_tables: Optional[Set[str]] = None,
    block_select_star: bool = True,
) -> Tuple[bool, List[str], Dict[str, Any], Optional[exp.Expression], Optional[str]]:
    reasons: List[str] = []

    if sql.strip().count(";") > 1:
        return False, ["MULTI_STATEMENT_NOT_ALLOWED"], {}, None, None

    try:
        tree = parse_sql(sql)
    except Exception as e:
        return False, ["PARSE_ERROR"], {}, None, f"PARSE_ERROR:{e}"

    stmt = detect_statement_type(tree)
    if stmt in FORBIDDEN_STATEMENTS:
        reasons.append(f"FORBIDDEN_STATEMENT:{stmt.upper()}")
    if stmt != "select":
        reasons.append("ONLY_SELECT_ALLOWED")

    tables = extract_tables(tree)
    cols = extract_columns(tree)

    if allow_tables is not None:
        bad_tables = [t for t in tables if t not in allow_tables]
        if bad_tables:
            reasons.append(f"DISALLOWED_TABLES:{','.join(bad_tables)}")

    if allow_columns is not None:
        bad_cols = [c for c in cols if c not in allow_columns]
        if bad_cols:
            reasons.append(f"DISALLOWED_COLUMNS:{','.join(bad_cols)}")

    star = has_select_star(tree)
    if block_select_star and star:
        if not tables or any(t not in STAR_ALLOWLIST for t in tables):
            reasons.append("SELECT_STAR_NOT_ALLOWED")

    if require_limit and not has_limit(tree):
        reasons.append("MISSING_LIMIT")

    if require_where_for_tables:
        if any(t in require_where_for_tables for t in tables) and not has_where(tree):
            reasons.append("MISSING_WHERE_FOR_RESTRICTED_TABLE")

    ast_features = {
        "statement": stmt,
        "tables": tables,
        "columns": cols,
        "join_count": count_joins(tree),
        "has_where": has_where(tree),
        "has_limit": has_limit(tree),
        "has_select_star": star,
    }

    return (len(reasons) == 0), reasons, ast_features, tree, None


# ----------------------------
# Performance check via EXPLAIN
# ----------------------------
def explain_query(con: duckdb.DuckDBPyConnection, sql: str) -> str:
    rows = con.execute(f"EXPLAIN {sql}").fetchall()
    return "\n".join(str(r[0]) for r in rows)


def performance_check(plan_text: str, ast_features: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    plan_upper = plan_text.upper()

    if "CROSS_PRODUCT" in plan_upper or "CROSS JOIN" in plan_upper:
        reasons.append("CROSS_PRODUCT_DETECTED")

    if ast_features.get("join_count", 0) >= 6:
        reasons.append("TOO_MANY_JOINS")

    if "SEQ_SCAN" in plan_upper and not ast_features.get("has_where", False):
        reasons.append("SEQ_SCAN_WITHOUT_FILTER")

    if ("SORT" in plan_upper or "ORDER BY" in plan_upper) and not ast_features.get("has_limit", False):
        reasons.append("SORT_WITHOUT_LIMIT")

    return (len(reasons) == 0), reasons


# ----------------------------
# Orchestrator
# ----------------------------
def validate_sql(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    allow_tables: Optional[Set[str]] = None,
    allow_columns: Optional[Set[str]] = None,
    require_where_for_tables: Optional[Set[str]] = None,
    require_limit: bool = True,
    block_select_star: bool = True,
) -> ValidationResult:
    t0 = time.time()

    is_safe, safety_reasons, ast_features, _tree, safety_err = safety_check(
        sql=sql,
        allow_tables=allow_tables,
        allow_columns=allow_columns,
        require_limit=require_limit,
        require_where_for_tables=require_where_for_tables,
        block_select_star=block_select_star,
    )

    if not is_safe:
        return ValidationResult(
            is_safe=False,
            is_performant=False,
            safety_reasons=safety_reasons,
            performance_reasons=[],
            ast_features=ast_features,
            plan_text=None,
            latency_ms=(time.time() - t0) * 1000,
            error=safety_err,
        )

    plan_text: Optional[str] = None
    perf_ok = False
    perf_reasons: List[str] = []
    err: Optional[str] = None

    try:
        plan_text = explain_query(con, sql)
        perf_ok, perf_reasons = performance_check(plan_text, ast_features)
    except Exception as e:
        perf_reasons = ["EXPLAIN_FAILED"]
        err = f"EXPLAIN_ERROR:{e}"

    return ValidationResult(
        is_safe=True,
        is_performant=perf_ok,
        safety_reasons=safety_reasons,
        performance_reasons=perf_reasons,
        ast_features=ast_features,
        plan_text=plan_text,
        latency_ms=(time.time() - t0) * 1000,
        error=err,
    )


# ----------------------------
# LangChain tool wrapper
# ----------------------------
_con = None


def _get_connection() -> duckdb.DuckDBPyConnection:
    """Lazily initialise a DuckDB connection with parquet views loaded."""
    global _con
    if _con is None:
        _con = connect_duckdb()
        load_parquet_views(_con, PARQUETS)
    return _con


@tool
def validate_sql_query(sql: str) -> str:
    """Validate a DuckDB SQL query for safety and performance issues.
    Call this after generating SQL to confirm it is safe before finalizing.
    Returns a summary of issues found, or confirms the query is valid."""
    con = _get_connection()

    allow_tables = set(PARQUETS.keys())
    schema_map = get_schema_map(con, PARQUETS)
    allow_columns = flatten_schema_map(schema_map)

    result = validate_sql(
        con=con,
        sql=sql,
        allow_tables=allow_tables,
        allow_columns=allow_columns,
        require_limit=False,
        block_select_star=True,
    )

    # Safety failures are blocking — the SQL must be fixed before proceeding.
    if not result.is_safe:
        lines = [f"- {i}" for i in result.safety_reasons]
        # Add LLM hint when concept table is disallowed
        for reason in result.safety_reasons:
            if "DISALLOWED_TABLES:" in reason:
                for table in reason.replace("DISALLOWED_TABLES:", "").split(","):
                    table = table.strip()
                    if table in DISALLOWED_TABLE_HINTS:
                        lines.append(f"- DO NOT USE THE TABLE: {DISALLOWED_TABLE_HINTS[table]}")
                        break
        return (
            "Safety issues found — fix the SQL and call validate_sql_query again:\n"
            + "\n".join(lines)
        )

    tables = result.ast_features.get("tables", [])

    # EXPLAIN failures are non-blocking — the SQL is structurally safe.
    if result.error:
        return (
            f"SQL passed safety checks. Tables referenced: {tables}. "
            "Performance check unavailable — proceed to get_data."
        )

    # Performance warnings are advisory only — allow the LLM to proceed.
    if result.performance_reasons:
        return (
            f"SQL passed safety checks. Tables referenced: {tables}. "
            "Performance warnings (non-blocking):\n"
            + "\n".join(f"- {i}" for i in result.performance_reasons)
            + "\nProceed to get_data."
        )

    return f"SQL is valid. Tables referenced: {tables}. Proceed to get_data."

# Testing
# if __name__ == "__main__":
#     con = _get_connection()

#     allow_tables = set(PARQUETS.keys())
#     schema_map = get_schema_map(con, PARQUETS)
#     allow_columns = flatten_schema_map(schema_map)

#     test_queries = {
#         "query_1": """
#             SELECT COUNT(person_id) AS count
#             FROM (
#                 SELECT *,
#                        trim(split(condition_source_value, '||')[4]) as Histo2
#                 FROM condition_occurrence
#             )
#             WHERE Histo2 ILIKE '%Signet ring%'
#         """,
#         "query_2": """
#             SELECT COUNT(*) as n, race_source_value
#             FROM person
#             GROUP BY race_source_value
#             ORDER BY n DESC
#         """,
#     }

#     for name, sql in test_queries.items():
#         print(f"\n=== {name} ===")
#         result = validate_sql(
#             con=con,
#             sql=sql,
#             allow_tables=allow_tables,
#             allow_columns=allow_columns,
#             require_limit=False,
#             block_select_star=True,
#         )
#         print("SQL:")
#         print(sql)
#         print("is_safe:", result.is_safe)
#         print("safety_reasons:", result.safety_reasons)
#         print("is_performant:", result.is_performant)
#         print("performance_reasons:", result.performance_reasons)
#         print("error:", result.error)
