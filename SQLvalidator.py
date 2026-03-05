from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import time

import duckdb
import sqlglot
from sqlglot import exp
import sys

# Config: data paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

PARQUETS: Dict[str, Path] = {
    "person": DATA_DIR / "person.parquet",
    "condition_occurrence": DATA_DIR / "condition_occurrence.parquet",
    "procedure_occurrence": DATA_DIR / "procedure_occurrence.parquet",
    "drug_exposure_cancerdrugs": DATA_DIR / "drug_exposure_cancerdrugs.parquet",
    "measurement_mutation": DATA_DIR / "measurement_mutation.parquet",
    "death": DATA_DIR / "death.parquet",
}

DUCKDB_DB_PATH = DATA_DIR / "nccs_cap26.db"  


# Result structure
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

# Load data into DuckDB
def connect_duckdb(use_db_file: bool = False) -> duckdb.DuckDBPyConnection:
    """
    If use_db_file=True, connects to ./data/nccs_cap26.db
    Else uses in-memory.
    """
    if use_db_file:
        return duckdb.connect(DUCKDB_DB_PATH.as_posix())
    return duckdb.connect(database=":memory:")


def load_parquet_views(con, parquet_map):
    # 1) Register the encryption key (name can be anything; use something readable)
    con.execute("PRAGMA add_parquet_key('nccs_key', 'MachuPicchu&K2BC');")

    # 2) Create views using encryption_config
    for table, path in parquet_map.items():
        con.execute(f"""
            CREATE OR REPLACE VIEW {table} AS
            SELECT * FROM read_parquet(
                '{path.as_posix()}',
                encryption_config = {{footer_key: 'nccs_key'}}
            );
        """)

# SQLGlot: parsing + safety + features
FORBIDDEN_STATEMENTS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "copy",
    "attach",
    "export",
}

# If you want to allow SELECT * on some small tables, add them here
STAR_ALLOWLIST: Set[str] = set()


def parse_sql(sql: str) -> exp.Expression:
    # DuckDB dialect is supported; fallback to "postgres" sometimes also works
    return sqlglot.parse_one(sql, read="duckdb")


def detect_statement_type(tree: exp.Expression) -> str:
    return tree.key.lower()


def extract_tables(tree: exp.Expression) -> List[str]:
    tables = []
    for t in tree.find_all(exp.Table):
        tables.append(t.name)
    return sorted(set(tables))


def extract_columns(tree: exp.Expression) -> List[str]:
    cols = []
    for c in tree.find_all(exp.Column):
        cols.append(c.name)
    return sorted(set(cols))


def count_joins(tree: exp.Expression) -> int:
    return len(list(tree.find_all(exp.Join)))


def has_where(tree: exp.Expression) -> bool:
    return tree.args.get("where") is not None


def has_limit(tree: exp.Expression) -> bool:
    return tree.args.get("limit") is not None


def has_select_star(tree: exp.Expression) -> bool:
    # Detect SELECT * or SELECT table.*
    for sel in tree.find_all(exp.Select):
        for proj in sel.expressions:
            if isinstance(proj, exp.Star):
                return True
            if isinstance(proj, exp.Column) and isinstance(proj.this, exp.Star):
                return True
    return False


def safety_check(
    sql: str,
    allow_tables: Optional[Set[str]] = None,
    allow_columns: Optional[Set[str]] = None,
    require_limit: bool = True,
    require_where_for_tables: Optional[Set[str]] = None,
    block_select_star: bool = True,
) -> Tuple[bool, List[str], Dict[str, Any], Optional[exp.Expression], Optional[str]]:
    """
    Returns:
      ok, reasons, ast_features, tree, error
    """
    reasons: List[str] = []

    # Simple multi-statement guard
    # (still parse AST after, but this is a fast early catch)
    if sql.strip().count(";") > 1:
        return False, ["MULTI_STATEMENT_NOT_ALLOWED"], {}, None, None

    try:
        tree = parse_sql(sql)
    except Exception as e:
        return False, ["PARSE_ERROR"], {}, None, f"PARSE_ERROR:{e}"

    stmt = detect_statement_type(tree)
    if stmt in FORBIDDEN_STATEMENTS:
        reasons.append(f"FORBIDDEN_STATEMENT:{stmt.upper()}")

    # MVP: only allow SELECT
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
        # Allow SELECT * only if all referenced tables are allowlisted for star
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


# DuckDB: performance checks via EXPLAIN
def explain_query(con: duckdb.DuckDBPyConnection, sql: str) -> str:
    """
    DuckDB EXPLAIN returns a table. We join all rows into a single plan string.
    """
    rows = con.execute(f"EXPLAIN {sql}").fetchall()
    return "\n".join(str(r[0]) for r in rows)


def performance_check(plan_text: str, ast_features: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Heuristic performance rules (MVP).
    You should tune these based on your dataset and what you consider "too expensive".
    """
    reasons: List[str] = []
    plan_upper = plan_text.upper()

    # Cartesian joins / cross products
    if "CROSS_PRODUCT" in plan_upper or "CROSS JOIN" in plan_upper:
        reasons.append("CROSS_PRODUCT_DETECTED")

    # Many joins often means expensive
    if ast_features.get("join_count", 0) >= 6:
        reasons.append("TOO_MANY_JOINS")

    # Sequential scan without filters (cheap heuristic)
    if "SEQ_SCAN" in plan_upper and not ast_features.get("has_where", False):
        reasons.append("SEQ_SCAN_WITHOUT_FILTER")

    # Sorting without limit can be expensive
    if ("SORT" in plan_upper or "ORDER BY" in plan_upper) and not ast_features.get("has_limit", False):
        reasons.append("SORT_WITHOUT_LIMIT")

    return (len(reasons) == 0), reasons

# Orchestrator
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

    plan_text: Optional[str] = None
    perf_ok = False
    perf_reasons: List[str] = []
    err: Optional[str] = safety_err

    # If unsafe, stop early
    if not is_safe:
        latency_ms = (time.time() - t0) * 1000
        return ValidationResult(
            is_safe=False,
            is_performant=False,
            safety_reasons=safety_reasons,
            performance_reasons=[],
            ast_features=ast_features,
            plan_text=None,
            latency_ms=latency_ms,
            error=err,
        )

    # Performance stage
    try:
        plan_text = explain_query(con, sql)
        perf_ok, perf_reasons = performance_check(plan_text, ast_features)
    except Exception as e:
        perf_ok = False
        perf_reasons = ["EXPLAIN_FAILED"]
        err = f"EXPLAIN_ERROR:{e}"

    latency_ms = (time.time() - t0) * 1000

    return ValidationResult(
        is_safe=True,
        is_performant=perf_ok,
        safety_reasons=safety_reasons,
        performance_reasons=perf_reasons,
        ast_features=ast_features,
        plan_text=plan_text,
        latency_ms=latency_ms,
        error=err,
    )


# Main: run a few tests
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python SQLvalidator.py \"<SQL>\"")
        return

    sql = sys.argv[1]

    con = connect_duckdb(use_db_file=False)
    load_parquet_views(con, PARQUETS)

    allow_tables = set(PARQUETS.keys())
    restricted = {"condition_occurrence", "drug_exposure_cancerdrugs", "measurement_mutation"}

    res = validate_sql(
        con=con,
        sql=sql,
        allow_tables=allow_tables,
        allow_columns=None,
        require_where_for_tables=restricted,
        require_limit=True,
        block_select_star=True,
    )

    print(asdict(res))

if __name__ == "__main__":
    main()
