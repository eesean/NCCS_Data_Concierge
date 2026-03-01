# Evaluate similarity between ground-truth SQL result set and LLM SQL result set using precision, recall, and f1 score

from __future__ import annotations
from typing import Any, List, Tuple, Optional
import itertools


# -----------------------------
# Helpers
# -----------------------------

def _clean(sql: str) -> str:
    return sql.strip().rstrip(";\n\t ")


def _fetch_rows(con, sql: str) -> Tuple[List[str], List[Tuple]]:
    """
    Execute SQL and return (column_names, rows_as_tuples).
    """
    sql = _clean(sql)
    cur = con.execute(sql)
    cols = [d[0] for d in (cur.description or [])]
    rows = cur.fetchall()
    return cols, rows


def _norm(v):
    """
    Normalize cell values for comparison.
    - Distinguish NULL
    - Apply float tolerance
    """
    if v is None:
        return "∅"

    if isinstance(v, float):
        return round(v, 3)  # adjust precision if needed

    return v


def _print_table(title: str, cols: List[str], rows: List[Tuple], max_rows: int = 10):
    print(f"\n--- {title} ---")
    print("Columns:", cols)

    if not rows:
        print("(no rows)")
        return

    for r in rows[:max_rows]:
        print(r)

    if len(rows) > max_rows:
        print(f"... ({len(rows)} rows total)")


# -----------------------------
# Column alignment logic
# To ensure that comparison is column-order invariant
# -----------------------------

def _col_fingerprint(rows: List[Tuple], col_idx: int) -> Tuple[str, ...]:
    """
    Fingerprint of a column based on its multiset of values (order-insensitive).
    """
    vals = [_norm(r[col_idx]) for r in rows]
    vals.sort()
    return tuple(vals)

#Unused for now, was used previously. Left here just in case requirements change again in future.
def _best_column_permutation(gold_rows: List[Tuple], pred_rows: List[Tuple], k: int) -> Tuple[int, ...]:
    """
    Find permutation p of [0..k-1] such that pred columns reordered by p
    best match gold columns by value fingerprints.
    """
    gold_fps = [_col_fingerprint(gold_rows, i) for i in range(k)]
    pred_fps = [_col_fingerprint(pred_rows, j) for j in range(k)]

    best_p = tuple(range(k))
    best_score = -1

    for p in itertools.permutations(range(k)):
        score = 0
        for i in range(k):
            if gold_fps[i] == pred_fps[p[i]]:
                score += 1
        if score > best_score:
            best_score = score
            best_p = p
            if best_score == k:
                break  # perfect alignment found

    return best_p


def _reorder_rows(rows: List[Tuple], perm: Tuple[int, ...]) -> List[Tuple]:
    return [tuple(r[j] for j in perm) for r in rows]


# -----------------------------
# Multiset row comparison
# -----------------------------

def _multiset_counts(rows: List[Tuple]) -> dict:
    counts = {}
    for r in rows:
        counts[r] = counts.get(r, 0) + 1
    return counts


def _multiset_diff_count(a: List[Tuple], b: List[Tuple]) -> int:
    """
    Count rows in A not present in B (duplicates respected).
    """
    ca = _multiset_counts(a)
    cb = _multiset_counts(b)
    diff = 0
    for row, n in ca.items():
        diff += max(0, n - cb.get(row, 0))
    return diff


# -----------------------------
# Main scoring function
# -----------------------------

from typing import Any, Tuple

def compare_results_f1(
    executor: Any,
    gold_sql: str,
    pred_sql: str,
    debug: bool = False
) -> Tuple[float, float, float]:
    """
    Comparison that is:
      - row-order invariant
      - column-alias invariant
      - column-order invariant
      - duplicate aware

    Returns:
        (precision, recall, f1) — always numeric in [0,1]
    """

    con = executor.con

    try:
        gold_cols, gold_rows = _fetch_rows(con, gold_sql)
        pred_cols, pred_rows = _fetch_rows(con, pred_sql)

        if debug:
            _print_table("Gold Result (raw)", gold_cols, gold_rows)
            _print_table("Pred Result (raw)", pred_cols, pred_rows)

        # -------------------------------------------------
        # Perfect empty match
        # -------------------------------------------------
        if len(gold_rows) == 0 and len(pred_rows) == 0:
            return 1.0, 1.0, 1.0

        # -------------------------------------------------
        # Column mismatch
        # -------------------------------------------------
        if gold_rows and pred_rows:
            k_gold = len(gold_rows[0])
            k_pred = len(pred_rows[0])

            # If prediction has fewer columns than gold → fail
            if k_pred < k_gold:
                if debug:
                    print("\nPrediction missing required columns.")
                return 0.0, 0.0, 0.0

        # -------------------------------------------------
        # Align columns (if both non-empty)
        # -------------------------------------------------
        if gold_rows and pred_rows:
            k_gold = len(gold_rows[0])
            k_pred = len(pred_rows[0])

            gold_fps = [_col_fingerprint(gold_rows, i) for i in range(k_gold)]
            pred_fps = [_col_fingerprint(pred_rows, j) for j in range(k_pred)]

            matched_indices = []
            used_pred_cols = set()

            for g_fp in gold_fps:
                match = None
                for j, p_fp in enumerate(pred_fps):
                    if j in used_pred_cols:
                        continue
                    if set(g_fp).issubset(set(p_fp)) or set(p_fp).issubset(set(g_fp)):
                        match = j
                        break

                if match is None:
                    if debug:
                        print("\nCould not match gold column in prediction.")
                    return 0.0, 0.0, 0.0

                matched_indices.append(match)
                used_pred_cols.add(match)

            # Project prediction onto matched gold columns
            pred_rows = [
                tuple(row[j] for j in matched_indices)
                for row in pred_rows
            ]

            if debug:
                aligned_cols = [pred_cols[j] for j in matched_indices]
                _print_table("Pred Result (aligned to gold)", aligned_cols, pred_rows)

        gold_n = len(gold_rows)
        pred_n = len(pred_rows)

        # -------------------------------------------------
        # Compute TP / FP / FN
        # -------------------------------------------------
        missing_n = _multiset_diff_count(gold_rows, pred_rows)   # FN
        extra_n = _multiset_diff_count(pred_rows, gold_rows)     # FP

        tp = gold_n - missing_n
        fp = extra_n
        fn = missing_n

        # -------------------------------------------------
        # Precision (zero-division safe)
        # -------------------------------------------------
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # -------------------------------------------------
        # Recall (zero-division safe)
        # -------------------------------------------------
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # -------------------------------------------------
        # F1
        # -------------------------------------------------
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return precision, recall, f1

    except Exception as e:
        if debug:
            print("\nExecution error:", e)
        return 0.0, 0.0, 0.0

# -----------------------------
# Tests
# -----------------------------
if __name__ == "__main__":

    from queryexecutor import QueryExecutor

    print("Initializing QueryExecutor...\n")
    qe = QueryExecutor(max_rows=100000)

    # 1) Scalar count changes (F1 should be 0.0 because the single row differs)
    gold_1 = """
    SELECT COUNT(*) AS death_count 
    FROM death 
    WHERE YEAR(death_date) = 2020 
    LIMIT 1;
    """

    pred_1 = """
    SELECT COUNT(*) AS death_count 
    FROM death 
    WHERE YEAR(death_date) = 2019
    LIMIT 1;
    """

    # 2) Same query but fewer rows via LIMIT (pred is a subset of gold -> recall < 1, precision ~ 1)
    # Good to test partial-credit behavior.
    gold_2 = """
    SELECT race_source_value, COUNT(*) AS race_count 
    FROM person 
    GROUP BY race_source_value 
    ORDER BY race_count DESC LIMIT 5;
    """

    pred_2 = """
    SELECT race_source_value, COUNT(*) AS race_count 
    FROM person 
    GROUP BY race_source_value 
    ORDER BY race_count DESC LIMIT 3;
    """

    # 3) Filter changed so results differ (often becomes empty or different counts -> score likely 0)
    gold_3 = """
    SELECT gender_source_value, COUNT(*) AS gender_count 
    FROM person 
    LEFT JOIN condition_occurrence USING (person_id) 
    WHERE ICD10 = 'C18.7' AND YEAR(condition_start_date) = 2021 
    GROUP BY gender_source_value;
    """

    pred_3 = """
    SELECT gender_source_value, COUNT(*) AS gender_count 
    FROM person 
    LEFT JOIN condition_occurrence USING (person_id) 
    WHERE ICD10 = 'C18.7' AND YEAR(condition_start_date) = 2020
    GROUP BY gender_source_value;
    """

    # 4) Scalar count changes by tweaking one condition (F1 should be 0.0)
    gold_4 = """
    SELECT COUNT(*) AS count 
    FROM drug_exposure_cancerdrugs 
    WHERE YEAR(drug_exposure_start_date) = 2020 
    AND YEAR(drug_exposure_end_date) = 2020;
    """

    pred_4 = """
    SELECT COUNT(*) AS count 
    FROM drug_exposure_cancerdrugs 
    WHERE YEAR(drug_exposure_start_date) = 2020 
    AND YEAR(drug_exposure_end_date) = 2019;
    """

    # 5) Same query, F1 should be 1.0
    gold_5 = """
    SELECT COUNT(*) AS count 
    FROM condition_occurrence 
    WHERE condition_source_value LIKE '%colon%';
    """

    pred_5 = """
    SELECT COUNT(*) AS count 
    FROM condition_occurrence 
    WHERE condition_source_value LIKE '%colon%';
    """

    # 6) Scalar count changes by changing gender filter (F1 should be 0.0)
    gold_6 = """
    SELECT COUNT(*) AS male_count 
    FROM person 
    JOIN condition_occurrence USING (person_id) 
    WHERE person.gender_source_value = 'male' 
    AND condition_occurrence.condition_source_value LIKE '%rectum%';
    """

    pred_6 = """
    SELECT COUNT(*) AS male_count 
    FROM person 
    JOIN condition_occurrence USING (person_id) 
    WHERE person.gender_source_value = 'female'
    AND condition_occurrence.condition_source_value LIKE '%rectum%';
    """

    # 7) Extra column returned, but aggregation is correct
    gold_7 = """
    SELECT COUNT(*) AS male_count 
    FROM person JOIN condition_occurrence USING (person_id) 
    WHERE person.gender_source_value = 'male' 
    AND condition_occurrence.condition_source_value LIKE '%rectum%';
    """

    pred_7 = """
    SELECT gender_source_value, COUNT(*) AS male_count 
    FROM person JOIN condition_occurrence USING (person_id) 
    WHERE person.gender_source_value = 'male' 
    AND condition_occurrence.condition_source_value LIKE '%rectum%'
    group by 1;
    """

    # 8) Scalar count changes by adding extra constraint (F1 should be 0.0)
    gold_8 = """
    SELECT COUNT(*) AS total_records
    FROM drug_exposure_cancerdrugs LEFT JOIN person USING (person_id) LEFT JOIN death USING (person_id)
    WHERE death_date IS NOT NULL AND YEAR(NOW()) - year_of_birth >= 55;
    """

    pred_8 = """
    SELECT COUNT(*) AS total_records
    FROM drug_exposure_cancerdrugs LEFT JOIN person USING (person_id) LEFT JOIN death USING (person_id)
    WHERE death_date IS NOT NULL AND YEAR(NOW()) - year_of_birth >= 55
    AND person.gender_source_value = 'male';
    """

    # 9) Extra rows predicted by increasing LIMIT (gold 1 row, pred 2 rows)
    # This should produce a F1 ~0.6667.
    gold_9 = """
    SELECT cause_source_value, COUNT(*) AS death_count 
    FROM death 
    GROUP BY cause_source_value 
    ORDER BY death_count DESC LIMIT 1;
    """

    pred_9 = """
    SELECT cause_source_value, COUNT(*) AS death_count 
    FROM death 
    GROUP BY cause_source_value 
    ORDER BY death_count DESC LIMIT 2;
    """

    # 9) Gold set has non-zero rows, while pred set returns 0 rows
    # This should produce precision, recall, and f1 = 0.0
    gold_10 = """
    SELECT race_source_value, COUNT(*) AS race_count
    FROM person
    GROUP BY race_source_value
    LIMIT 3;
    """

    pred_10 = """
    SELECT race_source_value, COUNT(*) AS race_count
    FROM person
    WHERE 1=0
    GROUP BY race_source_value
    """

    # 11) Columns in different order but same rows
    # This should produce precision, recall, and f1 = 1.0
    gold_11 = """
    SELECT gender_source_value, race_source_value, COUNT(*) AS total_count
    FROM person
    GROUP BY 1,2
    LIMIT 20;
    """

    pred_11 = """
    SELECT race_source_value, gender_source_value, COUNT(*) AS total_count
    FROM person
    GROUP BY 1,2
    LIMIT 20;
    """

    # 12) 2 extra columns returned, but aggregation is correct
    gold_12 = """
    SELECT COUNT(*) AS male_count 
    FROM person JOIN condition_occurrence USING (person_id) 
    WHERE person.gender_source_value = 'male' 
    AND condition_occurrence.condition_source_value LIKE '%rectum%';
    """

    pred_12 = """
    SELECT gender_source_value, gender_concept_id, COUNT(*) AS male_count 
    FROM person JOIN condition_occurrence USING (person_id) 
    WHERE person.gender_source_value = 'male' 
    AND condition_occurrence.condition_source_value LIKE '%rectum%'
    group by 1,2;
    """

    # 13) Duplicate-aware test: prediction has extra duplicate rows
    gold_13 = """
    SELECT gender_source_value
    FROM person
    WHERE gender_source_value = 'male'
    LIMIT 2;
    """

    pred_13 = """
    SELECT gender_source_value
    FROM person
    WHERE gender_source_value = 'male'
    LIMIT 3;
    """

    tests = [
        ("Test 1: Scalar count changed (TC1); Expected F1 = 0.0", gold_1, pred_1),
        ("Test 2: Fewer rows via LIMIT (TC2); Expected F1 > 0.0 and < 1.0", gold_2, pred_2),
        ("Test 3: Changed year filter (TC3); Expected F1 = 0.0", gold_3, pred_3),
        ("Test 4: Scalar count changed (TC4); Expected F1 = 0.0", gold_4, pred_4),
        ("Test 5: Same query (TC5); Expected F1 = 1.0", gold_5, pred_5),
        ("Test 6: Scalar count changed (TC6); Expected F1 = 0.0", gold_6, pred_6),
        ("Test 7: Identical baseline, but additional column (TC7); Expected F1 = 1.0", gold_7, pred_7),
        ("Test 8: Extra constraint added (TC8); Expected F1 = 0.0", gold_8, pred_8),
        ("Test 9: Extra rows via LIMIT (TC9); Expected F1 > 0.0 and < 1.0", gold_9, pred_9),
        ("Test 10: No rows returned in pred set; Expected F1 = 0.0", gold_10, pred_10),
        ("Test 11: Columns in different order but same rows; Expected F1 = 1.0", gold_11, pred_11),
        ("Test 12: 2 extra columns returned, but aggregation is correct; Expected F1 = 1.0", gold_12, pred_12),
        ("Test 13: Duplicate-aware test: prediction has extra duplicate rows; Expected F1 = ~0.8", gold_13, pred_13)
    ]

    for title, gold_sql, pred_sql in tests:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

        print("\nRunning execution comparison...\n")

        score = compare_results_f1(qe, gold_sql, pred_sql, debug=True)

        precision, recall, f1 = score
        print("\nPrecision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1:", round(f1, 4))

    print("\nAll tests completed.")

