from retrieval.pipeline import handle_question

if __name__ == "__main__":
    while True:
        q = input("Ask (NL): ").strip()
        if not q:
            continue
        out = handle_question(q)
        print("STATUS:", out["status"])
        if "message" in out:
            print("MESSAGE:", out["message"])

        if out.get("status") == "ok" and "value" in out and "metric" in out:
            print(f"{out['metric']}: {out['value']}")
        elif out["status"] == "ok":
            print("COLUMNS:", out["columns"])
            print("ROWS (first 10):", out["rows"][:10])
            print("ROW COUNT:", out["row_count"])
        else:
            print("REASONS:", out.get("reasons"))

