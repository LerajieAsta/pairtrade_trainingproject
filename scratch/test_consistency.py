import sqlite3
import os
import pandas as pd
import sys

sys.path.append(os.path.abspath("."))
from dashboard import extract_features_from_path, get_sector_mapping

def test():
    db_path = "results/result.db"
    if not os.path.exists(db_path):
        print("Database not found!")
        return

    print("Connecting to database...")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
    cursor = conn.cursor()

    print("Fetching strategy IDs...")
    cursor.execute("SELECT DISTINCT strategy_id FROM strategy_pairs;")
    all_strat_ids = sorted([row[0] for row in cursor.fetchall()])
    print(f"Total strategy IDs: {len(all_strat_ids)}")

    selected_strat = all_strat_ids[0]
    
    # Get sector mapping
    print("Loading sector mapping...")
    sec_map = get_sector_mapping()
    print(f"Loaded {len(sec_map)} mappings.")

    # Section 1 test
    print("\n--- Testing Section 1: Detail Query ---")
    pairs_query = """
    SELECT Ticker_A as "股票 A", Ticker_B as "股票 B", Period_Start as "交易期起點", Period_End as "交易期終點", Hedge_Ratio as "避險比例"
    FROM strategy_pairs
    WHERE strategy_id = ?
    ORDER BY Period_Start DESC
    """
    strat_pairs_df = pd.read_sql_query(pairs_query, conn, params=(selected_strat,))
    
    strat_pairs_df.insert(2, "股票 A 產業", strat_pairs_df["股票 A"].apply(lambda x: sec_map.get(str(x).upper(), "Unknown")))
    strat_pairs_df.insert(3, "股票 B 產業", strat_pairs_df["股票 B"].apply(lambda x: sec_map.get(str(x).upper(), "Unknown")))
    
    print(f"Details DataFrame successfully built with {len(strat_pairs_df)} rows.")
    print("Sample details:")
    print(strat_pairs_df.head(3))

    # Section 2 test
    print("\n--- Testing Section 2: Intersection ---")
    strat_a = all_strat_ids[0]
    cursor.execute("SELECT strategy_id FROM strategy_pairs WHERE Ticker_B = 'BRK-B' LIMIT 1")
    row_with_hyphen = cursor.fetchone()
    strat_b = row_with_hyphen[0] if row_with_hyphen else all_strat_ids[1]

    cursor.execute("SELECT DISTINCT Ticker_A, Ticker_B FROM strategy_pairs WHERE strategy_id = ?;", (strat_a,))
    pairs_a = {(row[0], row[1]) for row in cursor.fetchall()}
    
    cursor.execute("SELECT DISTINCT Ticker_A, Ticker_B FROM strategy_pairs WHERE strategy_id = ?;", (strat_b,))
    pairs_b = {(row[0], row[1]) for row in cursor.fetchall()}

    intersection = pairs_a.intersection(pairs_b)
    print(f"Intersection count: {len(intersection)}")

    if intersection:
        cursor.execute("SELECT Ticker_A, Ticker_B, Period_Start, Period_End FROM strategy_pairs WHERE strategy_id = ?;", (strat_a,))
        rows_a = cursor.fetchall()
        cursor.execute("SELECT Ticker_A, Ticker_B, Period_Start, Period_End FROM strategy_pairs WHERE strategy_id = ?;", (strat_b,))
        rows_b = cursor.fetchall()

        periods_a = {}
        for row in rows_a:
            pair_key = (row[0], row[1])
            if pair_key in intersection:
                periods_a.setdefault(pair_key, []).append(f"{row[2]} ~ {row[3]}")

        periods_b = {}
        for row in rows_b:
            pair_key = (row[0], row[1])
            if pair_key in intersection:
                periods_b.setdefault(pair_key, []).append(f"{row[2]} ~ {row[3]}")

        intersection_rows = []
        for p in sorted(list(intersection)):
            t_a, t_b = p
            sec_a = sec_map.get(t_a.upper(), "Unknown")
            sec_b = sec_map.get(t_b.upper(), "Unknown")
            
            p_list_a = sorted(list(set(periods_a.get(p, []))))
            p_list_b = sorted(list(set(periods_b.get(p, []))))
            
            intersection_rows.append({
                "股票 A": t_a,
                "股票 B": t_b,
                "股票 A 產業": sec_a,
                "股票 B 產業": sec_b,
                "Strategy A 交易期間 (年月日)": ", ".join(p_list_a) if p_list_a else "-",
                "Strategy B 交易期間 (年月日)": ", ".join(p_list_b) if p_list_b else "-"
            })
        intersection_df = pd.DataFrame(intersection_rows)
        print(f"Intersection DataFrame built with {len(intersection_df)} rows.")
        print("Sample intersection:")
        print(intersection_df.head(3))

    conn.close()
    print("\nSimulation test completed successfully!")

if __name__ == "__main__":
    test()
