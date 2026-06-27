"""
UMAP 隨機性診斷工具
測試不同 random_state 對 HDBSCAN UMAP / MultiScale 配對結果的影響。
使用縮短期間（單一形成期視窗）快速評估。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import pandas as pd
import numpy as np
from itertools import combinations

from strategies.preprocess_equity import DataProcessor

# ── 設定 ─────────────────────────────────────────────────────────────────
DB_PATH       = "./dataset/sp500_yF.db"
TABLE_NAME    = "Daily_Prices"
INFO_TABLE    = "Constituents"
TICKER_COL    = "Symbol"
SECTOR_COL    = "GICS_Sector"

# 縮短：只取一段形成期來測試（2015-01-01 ~ 2015-12-31，約 252 個交易日）
FORM_START    = "2015-01-01"
FORM_END      = "2015-12-31"
TOP_N         = 20
RANDOM_STATES = [0, 42, 123, 999, 2024]

HDBSCAN_COMMON = {
    "hdbscan_min_cluster_size": 5,
    "hdbscan_min_samples":      2,
    "hdbscan_metric":           "euclidean",
    "umap_n_components":        5,
    "umap_n_neighbors":         40,
    "umap_min_dist":            0.01,
    "reduce_method":            "umap",
}

UMAP_PARAMS   = {
    **HDBSCAN_COMMON,
    "adf_pvalue_threshold": 0.01,
    "min_corr":             0.50,
    "min_zero_crossings":   5,
    "hurst_threshold":      0.5,
    "halflife_min":         1.0,
    "halflife_max":         60.0,
    "roll_corr_window":     60,
    "max_beta_diff":        0.8,
    "max_vol_ratio":        3.0,
    "min_adv_ratio":        0.1,
    "use_mom1_filter":      True,
    "feature_mode":         "stats10",
}

MULTISCALE_PARAMS = {
    **HDBSCAN_COMMON,
    "adf_pvalue_threshold": 0.05,
    "adf_sub_pvalue":       0.10,
    "min_corr_mean":        0.50,
    "min_corr_min":         0.10,
    "max_corr_std":         0.30,
    "min_coint_pass_rate":  0.40,
    "max_regime_diff":      0.50,
    "max_vol_ratio_std":    0.80,
    "use_mom1_filter":      True,
}


def canonical(a, b):
    return (min(a, b), max(a, b))


def run_formation(module_name, price_df, sector_mapping, params, random_state):
    import importlib
    import inspect
    mod = importlib.import_module(module_name)
    cls = mod.Formation
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys())

    kwargs = {
        "price_df":       price_df,
        "form_start":     FORM_START,
        "form_end":       FORM_END,
        "top_n":          TOP_N,
        "sector_mapping": sector_mapping,
        "umap_random_state": random_state,
        **params,
    }
    kwargs = {k: v for k, v in kwargs.items() if k in valid}
    result = cls(**kwargs).run()
    if result is None or result.empty:
        return set()
    return {canonical(r["Ticker_A"], r["Ticker_B"]) for _, r in result.iterrows()}


def overlap_rate(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def print_matrix(title, states, results):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    n = len(states)
    # header
    header = f"{'seed':>6}" + "".join(f"{s:>8}" for s in states)
    print(header)
    print("-" * len(header))
    for i, s_i in enumerate(states):
        row = f"{s_i:>6}"
        for j, s_j in enumerate(states):
            if i == j:
                row += f"{'---':>8}"
            else:
                pairs_i = results[s_i]
                pairs_j = results[s_j]
                ov = overlap_rate(pairs_i, pairs_j)
                row += f"{ov:>8.1%}"
        n_pairs = len(results[s_i])
        print(f"{row}   ({n_pairs} pairs)")

    # 平均交叉重疊率
    all_overlaps = []
    for i, j in combinations(range(n), 2):
        ov = overlap_rate(results[states[i]], results[states[j]])
        all_overlaps.append(ov)
    print(f"\n  平均配對重疊率（Jaccard）: {np.mean(all_overlaps):.1%}")
    print(f"  最低重疊率: {np.min(all_overlaps):.1%}  最高: {np.max(all_overlaps):.1%}")


def main():
    print("=" * 60)
    print("  UMAP 隨機性診斷工具")
    print(f"  形成期: {FORM_START} ~ {FORM_END}")
    print(f"  測試 random_state: {RANDOM_STATES}")
    print("=" * 60)

    # 載入資料
    print("\n[1/2] 載入價格資料...")
    processor = DataProcessor(db_path=DB_PATH, table_name=TABLE_NAME)
    sector_mapping = processor.load_sector_mapping(INFO_TABLE, TICKER_COL, SECTOR_COL)

    conn = sqlite3.connect(DB_PATH)
    df_all = pd.read_sql_query(
        f"SELECT Date AS date, Symbol AS ticker, COALESCE(Adj_Close, Close) AS price "
        f"FROM {TABLE_NAME} WHERE date >= '2014-01-01' AND date <= '{FORM_END}' "
        f"ORDER BY date ASC", conn)
    memberships = pd.read_sql_query(
        "SELECT Symbol, start_date, end_date FROM index_memberships", conn)
    conn.close()

    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all["price"] = pd.to_numeric(df_all["price"], errors="coerce")
    df_all.dropna(subset=["price"], inplace=True)
    df_all = df_all[df_all["price"] > 0]

    price_pivot = df_all.pivot_table(index="date", columns="ticker", values="price", aggfunc="last")
    price_pivot = price_pivot.ffill(limit=5)

    # 只保留形成期且在成分股內的股票
    memberships["start_date"] = pd.to_datetime(memberships["start_date"])
    memberships["end_date"] = pd.to_datetime(memberships["end_date"])
    active = memberships[
        (memberships["start_date"] <= FORM_END) &
        ((memberships["end_date"].isna()) | (memberships["end_date"] >= FORM_START))
    ]
    active_symbols = set(active["Symbol"])
    valid_cols = [c for c in price_pivot.columns if c in active_symbols]
    form_prices = price_pivot.loc[FORM_START:FORM_END, valid_cols].dropna(axis=1)
    print(f"  有效成分股: {len(form_prices.columns)} 支")

    print("\n[2/2] 執行各策略 × 各 random_state...\n")

    for strat_name, module, params in [
        ("HDBSCAN UMAP",       "strategies.formation.HDBSCAN_UMAP",       UMAP_PARAMS),
        ("HDBSCAN MultiScale", "strategies.formation.HDBSCAN_MultiScale",  MULTISCALE_PARAMS),
    ]:
        print(f"\n>> {strat_name}")
        results = {}
        for rs in RANDOM_STATES:
            print(f"   random_state={rs} ...", end=" ", flush=True)
            pairs = run_formation(module, form_prices, sector_mapping, params, rs)
            results[rs] = pairs
            print(f"{len(pairs)} pairs")

        print_matrix(strat_name, RANDOM_STATES, results)

    # 額外：兩策略 seed=42 的交集
    print(f"\n{'='*60}")
    print("  Ensemble 交集分析（seed=42，兩策略比較）")
    print(f"{'='*60}")
    umap_42 = run_formation("strategies.formation.HDBSCAN_UMAP", form_prices,
                             sector_mapping, UMAP_PARAMS, 42)
    ms_42   = run_formation("strategies.formation.HDBSCAN_MultiScale", form_prices,
                             sector_mapping, MULTISCALE_PARAMS, 42)
    inter   = umap_42 & ms_42
    union   = umap_42 | ms_42
    print(f"  HDBSCAN UMAP pairs:       {len(umap_42)}")
    print(f"  HDBSCAN MultiScale pairs: {len(ms_42)}")
    print(f"  交集:  {len(inter)} 對  ({len(inter)/TOP_N:.0%} of top_n={TOP_N})")
    print(f"  聯集:  {len(union)} 對")
    print(f"  Jaccard: {len(inter)/len(union):.1%}" if union else "  N/A")
    if inter:
        print(f"  交集配對: {sorted(inter)}")


if __name__ == "__main__":
    main()
