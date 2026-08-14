"""
量測分群層本身的跨期週轉
======================================================================

論文 3.3.2 節指出一項量測空白：配對層級的週轉率無法佐證分群層的動態性
（靜態的 GICS 分組其配對重疊率 11.7%，與三種動態分群同一量級——週轉主要
由排序層與篩選層驅動）。而群標籤未落庫，故分群層自身的週轉幅度未被量測。

本工具補上該量測。作法是重跑形成期的**前兩層**：

    特徵萃取 → 分組          ← 本工具只跑到這裡
    （群內排序 → 統計篩選）    ← 跳過，那才是重運算

`ClusterFormation._build_feature_matrix()` 與 `._cluster()` 可獨立呼叫，
故不需要完整 `run()`，也不寫入任何資料庫。

量測指標
--------
**調整蘭德指數（Adjusted Rand Index, ARI）**——相鄰兩期在「共同存在的標的」
上的分群一致度。ARI = 1 代表兩期分群完全相同（靜態），
ARI = 0 代表與隨機分群無異。這是分群穩定度的標準指標，
且已對「隨機一致」作期望值校正，故不受群數變動影響。

同時報告群數、標的數、被排除比例，以及**共同標的中改變群歸屬的比例**
（較直觀但未校正）。

**量測對象為實際進入配對的分群結構**：HDBSCAN 的噪音點（標籤 −1）與
成員數低於 `min_cluster_size` 的過小群，依 `cluster_formation.run()` 的規則
併入 Unknown 後排除，不參與 ARI 計算。

排除的理由是定義而非方向：噪音標記的語意是「無法歸入任何群」，把它當成
一個群，量到的會是「無法歸類此一狀態的持續性」，而非分群結構的穩定度；
且該桶不進入配對，其穩定與否與配對品質無關。

至於納入噪音會使 ARI 偏高或偏低，取決於兩股相反力量的相對大小——
兩期皆為噪音者被算作「持續同群」而推高，在噪音與實群之間移動者則推低——
方向為實證問題，本工具不對其作假設。

GICS 對照的 ARI 恆為 1.0（產業分類不隨期間改變），故不必實跑；
本工具的意義在於量出三種動態分群距離那個上界有多遠。

用法：
    python tools/measure_cluster_churn.py                 # 三種分群法、全期
    python tools/measure_cluster_churn.py --every 4       # 每 4 期取樣（快速預覽）
    python tools/measure_cluster_churn.py --methods agglomerative
"""

import argparse
import os
import sqlite3
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

warnings.filterwarnings("ignore")

from sklearn.metrics import adjusted_rand_score  # noqa: E402

from strategies.config import (  # noqa: E402
    BACKTEST_END, BACKTEST_START, DB_PATH, FORMATION_WINDOW, FORWARD_DAYS,
    INFO_TABLE, SECTOR_COL, TABLE_NAME, TICKER_COL, base_params, _GRID_COMMON,
)
from strategies.preprocess_equity import DataProcessor  # noqa: E402
from strategies.formation.cluster_formation import Formation  # noqa: E402

# 檔名帶上期距，否則不同 --every 的執行會互相覆蓋（量測的是不同的東西）
OUT_TMPL = "results/analysis/cluster_churn_lag{lag}.csv"


def _periods(price_pivot, all_dates, total_days, first_idx, step):
    """與 run_formation 相同的滾動切分，確保期間定義一致。"""
    for idx in range(first_idx, total_days - FORWARD_DAYS + 1, step):
        s = idx - FORMATION_WINDOW
        yield (all_dates[s].strftime("%Y-%m-%d"),
               all_dates[idx - 1].strftime("%Y-%m-%d"),
               price_pivot.iloc[s:idx])


def _active(form_prices, memberships, form_end):
    """僅保留形成期結束日當時真實屬於 S&P 500 者（同 run_formation）。"""
    if memberships is None or memberships.empty:
        return form_prices.dropna(axis=1)
    act = memberships[(memberships.start_date <= form_end)
                      & (memberships.end_date.isna()
                         | (memberships.end_date >= form_end))]
    cols = [c for c in form_prices.columns if c in set(act.Symbol)]
    return form_prices[cols].dropna(axis=1)


def churn_for(method, price_pivot, all_dates, total_days, first_idx,
              step, sector_mapping, memberships, every):
    rows, prev = [], None
    periods = list(_periods(price_pivot, all_dates, total_days, first_idx, step))
    periods = periods[::every]
    print(f"\n── {method}：{len(periods)} 期 ──")

    for n, (f_start, f_end, prices) in enumerate(periods, 1):
        prices = _active(prices, memberships, f_end)
        if prices.shape[1] < 10:
            prev = None
            continue
        try:
            # 直接沿用主矩陣的 _GRID_COMMON，不逐鍵挑選。
            # 舊寫法從 base_params 撈 pca_n_components 等三個鍵，但那三個鍵位於
            # _GRID_COMMON，故該 dict 恆為空、全部落回 Formation 的預設值——
            # 其中 impute_scope 的預設為 "group"，與主管線的 "global" 不符，
            # 量到的會是另一組特徵上的分群。Formation 以 **kwargs 吸收多餘鍵，
            # 故排序/篩選相關的鍵一併傳入無妨（本工具不呼叫 run()）。
            fm = Formation(price_df=prices, form_start=f_start, form_end=f_end,
                           sector_mapping=sector_mapping, cluster_method=method,
                           **_GRID_COMMON)
            X, tickers = fm._build_feature_matrix()
            if len(tickers) < 10:
                prev = None
                continue
            raw = [int(x) for x in fm._cluster(X)]
        except Exception as e:
            print(f"  ⚠ {f_end} 失敗：{type(e).__name__}: {e}")
            prev = None
            continue

        # 套用與 cluster_formation.run() 相同的排除規則：HDBSCAN 噪音（−1）與
        # 過小群併入 Unknown、不進入配對。本工具繞過 run() 直接呼叫 _cluster()，
        # 若不在此重現該規則，噪音點會被當成一個合法群：n_clusters 多算一個，
        # 且約兩成標的被視為「彼此同群」——量到的會是噪音狀態的持續性而非
        # 分群結構的穩定度（偏誤方向見模組 docstring）。
        sizes = pd.Series(raw).value_counts()
        min_cs = int(_GRID_COMMON.get("min_cluster_size", 5))
        labels = {t: l for t, l in zip(tickers, raw)
                  if l != -1 and sizes[l] >= min_cs}
        if len(labels) < 10:
            prev = None
            continue

        row = {"method": method, "form_end": f_end,
               "n_tickers": len(labels), "n_clusters": len(set(labels.values())),
               "n_excluded": len(tickers) - len(labels),
               "excluded_pct": round(100 * (len(tickers) - len(labels)) / len(tickers), 2)}
        if prev is not None:
            common = sorted(set(prev) & set(labels))
            if len(common) >= 10:
                a = [prev[t] for t in common]
                b = [labels[t] for t in common]
                row["n_common"] = len(common)
                row["ARI"] = round(float(adjusted_rand_score(a, b)), 4)
                # 直觀但未校正的指標：共同標的中，群歸屬「相對關係」改變的比例。
                # 群編號本身無意義（每期重新編），故比對「兩兩是否同群」。
                A = np.equal.outer(a, a)
                B = np.equal.outer(b, b)
                iu = np.triu_indices(len(common), 1)
                row["同群關係改變%"] = round(float((A[iu] != B[iu]).mean() * 100), 2)
        rows.append(row)
        prev = labels
        if n % 25 == 0:
            print(f"  …{n}/{len(periods)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="agglomerative,hdbscan,kmeans")
    ap.add_argument("--every", type=int, default=1,
                    help="每 N 期取樣一次（>1 時 ARI 為 N 期間隔，非相鄰期）")
    a = ap.parse_args()

    print("載入價格與產業對照…")
    proc = DataProcessor(db_path=DB_PATH, table_name=TABLE_NAME)
    sector_mapping = proc.load_sector_mapping(INFO_TABLE, TICKER_COL, SECTOR_COL)
    price_pivot, all_dates, total_days, first_idx = proc.prepare_backtest_data(
        BACKTEST_START, BACKTEST_END, FORMATION_WINDOW)
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as con:
        mem = pd.read_sql("SELECT Symbol, start_date, end_date FROM index_memberships", con)
    mem["start_date"] = pd.to_datetime(mem.start_date).dt.strftime("%Y-%m-%d")
    mem["end_date"] = pd.to_datetime(mem.end_date).dt.strftime("%Y-%m-%d")

    step = int(base_params.get("rolling_step", 21))
    rows = []
    for m in [x.strip() for x in a.methods.split(",") if x.strip()]:
        rows += churn_for(m, price_pivot, all_dates, total_days, first_idx,
                          step, sector_mapping, mem, a.every)

    df = pd.DataFrame(rows)
    out = OUT_TMPL.format(lag=a.every)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print(f"分群層跨期週轉（相隔 {a.every} 期）")
    print("=" * 78)
    g = df.dropna(subset=["ARI"]).groupby("method")
    summary = pd.DataFrame({
        "期數": g.size(),
        "群數(中位)": g["n_clusters"].median(),
        "標的數(中位)": g["n_tickers"].median(),
        "排除%(中位)": g["excluded_pct"].median().round(2),
        "ARI(中位)": g["ARI"].median().round(4),
        "ARI(平均)": g["ARI"].mean().round(4),
        "ARI(10-90分位)": g["ARI"].quantile(.1).round(3).astype(str)
                          + "–" + g["ARI"].quantile(.9).round(3).astype(str),
        "同群關係改變%(中位)": g["同群關係改變%"].median().round(2),
    })
    print(summary.to_string())
    print("\n對照：GICS 靜態產業分類的 ARI 恆為 1.0000（分組不隨期間改變）")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
