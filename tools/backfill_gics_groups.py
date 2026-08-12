#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回填 GICS／不分組策略的 formation_groups 紀錄
======================================================================

`cluster_formation` 原本只對 ML 分群寫 `cluster_labels_`；`cluster_method`
為 "gics"（直接用真實產業）或 "none"（全市場單一組）時留空。結果是
**分組層的對照組反而沒有分組層的紀錄**——而命題 1 的核心正是
「ML 分群 vs GICS vs 不分組」，稽核時少了兩邊。

程式碼已修正（gics/none 現在也寫 `cluster_labels_`），但既有的窗口是舊版
跑出來的。重跑形成期不划算：`HSU25 (DTW)` 與 `(SDP)` 各要 19.5 小時，
而它們要記的東西只是「每檔股票屬於哪一組」——那是**確定性的查表**，
不需要分群、不需要 OLS、不需要任何統計檢定。

本腳本因此直接重算該對照：以 `formation_progress` 已記錄的窗口清單為權威
（而非重新枚舉，避免對不上），逐窗口重現 run_formation 的標的過濾：

    1. 取 [Period_Start, Period_Start+252) 的價格切片
    2. 動態成分股過濾（form_end 當日仍在 S&P 500 者）
    3. dropna(axis=1)（任一日缺值即剔除該檔）
    4. gics → 產業名；none → "All_Market"

用法：
    python -m tools.backfill_gics_groups --dry-run
    python -m tools.backfill_gics_groups --execute
"""
import argparse
import os
import sqlite3
import sys
import time

import pandas as pd

from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

FORMATION_DB = "formation_data/formation_pairs_sp500_Tiingo.db"


def targets() -> dict[str, str]:
    """{formation strategy_id: cluster_method}，僅取 gics / none 兩類。"""
    from strategies.config import strategies_raw_all
    out = {}
    for s in strategies_raw_all:
        if s.get("formation_strategy_id_base"):
            continue
        cm = s.get("params", {}).get("cluster_method")
        if cm in ("gics", "none"):
            out[f"{s['name']}_MSR0"] = cm
    return out


def main():
    ap = argparse.ArgumentParser(description="回填 GICS／不分組的 formation_groups")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    tgt = targets()
    print(f"目標策略 {len(tgt)} 條：")
    for k, v in tgt.items():
        print(f"    {k:<34} cluster_method={v}")

    con = sqlite3.connect(FORMATION_DB, timeout=120.0)
    have = pd.read_sql(
        "SELECT strategy_id, COUNT(DISTINCT Period_Start) n FROM formation_progress "
        f"WHERE strategy_id IN ({','.join('?' * len(tgt))}) GROUP BY strategy_id",
        con, params=list(tgt))
    print("\n已跑過的窗口數：")
    print(have.to_string(index=False) if len(have) else "  （無——形成期尚未合併完成）")

    already = pd.read_sql(
        "SELECT strategy_id, COUNT(DISTINCT Period_Start) n FROM formation_groups "
        f"WHERE strategy_id IN ({','.join('?' * len(tgt))}) GROUP BY strategy_id",
        con, params=list(tgt))
    print("\n已有分組紀錄的窗口數：")
    print(already.to_string(index=False) if len(already) else "  （全部缺）")

    if args.dry_run:
        con.close()
        return

    # ── 重現 run_formation 的資料前處理 ──────────────────────────────
    from strategies.preprocess_equity import DataProcessor
    from strategies.config import (DB_PATH, TABLE_NAME, INFO_TABLE, TICKER_COL,
                                   SECTOR_COL, BACKTEST_START, BACKTEST_END,
                                   FORMATION_WINDOW)
    print("\n載入價格與成分股名冊…")
    proc = DataProcessor(db_path=DB_PATH, table_name=TABLE_NAME)
    smap = proc.load_sector_mapping(INFO_TABLE, TICKER_COL, SECTOR_COL)
    price, all_dates, total_days, _ = proc.prepare_backtest_data(
        BACKTEST_START, BACKTEST_END, FORMATION_WINDOW)
    try:
        with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as c:
            mem = pd.read_sql_query(
                "SELECT Symbol, start_date, end_date FROM index_memberships", c)
    except Exception:
        mem = None
    print(f"  {total_days} 個交易日 × {price.shape[1]} 檔")

    idx_of = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(all_dates)}
    look = lambda t: smap.get(t.upper(), smap.get(t, "Unknown"))

    cur = con.cursor()
    t0 = time.time()
    for sid, cm in tgt.items():
        windows = [r[0] for r in cur.execute(
            "SELECT DISTINCT Period_Start FROM formation_progress WHERE strategy_id=?"
            " ORDER BY Period_Start", (sid,)).fetchall()]
        if not windows:
            print(f"  略過 {sid}（無已跑窗口）")
            continue
        cur.execute("DELETE FROM formation_groups WHERE strategy_id=?", (sid,))
        rows = []
        for ps in windows:
            i0 = idx_of.get(str(ps)[:10])
            if i0 is None:
                continue
            i1 = i0 + FORMATION_WINDOW
            form = price.iloc[i0:i1]
            fe = all_dates[min(i1 - 1, total_days - 1)].strftime("%Y-%m-%d")
            if mem is not None and not mem.empty:
                act = mem[(mem.start_date <= fe)
                          & (mem.end_date.isna() | (mem.end_date >= fe))]
                keep = set(act.Symbol.unique())
                form = form[[c for c in form.columns if c in keep]]
            form = form.dropna(axis=1)
            valid = [t for t in form.columns if form[t].notna().sum() >= 30]
            lab = (lambda t: "All_Market") if cm == "none" else look
            rows += [(sid, ps, t, lab(t)) for t in valid]
        cur.executemany(
            "INSERT OR REPLACE INTO formation_groups"
            " (strategy_id, Period_Start, Ticker, Cluster_Label) VALUES (?,?,?,?)",
            rows)
        con.commit()
        print(f"  ✔ {sid:<34} {len(windows):>3} 窗口 → {len(rows):>8,} 列"
              f"  ({time.time() - t0:.0f}s)")
    con.close()
    print("\n完成。")


if __name__ == "__main__":
    main()
