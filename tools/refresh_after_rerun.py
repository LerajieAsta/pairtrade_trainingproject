#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全量重跑後的快取失效與重算順序
======================================================================

`run_trading.py` 重跑之後，分析層的逐日快取仍裝著**重跑前**的序列。
`daily_returns_mainaxis.parquet` 只以 strategy_id 為鍵、沒有上游指紋，
`load_daily_sids` 看到欄位已存在就直接回傳舊資料——結果是新的形成期配對
配上舊的損益數字，**分析結果與重跑前逐位元相同，而且不報任何錯**。

本腳本把「該失效哪些、不該碰哪些」寫死，取代人工記憶：

  失效：現役策略（本次重跑過的）
  保留：封存策略——其 trade_logs 已於 2026-08-06 清除，逐日序列只存在於
        這份快取裡，刪掉就再也重建不回來（tools/archive_trade_logs.py）

`strategies/returns.py` 的另一份快取走指紋失效，不需要人工處理。

用法：
    python -m tools.refresh_after_rerun --dry-run
    python -m tools.refresh_after_rerun --execute
"""
import argparse
import os
import sqlite3
import sys

import pandas as pd

from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

RESULT_DB = "results/result.db"
CACHE = "results/analysis/daily_returns_mainaxis.parquet"

# 重算順序：命題 1 先，因為它決定其餘敘述怎麼寫（基本面回補後結論可能翻轉）
PIPELINE = [
    ("命題 1 主檢定",      "analysis.proposition1_daily_hac"),
    ("命題 1 粒度掃描",    "analysis.granularity_sweep"),
    ("命題 1 機制因子",    "analysis.proposition1_mechanism"),
    ("命題 1 特徵維度",    "analysis.prop1_feature_dimension"),
    ("命題 1 F09 重驗",    "analysis.prop1_f09_reverify"),
    ("命題 2 主檢定",      "analysis.proposition2_daily_hac"),
    ("命題 2 曝險對照",    "analysis.prop2_exposure_control"),
    ("命題 2 SKIP 置換",   "analysis.prop2_skip_permutation"),
    ("命題 2 標籤資訊",    "analysis.prop2_label_information"),
    ("命題 2 行為解析",    "analysis.drl_behavior"),
    ("組合系統",           "analysis.prop3_combined_system"),
    ("regime / 成本 / DSR", "analysis.regime_cost_dsr_eval"),
]


def main():
    ap = argparse.ArgumentParser(description="重跑後的快取失效")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    from strategies.config import strategies_raw_all
    live = {s["db_method"] for s in strategies_raw_all}

    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    sm = pd.read_sql("SELECT _path, METHOD FROM strategy_summaries", con)
    con.close()
    live_sids = set(sm[sm.METHOD.isin(live)]._path)
    arch_sids = set(sm[~sm.METHOD.isin(live)]._path)

    if not os.path.exists(CACHE):
        print(f"找不到 {CACHE}，無需失效")
        return
    cached = pd.read_parquet(CACHE)
    to_drop = [c for c in cached.columns if c in live_sids]
    protect = [c for c in cached.columns if c in arch_sids]
    other = [c for c in cached.columns
             if c not in live_sids and c not in arch_sids]

    print(f"逐日快取 {len(cached.columns)} 欄")
    print(f"  現役（本次重跑過）→ 失效  {len(to_drop):>5}")
    print(f"  封存（明細已刪）  → 保留  {len(protect):>5}   ← 刪了無法重建")
    print(f"  其他（不在摘要表）→ 保留  {len(other):>5}")

    if args.dry_run:
        print("\n--dry-run：未修改。重算順序：")
        for i, (name, mod) in enumerate(PIPELINE, 1):
            print(f"  {i:>2}. {name:<20} python -m {mod}")
        return

    if to_drop:
        cached.drop(columns=to_drop).to_parquet(CACHE)
        print(f"\n✔ 已移除 {len(to_drop)} 欄，剩 {len(cached.columns) - len(to_drop)} 欄")
    else:
        print("\n無需失效（現役策略尚未進快取）")

    print("\n接著依序重算（命題 1 先——基本面回補後其結論可能翻轉，"
          "而它決定其餘敘述怎麼寫）：")
    for i, (name, mod) in enumerate(PIPELINE, 1):
        print(f"  {i:>2}. {name:<20} python -m {mod}")


if __name__ == "__main__":
    main()
