# -*- coding: utf-8 -*-
"""
損益的出場別分解：收斂獲利 vs 期末強制平倉
======================================================================
論文 4.4.1 的來源。策略的全部損益是**兩股量級相近、方向相反的流量之殘差**：
收斂出場（`EXIT`）帶來獲利，期末強制平倉（`PERIOD_END_EXIT`）實現虧損。
本模組把兩股流量分開清點，並計算「淨額 ÷ 收斂獲利」隨分組限制強度的變化。

**為什麼補成模組。** 原本是一次性腳本、未進版本庫，輸出 `_exit_status_raw.csv`
自 2026-08-17 起未再更新；2026-09 的對沖口徑修正之後仍是舊檔，而論文照抄。
與 `forced_close_followup`／`split_half` 是同一類問題（見附錄 B.5.5）。

**效能。** `trade_logs` 有 3.34 億列。不可以寫成單一句
`GROUP BY strategy_id, Status` 的全表掃描——實測逾一小時仍未完成。
改為**逐 strategy_id 查詢**以吃到 `idx_trade_logs_strat`，
582 個基準格約數分鐘。
"""
import os
import re
import sqlite3
import sys
from pathlib import Path as _Path

import pandas as pd

_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULT_DB = "results/result.db"
OUT_DIR = "results/analysis"
_BASELINE_CELL = re.compile(r"TradeLogs_Top\d+_SL\d+_ZWin\d+_MSR\d+\.csv$")

#: 依分組限制強度排列（排除標的比例 0% → 42.0%），與論文 4.1.3 同序。
GROUPS = [("不分組", "NOGRP"), ("GICS 產業", "GICS"), ("HDBSCAN", "HDB"),
          ("Agglomerative", "AGG"), ("K-means", "KM")]
SORTS = ("SSD", "DTW", "SDP")


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    sm = pd.read_sql("SELECT _path, METHOD FROM strategy_summaries", con)
    base = sm[sm._path.map(lambda p: bool(_BASELINE_CELL.search(os.path.basename(p))))]
    print(f"基準格 {len(base)} 個，逐 strategy_id 聚合…", flush=True)

    out = []
    for n, (_, r) in enumerate(base.iterrows(), 1):
        d = pd.read_sql(
            "SELECT Status, COUNT(*) AS n, SUM(Trade_PnL) AS pnl FROM trade_logs "
            "WHERE strategy_id = ? AND Trade_PnL IS NOT NULL GROUP BY Status",
            con, params=(r._path,))
        if d.empty:
            continue
        d["sid"], d["METHOD"] = r._path, r.METHOD
        out.append(d)
        if n % 100 == 0:
            print(f"  … {n}/{len(base)}", flush=True)
    con.close()

    if not out:
        raise SystemExit("trade_logs 無明細——該批格可能已被 archive_trade_logs 清除")
    d = pd.concat(out, ignore_index=True)
    d.to_csv(f"{OUT_DIR}/_exit_status_raw.csv", index=False, encoding="utf-8-sig")
    print(f"\n出場別：{d.Status.value_counts().to_dict()}")

    rows = []
    for gn, g in GROUPS:
        ms = [f"Grid ({g}-{s})" for s in SORTS]
        x = d[d.METHOD.isin(ms)]
        if x.empty:
            continue
        conv = float(x[x.Status == "EXIT"].pnl.sum())
        pe = float(x[x.Status == "PERIOD_END_EXIT"].pnl.sum())
        tot = float(x.pnl.sum())
        rows.append({"分組": gn, "收斂獲利": round(conv), "期末強平": round(pe),
                     "淨額": round(tot),
                     "淨額÷收斂%": round(tot / conv * 100, 1) if conv else float("nan")})
    summ = pd.DataFrame(rows)
    print("\n== 淨額 ÷ 收斂獲利（依分組限制強度排列）==")
    print(summ.to_string(index=False))

    mono = summ["淨額÷收斂%"].is_monotonic_decreasing
    print(f"\n是否隨限制強度單調下降：{'是' if mono else '否（有反例，僅為趨勢）'}")

    # 論文 4.4.1 引用的個案：GICS-SSD 單臂 15 格平均
    x = d[d.METHOD == "Grid (GICS-SSD)"]
    if not x.empty:
        cells = x.sid.nunique()
        print(f"\n== GICS-SSD 單臂（{cells} 格平均）==")
        for st, g in x.groupby("Status"):
            print(f"  {st:18s} 筆數 {g.n.sum() / cells:9.1f}   損益 {g.pnl.sum() / cells:10.1f}")
        print(f"  {'合計':18s} {'':9}   損益 {x.pnl.sum() / cells:10.1f}")

    summ.to_csv(f"{OUT_DIR}/exit_status_decomp.csv", index=False, encoding="utf-8-sig")
    print(f"\n→ {OUT_DIR}/exit_status_decomp.csv、{OUT_DIR}/_exit_status_raw.csv")


if __name__ == "__main__":
    run()
