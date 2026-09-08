# -*- coding: utf-8 -*-
"""B：`CONCURRENT_PERIODS` 寫死 —— 讀端與引擎的並行期數不一致。

唯讀。用法：
    python dev/trading_arch/measure_concurrency_bug.py

## 錯誤

引擎逐策略推導（`run_trading.py`）：`_concurrent = trading_window // rolling_step`。
讀端在 2026-08-28 之前一律寫死全域 6（`db_utils.py:327`、`metrics.py:221`），
且 `metrics.traded_notional` 把窗長放成**預設參數**，兩個呼叫端都沒傳。

中招的兩支臂（形成期 DB 的實際窗長已核對過，見本檔末的重疊統計）：

    Grid (HAN4-MONTHLY)     trading_window=21, rolling_step=21 → 1 期
    Grid (NOGRP-DTW-TW63)   trading_window=63, rolling_step=21 → 3 期

## 修正後

`config.concurrent_periods()` 為唯一擁有者；引擎把實際值寫進
`strategy_summaries.Concurrent_Periods`；讀端經 `metrics.concurrent_of()`
取該權威值，取不到就拋錯而非猜。1,392 列已由
`tools/backfill_concurrency.py` 回填，其中 30 列數值有變動。

本腳本以修正前的備份 CSV 對照修正後的資料庫，故永遠可重現這次的位移。
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from strategies.config import INITIAL_CAPITAL  # noqa: E402
from strategies.metrics import CURRENT_FEE_SIDE, traded_notional  # noqa: E402

RESULT_DB = "results/result.db"
BACKUP = "results/analysis/strategy_summaries_backup_20260828.csv"
AFFECTED = {"Grid (HAN4-MONTHLY)": 1, "Grid (NOGRP-DTW-TW63)": 3}
LEGACY_ASSUMED = 6


def main() -> None:
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    try:
        now = pd.read_sql(
            'SELECT _path, METHOD, "TOP N", Avg_Utilization, Ann_Ret_Raw, '
            'Ann_Ret_Employed, Final_Equity, "Concurrent_Periods" '
            "FROM strategy_summaries WHERE METHOD IN (?, ?) ORDER BY METHOD, _path",
            con, params=tuple(AFFECTED))
    finally:
        con.close()

    if now.empty:
        print("result.db 內找不到受影響的臂")
        return

    old = None
    if os.path.exists(BACKUP):
        old = (pd.read_csv(BACKUP)
               .set_index("_path")[["Avg_Utilization", "Ann_Ret_Employed"]])

    print("受影響的臂與其真實並行期數：")
    for m, c in AFFECTED.items():
        print(f"  · {m:<24} {c} 期（讀端修正前一律當成 {LEGACY_ASSUMED} 期）")

    print(f"\n{'METHOD':<24}{'TOP N':>7}{'conc':>5}"
          f"{'Util 修正前':>12}{'Util 修正後':>12}"
          f"{'Emp 修正前':>12}{'Emp 修正後':>12}")
    for _, r in now.iterrows():
        u_old = e_old = np.nan
        if old is not None and r["_path"] in old.index:
            u_old = old.loc[r["_path"], "Avg_Utilization"]
            e_old = old.loc[r["_path"], "Ann_Ret_Employed"]
        print(f"{r['METHOD']:<24}{r['TOP N']:>7}{int(r['Concurrent_Periods']):>5}"
              f"{u_old:>12.4f}{r['Avg_Utilization']:>12.4f}"
              f"{e_old:>12.4f}{r['Ann_Ret_Employed']:>12.4f}")

    print("\n名目額與 break-even（同一批交易，只換並行期數口徑）：")
    print(f"{'METHOD':<24}{'TOP N':>7}"
          f"{'notional 錯':>14}{'notional 對':>14}{'BE% 錯':>9}{'BE% 對':>9}")
    for _, r in now.iterrows():
        top_n = int(str(r["TOP N"]).replace("Top", "").strip())
        conc = int(r["Concurrent_Periods"])
        n_bad = traded_notional(r["_path"], top_n, n_concurrent=LEGACY_ASSUMED)
        n_ok = traded_notional(r["_path"], top_n, n_concurrent=conc)
        if not np.isfinite(n_bad) or n_bad <= 0:
            continue
        net = float(r["Final_Equity"]) - INITIAL_CAPITAL
        be_bad = 2.0 * (CURRENT_FEE_SIDE + net / n_bad) * 100
        be_ok = 2.0 * (CURRENT_FEE_SIDE + net / n_ok) * 100
        print(f"{r['METHOD']:<24}{r['TOP N']:>7}"
              f"{n_bad:>14,.0f}{n_ok:>14,.0f}{be_bad:>9.4f}{be_ok:>9.4f}")

    print("\n註：Avg_Utilization 略超 100% 是期界對齊的短暫重疊（HAN4 於 6,287 個")
    print("    交易日中有 13 日同時跑 2 期；TW63 有 13 日跑到 4 期），非分母錯誤。")


if __name__ == "__main__":
    main()
