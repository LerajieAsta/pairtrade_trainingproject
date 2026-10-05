# -*- coding: utf-8 -*-
"""不分組 × DL-THR 的接線驗證 V2／V3（PREREGISTRATION.md §八；V1 為設定比對，另行執行）。

試跑 `Grid NOGRP-DTW DRL` 的 Top1/SL0 一格至試跑庫 results/pilot_nogrp_dlthr.db，**不碰** result.db：
  V2. 逐期讀到的配對與 Z-Score 對照組（result.db 的 NOGRP-DTW Top1/SL0）相同，交易期日期集合相同
  V3. SKIP 比例（整期 Status 皆為 HOLD_CASH (SKIP) 的配對期占比）落在既有五條 DL-THR 臂同一格的範圍附近，
      且不為 0% 或 100%

    python dev/nogrp_dlthr/pilot.py
"""
import copy
import os
import sqlite3
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import pandas as pd  # noqa: E402

import run_trading as RT  # noqa: E402
from strategies import config as C  # noqa: E402

PILOT_DB = "results/pilot_nogrp_dlthr.db"
ARM = "Grid NOGRP-DTW DRL"
LIST_KEYS = ("top_n_list", "stop_loss_list", "stop_loss_pct_list", "max_sector_ratio_list",
             "entry_z_list", "dynamic_stop_z_list", "max_holding_days_list",
             "exec_lag_list", "exit_z_list", "max_active_list")
EXISTING = ("GICS_SSD", "GICS_SDP", "HDB_SDP", "KM_SSD", "AGG_SSD")
SKIP_SQL = ("SELECT Period_Start, Ticker_A, Ticker_B, "
            "MIN(CASE WHEN Status = 'HOLD_CASH (SKIP)' THEN 1 ELSE 0 END) AS skip "
            "FROM trade_logs WHERE strategy_id = ? GROUP BY Period_Start, Ticker_A, Ticker_B")


def skip_share(db, path):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
    d = pd.read_sql(SKIP_SQL, con, params=(path,))
    con.close()
    return float(d.skip.mean()), len(d)


def main():
    raw = next(s for s in C.strategies_raw_all if s["name"] == ARM)
    p = copy.deepcopy(raw["params"])
    for key in LIST_KEYS:
        p.pop(key, None)
    p.update(top_n=1, stop_loss_pct=0.0, max_sector_ratio=0.0)
    cfg = {"name": f"{ARM}_Top1_SL0_MSR0", "trading_module": raw["trading_module"],
           "sub_dir": raw["sub_dir"], "db_method": raw["db_method"], "trade_method": "DRL",
           "params": p, "formation_strategy_id": f"{raw['formation_strategy_id_base']}_MSR0"}

    proc = RT.DataProcessor(db_path=C.DB_PATH, table_name=C.TABLE_NAME)
    sector = proc.load_sector_mapping(C.INFO_TABLE, C.TICKER_COL, C.SECTOR_COL)
    pv, dates, nd, first = proc.prepare_backtest_data(C.BACKTEST_START, C.BACKTEST_END,
                                                      C.FORMATION_WINDOW)
    if os.path.exists(PILOT_DB):
        os.remove(PILOT_DB)
    res = RT.worker_task(cfg, pv, dates, nd, first, sector,
                         "formation_data/formation_pairs_sp500_Tiingo.db",
                         tempfile.mkdtemp(prefix="nd_pilot_"), PILOT_DB, "Tiingo", {})
    print(cfg["name"], res.get("status") if isinstance(res, dict) else res, flush=True)

    fn = RT._build_filename(p)
    new_path = f"tiingo/{raw['sub_dir']}/{fn}"
    base_path = f"tiingo/Grid_NOGRP_DTW/{fn}"

    # V2：同一批配對、同一組日期
    q = "SELECT Period_Start, Date, Ticker_A, Ticker_B FROM trade_logs WHERE strategy_id = ?"
    con = sqlite3.connect(PILOT_DB)
    a = pd.read_sql(q, con, params=(new_path,))
    con.close()
    con = sqlite3.connect("file:results/result.db?mode=ro", uri=True, timeout=600)
    b = pd.read_sql(q, con, params=(base_path,))
    con.close()
    pa = set(map(tuple, a[["Period_Start", "Ticker_A", "Ticker_B"]].drop_duplicates().values))
    pb = set(map(tuple, b[["Period_Start", "Ticker_A", "Ticker_B"]].drop_duplicates().values))
    da, db_ = set(a.Date), set(b.Date)
    v2 = pa == pb and da == db_
    print(f"\n[V2] 配對期 DL-THR {len(pa)} vs Z-Score {len(pb)}，僅一方有："
          f"{len(pa ^ pb)}；交易日 {len(da)} vs {len(db_)}，僅一方有：{len(da ^ db_)} → "
          f"{'通過' if v2 else '失敗'}")

    # V3：SKIP 比例
    s_new, n_new = skip_share(PILOT_DB, new_path)
    ref = {arm: skip_share("results/result.db", f"tiingo/Grid_{arm}_DRL/{fn}")[0] for arm in EXISTING}
    lo, hi = min(ref.values()), max(ref.values())
    v3 = 0.0 < s_new < 1.0
    print(f"[V3] NOGRP-DTW DL-THR SKIP 比例 {s_new:.1%}（{n_new} 個配對期）；"
          f"既有五臂同格 {', '.join(f'{k} {v:.1%}' for k, v in ref.items())}"
          f"（範圍 {lo:.1%}–{hi:.1%}）→ {'通過' if v3 else '失敗'}"
          f"{'' if lo <= s_new <= hi else '（落在既有範圍外，需說明）'}")
    print("\n全部通過" if v2 and v3 else "\n未通過")


if __name__ == "__main__":
    main()
