# -*- coding: utf-8 -*-
"""跨期去重的接線驗證（跑網格之前必須全過）。

寫進試跑庫 results/pilot_dedup.db，**不碰** result.db、max_active.db 與正式 checkpoint：
  V1. GICS-SSD Top20、不去重  → 與 result.db 既有列逐位元相同（_simulate 改傳 ctx 未改變預設路徑）
  V2. GICS-SSD、NOGRP-DTW Top20 去重 → 同一配對（不分 A|B／B|A）同日被 ≥2 期持有的次數 = 0
      （NOGRP-DTW 臂內同一配對會以兩種順序出現，驗證不分順序的比對）
  V3. 去重格的權益帳：引擎的 pm.current_equity（worker_task 回傳）= 初始 + Σ 寫入的各 (期, 配對)
      最終已實現損益——驗證重新模擬後的補差額。不可用摘要的 Final_Equity：它本身就是由
      逐日紀錄加總（db_utils.py），拿來比對是循環論證。
  V4. NOGRP-SSD Top20 K=10 → 與 max_active.db 中以改寫前程式算出的列逐位元相同

    python dev/dedup/pilot.py
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
from strategies.dedup import pair_key  # noqa: E402

PILOT_DB = "results/pilot_dedup.db"
LIST_KEYS = ("top_n_list", "stop_loss_list", "stop_loss_pct_list", "max_sector_ratio_list",
             "entry_z_list", "dynamic_stop_z_list", "max_holding_days_list",
             "exec_lag_list", "exit_z_list", "max_active_list")
COLS = "Sharpe_Raw, Final_Equity, Entries"


def make_config(arm, top_n, dedup=False, k=0):
    raw = next(s for s in C.strategies_raw_all if s["name"] == arm)
    p = copy.deepcopy(raw["params"])
    for key in LIST_KEYS:
        p.pop(key, None)
    p.update(top_n=top_n, stop_loss_pct=0.0, max_sector_ratio=0.0, max_active=k)
    if dedup:
        p["dedup_across_periods"] = True
    name = (f"{arm}_Top{top_n}_SL0_MSR0" + (f"_MA{k}" if C.max_active_of(p) else "")
            + ("_DD" if dedup else ""))
    return {"name": name, "trading_module": raw["trading_module"], "sub_dir": raw["sub_dir"],
            "db_method": raw["db_method"], "trade_method": raw.get("trade_method", "Z-Score"),
            "params": p, "formation_strategy_id": f"{arm}_MSR0"}


def row(db, path):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
    r = con.execute(f"SELECT {COLS} FROM strategy_summaries WHERE _path=?", (path,)).fetchone()
    con.close()
    return r


def duplicates(path):
    con = sqlite3.connect(PILOT_DB)
    df = pd.read_sql("SELECT Period_Start, Date, Ticker_A, Ticker_B, Position, Realized_PnL "
                     "FROM trade_logs WHERE strategy_id=?", con, params=(path,))
    con.close()
    df["key"] = [pair_key((a, b)) for a, b in zip(df.Ticker_A, df.Ticker_B)]
    live = df[df.Position != 0]
    n = live.groupby(["Date", "key"]).size()
    final = df.groupby(["Period_Start", "Ticker_A", "Ticker_B"]).Realized_PnL.last().sum()
    return int((n > 1).sum()), int(len(n)), float(final)


def main():
    proc = RT.DataProcessor(db_path=C.DB_PATH, table_name=C.TABLE_NAME)
    sector = proc.load_sector_mapping(C.INFO_TABLE, C.TICKER_COL, C.SECTOR_COL)
    pv, dates, nd, first = proc.prepare_backtest_data(C.BACKTEST_START, C.BACKTEST_END,
                                                      C.FORMATION_WINDOW)
    form_db = "formation_data/formation_pairs_sp500_Tiingo.db"
    out_root = tempfile.mkdtemp(prefix="dd_pilot_")
    if os.path.exists(PILOT_DB):
        os.remove(PILOT_DB)

    cells = {
        "base": make_config("Grid GICS-SSD", 20),
        "dd_ssd": make_config("Grid GICS-SSD", 20, dedup=True),
        "dd_dtw": make_config("Grid NOGRP-DTW", 20, dedup=True),
        "ma10": make_config("Grid NOGRP-SSD", 20, k=10),
    }
    paths, engine_eq = {}, {}
    for key, cfg in cells.items():
        res = RT.worker_task(cfg, pv, dates, nd, first, sector, form_db, out_root,
                             PILOT_DB, "Tiingo", {})
        paths[key] = f"tiingo/{cfg['sub_dir']}/{RT._build_filename(cfg['params'])}"
        engine_eq[key] = res.get("final_equity") if isinstance(res, dict) else None
        print(cfg["name"], res.get("status") if isinstance(res, dict) else res, flush=True)

    ok = {}
    prod, new = row("results/result.db", paths["base"]), row(PILOT_DB, paths["base"])
    ok["V1"] = prod == new
    print(f"\n[V1] result.db {prod}\n     試跑庫    {new}\n     相同={ok['V1']}")

    base_dup, base_n, base_final = duplicates(paths["base"])
    print(f"[V2] 對照（不去重）GICS-SSD：重複持有 {base_dup}/{base_n} 個 (日, 配對)")
    for key in ("dd_ssd", "dd_dtw"):
        dup, n, final = duplicates(paths[key])
        eng = engine_eq[key]
        eq = C.INITIAL_CAPITAL + final
        ok[f"V2 {key}"] = dup == 0
        ok[f"V3 {key}"] = eng is not None and abs(eng - eq) < 1e-6 * max(1.0, abs(eq))
        print(f"[V2] {key}：重複持有 {dup}/{n} 個 (日, 配對)  → {'通過' if dup == 0 else '失敗'}")
        print(f"[V3] {key}：引擎權益 {eng:,.4f} vs 初始+Σ損益 {eq:,.4f}"
              f"  → {'通過' if ok[f'V3 {key}'] else '失敗'}")
    eq_b = C.INITIAL_CAPITAL + base_final
    print(f"     （同一恆等式在不去重格：{engine_eq['base']:,.4f} vs {eq_b:,.4f}）")

    ma_ref, ma_new = row("results/max_active.db", paths["ma10"]), row(PILOT_DB, paths["ma10"])
    ok["V4"] = ma_ref is not None and ma_ref == ma_new
    print(f"[V4] max_active.db {ma_ref}\n     試跑庫          {ma_new}\n     相同={ok['V4']}")

    for key in ("base", "dd_ssd", "dd_dtw"):
        r = row(PILOT_DB, paths[key])
        print(f"     {key}: Sharpe={r[0]:+.4f} Final={r[1]:,.0f} Entries={r[2]}")
    print("\n全部通過" if all(ok.values()) else f"\n未通過：{[k for k, v in ok.items() if not v]}")


if __name__ == "__main__":
    main()
