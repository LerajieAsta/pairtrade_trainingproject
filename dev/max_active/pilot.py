# -*- coding: utf-8 -*-
"""max_active（同一交易期同時持倉上限 K）的接線驗證。

跑三格 `Grid GICS-SSD`（SL0）寫進試跑庫，**不碰** result.db 與正式 checkpoint：
  A. Top20、未設 K      → 必須與 result.db 既有列逐位元相同（Sharpe / Final_Equity / 列數）
  B. Top20、K=10        → 每期每日持倉數 ≤ 10；槽位分母 = 10 × 6
  C. Top10、未設 K      → 對照：與 B 同一每對資金，但候選固定 10 對

    python dev/max_active/pilot.py
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
from strategies.metrics import metrics  # noqa: E402

PILOT_DB = "results/pilot_max_active.db"
ARM = "Grid GICS-SSD"
LIST_KEYS = ("top_n_list", "stop_loss_list", "stop_loss_pct_list", "max_sector_ratio_list",
             "entry_z_list", "dynamic_stop_z_list", "max_holding_days_list",
             "exec_lag_list", "exit_z_list", "max_active_list")


def make_config(raw, top_n, k):
    p = copy.deepcopy(raw["params"])
    for key in LIST_KEYS:
        p.pop(key, None)
    p.update(top_n=top_n, stop_loss_pct=0.0, max_sector_ratio=0.0, max_active=k)
    name = f"{raw['name']}_Top{top_n}_SL0_MSR0" + (f"_MA{k}" if C.max_active_of(p) else "")
    return {"name": name, "trading_module": raw["trading_module"], "sub_dir": raw["sub_dir"],
            "db_method": raw["db_method"], "trade_method": raw.get("trade_method", "Z-Score"),
            "params": p, "formation_strategy_id": f"{raw['name']}_MSR0"}


def main():
    raw = next(s for s in C.strategies_raw_all if s["name"] == ARM)
    proc = RT.DataProcessor(db_path=C.DB_PATH, table_name=C.TABLE_NAME)
    sector = proc.load_sector_mapping(C.INFO_TABLE, C.TICKER_COL, C.SECTOR_COL)
    pv, dates, nd, first = proc.prepare_backtest_data(C.BACKTEST_START, C.BACKTEST_END,
                                                      C.FORMATION_WINDOW)
    form_db = "formation_data/formation_pairs_sp500_Tiingo.db"
    out_root = tempfile.mkdtemp(prefix="ma_pilot_")
    if os.path.exists(PILOT_DB):
        os.remove(PILOT_DB)

    cells = [(20, 0), (20, 10), (10, 0)]
    paths = {}
    for tn, k in cells:
        cfg = make_config(raw, tn, k)
        res = RT.worker_task(cfg, pv, dates, nd, first, sector, form_db, out_root,
                             PILOT_DB, "Tiingo", {})
        paths[(tn, k)] = f"tiingo/{cfg['sub_dir']}/{RT._build_filename(cfg['params'])}"
        print(cfg["name"], res.get("status") if isinstance(res, dict) else res, flush=True)

    # A：逐位元同於正式庫
    pa = paths[(20, 0)]
    con = sqlite3.connect(f"file:results/result.db?mode=ro", uri=True, timeout=600)
    prod = con.execute('SELECT Sharpe_Raw, Final_Equity, Entries FROM strategy_summaries '
                       'WHERE _path=?', (pa,)).fetchone()
    con.close()
    con = sqlite3.connect(PILOT_DB)
    new = con.execute('SELECT Sharpe_Raw, Final_Equity, Entries FROM strategy_summaries '
                      'WHERE _path=?', (pa,)).fetchone()
    print(f"\n[A] 正式庫 {prod}\n    試跑庫 {new}\n    相同={prod == new}")

    # B：每期每日持倉 ≤ K
    pb = paths[(20, 10)]
    df = pd.read_sql("SELECT Period_Start, Date, Position FROM trade_logs WHERE strategy_id=?",
                     con, params=(pb,))
    occ = df[df.Position != 0].groupby(["Period_Start", "Date"]).size()
    row = con.execute('SELECT Max_Active, Avg_Utilization, Sharpe_Raw, Entries FROM '
                      'strategy_summaries WHERE _path=?', (pb,)).fetchone()
    print(f"[B] 每期單日最大持倉 = {occ.max()}（上限 10）；"
          f"觸頂日比例 = {(occ == 10).mean():.3f}；Max_Active/Util/Sharpe/Entries = {row}")
    for key in ((20, 10), (10, 0), (20, 0)):
        r = con.execute('SELECT Sharpe_Raw, Ann_Ret_Raw, MDD_Raw, Entries, Avg_Utilization '
                        'FROM strategy_summaries WHERE _path=?', (paths[key],)).fetchone()
        print(f"    Top{key[0]} K={key[1] or '-'}: Sharpe={r[0]:+.4f} Ann={r[1]:+.4%} "
              f"MDD={r[2]:+.2%} Entries={r[3]} Util={r[4]:.3f}")
    con.close()

    # 讀端：metrics() 以 Max_Active 算槽位，必須與引擎落庫的利用率一致
    m = metrics(pb, top_n=20, result_db=PILOT_DB)
    print(f"[B'] metrics() 利用率 {m['Avg_Utilization']:.6f} vs 落庫 {row[1]:.6f}")


if __name__ == "__main__":
    main()
