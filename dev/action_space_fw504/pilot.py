# -*- coding: utf-8 -*-
"""504 版動作空間消融的接線驗證 V1／V2（PREREGISTRATION.md §五；V3 為參數比對，另行執行）。

  python dev/action_space_fw504/pilot.py v1   # 開關未設：Grid GICS-SDP Top1/SL0 與 result.db 逐位元相同
  python dev/action_space_fw504/pilot.py v2   # 開關開啟、起點 2008-07：Z-Score 臂恰跑那 108 期且形成期皆完整

期間與開關在 import config 之前設定（config 於 import 時讀環境變數），故兩項分開執行。
結果寫入 results/pilot_abl504.db，不碰 result.db。
"""
import copy
import os
import sqlite3
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

MODE = sys.argv[1] if len(sys.argv) > 1 else ""
if MODE == "v2":
    os.environ.update(SKIP_TRUNCATED_FORMATION="1", BACKTEST_START="2008-07", BACKTEST_END="2018-12")
elif MODE == "v1":
    for k in ("SKIP_TRUNCATED_FORMATION", "BACKTEST_START", "BACKTEST_END"):
        os.environ.pop(k, None)
else:
    raise SystemExit("用法：pilot.py v1 | v2")

import pandas as pd  # noqa: E402

import run_trading as RT  # noqa: E402
from strategies import config as C  # noqa: E402

PILOT_DB = "results/pilot_abl504.db"
LIST_KEYS = ("top_n_list", "stop_loss_list", "stop_loss_pct_list", "max_sector_ratio_list",
             "entry_z_list", "dynamic_stop_z_list", "max_holding_days_list",
             "exec_lag_list", "exit_z_list", "max_active_list")


def cell(raw, form_base):
    p = copy.deepcopy(raw["params"])
    for k in LIST_KEYS:
        p.pop(k, None)
    p.update(top_n=1, stop_loss_pct=0.0, max_sector_ratio=0.0)
    return {"name": f"{raw['name']}_Top1_SL0_MSR0", "trading_module": raw["trading_module"],
            "sub_dir": raw["sub_dir"], "db_method": raw["db_method"],
            "trade_method": raw.get("trade_method", "Z-Score"), "params": p,
            "formation_strategy_id": f"{form_base}_MSR0"}


def run(cfg):
    proc = RT.DataProcessor(db_path=C.DB_PATH, table_name=C.TABLE_NAME)
    sector = proc.load_sector_mapping(C.INFO_TABLE, C.TICKER_COL, C.SECTOR_COL)
    pv, dates, nd, first = proc.prepare_backtest_data(C.BACKTEST_START, C.BACKTEST_END,
                                                      C.FORMATION_WINDOW)
    res = RT.worker_task(cfg, pv, dates, nd, first, sector,
                         "formation_data/formation_pairs_sp500_Tiingo.db",
                         tempfile.mkdtemp(prefix="abl504_pilot_"), PILOT_DB, "Tiingo", {})
    print(cfg["name"], res.get("status") if isinstance(res, dict) else res, flush=True)
    return f"tiingo/{cfg['sub_dir']}/{RT._build_filename(cfg['params'])}", pv.index[0]


def summary(db, path):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
    r = con.execute("SELECT Sharpe_Raw, Final_Equity, Entries FROM strategy_summaries WHERE _path=?",
                    (path,)).fetchone()
    con.close()
    return r


if MODE == "v1":
    raw = next(s for s in C.strategies_raw_all if s["name"] == "Grid GICS-SDP")
    path, _ = run(cell(raw, raw["name"]))
    a, b = summary("results/result.db", path), summary(PILOT_DB, path)
    print(f"[V1] result.db {a}\n     試跑庫    {b}\n     相同={a == b} → {'通過' if a == b else '失敗'}")
else:
    from dev.action_space_fw504.candidate_strategies import build
    zs = next(s for s in build() if s["db_method"] == "Grid (GICS-SDP-FW504-DOLLAR)")
    path, idx0 = run(cell(zs, zs["formation_strategy_id_base"]))
    con = sqlite3.connect(PILOT_DB)
    got = pd.to_datetime(pd.read_sql("SELECT DISTINCT Period_Start FROM strategy_pairs WHERE strategy_id=?",
                                     con, params=(path,)).Period_Start)
    con.close()
    con = sqlite3.connect("file:results/result.db?mode=ro", uri=True, timeout=600)
    want = pd.to_datetime(pd.read_sql("SELECT DISTINCT Period_Start FROM strategy_pairs "
                                      "WHERE strategy_id LIKE 'tiingo/Grid_GICS_SDP_DRL_V1/%'", con).Period_Start)
    con.close()
    f = sqlite3.connect("file:formation_data/formation_pairs_sp500_Tiingo.db?mode=ro", uri=True, timeout=600)
    fp = pd.read_sql("SELECT DISTINCT Period_Start, Trade_Start FROM formation_pairs "
                     "WHERE strategy_id='Grid GICS-SDP-FW504_MSR0'", f)
    f.close()
    form_start = pd.to_datetime(fp.set_index(pd.to_datetime(fp.Trade_Start)).Period_Start)
    starts = form_start.reindex(got)
    g, w = set(got), set(want)
    ok_set = g == w
    ok_form = bool((starts >= idx0).all()) and not starts.isna().any()
    print(f"[V2] 執行的交易期 {len(g)}（{min(g).date()}～{max(g).date()}）；與 252 版共同期：交集 {len(g & w)}、"
          f"缺 {len(w - g)}、多 {len(g - w)} → {'通過' if ok_set else '失敗'}")
    print(f"     價格索引起點 {idx0.date()}；最早的形成期起點 {starts.min().date()}；"
          f"形成期皆完整 → {'通過' if ok_form else '失敗'}")
