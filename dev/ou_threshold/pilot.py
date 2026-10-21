# -*- coding: utf-8 -*-
"""OU 解析最適門檻的接線驗證（PREREGISTRATION.md §六）。結果寫入 results/pilot_ou.db，不碰 result.db。

  V1. Kalman 交易端（同名參數 spread_mode）仍可建構，且 NOGRP-DTW-KAL Top1/SL0 與 result.db 逐位元相同
  V2. ou_force_baseline：GICS-SDP、NOGRP-DTW 的 Top1/SL0 與 result.db 的 Z-Score 同格逐位元相同
  V3. NOGRP-DTW Top3/SL0：以獨立程式（np.polyfit 估 AR(1)）重算數個配對期的 θ、σ、c_z、
      各動作 μ 與所選動作，與引擎記錄的 OU_Theta／OU_Sigma／OU_Cz／Sel_Entry_Z／Sel_Exit_Z 一致
  V4. 同一格的 SKIP 比例不為 0% 或 100%，所選動作不全相同

    python dev/ou_threshold/pilot.py
"""
import copy
import json
import os
import sqlite3
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import erfi  # noqa: E402

import run_trading as RT  # noqa: E402
from strategies import config as C  # noqa: E402
from strategies.trading import ou_threshold_trading as OU  # noqa: E402
from strategies.trading.zscore_trading import clean_prices  # noqa: E402

PILOT_DB = "results/pilot_ou.db"
FORM_DB = "formation_data/formation_pairs_sp500_Tiingo.db"
LIST_KEYS = ("top_n_list", "stop_loss_list", "stop_loss_pct_list", "max_sector_ratio_list",
             "entry_z_list", "dynamic_stop_z_list", "max_holding_days_list",
             "exec_lag_list", "exit_z_list", "max_active_list")


def cell(name, top_n, module=None, extra=None, sub=None):
    raw = next(s for s in C.strategies_raw_all if s["name"] == name)
    p = copy.deepcopy(raw["params"])
    for k in LIST_KEYS:
        p.pop(k, None)
    p.update(top_n=top_n, stop_loss_pct=0.0, max_sector_ratio=0.0, **(extra or {}))
    return {"name": f"{name}_Top{top_n}_SL0_MSR0{sub or ''}",
            "trading_module": module or raw["trading_module"],
            "sub_dir": raw["sub_dir"] + (sub or ""), "db_method": raw["db_method"],
            "trade_method": raw.get("trade_method", "Z-Score"), "params": p,
            "formation_strategy_id": f"{raw.get('formation_strategy_id_base') or name}_MSR0"}


def summary(db, path):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
    r = con.execute("SELECT Sharpe_Raw, Final_Equity, Entries FROM strategy_summaries WHERE _path=?",
                    (path,)).fetchone()
    con.close()
    return r


def main():
    proc = RT.DataProcessor(db_path=C.DB_PATH, table_name=C.TABLE_NAME)
    sector = proc.load_sector_mapping(C.INFO_TABLE, C.TICKER_COL, C.SECTOR_COL)
    pv, dates, nd, first = proc.prepare_backtest_data(C.BACKTEST_START, C.BACKTEST_END, C.FORMATION_WINDOW)
    out_root = tempfile.mkdtemp(prefix="ou_pilot_")
    if os.path.exists(PILOT_DB):
        os.remove(PILOT_DB)
    ok = {}

    def run(cfg):
        res = RT.worker_task(cfg, pv, dates, nd, first, sector, FORM_DB, out_root, PILOT_DB, "Tiingo", {})
        print(cfg["name"], res.get("status") if isinstance(res, dict) else res, flush=True)
        return f"tiingo/{cfg['sub_dir']}/{RT._build_filename(cfg['params'])}"

    # V1：Kalman
    if any(s["name"] == "Grid NOGRP-DTW-KAL" for s in C.strategies_raw_all):
        c = cell("Grid NOGRP-DTW-KAL", 1)
        path = run(c)
        a, b = summary("results/result.db", path), summary(PILOT_DB, path)
        ok["V1 Kalman"] = a is not None and a == b
        print(f"[V1] Kalman：result.db {a} | 試跑庫 {b} → {'通過' if ok['V1 Kalman'] else '失敗'}")

    # V2：強制基準動作
    for name in ("Grid GICS-SDP", "Grid NOGRP-DTW"):
        c = cell(name, 1, module="strategies.trading.ou_threshold_trading",
                 extra={"ou_force_baseline": True}, sub="_OUFORCE")
        path = run(c)
        base = path.replace("_OUFORCE/", "/")
        a, b = summary("results/result.db", base), summary(PILOT_DB, path)
        ok[f"V2 {name}"] = a == b
        print(f"[V2] {name}：Z-Score {a} | 強制 (2.0,0) {b} → {'通過' if a == b else '失敗'}")

    # V3／V4：正常選擇
    c = cell("Grid NOGRP-DTW", 3, module="strategies.trading.ou_threshold_trading", sub="_OUTHR")
    path = run(c)
    con = sqlite3.connect(PILOT_DB)
    tl = pd.read_sql("SELECT Period_Start, Ticker_A, Ticker_B, MIN(Status='HOLD_CASH (SKIP)') skip, "
                     "AVG(Sel_Entry_Z) e, AVG(Sel_Exit_Z) x, AVG(OU_Theta) th, AVG(OU_Sigma) sg, AVG(OU_Cz) cz "
                     "FROM trade_logs WHERE strategy_id=? GROUP BY 1,2,3", con, params=(path,))
    con.close()
    f = sqlite3.connect(f"file:{FORM_DB}?mode=ro", uri=True, timeout=600)
    fp = pd.read_sql("SELECT Period_Start, Trade_Start, Ticker_A, Ticker_B, Formation_Params FROM formation_pairs "
                     "WHERE strategy_id='Grid NOGRP-DTW_MSR0'", f)
    f.close()
    clean = clean_prices(pv)
    friction = float(c["params"].get("fee_rate", 0)) + float(c["params"].get("slippage_rate", 0))
    worst = {"th": 0.0, "sg": 0.0, "cz": 0.0}
    mismatch, checked = 0, 0
    for r in tl.dropna(subset=["th"]).head(8).itertuples(index=False):
        row = fp[(fp.Trade_Start == r.Period_Start) & (fp.Ticker_A == r.Ticker_A) & (fp.Ticker_B == r.Ticker_B)].iloc[0]
        q = json.loads(row.Formation_Params)
        fs, ts = pd.Timestamp(row.Period_Start), pd.Timestamp(row.Trade_Start)
        w = clean.loc[fs:ts, [r.Ticker_A, r.Ticker_B]]
        w = w[w.index < ts].dropna()
        na = (np.log(w[r.Ticker_A].values) - q["Log_Mean_A"]) / q["Log_Std_A"]
        nb = (np.log(w[r.Ticker_B].values) - q["Log_Mean_B"]) / q["Log_Std_B"]
        z = (na - q["Hedge_Ratio"] * nb - q["Spread_Mean"]) / q["Spread_Std"]
        phi, cc = np.polyfit(z[:-1], z[1:], 1)
        th = -np.log(phi)
        sg = np.std(z[1:] - cc - phi * z[:-1], ddof=1) * np.sqrt(2 * th / (1 - phi ** 2))
        cz = 2 * friction * (1 / q["Log_Std_A"] + abs(q["Hedge_Ratio"]) / q["Log_Std_B"]) / q["Spread_Std"]
        k = np.sqrt(th) / sg
        mus = {(e, x): ((e - x) - cz) / ((np.pi / th) * (erfi(-x * k) - erfi(-e * k))) for e, x in OU.MENU}
        best = max(mus, key=mus.get)
        pick = best if mus[best] > 0 else None
        worst["th"] = max(worst["th"], abs(th - r.th))
        worst["sg"] = max(worst["sg"], abs(sg - r.sg))
        worst["cz"] = max(worst["cz"], abs(cz - r.cz))
        got = None if r.skip else (r.e, r.x)
        mismatch += int(got != pick)
        checked += 1
    ok["V3"] = checked > 0 and mismatch == 0 and max(worst.values()) < 1e-9
    print(f"[V3] 獨立重算 {checked} 個配對期：|Δθ| {worst['th']:.1e}、|Δσ| {worst['sg']:.1e}、|Δc_z| {worst['cz']:.1e}；"
          f"所選動作不符 {mismatch} → {'通過' if ok['V3'] else '失敗'}")
    skip = float(tl.skip.mean())
    dist = tl[tl.skip == 0].groupby(["e", "x"]).size()
    ok["V4"] = 0 < skip < 1 and len(dist) > 1
    print(f"[V4] SKIP 比例 {skip:.1%}；所選動作分布 {dist.to_dict()} → {'通過' if ok['V4'] else '失敗'}")
    print("\n全部通過" if all(ok.values()) else f"\n未通過：{[k for k, v in ok.items() if not v]}")


if __name__ == "__main__":
    main()
