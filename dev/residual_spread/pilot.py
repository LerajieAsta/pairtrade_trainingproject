# -*- coding: utf-8 -*-
"""殘差報酬價差的接線驗證（PREREGISTRATION.md §六）。結果寫入 results/pilot_rr.db，不碰 result.db。

  V1. spread_mode 未設：Grid GICS-SDP Top1/SL0、Grid NOGRP-SSD Top3/SL0 與 result.db 逐位元相同
  V2. Grid NOGRP-DTW Top1/SL0 以殘差報酬價差試跑；以 np.polyfit（不同於引擎的 lstsq）獨立重算
      某配對期的 β、X、Z，兩層比對：(a) 直接呼叫引擎的 _residual_spread，完整精度容差 1e-9；
      (b) 獨立值四捨五入到小數 4 位後與 trade_logs 記錄值相同（zscore_trading 輸出時對
      Hedge_Ratio／ZScore 等欄 round(4)，故記錄值不可能以 1e-9 比對——初版即因此誤判失敗）
  V3. 直接攔截 _execute_entry：進場後兩腳名目額比 = 1:|β|、總名目額 = 每對資金
  V4. 15 臂全部形成期配對中，形成期日報酬少於 60 筆者的數目

    python dev/residual_spread/pilot.py
"""
import copy
import os
import sqlite3
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import types  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_trading as RT  # noqa: E402
from strategies import config as C  # noqa: E402
from strategies.trading import zscore_trading as ZT  # noqa: E402

PILOT_DB = "results/pilot_rr.db"
FORM_DB = "formation_data/formation_pairs_sp500_Tiingo.db"
LIST_KEYS = ("top_n_list", "stop_loss_list", "stop_loss_pct_list", "max_sector_ratio_list",
             "entry_z_list", "dynamic_stop_z_list", "max_holding_days_list",
             "exec_lag_list", "exit_z_list", "max_active_list")


def cell(name, top_n, rr=False):
    raw = next(s for s in C.strategies_raw_all if s["name"] == name)
    p = copy.deepcopy(raw["params"])
    for k in LIST_KEYS:
        p.pop(k, None)
    p.update(top_n=top_n, stop_loss_pct=0.0, max_sector_ratio=0.0)
    if rr:
        p["spread_mode"] = "residual_return"
    return {"name": f"{name}_Top{top_n}_SL0_MSR0" + ("_RR" if rr else ""),
            "trading_module": raw["trading_module"], "sub_dir": raw["sub_dir"],
            "db_method": raw["db_method"], "trade_method": "Z-Score", "params": p,
            "formation_strategy_id": f"{name}_MSR0"}


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
    out_root = tempfile.mkdtemp(prefix="rr_pilot_")
    if os.path.exists(PILOT_DB):
        os.remove(PILOT_DB)
    ok = {}

    def run(cfg):
        res = RT.worker_task(cfg, pv, dates, nd, first, sector, FORM_DB, out_root, PILOT_DB, "Tiingo", {})
        print(cfg["name"], res.get("status") if isinstance(res, dict) else res, flush=True)
        return f"tiingo/{cfg['sub_dir']}/{RT._build_filename(cfg['params'])}"

    # V1
    for name, tn in (("Grid GICS-SDP", 1), ("Grid NOGRP-SSD", 3)):
        path = run(cell(name, tn))
        a, b = summary("results/result.db", path), summary(PILOT_DB, path)
        ok[f"V1 {name}"] = a == b
        print(f"[V1] {name} Top{tn}：result.db {a} | 試跑庫 {b} → {'通過' if a == b else '失敗'}")

    # V2：引擎紀錄 vs 獨立重算
    path = run(cell("Grid NOGRP-DTW", 1, rr=True))
    con = sqlite3.connect(PILOT_DB)
    tl = pd.read_sql("SELECT Period_Start, Date, Ticker_A, Ticker_B, ZScore, Hedge_Ratio, Position "
                     "FROM trade_logs WHERE strategy_id=?", con, params=(path,))
    con.close()
    tl["Date"] = pd.to_datetime(tl.Date)
    traded = tl[tl.Position != 0].groupby(["Period_Start", "Ticker_A", "Ticker_B"]).size()
    f = sqlite3.connect(f"file:{FORM_DB}?mode=ro", uri=True, timeout=600)
    fp = pd.read_sql("SELECT DISTINCT Period_Start, Trade_Start, Ticker_A, Ticker_B FROM formation_pairs "
                     "WHERE strategy_id='Grid NOGRP-DTW_MSR0'", f)
    f.close()
    clean = ZT.clean_prices(pv)
    worst_z = worst_b = worst_rz = worst_rb = 0.0
    checked = 0
    for (ps, a, b), _ in list(traded.items())[:5]:          # 前 5 個有交易的配對期
        row = fp[(fp.Trade_Start == ps) & (fp.Ticker_A == a) & (fp.Ticker_B == b)].iloc[0]
        fs, ts = pd.Timestamp(row.Period_Start), pd.Timestamp(row.Trade_Start)
        form = clean.loc[fs:ts, [a, b]]
        form = form[form.index < ts].dropna()
        ra, rb = np.diff(np.log(form[a].values)), np.diff(np.log(form[b].values))
        beta, alpha = np.polyfit(rb, ra, 1)
        xf = np.cumsum(ra - alpha - beta * rb)
        g = tl[(tl.Period_Start == ps) & (tl.Ticker_A == a) & (tl.Ticker_B == b)].sort_values("Date")
        # 交易期價格取引擎同一來源：該期延伸價格切片的清洗結果
        tsl = ZT.clean_prices(pv.loc[ts:g.Date.max()])[[a, b]].loc[g.Date]
        xa = np.log(np.concatenate([[form[a].values[-1]], tsl[a].values]))
        xb = np.log(np.concatenate([[form[b].values[-1]], tsl[b].values]))
        xt = xf[-1] + np.cumsum(np.diff(xa) - alpha - beta * np.diff(xb))
        z = np.clip((xt - xf.mean()) / xf.std(ddof=1), -10, 10)
        # (a) 完整精度：以最小替身物件直接呼叫引擎方法
        ns = types.SimpleNamespace(_form_px=clean, formation_start=str(row.Period_Start),
                                   formation_end=str(row.Trade_Start), min_spread_std=1e-6,
                                   zscore_clip=10.0, MIN_FORM_RETURNS=ZT.Trading.MIN_FORM_RETURNS)
        ez, eb = ZT.Trading._residual_spread(ns, a, b, tsl[a], tsl[b])
        worst_z = max(worst_z, float(np.max(np.abs(z - ez.values))))
        worst_b = max(worst_b, abs(beta - eb))
        # (b) 記錄值：輸出時 round(4)
        worst_rz = max(worst_rz, float(np.max(np.abs(np.round(z, 4) - g.ZScore.values))))
        worst_rb = max(worst_rb, float(np.max(np.abs(round(beta, 4) - g.Hedge_Ratio.values))))
        checked += 1
    ok["V2"] = checked > 0 and worst_z < 1e-9 and worst_b < 1e-9 and worst_rz < 1e-12 and worst_rb < 1e-12
    print(f"[V2] 獨立重算 {checked} 個配對期：(a) 對引擎方法 |ΔZ| {worst_z:.2e}、|Δβ| {worst_b:.2e}；"
          f"(b) 四捨五入後對記錄值 |ΔZ| {worst_rz:.2e}、|Δβ| {worst_rb:.2e} → {'通過' if ok['V2'] else '失敗'}")

    # V3：攔截進場
    caught = []
    orig = ZT.Trading._execute_entry

    def spy(self, state, z, p_a, p_b, hedge_ratio):
        r = orig(self, state, z, p_a, p_b, hedge_ratio)
        if r[0]:
            caught.append((abs(state.shares_a) * p_a, abs(state.shares_b) * p_b, hedge_ratio,
                           self.capital_per_pair))
        return r
    ZT.Trading._execute_entry = spy
    try:
        run(cell("Grid NOGRP-SSD", 3, rr=True))
    finally:
        ZT.Trading._execute_entry = orig
    err_ratio = max(abs(vb / va - abs(h)) for va, vb, h, _ in caught)
    err_cap = max(abs(va + vb - c) / c for va, vb, _, c in caught)
    ok["V3"] = len(caught) > 0 and err_ratio < 1e-9 and err_cap < 1e-9
    print(f"[V3] 攔截 {len(caught)} 次進場：|v_B/v_A − |β||最大 {err_ratio:.2e}、"
          f"總名目額相對誤差最大 {err_cap:.2e} → {'通過' if ok['V3'] else '失敗'}")

    # V4：全部 15 臂的形成期報酬筆數
    f = sqlite3.connect(f"file:{FORM_DB}?mode=ro", uri=True, timeout=600)
    allp = pd.read_sql("SELECT DISTINCT Period_Start, Trade_Start, Ticker_A, Ticker_B FROM formation_pairs "
                       "WHERE strategy_id IN (" + ",".join(
                           f"'Grid {g}-{r}_MSR0'" for g in ("NOGRP", "GICS", "HDB", "AGG", "KM")
                           for r in ("SSD", "DTW", "SDP")) + ")", f)
    f.close()
    short = 0
    for (fs, ts), grp in allp.groupby(["Period_Start", "Trade_Start"]):
        win = clean.loc[pd.Timestamp(fs):pd.Timestamp(ts)]
        win = win[win.index < pd.Timestamp(ts)]
        for a, b in zip(grp.Ticker_A, grp.Ticker_B):
            if a not in win or b not in win or len(win[[a, b]].dropna()) - 1 < ZT.Trading.MIN_FORM_RETURNS:
                short += 1
    ok["V4"] = short == 0
    print(f"[V4] {len(allp)} 個配對期中，形成期日報酬 < {ZT.Trading.MIN_FORM_RETURNS} 筆者：{short} → "
          f"{'通過' if short == 0 else '需逐一說明'}")
    print("\n全部通過" if all(ok.values()) else f"\n未通過：{[k for k, v in ok.items() if not v]}")


if __name__ == "__main__":
    main()
