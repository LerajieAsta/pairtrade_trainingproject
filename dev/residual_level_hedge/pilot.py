# -*- coding: utf-8 -*-
"""殘差報酬訊號＋水準部位的接線驗證（PREREGISTRATION.md §六）。試跑寫入 results/pilot_rrl.db。

  V1. residual_hedge 未設：_RR 的 GICS-SSD、NOGRP-DTW Top1/SL0 摘要與 residual_spread.db 逐位元相同；
      spread_mode 未設：GICS-SSD Top1/SL0 與 result.db 逐位元相同
  V2. RRL 試跑格的 ZScore 欄與 _RR 同格逐列相同
  V3. RRL 每次進場：傳入的避險比率 = 形成期水準 β；兩腳名目額比 = s_B|β|/s_A（s 由形成期參數獨立求得）；
      總名目額 = 每對資金。GICS-SSD（signal 權重）與 HDB-SSD 各一格
  V4. 進出場日與現行完全相同的配對期，逐日損益比 RRL／現行在期內為常數（每對資金隨權益複利，兩邊只差資金倍數；
      僅取 |現行日損益| > 5 美元的日子，使 4 位捨入誤差 < 1e-4 相對）

    python dev/residual_level_hedge/pilot.py
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

import run_trading as RT  # noqa: E402
from strategies import config as C  # noqa: E402
from strategies.trading import zscore_trading as ZT  # noqa: E402

PILOT_DB = "results/pilot_rrl.db"
RR_DB = "results/residual_spread.db"
FORM_DB = "formation_data/formation_pairs_sp500_Tiingo.db"
LIST_KEYS = ("top_n_list", "stop_loss_list", "stop_loss_pct_list", "max_sector_ratio_list",
             "entry_z_list", "dynamic_stop_z_list", "max_holding_days_list",
             "exec_lag_list", "exit_z_list", "max_active_list")
SUMMARY_COLS = ("Sharpe_Raw", "Final_Equity", "Entries", "Win_Rate", "MDD_Raw")

# ── V3 的探針：記錄每次進場的 (配對, 傳入 β, 腳權重, 兩腳名目額) ───────────────
ENTRIES = []
_orig_sim, _orig_entry = ZT.Trading._simulate_pair, ZT.Trading._execute_entry


def _sim(self, period_start, period_end, sector, ticker_a, ticker_b, *a, **k):
    self._probe = (period_start, ticker_a, ticker_b, k.get("ols_alpha"))
    return _orig_sim(self, period_start, period_end, sector, ticker_a, ticker_b, *a, **k)


def _entry(self, state, z, p_a, p_b, hedge_ratio):
    out = _orig_entry(self, state, z, p_a, p_b, hedge_ratio)
    if out[0] and getattr(self, "spread_mode", "level") == "residual_return":
        ENTRIES.append({"key": self._probe, "beta_in": hedge_ratio, "na": abs(state.shares_a) * p_a,
                        "nb": abs(state.shares_b) * p_b, "cap": self.capital_per_pair, "hedge": self.residual_hedge})
    return out


ZT.Trading._simulate_pair, ZT.Trading._execute_entry = _sim, _entry


def cell(name, extra=None):
    raw = next(s for s in C.strategies_raw_all if s["name"] == name)
    p = copy.deepcopy(raw["params"])
    for k in LIST_KEYS:
        p.pop(k, None)
    p.update(top_n=1, stop_loss_pct=0.0, max_sector_ratio=0.0, **(extra or {}))
    return {"name": f"{name}_Top1_SL0_MSR0", "trading_module": raw["trading_module"],
            "sub_dir": raw["sub_dir"], "db_method": raw["db_method"],
            "trade_method": raw.get("trade_method", "Z-Score"), "params": p,
            "formation_strategy_id": f"{raw.get('formation_strategy_id_base') or name}_MSR0"}


def q(db, sql, params):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
    try:
        return pd.read_sql(sql, con, params=params)
    finally:
        con.close()


def summary(db, path):
    d = q(db, f"SELECT {','.join(SUMMARY_COLS)} FROM strategy_summaries WHERE _path=?", (path,))
    return tuple(d.iloc[0]) if len(d) else None


def main():
    proc = RT.DataProcessor(db_path=C.DB_PATH, table_name=C.TABLE_NAME)
    sector = proc.load_sector_mapping(C.INFO_TABLE, C.TICKER_COL, C.SECTOR_COL)
    pv, dates, nd, first = proc.prepare_backtest_data(C.BACKTEST_START, C.BACKTEST_END, C.FORMATION_WINDOW)
    out_root = tempfile.mkdtemp(prefix="rrl_pilot_")
    if os.path.exists(PILOT_DB):
        os.remove(PILOT_DB)
    ok = {}

    def run(cfg):
        res = RT.worker_task(cfg, pv, dates, nd, first, sector, FORM_DB, out_root, PILOT_DB, "Tiingo", {})
        print(cfg["name"], res.get("status") if isinstance(res, dict) else res, flush=True)
        return f"tiingo/{cfg['sub_dir']}/{RT._build_filename(cfg['params'])}"

    # V1
    path = run(cell("Grid GICS-SSD"))
    a, b = summary("results/result.db", path), summary(PILOT_DB, path)
    ok["V1 level"] = a is not None and a == b
    print(f"[V1] 現行 GICS-SSD：{'相同' if ok['V1 level'] else f'不同 {a} vs {b}'}")
    for name in ("Grid GICS-SSD", "Grid NOGRP-DTW"):
        path = run(cell(name, {"spread_mode": "residual_return"}))
        a, b = summary(RR_DB, path), summary(PILOT_DB, path)
        ok[f"V1 RR {name}"] = a is not None and a == b
        print(f"[V1] _RR {name}：{'相同' if ok[f'V1 RR {name}'] else f'不同 {a} vs {b}'}")

    # V2–V4
    ENTRIES.clear()
    fp_all = {}
    for name in ("Grid GICS-SSD", "Grid HDB-SSD"):
        path = run(cell(name, {"spread_mode": "residual_return", "residual_hedge": "level"}))
        rr_path = path.replace("_RRL.csv", "_RR.csv")
        cols = "Period_Start, Ticker_A, Ticker_B, Date, ZScore, Position, Trade_PnL, Daily_Delta"
        mine = q(PILOT_DB, f"SELECT {cols} FROM trade_logs WHERE strategy_id=? ORDER BY 1,2,3,4", (path,))
        rr = q(RR_DB, f"SELECT {cols} FROM trade_logs WHERE strategy_id=? ORDER BY 1,2,3,4", (rr_path,))
        same_z = len(mine) == len(rr) and np.array_equal(mine.ZScore.values, rr.ZScore.values)
        ok[f"V2 {name}"] = same_z
        print(f"[V2] {name}：RRL {len(mine)} 列 vs _RR {len(rr)} 列，ZScore 逐列相同 = {same_z}")

        base_path = path.replace("_RRL.csv", ".csv")
        base = q("results/result.db", f"SELECT {cols} FROM trade_logs WHERE strategy_id=? ORDER BY 1,2,3,4",
                 (base_path,))
        key = ["Period_Start", "Ticker_A", "Ticker_B"]
        # 每對資金 = 當下權益 ÷ 槽位（複利），兩邊資金不同；部位口徑相同 ⇔ 逐日損益比在配對期內為常數
        n_same, worst = 0, 0.0
        for k, g in mine.groupby(key):
            h = base[(base.Period_Start == k[0]) & (base.Ticker_A == k[1]) & (base.Ticker_B == k[2])]
            if len(h) != len(g) or not (g.Position != 0).any():
                continue
            if np.array_equal(g.Position.values, h.Position.values):
                m = h.Daily_Delta.abs().values > 5.0
                if m.sum() < 5:
                    continue
                ratio = g.Daily_Delta.values[m] / h.Daily_Delta.values[m]
                n_same += 1
                worst = max(worst, float(np.max(np.abs(ratio / np.median(ratio) - 1))))
        ok[f"V4 {name}"] = n_same > 0 and worst < 1e-3
        print(f"[V4] {name}：部位序列與現行完全相同的配對期 {n_same} 個，逐日損益比偏離中位數的最大相對差 {worst:.1e}")

        sid = next(s for s in C.strategies_raw_all if s["name"] == name)
        fsid = f"{sid.get('formation_strategy_id_base') or name}_MSR0"
        fp = q(FORM_DB, "SELECT Trade_Start, Ticker_A, Ticker_B, Formation_Params FROM formation_pairs "
                        "WHERE strategy_id=?", (fsid,))
        fp_all.update({(r.Trade_Start, r.Ticker_A, r.Ticker_B): json.loads(r.Formation_Params)
                       for r in fp.itertuples(index=False)})

    # V3
    worst_b, worst_r, worst_c, n_ols, n_sig = 0.0, 0.0, 0.0, 0, 0
    for e in ENTRIES:
        if e["hedge"] != "level":
            continue
        ps, ta, tb, ols = e["key"]
        p = fp_all[(ps, ta, tb)]
        beta = p["Hedge_Ratio"]
        if ols is not None:
            sa, sb = 1.0, 1.0
            n_ols += 1
        else:
            sa, sb = 1.0 / p["Log_Std_A"], 1.0 / p["Log_Std_B"]
            n_sig += 1
        worst_b = max(worst_b, abs(e["beta_in"] - beta))
        worst_r = max(worst_r, abs(e["nb"] / e["na"] - sb * abs(beta) / sa))
        worst_c = max(worst_c, abs(e["na"] + e["nb"] - e["cap"]) / e["cap"])
    ok["V3"] = n_sig > 0 and worst_b < 1e-9 and worst_r < 1e-9 and worst_c < 1e-9
    print(f"[V3] RRL 進場 {n_sig + n_ols} 次（signal 權重 {n_sig}、OLS 路徑 {n_ols}）：|β 傳入 − 水準 β| {worst_b:.1e}、"
          f"名目額比差 {worst_r:.1e}、總名目額相對差 {worst_c:.1e}")
    print("\n全部通過" if all(ok.values()) else f"\n未通過：{[k for k, v in ok.items() if not v]}")


if __name__ == "__main__":
    main()
