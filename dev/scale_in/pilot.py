# -*- coding: utf-8 -*-
"""分層加碼的接線驗證（PREREGISTRATION.md §六）。網格層的試跑寫入 results/pilot_scale_in.db，不碰 result.db。

  V1. scale_in_z = 0（預設）：GICS-SSD、NOGRP-DTW 的 Top1/SL0 摘要與 result.db 逐位元相同
  V2. 配對層：scale_in_z = 1e9、frac = 0.5、停損 0，逐日損益恰為現行的 0.5 倍
      （trade_logs 輸出四捨五入到 4 位，故容差為 1e-4 美元／日；另比對期末累積損益）
  V3. 配對層：以獨立的逐日迴圈（不呼叫引擎）重算含加碼事件的配對期，逐日累積損益與引擎一致
  V4. scale_in_frac = 1.0：GICS-SSD Top1/SL0 摘要與 result.db 逐位元相同；回報 V3 配對期的加碼觸發率

    python dev/scale_in/pilot.py
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
from strategies.trading.zscore_trading import Trading, clean_prices  # noqa: E402

PILOT_DB = "results/pilot_scale_in.db"
FORM_DB = "formation_data/formation_pairs_sp500_Tiingo.db"
LIST_KEYS = ("top_n_list", "stop_loss_list", "stop_loss_pct_list", "max_sector_ratio_list",
             "entry_z_list", "dynamic_stop_z_list", "max_holding_days_list",
             "exec_lag_list", "exit_z_list", "max_active_list")
SUMMARY_COLS = ("Sharpe_Raw", "Final_Equity", "Entries", "Win_Rate", "MDD_Raw", "Avg_Utilization")
CAP = 1000.0


def cell(name, top_n, extra=None):
    raw = next(s for s in C.strategies_raw_all if s["name"] == name)
    p = copy.deepcopy(raw["params"])
    for k in LIST_KEYS:
        p.pop(k, None)
    p.update(top_n=top_n, stop_loss_pct=0.0, max_sector_ratio=0.0, **(extra or {}))
    return {"name": f"{name}_Top{top_n}_SL0_MSR0", "trading_module": raw["trading_module"],
            "sub_dir": raw["sub_dir"], "db_method": raw["db_method"],
            "trade_method": raw.get("trade_method", "Z-Score"), "params": p,
            "formation_strategy_id": f"{raw.get('formation_strategy_id_base') or name}_MSR0"}


def summary(db, path):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
    r = con.execute(f"SELECT {','.join(SUMMARY_COLS)} FROM strategy_summaries WHERE _path=?", (path,)).fetchone()
    con.close()
    return r


def sim_engine(clean, pair, scale_in_z, frac):
    q = json.loads(pair.Formation_Params)
    dates = clean.loc[pair.Trade_Start:pair.Trade_End].index
    fr = float(C.base_params["fee_rate"]), float(C.base_params["slippage_rate"])
    t = Trading(clean, dates, pd.DataFrame(), CAP, fr[0], fr[1], 0.0, 2.0, 0.0, 0,
                scale_in_z=scale_in_z, scale_in_frac=frac)
    return t._simulate_pair(pair.Trade_Start, pair.Trade_End, "x", pair.Ticker_A, pair.Ticker_B, 1,
                            q["Hedge_Ratio"], q["Spread_Mean"], q["Spread_Std"],
                            q["Log_Mean_A"], q["Log_Std_A"], q["Log_Mean_B"], q["Log_Std_B"]), q


def sim_independent(clean, pair, q, f):
    """不呼叫引擎：z 由形成期參數重算，部位依 PREREGISTRATION.md §二逐日模擬，回傳逐日累積損益。"""
    w = clean.loc[pair.Trade_Start:pair.Trade_End, [pair.Ticker_A, pair.Ticker_B]].dropna()
    pa, pb = w[pair.Ticker_A].values, w[pair.Ticker_B].values
    na = (np.log(pa) - q["Log_Mean_A"]) / q["Log_Std_A"]
    nb = (np.log(pb) - q["Log_Mean_B"]) / q["Log_Std_B"]
    z = np.clip((na - q["Hedge_Ratio"] * nb - q["Spread_Mean"]) / q["Spread_Std"], -10, 10)
    wa, wb = 1 / q["Log_Std_A"], abs(q["Hedge_Ratio"]) / q["Log_Std_B"]
    va, vb = CAP * wa / (wa + wb), CAP * wb / (wa + wb)
    legs = []                       # 每層 (股數 A, 股數 B, 進場價 A, 進場價 B)
    side, realized, fees, cum = 0, 0.0, 0.0, []
    for i in range(len(z)):
        mark = 0.0
        if side:
            gross = sum(sa * (pa[i] - ea) + sb * (pb[i] - eb) for sa, sb, ea, eb in legs)
            exit_fee = sum(abs(sa) for sa, *_ in legs) * pa[i] * f + sum(abs(sb) for _, sb, *_ in legs) * pb[i] * f
            pnl = gross - fees - exit_fee
            if (side == -1 and z[i] <= 0) or (side == 1 and z[i] >= 0):
                realized += pnl
                side, legs, fees = 0, [], 0.0
            else:
                if len(legs) == 1 and side * z[i] < -3.0:
                    sa, sb = side * 0.5 * va / pa[i], -side * 0.5 * vb / pb[i]
                    legs.append((sa, sb, pa[i], pb[i]))
                    fees += (abs(sa) * pa[i] + abs(sb) * pb[i]) * f
                    exit_fee = sum(abs(x[0]) for x in legs) * pa[i] * f + sum(abs(x[1]) for x in legs) * pb[i] * f
                    gross = sum(sa * (pa[i] - ea) + sb * (pb[i] - eb) for sa, sb, ea, eb in legs)
                    pnl = gross - fees - exit_fee
                mark = pnl
        elif abs(z[i]) > 2.0:
            side = -1 if z[i] > 0 else 1
            k = 1.0 if abs(z[i]) > 3.0 else 0.5
            sa, sb = side * k * va / pa[i], -side * k * vb / pb[i]
            if k == 1.0:            # 整份＝兩層同時，拆成兩個半份記帳（成本相同）
                legs = [(sa / 2, sb / 2, pa[i], pb[i]), (sa / 2, sb / 2, pa[i], pb[i])]
            else:
                legs = [(sa, sb, pa[i], pb[i])]
            fees = (abs(sa) * pa[i] + abs(sb) * pb[i]) * f
            mark = -fees
        cum.append(realized + mark)
    if side:                        # 期末強制平倉
        gross = sum(sa * (pa[-1] - ea) + sb * (pb[-1] - eb) for sa, sb, ea, eb in legs)
        exit_fee = sum(abs(x[0]) for x in legs) * pa[-1] * f + sum(abs(x[1]) for x in legs) * pb[-1] * f
        cum[-1] = realized + gross - fees - exit_fee
    return np.array(cum)


def main():
    proc = RT.DataProcessor(db_path=C.DB_PATH, table_name=C.TABLE_NAME)
    sector = proc.load_sector_mapping(C.INFO_TABLE, C.TICKER_COL, C.SECTOR_COL)
    pv, dates, nd, first = proc.prepare_backtest_data(C.BACKTEST_START, C.BACKTEST_END, C.FORMATION_WINDOW)
    out_root = tempfile.mkdtemp(prefix="si_pilot_")
    if os.path.exists(PILOT_DB):
        os.remove(PILOT_DB)
    ok = {}

    def run(cfg):
        res = RT.worker_task(cfg, pv, dates, nd, first, sector, FORM_DB, out_root, PILOT_DB, "Tiingo", {})
        print(cfg["name"], res.get("status") if isinstance(res, dict) else res, flush=True)
        return f"tiingo/{cfg['sub_dir']}/{RT._build_filename(cfg['params'])}"

    # V1
    for name in ("Grid GICS-SSD", "Grid NOGRP-DTW"):
        path = run(cell(name, 1))
        a, b = summary("results/result.db", path), summary(PILOT_DB, path)
        ok[f"V1 {name}"] = a is not None and a == b
        print(f"[V1] {name}：result.db {a}\n      試跑庫    {b} → {'通過' if ok[f'V1 {name}'] else '失敗'}")

    # V4（網格層）
    path = run(cell("Grid GICS-SSD", 1, {"scale_in_z": 3.0, "scale_in_frac": 1.0}))
    a, b = summary("results/result.db", path.replace("_SI30F100", "")), summary(PILOT_DB, path)
    ok["V4 frac=1"] = a is not None and a == b
    print(f"[V4] frac=1.0：result.db {a}\n      試跑庫    {b} → {'通過' if ok['V4 frac=1'] else '失敗'}")

    # V2／V3（配對層）
    clean = clean_prices(pv)
    f = float(C.base_params["fee_rate"]) + float(C.base_params["slippage_rate"])
    con = sqlite3.connect(f"file:{FORM_DB}?mode=ro", uri=True, timeout=600)
    pairs = pd.read_sql("SELECT * FROM formation_pairs WHERE strategy_id='Grid GICS-SSD_MSR0' "
                        "AND Pair_Rank <= 5 ORDER BY Trade_Start", con)
    con.close()
    pairs = pairs.iloc[::12].head(40)
    worst2, worst2_end, worst3, n_si, n_traded, n3 = 0.0, 0.0, 0.0, 0, 0, 0
    for pr in pairs.itertuples(index=False):
        base, q = sim_engine(clean, pr, 0.0, 0.5)
        half, _ = sim_engine(clean, pr, 1e9, 0.5)
        si, _ = sim_engine(clean, pr, 3.0, 0.5)
        if base.empty:
            continue
        worst2 = max(worst2, float(np.max(np.abs(half.Daily_Delta.values - 0.5 * base.Daily_Delta.values))))
        worst2_end = max(worst2_end, abs(half.Cumulative_PnL.iloc[-1] - 0.5 * base.Cumulative_PnL.iloc[-1]))
        n_traded += int((si.Position != 0).any())
        has = bool((si.Status == "SCALE_IN").any() or (si.SI_Tier == 2).any())
        n_si += int(has)
        if has:
            ind = sim_independent(clean, pr, q, f)
            worst3 = max(worst3, float(np.max(np.abs(ind - si.Cumulative_PnL.values))))
            n3 += 1
    ok["V2"] = worst2 <= 1e-4 + 1e-12 and worst2_end <= 1e-3
    print(f"[V2] {len(pairs)} 個配對期：|半份 − 0.5×現行| 逐日最大 {worst2:.1e}、期末最大 {worst2_end:.1e}"
          f" → {'通過' if ok['V2'] else '失敗'}")
    ok["V3"] = n3 > 0 and worst3 < 1e-3
    print(f"[V3] 含加碼（含進場即整份）的配對期 {n3} 個：獨立重算的逐日累積損益最大差 {worst3:.1e}"
          f"（引擎輸出四捨五入至 4 位）→ {'通過' if ok['V3'] else '失敗'}")
    rate = n_si / max(n_traded, 1)
    ok["V4 rate"] = 0 < rate < 1
    print(f"[V4] 有進場的配對期中出現第二層者 {n_si}/{n_traded} = {rate:.1%} → {'通過' if ok['V4 rate'] else '失敗'}")
    print("\n全部通過" if all(ok.values()) else f"\n未通過：{[k for k, v in ok.items() if not v]}")


if __name__ == "__main__":
    main()
