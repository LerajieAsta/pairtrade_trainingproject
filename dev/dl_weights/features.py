# -*- coding: utf-8 -*-
"""每個 (配對, 期) 的形成期特徵與交易期結果。設計見同目錄 PREREGISTRATION.md §二、§三。

結果（標的）以 Top 20 的 trade_logs 計算：引擎逐配對獨立模擬、損益對名目額線性，
故「以每對資金為分母的報酬」與 top_n 無關，可套用到任何 Top N。
"""
import json
import os
import pickle
import sqlite3
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from strategies import config as C  # noqa: E402

RESULT_DB = "results/result.db"
FORM_DB = "formation_data/formation_pairs_sp500_Tiingo.db"
CACHE = "tmp/dl_weights"
SEQ_LEN = 252
TRAIL_WIN = 12

ARMS = [f"Grid {g}-{r}" for g in ("NOGRP", "GICS", "HDB", "AGG", "KM")
        for r in ("SSD", "DTW", "SDP")]

TAB_COLS = ["rank", "log_ssd", "log_dtw", "has_dtw", "spread_std", "hedge", "same_sector",
            "z_last", "abs_z_last", "zero_cross", "ar1", "dz_std", "vol_a", "vol_b",
            "ret_corr", "has_hist", "hist_r", "hist_v", "hist_n"]


def price_pivot() -> pd.DataFrame:
    """與引擎同一份清洗後價格（DataProcessor.prepare_backtest_data），快取於 tmp/。"""
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, "price_pivot.pkl")
    if os.path.exists(f):
        with open(f, "rb") as fh:
            return pickle.load(fh)
    from strategies.preprocess_equity import DataProcessor
    proc = DataProcessor(db_path=C.DB_PATH, table_name=C.TABLE_NAME)
    pv, _, _, _ = proc.prepare_backtest_data(C.BACKTEST_START, C.BACKTEST_END, C.FORMATION_WINDOW)
    with open(f, "wb") as fh:
        pickle.dump(pv, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return pv


def cell_path(arm: str, top_n: int) -> str:
    sub = arm.replace("Grid ", "Grid_").replace("-", "_")
    return f"tiingo/{sub}/TradeLogs_Top{top_n}_SL0_ZWin0_MSR0.csv"


def outcomes(arm: str) -> pd.DataFrame:
    """每個 (配對, 期) 的 r（期報酬 / 每對資金）與 v（逐日報酬標準差）。

    回傳的 Period_Start 沿用 trade_logs 的欄名，但其值是**交易期起日**。
    """
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True, timeout=600)
    df = pd.read_sql("SELECT Period_Start, Ticker_A, Ticker_B, Daily_Delta FROM trade_logs "
                     "WHERE strategy_id = ?", con, params=(cell_path(arm, 20),))
    conc, = con.execute('SELECT Concurrent_Periods FROM strategy_summaries WHERE _path=?',
                        (cell_path(arm, 20),)).fetchone()
    con.close()
    if df.empty:
        raise RuntimeError(f"{arm} Top20 無 trade_logs")
    g = (df.groupby(["Period_Start", "Ticker_A", "Ticker_B"], sort=False)
           .Daily_Delta.agg(pnl="sum", sd="std").reset_index())
    # 引擎的每對資金 = 期初權益 / (20 × 並行期數)；期初權益 = 初始 + 先前各期的最終已實現損益
    # （run_trading.py：每期模擬完即 process_closed_trade，下一期配置時已計入）
    per = g.groupby("Period_Start").pnl.sum().sort_index()
    eq = C.INITIAL_CAPITAL + per.cumsum().shift(1, fill_value=0.0)
    cap = eq / (20 * int(conc))
    g["cap"] = g.Period_Start.map(cap)
    g["r"] = g.pnl / g.cap
    g["v"] = g.sd.fillna(0.0) / g.cap
    return g[["Period_Start", "Ticker_A", "Ticker_B", "r", "v", "pnl", "cap"]]


def _form_rows(arm: str) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{FORM_DB}?mode=ro", uri=True)
    fp = pd.read_sql("SELECT Period_Start, Trade_Start, Ticker_A, Ticker_B, Sector_A, Sector_B, "
                     "Pair_Rank, Formation_Params FROM formation_pairs WHERE strategy_id = ?",
                     con, params=(f"{arm}_MSR0",))
    con.close()
    return fp


def _seq_and_stats(pa, pb, fpar, ignore_alpha):
    """形成窗的 z 序列（與交易端 _compute_spread 同定義）與兩腳日報酬。"""
    la, lb = np.log(pa), np.log(pb)
    beta = float(fpar.get("Hedge_Ratio", 1.0))
    alpha = fpar.get("OLS_Alpha")
    sd = max(float(fpar.get("Spread_Std", 1.0) or 1.0), 1e-6)
    mu = float(fpar.get("Spread_Mean", 0.0) or 0.0)
    if alpha is not None and not ignore_alpha:
        spread = la - float(alpha) - beta * lb                     # 路徑 A
    else:
        na = (la - float(fpar["Log_Mean_A"])) / float(fpar["Log_Std_A"])
        nb = (lb - float(fpar["Log_Mean_B"])) / float(fpar["Log_Std_B"])
        spread = na - beta * nb                                    # 路徑 B
    z = np.clip((spread - mu) / sd, -10, 10)
    ra, rb = np.diff(la, prepend=la[0]), np.diff(lb, prepend=lb[0])
    return z, ra, rb


def build(arm: str, pv: pd.DataFrame, lag: int = 6):
    """回傳 (meta DataFrame, seq ndarray [n, 3, SEQ_LEN], tab ndarray [n, len(TAB_COLS)])。

    meta 含 pi（期序號）、r、v；r/v 為 NaN 表示該配對無交易紀錄（模擬失敗或無價格）。
    """
    f = os.path.join(CACHE, f"feat_{arm.replace(' ', '_')}.pkl")
    if os.path.exists(f):
        with open(f, "rb") as fh:
            return pickle.load(fh)

    if pv is None:
        raise RuntimeError(f"{arm} 特徵快取不存在，需傳入價格矩陣")
    fp = _form_rows(arm)
    out = outcomes(arm)
    # trade_logs 的 Period_Start 其實是交易期起日（= formation_pairs 的 Trade_Start）
    out = out.rename(columns={"Period_Start": "Trade_Start"})
    fp = fp.merge(out, on=["Trade_Start", "Ticker_A", "Ticker_B"], how="left")
    periods = sorted(fp.Period_Start.unique())
    pos = {p: i for i, p in enumerate(periods)}
    fp["pi"] = fp.Period_Start.map(pos)
    fp = fp.sort_values(["pi", "Pair_Rank"]).reset_index(drop=True)

    ignore_alpha = bool(next(s for s in C.strategies_raw_all if s["name"] == arm)["params"]
                        .get("ignore_ols_alpha", False))
    idx = pv.index
    n = len(fp)
    seq = np.zeros((n, 3, SEQ_LEN), dtype=np.float32)
    tab = np.full((n, len(TAB_COLS)), np.nan, dtype=np.float64)

    # 自身戰績：同一 (A, B) 在 pj ≤ pi − lag 的最近 TRAIL_WIN 期
    hist = {}
    for (a, b), g in fp.dropna(subset=["r"]).groupby(["Ticker_A", "Ticker_B"]):
        hist[(a, b)] = (g.pi.to_numpy(), g.r.to_numpy(), g.v.to_numpy())

    for i, row in enumerate(fp.itertuples(index=False)):
        fpar = json.loads(row.Formation_Params)
        lo = idx.searchsorted(pd.Timestamp(row.Period_Start))
        hi = idx.searchsorted(pd.Timestamp(row.Trade_Start))   # 不含交易首日
        if row.Ticker_A not in pv.columns or row.Ticker_B not in pv.columns:
            continue
        pa = pv[row.Ticker_A].to_numpy()[lo:hi]
        pb = pv[row.Ticker_B].to_numpy()[lo:hi]
        ok = np.isfinite(pa) & np.isfinite(pb) & (pa > 0) & (pb > 0)
        if ok.sum() < 30:
            continue
        pa, pb = pa[ok], pb[ok]
        try:
            z, ra, rb = _seq_and_stats(pa, pb, fpar, ignore_alpha)
        except (KeyError, TypeError, ValueError):
            continue
        L = min(len(z), SEQ_LEN)
        seq[i, 0, -L:] = z[-L:]
        seq[i, 1, -L:] = ra[-L:] * 10
        seq[i, 2, -L:] = rb[-L:] * 10

        dz = np.diff(z)
        zc = np.mean(np.sign(z[1:]) != np.sign(z[:-1]))
        ar1 = np.corrcoef(z[1:], z[:-1])[0, 1] if np.std(z) > 0 else 1.0
        dtw = fpar.get("DTW_Dist")
        h = hist.get((row.Ticker_A, row.Ticker_B))
        if h is not None:
            m = h[0] <= row.pi - lag
            take = np.nonzero(m)[0][-TRAIL_WIN:]
        else:
            take = np.array([], dtype=int)
        tab[i] = [
            row.Pair_Rank / 20.0,
            np.log1p(float(fpar.get("SSD", 0.0) or 0.0)),
            np.log1p(float(dtw)) if dtw is not None else 0.0,
            1.0 if dtw is not None else 0.0,
            float(fpar.get("Spread_Std", 0.0) or 0.0),
            float(fpar.get("Hedge_Ratio", 1.0) or 1.0),
            1.0 if row.Sector_A == row.Sector_B else 0.0,
            z[-1], abs(z[-1]), zc, ar1, np.std(dz),
            np.std(ra[-60:]) * np.sqrt(252), np.std(rb[-60:]) * np.sqrt(252),
            np.corrcoef(ra[1:], rb[1:])[0, 1] if np.std(ra) > 0 and np.std(rb) > 0 else 0.0,
            1.0 if len(take) else 0.0,
            float(np.mean(h[1][take])) if len(take) else 0.0,
            float(np.mean(h[2][take])) if len(take) else 0.0,
            len(take) / TRAIL_WIN,
        ]

    meta = fp[["Period_Start", "Trade_Start", "Ticker_A", "Ticker_B", "Pair_Rank", "pi", "r", "v"]].copy()
    meta["feat_ok"] = np.isfinite(tab).all(axis=1)
    res = (meta, seq, tab)
    os.makedirs(CACHE, exist_ok=True)
    with open(f, "wb") as fh:
        pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return res
