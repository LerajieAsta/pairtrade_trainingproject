# -*- coding: utf-8 -*-
"""部分共整合（Clegg & Krauss 2018）的可行性診斷：資料建構。未預先登記、不跑回測、不入試驗宇宙。

對 `Grid GICS-SSD-FW504` 的每一期（295 期，formation DB 的期別與日期）：
  1. 以引擎同一形成期程式重跑，取 SSD 前 40 名（前 20 名須與 formation DB 逐名次相同，作為接線檢查）；
  2. 在 504 日形成期價差 S_t = P̃_A − β P̃_B 上估計部分 AR(1)：
         S_t = M_t + R_t，M_t = ρ M_{t−1} + ε^M，R_t = R_{t−1} + ε^R
     以 Kalman 濾波的最大概似估 (ρ, σ_M, σ_R)，β 固定為 SSD 臂的避險比率（原文聯合估計 β，此為簡化）；
     R²_MR = 2σ_M² / (2σ_M² + (1+ρ)σ_R²)（原文定義：ΔS 變異中均值回歸成分的比例）；
  3. 交易期 126 日的引擎忠實標籤：z 以形成期參數標準化，首次 |z| > 2 進場，之後穿越 0 → 收斂；
     另記首筆交易的報酬代理 = 進場方向 × (z_進 − z_出) × sd(S)/D − 2f（z_出 = 穿越 0 或期末）。

輸出 results/analysis/pci_feasibility.pkl（每列一個候選對-期）。

    python dev/pci_feasibility/build.py
"""
import inspect
import os
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numba import njit  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

FORM_DB = "formation_data/formation_pairs_sp500_Tiingo.db"
SID = "Grid GICS-SSD-FW504"
OUT = "results/analysis/pci_feasibility.pkl"
TOP_K = 40


@njit(cache=True)
def _negll(theta, s):
    rho = np.tanh(theta[0])
    vm = np.exp(2.0 * theta[1])
    vr = np.exp(2.0 * theta[2])
    m, r = 0.0, s[0]
    pmm, pmr, prr = vm / max(1.0 - rho * rho, 1e-6), 0.0, 0.0
    ll = 0.0
    for t in range(1, s.shape[0]):
        # 預測
        m = rho * m
        pmm, pmr, prr = rho * rho * pmm + vm, rho * pmr, prr + vr
        # 更新（觀測 S = M + R，無量測誤差）
        f = pmm + 2.0 * pmr + prr
        if f <= 1e-14:
            f = 1e-14
        v = s[t] - m - r
        km, kr = (pmm + pmr) / f, (pmr + prr) / f
        m, r = m + km * v, r + kr * v
        pmm, pmr, prr = pmm - km * (pmm + pmr), pmr - km * (pmr + prr), prr - kr * (pmr + prr)
        ll += np.log(f) + v * v / f
    return 0.5 * ll


def fit_pci(s):
    sd = np.std(np.diff(s))
    best = None
    for rho0 in (0.5, 0.9):
        for share in (0.9, 0.5, 0.1):
            x0 = np.array([np.arctanh(rho0), np.log(sd * np.sqrt(share) + 1e-9),
                           np.log(sd * np.sqrt(1 - share) + 1e-9)])
            res = minimize(_negll, x0, args=(s,), method="L-BFGS-B",
                           bounds=[(-3.8, 3.8), (np.log(sd) - 12, np.log(sd) + 3), (np.log(sd) - 12, np.log(sd) + 3)])
            if best is None or res.fun < best.fun:
                best = res
    rho, sm, sr = np.tanh(best.x[0]), np.exp(best.x[1]), np.exp(best.x[2])
    r2 = 2 * sm ** 2 / (2 * sm ** 2 + (1 + rho) * sr ** 2)
    return rho, sm, sr, r2, best.fun


def label(z, rpz, f2):
    hit = np.nonzero(np.abs(z) > 2.0)[0]
    if len(hit) == 0:
        return np.nan, np.nan
    i = hit[0]
    side = -np.sign(z[i])                       # z>0 → 做空價差
    after = z[i + 1:]
    cross = np.nonzero(after <= 0)[0] if side < 0 else np.nonzero(after >= 0)[0]
    if len(cross):
        z_out, conv = 0.0, 1.0
    else:
        z_out, conv = z[-1], 0.0
    cap = (z[i] - z_out) * (-side) * rpz - f2
    return conv, cap


_G = {}


def _init():
    from strategies import config as C
    from strategies.preprocess_equity import DataProcessor
    proc = DataProcessor(db_path=C.DB_PATH, table_name=C.TABLE_NAME)
    _G["sector"] = proc.load_sector_mapping(C.INFO_TABLE, C.TICKER_COL, C.SECTOR_COL)
    pv, *_ = proc.prepare_backtest_data("2000-01", C.BACKTEST_END, 252)
    from strategies.trading.zscore_trading import clean_prices
    # clean_prices 會把結果快取在傳入物件的 attrs；pandas 切片會深拷貝 attrs，故對副本清洗，pv 保持無 attrs
    _G["pv"], _G["clean"] = pv, clean_prices(pv.copy())
    pv.attrs.clear()
    con = sqlite3.connect(f"file:{C.DB_PATH}?mode=ro", uri=True)
    _G["mem"] = pd.read_sql("SELECT Symbol, start_date, end_date FROM index_memberships", con)
    con.close()
    raw = next(s for s in C.strategies_raw_all if s["name"] == SID)
    from strategies.formation.cluster_formation import Formation as CF
    _G["CF"], _G["params"] = CF, dict(raw["params"])
    _G["f2"] = 2 * (float(C.base_params["fee_rate"]) + float(C.base_params["slippage_rate"]))
    del proc
    import gc
    gc.collect()


def run_period(args):
    ps, ts, te = args
    pv, mem, CF = _G["pv"], _G["mem"], _G["CF"]
    dates = pv.index
    i_ts = dates.get_loc(pd.Timestamp(ts))
    fp = pv.iloc[i_ts - 504:i_ts]
    assert fp.index[0] == pd.Timestamp(ps), (ps, fp.index[0])
    fe = fp.index[-1].strftime("%Y-%m-%d")
    act = set(mem[(mem.start_date <= fe) & (mem.end_date.isna() | (mem.end_date >= fe))].Symbol)
    fp = fp[[c for c in fp.columns if c in act]].dropna(axis=1)
    p = dict(_G["params"])
    p["top_n"] = TOP_K
    ok = set(inspect.signature(CF.__init__).parameters)
    kw = {k: v for k, v in {"price_df": fp, "form_start": ps, "form_end": fe,
                            "sector_mapping": _G["sector"], **p}.items() if k in ok}
    sel = CF(**kw).run()
    if sel is None or len(sel) == 0:
        return pd.DataFrame()
    tp = _G["clean"].loc[ts:te]
    rows = []
    for rank, r in enumerate(sel.itertuples(index=False), 1):
        a, b, beta = r.Ticker_A, r.Ticker_B, float(r.Hedge_Ratio)
        la, lb = np.log(fp[a].values), np.log(fp[b].values)
        mua, sda, mub, sdb = la.mean(), la.std(), lb.mean(), lb.std()
        s = (la - mua) / sda - beta * (lb - mub) / sdb
        rho, sm, sr, r2, nll = fit_pci(s.astype(np.float64))
        st = (np.log(tp[a].values) - mua) / sda - beta * (np.log(tp[b].values) - mub) / sdb
        mean, sd = s.mean(), s.std(ddof=1)
        z = np.clip((st - mean) / sd, -10, 10)
        rpz = sd / (1 / sda + abs(beta) / sdb)
        conv, cap = label(z, rpz, _G["f2"])
        rows.append({"Period_Start": ps, "rank": rank, "Ticker_A": a, "Ticker_B": b, "Sector": r.Sector,
                     "beta": beta, "SSD": r.SSD, "rho": rho, "sigma_M": sm, "sigma_R": sr, "R2_MR": r2,
                     "phi_ar1": np.corrcoef(s[:-1], s[1:])[0, 1], "conv": conv, "cap": cap})
    return pd.DataFrame(rows)


def main():
    con = sqlite3.connect(f"file:{FORM_DB}?mode=ro", uri=True, timeout=600)
    per = con.execute("SELECT DISTINCT Period_Start, Trade_Start, Trade_End FROM formation_pairs "
                      "WHERE strategy_id=? ORDER BY Period_Start", (f"{SID}_MSR0",)).fetchall()
    db = pd.read_sql("SELECT Period_Start, Ticker_A, Ticker_B, Pair_Rank FROM formation_pairs WHERE strategy_id=?",
                     con, params=(f"{SID}_MSR0",))
    con.close()
    if len(sys.argv) > 1:
        per = per[:int(sys.argv[1])]
    with ProcessPoolExecutor(max_workers=4, initializer=_init) as ex:
        parts = list(ex.map(run_period, per, chunksize=2))
    x = pd.concat(parts, ignore_index=True)
    # 接線檢查：前 20 名與 formation DB 逐名次相同
    top = x[x["rank"] <= 20].merge(db, left_on=["Period_Start", "rank"], right_on=["Period_Start", "Pair_Rank"],
                                   suffixes=("", "_db"), how="left")
    same = ((top.Ticker_A == top.Ticker_A_db) & (top.Ticker_B == top.Ticker_B_db)).mean()
    print(f"期數 {x.Period_Start.nunique()}／{len(per)}；候選列 {len(x)}；前 20 名與 formation DB 相同比例 {same:.4f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    x.to_pickle(OUT)


if __name__ == "__main__":
    main()
