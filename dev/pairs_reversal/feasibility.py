# -*- coding: utf-8 -*-
"""Chen, Chen, Chen & Li (2019, Management Science) 配對策略的可行性診斷。未預先登記、不入試驗宇宙（若採用另行登記）。

與本研究主軸（距離法、等收斂）的機制不同：每月對每檔股票，取過去 60 個月月報酬相關最高的 50 檔為其「配對組合」，
訊號 RetDiff = β_C × Cret_t − Lret_t（β_C 為該股對配對組合的 60 個月迴歸斜率，Cret 為配對組合當月等權報酬，
Lret 為自身當月報酬；原文另減無風險利率，此處略去）。依 RetDiff 分十組，做多第 10 組（相對配對組合落後者）、
放空第 1 組，持有下一個月。原文報告其獲利主要來自短期反轉與一個月產業動量。

母體：每月月底為 S&P 500 成分股、且有 60 個月完整月報酬者（Tiingo 調整後收盤價；2010 年前覆蓋較少，見記憶）。
成本：每月以權重變動量 × 單邊摩擦 0.29% 計（與主軸相同的摩擦率）。
對照：單純短期反轉（只用 −Lret 排序），以檢查配對組合是否有增量。

    python dev/pairs_reversal/feasibility.py
"""
import os
import sqlite3
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FRICTION = 0.0029
LOOK, NPAIR, NQ = 60, 50, 10
HALF = pd.Timestamp("2014-01-01")
OUT = "results/analysis/pairs_reversal_feasibility.csv"


def load():
    con = sqlite3.connect("file:dataset/price/sp500_Tiingo.db?mode=ro", uri=True)
    d = pd.read_sql("SELECT Date, Symbol, COALESCE(Adj_Close, Close) C FROM Daily_Prices "
                    "WHERE COALESCE(Adj_Close, Close) > 0", con)
    mem = pd.read_sql("SELECT Symbol, start_date, end_date FROM index_memberships", con)
    con.close()
    d["Date"] = pd.to_datetime(d.Date)
    px = d.pivot(index="Date", columns="Symbol", values="C").sort_index()
    m = px.resample("ME").last()
    last_obs = px.apply(lambda s: s.last_valid_index())
    ret = m.pct_change(fill_method=None)
    # 下市月之後不再有報酬（不以 ffill 造出 0 報酬）
    for s in ret.columns:
        lo = last_obs[s]
        if pd.notna(lo):
            ret.loc[ret.index > lo + pd.offsets.MonthEnd(0), s] = np.nan
    return ret.clip(-0.9, 3.0), mem


def members_at(mem, dt):
    d = dt.strftime("%Y-%m-%d")
    return set(mem[(mem.start_date <= d) & (mem.end_date.isna() | (mem.end_date >= d))].Symbol)


def legs(score, k):
    q = pd.qcut(score.rank(method="first"), NQ, labels=False)
    long_, short = score.index[q == NQ - 1], score.index[q == 0]
    w = pd.Series(0.0, index=score.index)
    w[long_] = 1.0 / len(long_)
    w[short] = -1.0 / len(short)
    return w


def main():
    ret, mem = load()
    months = ret.index
    rows, w_prev = [], {"pairs": None, "rev": None}
    for i in range(LOOK, len(months) - 1):
        t, t1 = months[i], months[i + 1]
        hist = ret.iloc[i - LOOK + 1:i + 1]
        ok = [s for s in hist.columns if hist[s].notna().all() and s in members_at(mem, t)]
        if len(ok) < 100:
            continue
        H = hist[ok]
        corr = np.corrcoef(H.values.T)
        np.fill_diagonal(corr, -np.inf)
        top = np.argsort(-corr, axis=1)[:, :NPAIR]
        Hv = H.values
        C = Hv[:, top].mean(axis=2)                 # (LOOK, n)：各股配對組合的歷史月報酬
        L = Hv
        cm, lm = C - C.mean(0), L - L.mean(0)
        beta = (cm * lm).sum(0) / (cm * cm).sum(0)
        retdiff = pd.Series(beta * C[-1] - L[-1], index=ok)
        rev = pd.Series(-L[-1], index=ok)
        nxt = ret.loc[t1, ok]
        row = {"month": t1, "n": len(ok)}
        for name, sc in (("pairs", retdiff), ("rev", rev)):
            w = legs(sc, NQ)
            valid = nxt.notna()
            # 下個月下市者以 0 報酬計、權重仍計入（保守）
            g = float((w * nxt.fillna(0.0)).sum())
            wp = w_prev[name]
            turn = float((w - wp.reindex(w.index).fillna(0.0)).abs().sum()
                         + wp[~wp.index.isin(w.index)].abs().sum()) if wp is not None else float(w.abs().sum())
            row[f"{name}_gross"], row[f"{name}_net"] = g, g - turn * FRICTION
            row[f"{name}_turn"] = turn
            row[f"{name}_missing"] = int((~valid & (w != 0)).sum())
            w_prev[name] = w
        rows.append(row)
    x = pd.DataFrame(rows).set_index("month")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    x.to_csv(OUT, encoding="utf-8-sig")

    def stat(s):
        return f"月均 {s.mean() * 100:+.3f}%（t = {s.mean() / s.std() * np.sqrt(len(s)):+.2f}）、年化 Sharpe {s.mean() / s.std() * np.sqrt(12):+.2f}"
    print(f"期間 {x.index[0]:%Y-%m} ～ {x.index[-1]:%Y-%m}，{len(x)} 個月，每月母體中位 {x.n.median():.0f} 檔")
    for name, lab in (("pairs", "Chen 等配對訊號"), ("rev", "單純短期反轉")):
        print(f"\n== {lab}（多空十分位）== 月換手率中位 {x[f'{name}_turn'].median():.2f}")
        for part, sel in (("全期", x.index == x.index), ("前半 <2014", x.index < HALF), ("後半 ≥2014", x.index >= HALF)):
            g, n = x.loc[sel, f"{name}_gross"], x.loc[sel, f"{name}_net"]
            print(f"  {part:<10} 毛：{stat(g)}｜淨：{stat(n)}")
    d = x.pairs_gross - x.rev_gross
    print(f"\n配對訊號 − 短期反轉（毛）：{stat(d)}；相關 {x[['pairs_gross', 'rev_gross']].corr().iloc[0, 1]:.2f}")


if __name__ == "__main__":
    main()
