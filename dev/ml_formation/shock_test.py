"""可行性檢定：交易期的單腳特異衝擊能否預測「不收斂」。

與今天形成期那組檢定的關鍵差別：訊號在**交易期內**到達，故用途是提早出場
而非挑配對。這也是它可能有效的原因——形成期的價格歷史裡本來就沒有「下一季
會發生什麼事」的資訊，但事件發生當下的成交量與跳空會洩漏。

特徵（皆為進場後前 K 日內的極值，只用當下已知資訊）：
    abn_vol   異常成交量 = (ln V - 過去60日中位) / 過去60日 MAD，取兩腳較大者
    gap       |ln(Open_t) - ln(Close_{t-1})|，取兩腳較大者
    asym      兩腳異常量之差的絕對值——單腳衝擊（配對關係破裂）而非雙腳同動
    park      Parkinson 波幅 (ln H - ln L)，取兩腳較大者
標籤：not_converged（沿用 build.py 的引擎忠實定義）
"""
from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dev.ml_formation.build import CACHE

LOOK = 60          # 異常量的參照窗
AFTER = 10         # 進場後觀察天數


def load_ohlcv():
    import sqlite3
    c = sqlite3.connect("file:dataset/price/sp500_Tiingo.db?mode=ro", uri=True)
    d = pd.read_sql_query(
        "select Date,Symbol,Open,High,Low,COALESCE(Adj_Close,Close) C,Volume "
        "from Daily_Prices where COALESCE(Adj_Close,Close)>0", c)
    d["Date"] = pd.to_datetime(d.Date)
    piv = lambda col: d.pivot_table(index="Date", columns="Symbol", values=col,
                                    aggfunc="last").sort_index()
    return piv("Open"), piv("High"), piv("Low"), piv("C"), piv("Volume")


def main():
    t0 = time.time()
    O, H, L, C, V = load_ohlcv()
    print(f"OHLCV 載入 {C.shape[0]} 日 x {C.shape[1]} 檔  ({time.time()-t0:.0f}s)", flush=True)

    lv = np.log(V.replace(0, np.nan))
    med = lv.rolling(LOOK, min_periods=20).median()
    mad = (lv - med).abs().rolling(LOOK, min_periods=20).median()
    ABN = (lv - med) / (mad.replace(0, np.nan) * 1.4826)          # 穩健 z
    GAP = (np.log(O) - np.log(C.shift(1))).abs()
    PARK = np.log(H) - np.log(L)

    P = pd.read_parquet(os.path.join(CACHE, "train.parquet"))
    P = P[P.label_valid & P.adf_pass].copy()
    P["Trade_Start"] = pd.to_datetime(P.Trade_Start)
    dates = C.index
    print(f"樣本 {len(P):,} 個配對-期（ADF 通過、可標記）", flush=True)

    rows = []
    for ts, g in P.groupby("Trade_Start"):
        i0 = dates.searchsorted(ts)
        for r in g.itertuples(index=False):
            a, b = r.Ticker_A, r.Ticker_B
            if a not in C.columns or b not in C.columns:
                continue
            s = i0 + int(r.entry_day)                      # 進場日
            e = min(s + AFTER, len(dates))
            if s >= len(dates) or e <= s:
                continue
            sl = slice(s, e)
            va, vb = ABN[a].values[sl], ABN[b].values[sl]
            ga, gb = GAP[a].values[sl], GAP[b].values[sl]
            pa, pb = PARK[a].values[sl], PARK[b].values[sl]
            with np.errstate(invalid="ignore"):
                rows.append({
                    "not_converged": r.not_converged,
                    "abn_vol": np.nanmax(np.maximum(va, vb)) if len(va) else np.nan,
                    "abn_asym": np.nanmax(np.abs(va - vb)) if len(va) else np.nan,
                    "gap": np.nanmax(np.maximum(ga, gb)) if len(ga) else np.nan,
                    "gap_asym": np.nanmax(np.abs(ga - gb)) if len(ga) else np.nan,
                    "park": np.nanmax(np.maximum(pa, pb)) if len(pa) else np.nan,
                    "SSD": r.SSD, "pi_date": ts,
                })
    D = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"可用 {len(D):,}  未收斂率 {D.not_converged.mean():.3f}  ({time.time()-t0:.0f}s)\n")

    from sklearn.metrics import roc_auc_score
    print(f"{'特徵':<12}{'AUC(預測不收斂)':>18}")
    for c_ in ["abn_vol", "abn_asym", "gap", "gap_asym", "park", "SSD"]:
        au = roc_auc_score(D.not_converged, D[c_])
        print(f"{c_:<12}{max(au,1-au):>18.4f}" + ("  (反向)" if au < .5 else ""))

    # 逐期（避免跨期基準率差異灌水）
    per = []
    for _p, g in D.groupby("pi_date"):
        if g.not_converged.nunique() < 2 or len(g) < 30:
            continue
        per.append({c_: roc_auc_score(g.not_converged, g[c_])
                    for c_ in ["abn_vol", "abn_asym", "gap", "gap_asym", "park", "SSD"]})
    A = pd.DataFrame(per)
    print(f"\n逐期 AUC（{len(A)} 期）：")
    for c_ in A.columns:
        print(f"  {c_:<10} {A[c_].mean():.4f} ± {A[c_].std()/np.sqrt(len(A)):.4f}")
    D.to_parquet(os.path.join(CACHE, "shock.parquet"))


if __name__ == "__main__":
    main()
