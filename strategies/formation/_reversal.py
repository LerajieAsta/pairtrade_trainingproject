# -*- coding: utf-8 -*-
"""
群內短期反轉排序（Han, He & Toh 2021 的選對準則）
======================================================================
本研究既有的三種排序準則（SSD / DTW / SSD-DTW-PCA）皆為**歷史相似度**度量：
問「這兩檔過去走得多近」。Han, He & Toh (2021) *Pairs Trading via Unsupervised
Learning* 的選對準則不同——它問「這兩檔**上個月**分開了多少」：

  > we open long and short positions if the return diﬀerence is greater than
  > one cross-sectional standard deviation of the diﬀerence of past one-month
  > returns. The long-short portfolio is rebalanced and reinvested at the end
  > of every month.

即：群內兩兩計算過去一個月的報酬差 d_ij = r_i − r_j，取 |d_ij| 超過該期
橫斷面標準差者，做多輸家、做空贏家，持有一個月後全部重開。

押注的經濟機制與本研究其他排序準則**不同**：
  - SSD/DTW/SDP + 共整合篩選 → 押「價差平穩，現在偏離均值會回歸」
  - 本模組              → 押「相似的兩檔上月分開，下月會收斂」（短期反轉）

方向約定
--------
`zscore_trading._execute_entry` 於 z > entry_z 時做空 A、做多 B。搭配
`distance_trading` 的 spread D = P̃_A − P̃_B（標準化 log price、hedge=1），
故本模組把**上月贏家放 Ticker_A**，使交易端自然執行「空贏家、多輸家」。

對沖比例固定 1.0（等金額），對應原文 "the stocks in a pair should be equally
weighted, i.e., buy one stock and sell the other for the same amount"。

已知歧義
--------
原文 "cross-sectional standard deviation" 未明言是群內或全體。本模組預設
**全體候選配對池**（pooled，`sd_scope="pooled"`），並提供 `sd_scope="group"`
供敏感性檢查——兩者結論不同時須據實揭露。
"""
import itertools

import numpy as np
import pandas as pd

from strategies.formation._cointegration import screen_pair


def rank_by_reversal(
    price_df: pd.DataFrame,
    form_start: str,
    form_end: str,
    group_map: dict,
    top_n: int,
    lookback: int = 21,              # 「過去一個月」的交易日數
    sd_mult: float = 1.0,            # 發散門檻（幾倍橫斷面 SD）
    sd_scope: str = "pooled",        # "pooled"（全體候選）| "group"（群內）
    min_tickers_for_pairing: int = 2,
    adf_pvalue_threshold: float = 0.05,
    trading_window: int = 126,
    enable_filters: bool = False,    # Han et al. 不施加統計篩選 → 預設關閉
    **_ignored,
) -> pd.DataFrame:
    """群內短期反轉排序，回傳與其他排序 backend 同構的 DataFrame。"""
    px = price_df.dropna(axis=1, how="any")
    if px.shape[1] < min_tickers_for_pairing or len(px) < lookback + 2:
        return pd.DataFrame()

    log_px = np.log(px)
    mean_p, std_p = log_px.mean(), log_px.std(ddof=1)
    # 過去一個月報酬（形成窗末端 lookback 個交易日）
    ret_1m = (log_px.iloc[-1] - log_px.iloc[-1 - lookback]).to_dict()

    # 群內兩兩列舉；"Unknown" 組與過小組跳過（與其他 backend 一致）
    groups = {}
    for t in px.columns:
        g = group_map.get(t, "Unknown")
        if g != "Unknown":
            groups.setdefault(g, []).append(t)

    recs = []
    for g, members in groups.items():
        if len(members) < min_tickers_for_pairing:
            continue
        for t1, t2 in itertools.combinations(members, 2):
            d = ret_1m[t1] - ret_1m[t2]
            # 上月贏家放 Ticker_A → 交易端做空 A、做多 B
            a, b = (t1, t2) if d > 0 else (t2, t1)
            recs.append({"Sector": g, "Ticker_A": a, "Ticker_B": b,
                         "Divergence": abs(d)})
    if not recs:
        return pd.DataFrame()

    df = pd.DataFrame(recs)

    # 發散門檻：|d| > sd_mult × 橫斷面 SD
    if sd_scope == "group":
        thr = df.groupby("Sector").Divergence.transform(lambda s: s.std(ddof=1))
    else:
        thr = pd.Series(df.Divergence.std(ddof=1), index=df.index)
    df = df[df.Divergence > sd_mult * thr.fillna(np.inf)]
    if df.empty:
        return pd.DataFrame()

    # 發散越大越優先（原文以門檻取「足夠發散者」，本研究需固定 top_n 名額）
    df = df.sort_values("Divergence", ascending=False).reset_index(drop=True)

    # 可選的統計篩選（Han et al. 不做；保留供消融）
    out, halflife_max = [], max(2.0, trading_window / 3.0)
    for _, r in df.iterrows():
        ta, tb = r.Ticker_A, r.Ticker_B
        # spread 與交易端一致：標準化 log price 之差，hedge = 1
        sa = (log_px[ta].values - mean_p[ta]) / (std_p[ta] or 1.0)
        sb = (log_px[tb].values - mean_p[tb]) / (std_p[tb] or 1.0)
        spread = sa - sb
        passed, _ = screen_pair(
            spread, adf_pvalue_threshold=adf_pvalue_threshold,
            halflife_min=1.0, halflife_max=halflife_max,
            hurst_threshold=0.50, adf_max_lags=1, enabled=enable_filters)
        if not passed:
            continue
        out.append({
            "Form_Start": form_start, "Form_End": form_end,
            "Sector": r.Sector, "Ticker_A": ta, "Ticker_B": tb,
            "Divergence": round(float(r.Divergence), 6),
            "Hedge_Ratio": 1.0,
            "Spread_Mean": round(float(np.mean(spread)), 6),
            "Spread_Std": round(float(np.std(spread, ddof=1)), 6),
            "Log_Mean_A": float(mean_p[ta]), "Log_Std_A": float(std_p[ta]),
            "Log_Mean_B": float(mean_p[tb]), "Log_Std_B": float(std_p[tb]),
        })
        if len(out) >= top_n:
            break

    if not out:
        return pd.DataFrame()
    res = pd.DataFrame(out)
    res["Rank"] = range(1, len(res) + 1)
    return res
