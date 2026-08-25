"""
形成期共用排序層（與特徵、分群無關）
======================================================================
把「群標籤 → 群內共整合篩選 + 距離排序 → 選 top_n 配對」抽成中性
dispatcher，讓排序準則可插拔：
    pairs = rank_within_groups(method, price_df, form_start, form_end,
                               group_map, top_n, **params)

group_map = {ticker: 群標籤}（噪音/未分群 → "Unknown"，各 backend 自動跳過）。

backend 對應現役排序核心（本身亦為獨立現役策略，故以委派而非搬移）：
    "ssd"         → ssd_rolling.Formation
                    （min-SSD 排序 + Engle-Granger ADF + 半衰期 + Hurst）
    "dtw"         → DTW_Cointegration_Paper.Formation（method="dtw"）
                    （雙向 OLS + ADF + Sakoe-Chiba DTW 距離升序）
    "ssd_dtw_pca" → 同上（method="ssd_dtw_pca"，SSD⊕DTW 之 PCA 第一主成分排序）

任何分群策略（HDBSCAN / Agglomerative / K-means / …）都經此統一介面選排序
準則，不必各自 import 具體排序策略模組。
"""
import pandas as pd


def rank_within_groups(
    method: str,
    price_df: pd.DataFrame,
    form_start: str,
    form_end: str,
    group_map: dict,
    top_n: int,
    min_tickers_for_pairing: int = 2,
    adf_pvalue_threshold: float = None,
    trading_window: int = 126,
    dtw_window: int = 15,
    enable_filters: bool = True,   # False → 跳過三道統計過濾（純排序消融）
    trace: list = None,            # 非 None 時，各後端附上排序分數與三道檢定統計量
    adf_only: bool = False,        # True → 篩選層只做 ADF
    normalize_mode: str = "zscore_log",   # "ggr_index" → GGR 累積報酬指數（見 dev/ggr/SPEC.md）
    **_ignored,
) -> pd.DataFrame:
    """
    在給定分組（group_map）內做共整合篩選與距離排序，回傳 top_n 配對。

        method: "ssd" | "dtw" | "ssd_dtw_pca"
        group_map: {ticker: 群標籤}，"Unknown" 組會被排序端跳過
    回傳的 DataFrame 由對應排序策略產生（欄位含 Ticker_A/B、Hedge_Ratio、
      Spread_Mean/Std、Log_Mean/Std、以及 DTW 系的 OLS_Alpha/DTW_Dist 等）。
    """
    if method == "ssd":
        from strategies.formation.ssd_rolling import Formation as _SSDFormation
        kw = dict(
            price_df=price_df,
            form_start=form_start,
            form_end=form_end,
            top_n=top_n,
            sector_mapping=group_map,
            min_tickers_for_pairing=min_tickers_for_pairing,
            trading_window=trading_window,
            enable_filters=enable_filters,
            trace=trace,
            adf_only=adf_only,
            normalize_mode=normalize_mode,
        )
        if adf_pvalue_threshold is not None:
            kw["adf_pvalue_threshold"] = adf_pvalue_threshold
        return _SSDFormation(**kw).run()

    if method in ("dtw", "ssd_dtw_pca"):
        from strategies.formation.DTW_Cointegration_Paper import Formation as _DTWFormation
        kw = dict(
            price_df=price_df,
            form_start=form_start,
            form_end=form_end,
            top_n=top_n,
            sector_mapping=group_map,
            min_tickers_for_pairing=min_tickers_for_pairing,
            dtw_window=dtw_window,
            method=method,
            trading_window=trading_window,
            enable_filters=enable_filters,
            trace=trace,
            adf_only=adf_only,
            normalize_mode=normalize_mode,
        )
        if adf_pvalue_threshold is not None:
            kw["adf_pvalue_threshold"] = adf_pvalue_threshold
        return _DTWFormation(**kw).run()

    if method == "reversal":
        # Han, He & Toh (2021) 的選對準則：群內短期反轉（過去一個月報酬發散）。
        # 與 ssd/dtw/sdp 的「歷史相似度」不同——押的是單月反轉而非價差平穩性。
        # 對沖固定 1.0，故須搭配 distance_trading（等權距離 spread）使用。
        from strategies.formation._reversal import rank_by_reversal
        return rank_by_reversal(
            price_df=price_df,
            form_start=form_start,
            form_end=form_end,
            group_map=group_map,
            top_n=top_n,
            min_tickers_for_pairing=min_tickers_for_pairing,
            adf_pvalue_threshold=(adf_pvalue_threshold
                                  if adf_pvalue_threshold is not None else 0.05),
            trading_window=trading_window,
            enable_filters=enable_filters,
            trace=trace,
            adf_only=adf_only,
            **{k: v for k, v in _ignored.items()
               if k in ("lookback", "sd_mult", "sd_scope")},
        )

    raise ValueError(
        f"未知排序方法 '{method}'，可選：ssd / dtw / ssd_dtw_pca / reversal")
