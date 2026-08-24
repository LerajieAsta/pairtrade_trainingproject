"""GICS-SSD 全候選池的唯讀重建。

管線本身只保留最終選中的 20 組（formation_pairs），且 select_pairs 有提前中止
——候選依 SSD 遞增處理，湊滿 top_n 個通過 ADF 者即 break，故池子的後段從未被
碰過。ML 排序器要對全部候選評分，必須自己把池子建出來。

不走 FORMATION_TRACE=1 重跑的理由：那要嘛被 formation_progress 全數跳過，
要嘛得開 FORCE_RERUN 覆寫現有 formation_pairs，拿既有結果去換一批本來就不打算
餵進模型的 ADF/半衰期欄位。此處全程唯讀。

對帳依據（見 ssd_rolling.py:62-118）：兩條對數價序列都在形成窗內 z-score 化，
故 OLS 斜率恰為相關係數，且
    SSD = sum_t (P'_A - P'_B)^2 = 2(T-1)(1 - rho)
    beta = rho,  sigma_s^2 = 1 - rho^2
儲存的 Hedge_Ratio 為 round(beta, 4)，1-rho 很小時相對誤差被放大，對帳要用
絕對容差而非相對容差。
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd

FORMATION_WINDOW = 252
FORWARD_DAYS = 126
ROLLING_STEP = 21
STRAT_FORM = "Grid GICS-SSD_MSR0"
FORM_DB = "formation_data/formation_pairs_sp500_Tiingo.db"


def load_prices():
    """與 run_formation.py:535 完全相同的前處理路徑。"""
    from strategies.preprocess_equity import DataProcessor
    from strategies.config import (DB_PATH, TABLE_NAME, BACKTEST_START, BACKTEST_END)
    proc = DataProcessor(db_path=DB_PATH, table_name=TABLE_NAME)
    pivot, dates, total, first_idx = proc.prepare_backtest_data(
        BACKTEST_START, BACKTEST_END, FORMATION_WINDOW)
    return pivot, pd.DatetimeIndex(dates), total, first_idx


def roll_indices(total_days: int, first_idx: int) -> list[int]:
    """run_formation.py:140-143 的視窗推進，含最後一格的補齊。"""
    idxs = list(range(first_idx, total_days - FORWARD_DAYS + 1, ROLLING_STEP))
    last = total_days - FORWARD_DAYS
    if idxs and idxs[-1] != last:
        idxs.append(last)
    return idxs


def load_groups(strategy_id: str = STRAT_FORM) -> dict:
    """{Period_Start(str): {ticker: 群標籤}}，直接取自管線落地的分組結果。"""
    with sqlite3.connect(f"file:{FORM_DB}?mode=ro", uri=True) as c:
        g = pd.read_sql_query(
            "SELECT Period_Start, Ticker, Cluster_Label FROM formation_groups "
            "WHERE strategy_id = ?", c, params=(strategy_id,))
    return {p: dict(zip(sub.Ticker, sub.Cluster_Label))
            for p, sub in g.groupby("Period_Start")}


def window_pairs(form_prices: pd.DataFrame, group_map: dict) -> pd.DataFrame:
    """單一形成窗的群內全候選對。

    剔除窗內有任何缺值的標的：管線雖以「有效日 >= 30」納入，但含 NaN 者的
    SSD 與 beta 皆為 NaN，排序時落到最後、再被提前中止排除，故等價。
    """
    log_px = np.log(form_prices.where(form_prices > 0))
    usable = [t for t in log_px.columns
              if group_map.get(t, "Unknown") != "Unknown"
              and log_px[t].notna().all()]
    if len(usable) < 2:
        return pd.DataFrame()

    lp = log_px[usable]
    norm = (lp - lp.mean()) / (lp.std() + 1e-12)      # ssd_rolling.normalize_prices

    out = []
    by_group: dict[str, list[str]] = {}
    for t in usable:
        by_group.setdefault(group_map[t], []).append(t)

    for grp, members in by_group.items():
        if len(members) < 2:
            continue
        V = norm[members].values.T                    # (N, T)
        T = V.shape[1]
        R = np.corrcoef(V)                            # beta = rho（單位變異）
        iu, ju = np.triu_indices(len(members), k=1)
        # 外層 i = Ticker_B、內層 j = Ticker_A（ssd_rolling.py:101-108）
        rho = R[iu, ju]
        out.append(pd.DataFrame({
            "Group": grp,
            "Ticker_B": [members[i] for i in iu],
            "Ticker_A": [members[j] for j in ju],
            "rho": rho,
            "SSD": 2.0 * (T - 1) * (1.0 - rho),
        }))
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    df["Hedge_Ratio"] = df["rho"]
    df["Spread_Std"] = np.sqrt(np.clip(1.0 - df["rho"] ** 2, 0.0, None))
    return df
