# -*- coding: utf-8 -*-
"""策略績效指標的唯一定義（唯讀）。

    metrics_from_pnl(daily_pnl)       純函式核心（輸入為損益金額）
    metrics_from_returns(returns)     純函式核心（輸入為報酬率）
    metrics(path_key, dates=None)     取 trade_logs 並包裝核心
    traded_notional(path_key, top_n)  實付手續費所依據的名目額

⚠ 三種輸入型別對應三個入口。**不要為了統一而強轉**——報酬率序列不帶資金規模，
硬轉成損益會憑空造出一條權益曲線。

## 為何要有這支

「一個策略表現如何」原本沒有 seam。`db_utils.calculate_metrics_from_params`
是權威（它產生 `result.db` 的欄位），但另有 13 處各自重算，且已經分歧：

  年化 —— `db_utils` 用**月頻**、13 處讀端用**日頻**，三種算法無任何兩種完全一致
  Sharpe 最小長度守衛 —— `len>20` / `len>100` / **無守衛** 三種並存
  MDD —— 一致（`(E−max)/max` 與 `E/max−1` 代數相同）

`dashboard.py:1493` 的註解「口徑與 db_utils 一致」是靠人手維護的宣稱，
而 `compute_range_metrics` 的 docstring 也只能說「口徑一致」——沒有結構保證。

代價已經付過兩次，都是名目額算錯：

  2026-08-26  `analysis/regime_cost_dsr_eval.py` —— 名目額漏 ×6 並行因子、
              事件數漏強平與停損，10 條主力臂的 break-even 全數低估 0.20–0.40 pp
  2026-08-27  `analysis/regime_cost_ew.py` —— **同一組錯誤的第二份副本**，
              修第一份時漏掉。它產生的 breakeven_ew.csv 為論文 5.3
              「成本餘裕過薄」所引用；修正後區間由 0.541–0.669% 變 0.391–0.924%

## 契約：精確重現 db_utils

本 module **不定義自己的口徑**，只把 `db_utils` 那一套抽出來供讀端共用。
驗證條件因此是客觀的：`metrics(p)` 對 `strategy_summaries` 的每一列逐欄比對。

這表示 `db_utils` 的幾個非顯然慣例一併移植，**不是疏忽而是刻意**：

  · 年化取**月頻**：`(1+cum_ret)^(12/n_months) − 1`，而 `cum_ret` 為月頻報酬
    連乘且第一個月 `fillna(0)`——故其基準是**第一個月底權益**，不是初始資金
  · 日報酬分母為 `Equity.shift(1).fillna(INITIAL_CAPITAL)`
  · Sharpe/Sortino 的守衛是 `std != 0`，**沒有最小長度限制**
  · 交易統計的分組鍵含 `Period_Start`（缺少它會使 Entries 高估、Forced_Closes 低估，
    見 `db_utils.py:344` 的說明）

若日後要改口徑，改 `db_utils` 並重跑，不要在此處分岔。
"""

import os
import sqlite3

import numpy as np
import pandas as pd

from strategies.config import INITIAL_CAPITAL, RF_ANNUAL, CONCURRENT_PERIODS

RESULT_DB = "results/result.db"
TRADING_DAYS = 252

#: 現行單邊摩擦成本（Do & Faff 2012 對美股 pairs trading 的估計）
CURRENT_FEE_SIDE = 0.0029


def metrics_from_pnl(daily_pnl: pd.Series,
                     initial_capital: float = INITIAL_CAPITAL) -> pd.Series:
    """由逐日損益金額算報酬類指標。純函式，不碰 DB。

    Parameters
    ----------
    daily_pnl
        index 為日期、值為當日損益**金額**（非報酬率）。
        反事實分析（重新縮放、加回費用、政體遮罩）改造過的序列亦可直接傳入——
        這正是此低階入口存在的理由。

    Returns
    -------
    Series
        `Cum_Ret_Raw`, `Ann_Ret_Raw`, `Sharpe_Raw`, `Sortino_Raw`,
        `MDD_Raw`, `Calmar_Raw`, `Final_Equity`

    Notes
    -----
    口徑逐項對齊 `db_utils.calculate_metrics_from_params`，包含其月頻年化與
    `fillna(INITIAL_CAPITAL)` 的日報酬分母。差異即為 bug，不是改良。
    """
    s = pd.Series(daily_pnl).dropna().sort_index()
    if s.empty:
        return pd.Series({k: 0.0 for k in (
            "Cum_Ret_Raw", "Ann_Ret_Raw", "Sharpe_Raw", "Sortino_Raw",
            "MDD_Raw", "Calmar_Raw", "Final_Equity")})

    equity = initial_capital + s.cumsum()
    equity.index = pd.to_datetime(equity.index)

    # 年化：月頻。cum_ret 的基準是第一個月底權益（第一個月報酬被 fillna(0)）。
    monthly_equity = equity.resample("ME").last().dropna()
    if len(monthly_equity) > 0:
        monthly_returns = monthly_equity.pct_change().fillna(0)
        cum_ret = float(np.prod(1 + monthly_returns) - 1)
        n_months = len(monthly_returns)
        ann_ret = ((1 + cum_ret) ** (12 / n_months)) - 1 if n_months > 0 else 0.0
    else:
        cum_ret = ann_ret = 0.0

    prev_equity = equity.shift(1).fillna(initial_capital)
    daily_returns = s / prev_equity
    sd = daily_returns.std()
    sharpe = float(np.sqrt(TRADING_DAYS) * daily_returns.mean() / sd) if sd != 0 else 0.0

    neg = daily_returns[daily_returns < 0]
    sortino = (float(np.sqrt(TRADING_DAYS) * daily_returns.mean() / neg.std())
               if (len(neg) > 0 and neg.std() != 0) else 0.0)

    roll_max = equity.cummax()
    mdd = float(((equity - roll_max) / roll_max).min())
    calmar = float(ann_ret / abs(mdd)) if mdd != 0 else 0.0

    return pd.Series({
        "Cum_Ret_Raw": float(cum_ret), "Ann_Ret_Raw": float(ann_ret),
        "Sharpe_Raw": sharpe, "Sortino_Raw": sortino,
        "MDD_Raw": mdd, "Calmar_Raw": calmar,
        "Final_Equity": float(equity.iloc[-1]),
    })


def metrics_from_returns(daily_returns: pd.Series) -> pd.Series:
    """由逐日**報酬率**算報酬類指標。

    第三種輸入型別。`strategies.returns.daily_returns()` 的輸出直接適用——
    多支分析腳本（`compare_fw504` / `compare_tw63` / `gate1_overlay` /
    兩支 `gate2_backtest`）拿到的都是報酬率而非損益金額。

    與 `metrics_from_pnl` 的關係：`db_utils` 的日報酬定義為
    `Daily_Delta / Equity.shift(1).fillna(INITIAL_CAPITAL)`，故本函式的輸入
    恰為該序列。Sharpe 與 Sortino 因此逐位一致；**年化則由複利連乘導出**
    （$\prod(1+r)^{252/n} - 1$），這是報酬率序列唯一自洽的年化方式，
    與 `metrics_from_pnl` 的月頻年化在數值上極接近（實測差 ±0.02 pp）但非同一式。

    ⚠ **兩者不可混用於同一張表。** 要並列比較時，一律走 `metrics_from_pnl`
    或一律走本函式。

    Returns
    -------
    Series
        `Ann_Ret_Raw`, `Sharpe_Raw`, `Sortino_Raw`, `MDD_Raw`, `Calmar_Raw`
        （不含 `Final_Equity`／`Cum_Ret_Raw`——報酬率序列不帶資金規模資訊）
    """
    r = pd.Series(daily_returns).dropna()
    if len(r) == 0:
        return pd.Series({k: np.nan for k in (
            "Ann_Ret_Raw", "Sharpe_Raw", "Sortino_Raw", "MDD_Raw", "Calmar_Raw")})

    ann = float((1.0 + r).prod() ** (TRADING_DAYS / len(r)) - 1.0)
    sd = r.std()
    sharpe = float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd != 0 else 0.0
    neg = r[r < 0]
    sortino = (float(r.mean() / neg.std() * np.sqrt(TRADING_DAYS))
               if (len(neg) > 0 and neg.std() != 0) else 0.0)
    eq = (1.0 + r).cumprod()
    mdd = float((eq / eq.cummax() - 1.0).min())
    calmar = float(ann / abs(mdd)) if mdd != 0 else 0.0
    return pd.Series({"Ann_Ret_Raw": ann, "Sharpe_Raw": sharpe,
                      "Sortino_Raw": sortino, "MDD_Raw": mdd, "Calmar_Raw": calmar})


def _daily_frame(path_key: str, result_db: str = RESULT_DB) -> pd.DataFrame:
    """逐日彙總。**必須在 pandas 端聚合，不可下推到 SQL。**

    SQLite 的 `SUM()` 與 pandas 的 `.sum()` 加總順序不同，末位可能差一個 ulp
    （實測 6,287 個交易日中約 26% 不逐位相同，最大差 3e-14）。這對 Sharpe 無影響，
    但 `Sortino` 的分母是 `daily_returns[daily_returns < 0].std()`——
    **子集成員資格對零的浮點比較敏感**。

    實測 `Grid (GICS-SSD) Top20/SL5/EZ22`：某日加總在 SQL 下為 −1.13e-16、
    在 pandas 下為 0.0，該日因此在負值子集中進出，使 Sortino 位移 2.65e-05。
    全庫 1,392 列中有約 10 列受此影響（其餘九欄不受影響，皆 100% 精確）。

    故此處讀原始列後在 pandas 聚合，與 `db_utils` 逐位一致。
    """
    con = sqlite3.connect(f"file:{result_db}?mode=ro", uri=True)
    try:
        raw = pd.read_sql(
            "SELECT Date, Daily_Delta, Position FROM trade_logs WHERE strategy_id = ?",
            con, params=(path_key,))
    finally:
        con.close()
    if raw.empty:
        return raw
    d = raw.groupby("Date")["Daily_Delta"].sum().sort_index()
    n_open = (raw[raw["Position"] != 0].groupby("Date").size()
              .reindex(d.index, fill_value=0))
    out = pd.DataFrame({"d": d, "n_open": n_open})
    out.index = pd.to_datetime(out.index)
    return out


def metrics(path_key: str, dates=None, top_n: int = None,
            result_db: str = RESULT_DB) -> pd.Series:
    """該策略的績效指標，欄名與 `result.db` 的 `strategy_summaries` 一致。

    Parameters
    ----------
    dates
        None 為全期；給一組日期（DatetimeIndex 或可轉換者）則只計那些交易日。
        區間分析與政體遮罩兩種需求由此同一參數涵蓋。
    top_n
        提供時一併計算 `Avg_Utilization` / `Ann_Ret_Employed` / `Excess_Ret_RF`
        （需要槽位數 = top_n × 並行期數才定義得出）。
    """
    df = _daily_frame(path_key, result_db)
    if df.empty:
        return metrics_from_pnl(pd.Series(dtype=float))
    if dates is not None:
        df = df.loc[df.index.isin(pd.DatetimeIndex(dates))]
        if df.empty:
            return metrics_from_pnl(pd.Series(dtype=float))

    out = metrics_from_pnl(df["d"])

    if top_n:
        max_pairs = int(top_n) * CONCURRENT_PERIODS
        years = len(df) / float(TRADING_DAYS)
        final_pnl = float(df["d"].sum())
        if years > 0 and max_pairs > 0:
            mean_open = float(df["n_open"].mean())
            out["Avg_Utilization"] = mean_open / max_pairs
            ann_arith = final_pnl / INITIAL_CAPITAL / years
            avg_employed = mean_open * (INITIAL_CAPITAL / max_pairs)
            out["Ann_Ret_Employed"] = (final_pnl / avg_employed / years
                                       if avg_employed > 0 else 0.0)
            out["Excess_Ret_RF"] = ann_arith - RF_ANNUAL * out["Avg_Utilization"]
    return out


def traded_notional(path_key: str, top_n: int, trading_window: int = 126,
                    rolling_step: int = 21, result_db: str = RESULT_DB) -> float:
    """該策略全期進出場的累計名目額（break-even 的分母）。

    每筆進場的名目額恰為當時的 `capital_per_pair`
    （`_execute_entry` 中 `v_a + v_b = cap`，而 `|shares_a|·p_a = v_a`），
    而 `capital_per_pair = current_equity / (top_n × 並行期數)`
    （`portfolio_manager.py:84`）。每次進場恰對應一次平倉，故 ×2。

    **兩處曾被算錯，皆已在此收攏**：
      · 資金基礎漏掉並行期數（名目額高估 6 倍），且以初始資金取代逐日權益
      · 事件數用 `Entries + Exits`，漏計停損與強制平倉的出場費
        （實測 `Entries = Exits + Stop_Losses + Forced_Closes`）

    自洽檢驗：把費率設為本函式推得的 break-even，重算期末淨利應歸零
    （實測 −$9 / −$2，見 `dev/breakeven_fix/`）。
    """
    conc = max(1, int(trading_window) // max(1, int(rolling_step)))
    max_pairs = max(1, int(top_n) * conc)
    con = sqlite3.connect(f"file:{result_db}?mode=ro", uri=True)
    try:
        pnl = pd.read_sql(
            "SELECT Date, SUM(Daily_Delta) d FROM trade_logs "
            "WHERE strategy_id = ? GROUP BY Date ORDER BY Date", con, params=(path_key,))
        ent = pd.read_sql(
            "SELECT Date, COUNT(*) n FROM trade_logs "
            "WHERE strategy_id = ? AND Status LIKE 'ENTER%' GROUP BY Date", con,
            params=(path_key,))
    finally:
        con.close()
    if pnl.empty or ent.empty:
        return float("nan")
    pnl["Date"] = pd.to_datetime(pnl.Date); ent["Date"] = pd.to_datetime(ent.Date)
    eq = INITIAL_CAPITAL + pnl.set_index("Date").d.cumsum()
    cap_t = (eq / max_pairs).reindex(ent.Date).ffill().bfill()
    return float((ent.set_index("Date").n * cap_t).sum()) * 2.0


def breakeven_roundtrip(path_key: str, top_n: int, final_equity: float = None,
                        result_db: str = RESULT_DB) -> float:
    """往返 break-even 成本：使淨利歸零的成本水準。

    $\\text{BE} = 2\\left(\\text{fee}_{\\text{side}} + p_{\\text{net}} / \\text{notional}\\right)$

    淨利為正時，修小名目額使 BE **上升**；為負時**下降**——兩個方向都正確。
    """
    n = traded_notional(path_key, top_n, result_db=result_db)
    if not np.isfinite(n) or n <= 0:
        return float("nan")
    if final_equity is None:
        final_equity = float(metrics(path_key, result_db=result_db)["Final_Equity"])
    return 2.0 * (CURRENT_FEE_SIDE + (final_equity - INITIAL_CAPITAL) / n)
