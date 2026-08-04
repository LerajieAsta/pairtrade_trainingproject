# -*- coding: utf-8 -*-
"""
策略報酬序列的單一擁有者
======================================================================
本模組擁有「一條策略的逐日損益 / 報酬序列到底是什麼」這個定義。在此之前，
同一個問題散在三處各自作答，答案互不相同：

    strategies/db_utils.py           Daily_Delta / prev_equity，日曆＝該變體有列的日期
    analysis/regime_cost_dsr_eval.py ΣDaily_Delta / INITIAL_CAPITAL，日曆同上
    analysis/proposition2_daily_hac.py 原始 PnL，日曆＝「你剛好查了哪些策略」的聯集

第三者的聯集補 0 會替晚上線的策略捏造報酬——F09 系列 2009 才上線，卻被補出
2001–2008 共八年的零，使其 |Sharpe| 被稀釋 21%（−0.689 → −0.568）。前兩者的
差異則在分母：回測引擎確實再投入（portfolio_manager 的
capital_per_pair = current_equity / max_pairs），故複利才是「實際承擔風險的
資本」上的報酬。

本模組的三項規則
----------------
1. 日曆 = 價格交易日曆 ∩ [該策略首個交易日, 末個交易日]。
   生命期**內**的空手日補 0（策略存在，只是選擇不持倉）；
   生命期**外**留 NaN（策略當時不存在，補 0 是捏造）。
2. 報酬 = 複利（Daily_Delta / 前一日權益），與引擎的部位規模一致。
3. 跨策略比較預設取生命期**交集**；重疊不足直接拋錯，不靜默補 0。

PnL 與報酬是兩個不同的量，不是同一個量的兩種口徑：
  · daily_pnl        逐日美元損益。兩臂差分為尺度自由（IR = μ/σ 中分母對消），
                     供 analysis/ 的 HAC 檢定使用。
  · daily_returns    複利報酬。供 Sharpe / DSR / strategy_summaries 使用。
  · pair_period_pnl  配對×期聚合。不套用報酬語意，只共用新鮮度保證。

分層
----
純函式核心（pnl_from_log / equity_from_pnl / returns_from_pnl / align）不碰
資料庫，可用合成資料在毫秒內驗證 —— 見 tools/check_returns.py。
資料庫包覆層（daily_pnl / daily_returns / pair_period_pnl）負責讀取與快取。

快取一致性
----------
聚合結果快取到 parquet，並附一份指紋。指紋取自 strategy_summaries 的
(Final_Equity, Entries, Exits, Gross_Profit) —— 該列與 trade_logs 由
db_utils.export_df_to_db 在**同一個交易**中寫入（失敗時 _purge_path_key 三表
一起清），故它是可信的新鮮度訊號。重跑過的策略指紋會變，快取自動失效。

（舊機制要求作者手動呼叫 invalidate_cached_sids，忘了就靜默拿到舊序列；
  prop2_skip_permutation 的快取更是只增不失效，連手動失效都沒有。）
"""
import json
import os
import sqlite3

import numpy as np
import pandas as pd

from strategies.config import DB_PATH, INITIAL_CAPITAL, TABLE_NAME

RESULT_DB = "results/result.db"
CACHE_DIR = "results/analysis/_returns_cache"

# 指紋欄位：任何一欄變動都代表該策略被重跑且結果不同
_FP_COLS = ("Final_Equity", "Entries", "Exits", "Gross_Profit")


# ════════════════════════════════════════════════════════════════════
# 純函式核心 —— 不碰資料庫，可用合成資料驗證
# ════════════════════════════════════════════════════════════════════

def pnl_from_log(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
    """
    交易明細 → 逐日損益序列，套用生命期規則。

        df:       需含 Date 與 Daily_Delta（同一策略的所有列）
        calendar: 完整交易日曆（通常為價格資料的日曆）

    回傳以 calendar 為索引的 Series：
        生命期內（首個交易日 ~ 末個交易日）缺漏日補 0.0
        生命期外                          留 NaN

    生命期由 df 實際出現的日期界定，而非由 calendar 界定 —— 這正是晚上線的
    策略不會被補出上線前零報酬的原因。
    """
    if df.empty:
        return pd.Series(np.nan, index=calendar, dtype=float)

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    daily = d.groupby("Date")["Daily_Delta"].sum().sort_index()

    s = daily.reindex(calendar)
    first, last = daily.index.min(), daily.index.max()
    inside = (calendar >= first) & (calendar <= last)
    s[inside] = s[inside].fillna(0.0)
    return s


def equity_from_pnl(pnl: pd.Series, initial_capital: float = INITIAL_CAPITAL) -> pd.Series:
    """
    逐日損益 → 權益曲線。生命期外維持 NaN。

    權益自 initial_capital 起算並累加生命期內的損益；空手日權益不變。
    """
    inside = pnl.notna()
    eq = pd.Series(np.nan, index=pnl.index, dtype=float)
    eq[inside] = initial_capital + pnl[inside].cumsum()
    return eq


def returns_from_pnl(pnl: pd.Series, initial_capital: float = INITIAL_CAPITAL) -> pd.Series:
    """
    逐日損益 → **複利**日報酬：Daily_Delta / 前一日權益。

    首日的前一日權益取 initial_capital。複利而非除以固定初始資本，是因為
    回測引擎的部位規模本身就跟著權益走（portfolio_manager.allocate_capital：
    capital_per_pair = current_equity / max_pairs），除以固定分母算出來的
    並不是「實際承擔風險的資本」上的報酬。
    """
    inside = pnl.notna()
    eq = equity_from_pnl(pnl, initial_capital)
    prev = eq[inside].shift(1)
    prev.iloc[0:1] = initial_capital

    r = pd.Series(np.nan, index=pnl.index, dtype=float)
    r[inside] = pnl[inside] / prev
    return r


def align(df: pd.DataFrame, how: str = "intersect", min_overlap: int = 252) -> pd.DataFrame:
    """
    多條序列對齊。

        how="intersect"（預設）：裁到所有序列都存在的日期區間。
        how="union"            ：保留聯集並把生命期外補 0 —— 僅在你**確知**
                                 各臂上線日相同時使用；用錯會替晚上線的策略
                                 捏造報酬。

    重疊日數少於 min_overlap 時拋 ValueError，而非回傳一份看似正常、
    實則幾乎沒有共同樣本的結果。
    """
    if df.empty or df.shape[1] == 0:
        return df

    if how == "union":
        return df.fillna(0.0)
    if how != "intersect":
        raise ValueError(f"how 必須是 'intersect' 或 'union'，收到 {how!r}")

    mask = df.notna().all(axis=1)
    n = int(mask.sum())
    if n < min_overlap:
        spans = {c: (df[c].first_valid_index(), df[c].last_valid_index())
                 for c in df.columns}
        detail = "; ".join(
            f"{c.split('/')[-2] if '/' in c else c}: "
            f"{a.date() if a is not None else '?'}~{b.date() if b is not None else '?'}"
            for c, (a, b) in spans.items())
        raise ValueError(
            f"生命期交集只有 {n} 個交易日（門檻 {min_overlap}）。各序列生命期：{detail}。"
            f"若確定要在不重疊的區間上比較，請顯式指定較小的 min_overlap。")

    idx = df.index[mask]
    return df.loc[idx[0]:idx[-1]]


# ════════════════════════════════════════════════════════════════════
# 交易日曆
# ════════════════════════════════════════════════════════════════════

_CAL_CACHE = os.path.join(CACHE_DIR, "calendar.parquet")
_calendar_memo = None


def price_calendar(price_db: str = DB_PATH, table: str = TABLE_NAME,
                   use_cache: bool = True) -> pd.DatetimeIndex:
    """價格資料的交易日曆。行程內記憶 + 落地快取（日曆極少變動）。"""
    global _calendar_memo
    if _calendar_memo is not None:
        return _calendar_memo

    if use_cache and os.path.exists(_CAL_CACHE):
        _calendar_memo = pd.DatetimeIndex(pd.read_parquet(_CAL_CACHE)["Date"])
        return _calendar_memo

    con = sqlite3.connect(f"file:{price_db}?mode=ro", uri=True)
    d = pd.read_sql(f"SELECT DISTINCT Date FROM {table} ORDER BY Date", con)
    con.close()
    cal = pd.DatetimeIndex(pd.to_datetime(d["Date"]))

    os.makedirs(CACHE_DIR, exist_ok=True)
    pd.DataFrame({"Date": cal}).to_parquet(_CAL_CACHE)
    _calendar_memo = cal
    return cal


# ════════════════════════════════════════════════════════════════════
# 快取與指紋
# ════════════════════════════════════════════════════════════════════

def fingerprints(sids: list, result_db: str = RESULT_DB) -> dict:
    """自 strategy_summaries 取各策略的新鮮度指紋（1899 列，瞬讀）。"""
    if not sids:
        return {}
    con = sqlite3.connect(f"file:{result_db}?mode=ro", uri=True)
    cols = ", ".join(f'"{c}"' for c in _FP_COLS)
    q = (f'SELECT _path, {cols} FROM strategy_summaries '
         f'WHERE _path IN ({",".join("?" * len(sids))})')
    df = pd.read_sql(q, con, params=list(sids))
    con.close()
    # 不用 itertuples——「_path」是底線開頭，會被 pandas 改名成位置代號
    return {row["_path"]: "|".join(f"{float(row[c]):.10g}" for c in _FP_COLS)
            for _, row in df.iterrows()}


def _cache_paths(name: str) -> tuple:
    return (os.path.join(CACHE_DIR, f"{name}.parquet"),
            os.path.join(CACHE_DIR, f"{name}.fp.json"))


def _load_cache(name: str, sids: list, fps: dict) -> tuple:
    """回傳 (仍新鮮的快取內容, 需重算的 sid 清單)。"""
    pq, fp_path = _cache_paths(name)
    if not (os.path.exists(pq) and os.path.exists(fp_path)):
        return None, list(sids)

    with open(fp_path, encoding="utf-8") as f:
        old = json.load(f)
    cached = pd.read_parquet(pq)

    fresh = [s for s in sids
             if s in old and old[s] == fps.get(s) and s in _cols_of(cached, name)]
    stale = [s for s in sids if s not in fresh]
    return cached, stale


def _cols_of(cached: pd.DataFrame, name: str) -> set:
    return set(cached.columns) if name == "pnl" else set(cached["strategy_id"].unique())


def _save_cache(name: str, data: pd.DataFrame, fps: dict) -> None:
    pq, fp_path = _cache_paths(name)
    os.makedirs(CACHE_DIR, exist_ok=True)
    data.to_parquet(pq)
    old = {}
    if os.path.exists(fp_path):
        with open(fp_path, encoding="utf-8") as f:
            old = json.load(f)
    old.update(fps)
    with open(fp_path, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=1)


# ════════════════════════════════════════════════════════════════════
# 資料庫包覆層
# ════════════════════════════════════════════════════════════════════

def daily_pnl(sids, result_db: str = RESULT_DB, calendar=None,
              use_cache: bool = True) -> pd.DataFrame:
    """
    逐日**美元損益**，index=交易日曆，columns=strategy_id。

    生命期外為 NaN。供 HAC 兩臂差分使用 —— 差分的 IR 與 p 值對任何常數
    分母不變，故此處刻意不除以資本。
    """
    sids = list(dict.fromkeys(sids))
    cal = price_calendar() if calendar is None else calendar
    fps = fingerprints(sids, result_db)

    cached, stale = (None, sids)
    if use_cache:
        cached, stale = _load_cache("pnl", sids, fps)

    if stale:
        con = sqlite3.connect(f"file:{result_db}?mode=ro", uri=True)
        q = (f"SELECT strategy_id, Date, SUM(Daily_Delta) AS Daily_Delta "
             f"FROM trade_logs WHERE strategy_id IN ({','.join('?' * len(stale))}) "
             f"GROUP BY strategy_id, Date")
        raw = pd.read_sql(q, con, params=stale)
        con.close()

        built = {s: pnl_from_log(g, cal) for s, g in raw.groupby("strategy_id")}
        new = pd.DataFrame(built, index=cal) if built else pd.DataFrame(index=cal)
        if cached is not None and not cached.empty:
            keep = [c for c in cached.columns if c not in new.columns]
            merged = pd.concat([cached[keep], new], axis=1) if keep else new
        else:
            merged = new
        if use_cache:
            _save_cache("pnl", merged, {s: fps[s] for s in stale if s in fps})
        cached = merged

    have = [s for s in sids if s in cached.columns]
    return cached[have]


def daily_returns(sids, result_db: str = RESULT_DB, calendar=None,
                  use_cache: bool = True,
                  initial_capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    """逐日**複利報酬**，index=交易日曆，columns=strategy_id。生命期外為 NaN。"""
    pnl = daily_pnl(sids, result_db, calendar, use_cache)
    return pd.DataFrame({c: returns_from_pnl(pnl[c], initial_capital)
                         for c in pnl.columns}, index=pnl.index)


def equity_curves(sids, result_db: str = RESULT_DB, calendar=None,
                  use_cache: bool = True,
                  initial_capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    """逐日權益曲線。生命期外為 NaN。"""
    pnl = daily_pnl(sids, result_db, calendar, use_cache)
    return pd.DataFrame({c: equity_from_pnl(pnl[c], initial_capital)
                         for c in pnl.columns}, index=pnl.index)


def pair_period_pnl(sids, result_db: str = RESULT_DB,
                    use_cache: bool = True) -> pd.DataFrame:
    """
    每個 (strategy_id, 配對, 期) 一列：是否曾被 SKIP、該配對期總損益。

    與逐日層共用同一個指紋機制 —— 這是它被收進本模組的唯一理由，其餘語意
    （生命期、報酬分母）都不適用於這個粒度。
    """
    sids = list(dict.fromkeys(sids))
    fps = fingerprints(sids, result_db)

    cached, stale = (None, sids)
    if use_cache:
        cached, stale = _load_cache("pairperiod", sids, fps)

    if stale:
        con = sqlite3.connect(f"file:{result_db}?mode=ro", uri=True)
        q = (f"SELECT strategy_id, Ticker_A, Ticker_B, Period_Start, "
             f"MAX(CASE WHEN Status = 'HOLD_CASH (SKIP)' THEN 1 ELSE 0 END) AS skip, "
             f"SUM(Daily_Delta) AS pnl "
             f"FROM trade_logs WHERE strategy_id IN ({','.join('?' * len(stale))}) "
             f"GROUP BY strategy_id, Ticker_A, Ticker_B, Period_Start")
        new = pd.read_sql(q, con, params=stale)
        con.close()

        if cached is not None and not cached.empty:
            merged = pd.concat([cached[~cached.strategy_id.isin(stale)], new],
                               ignore_index=True)
        else:
            merged = new
        if use_cache:
            _save_cache("pairperiod", merged, {s: fps[s] for s in stale if s in fps})
        cached = merged

    return cached[cached.strategy_id.isin(sids)]
