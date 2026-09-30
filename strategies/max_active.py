# -*- coding: utf-8 -*-
"""單一交易期內「同時持倉配對數上限 K」的衝突化解。

來源：彭鈺玶（2025）元智碩論，內文 p.20 的「最大同時持倉 10 組」。
原文的 10 是**同一交易期內**的上限：取 Top 20 為候選時，同時最多也只開 10 對。
本專案把它做成參數 `max_active`（K），設計見 `dev/max_active/PREREGISTRATION.md`。

為何需要反覆模擬
----------------
引擎逐配對獨立模擬整個交易期（`Trading._simulate_pair`），配對彼此不知道
對方有沒有持倉。上限 K 卻是跨配對的約束。本模組以「先到先得」精確化解：

  1. 全部候選各自模擬一次（無上限）。
  2. 依日期前進，找第一個「持倉數 > K」的日子 d。
     d 當天的持倉者分兩類：d 之前就在倉的（續抱者）與 d 當天新進場的。
     續抱者數 ≤ K（前一日已合法），故只需在新進場者中依形成期排名
     （`Pair_Rank` 小者優先）保留 K − 續抱者 個，其餘在 d 當天被擋。
  3. 被擋的配對以 `entry_gate[d] = False` 重新模擬——該閘門只禁止「新開倉」，
     故 d 之前的路徑逐日不變；d 之後它可以在訊號仍在帶外、且有空位時再進場。
  4. 從 d 繼續往後掃，直到沒有任何一天超過 K。

每一輪只改動 d 當天及之後，且 d 單調不減，故必定終止。
結果與「逐日同步模擬所有配對、每日先平倉再依排名補位」完全等價。
"""
from __future__ import annotations

from typing import Callable, Hashable

import numpy as np
import pandas as pd


def _position_row(df, dates: pd.DatetimeIndex) -> np.ndarray:
    """單一配對在各日期的持倉旗標；該配對當日無資料（尚未上市／已下市）記 0。"""
    row = np.zeros(len(dates), dtype=np.int8)
    if df is None or df.empty:
        return row
    loc = dates.get_indexer(pd.DatetimeIndex(df["Date"]))
    ok = loc >= 0
    row[loc[ok]] = (df["Position"].to_numpy()[ok] != 0)
    return row


def resolve_max_active(
    logs: dict,
    ranks: dict,
    k: int,
    resimulate: Callable[[Hashable, set], pd.DataFrame],
    max_rounds: int = 100_000,
) -> tuple[dict, dict]:
    """化解同一交易期內的持倉數上限。

    Parameters
    ----------
    logs
        {pair: 無上限時的逐日交易紀錄}（需含 `Date`、`Position`）。
    ranks
        {pair: 形成期排名}，同日多個新進場者時小者優先。
    k
        同時持倉上限。
    resimulate
        `resimulate(pair, blocked_dates) -> df`：以「這些日子禁止新開倉」重新模擬。

    Returns
    -------
    (logs, blocked)
        化解後的紀錄，以及每個配對被擋的日期集合（供診斷）。
    """
    if k <= 0:
        raise ValueError(f"max_active 必須 ≥ 1，收到 {k}")
    logs = dict(logs)
    pairs = sorted(logs, key=lambda p: (ranks.get(p, np.inf), str(p)))
    blocked: dict = {p: set() for p in pairs}
    if len(pairs) <= k:
        return logs, blocked

    all_dates = [pd.DatetimeIndex(df["Date"]) for df in logs.values()
                 if df is not None and not df.empty]
    if not all_dates:
        return logs, blocked
    dates = all_dates[0]
    for d in all_dates[1:]:
        dates = dates.union(d)

    # 每輪只有被重新模擬的配對會變，只更新那幾列
    mat = np.stack([_position_row(logs.get(p), dates) for p in pairs])
    t0 = 0
    for _ in range(max_rounds):
        occ = mat.sum(axis=0)
        over = np.nonzero(occ[t0:] > k)[0]
        if len(over) == 0:
            return logs, blocked
        t = t0 + int(over[0])
        prev = mat[:, t - 1] if t > 0 else np.zeros(len(pairs), dtype=np.int8)
        entrants = [i for i in range(len(pairs)) if mat[i, t] and not prev[i]]
        holders = int(occ[t]) - len(entrants)
        # 續抱者前一日已合法（≤ k），故 room ≥ 0
        room = k - holders
        assert room >= 0, "續抱者超過上限：前一日的化解不完整"
        # pairs 已依排名排序，entrants 保留該順序
        d = dates[t]
        for i in entrants[room:]:
            p = pairs[i]
            blocked[p].add(d)
            logs[p] = resimulate(p, blocked[p])
            mat[i] = _position_row(logs[p], dates)
        t0 = t
    raise RuntimeError(f"max_active 化解超過 {max_rounds} 輪仍未收斂")
