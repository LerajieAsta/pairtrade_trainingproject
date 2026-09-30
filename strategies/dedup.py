# -*- coding: utf-8 -*-
"""跨期去重：同一配對同一日最多只被一個重疊交易期持有。

設計與判準見 `dev/dedup/PREREGISTRATION.md`。

問題
----
每 21 日形成一次、交易期 126 日 → 任一時點 6 期並行。引擎逐期、逐配對獨立模擬，
配對 A 若連續入選，會被多期同時持有、各拿一份資金（`dev/repeat_pairs/FINDING.md`）。

化解方式
--------
把每個 (期序, 配對) 當成一個單位，同一配對（**不分 A|B／B|A**：DTW／SDP 臂內同一條價差
會以兩種順序出現）的單位成一組，組內同日持有數上限 1，直接套用
`strategies.max_active.resolve_max_active`（k = 1）：

* 同日多個單位要新開倉時，**較早開始的期優先**，同期再依形成期排名。
* 已持有者永不被強制平倉；被擋的單位以 entry_gate 在該日禁止新開倉，之後若訊號仍在帶外
  且該配對已無人持有，可再進場。
* 結果與「所有期逐日同步模擬、先平倉再依優先序補位」等價（證明同 max_active 模組）。

引擎依期序逐期加入新單位；新期的部位只從它的交易期起日開始，先前的視窗已無衝突，
故每次只需化解**含新單位**的組。被擋日期須跨呼叫累積（`blocked`），否則重新模擬
舊單位時會遺失先前的阻擋。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Hashable

from strategies.max_active import resolve_max_active


def pair_key(pair: tuple) -> tuple:
    """不分順序的配對鍵。"""
    a, b = pair
    return (a, b) if a <= b else (b, a)


def resolve_new_period(
    units: dict,
    new_units: set,
    resimulate: Callable[[Hashable, set], "object"],
) -> tuple[dict, int]:
    """化解新加入的單位與視窗內既有單位的跨期重複持有。

    Parameters
    ----------
    units
        {(期序, 配對): {"df": 逐日紀錄, "rank": 形成期排名, "blocked": set, ...}}。
        就地更新 "df" 與 "blocked"。
    new_units
        本期新加入的單位。
    resimulate
        `resimulate(unit, blocked_dates) -> df`：以「這些日子禁止新開倉」重新模擬該單位。
        傳入的 blocked_dates 已含該單位先前累積的阻擋。

    Returns
    -------
    (changed, n_new_blocks)
        changed：{unit: 化解前的 df}（供呼叫端調整權益）；n_new_blocks：本次新增的被擋日數。
    """
    groups = defaultdict(list)
    for u in units:
        groups[pair_key(u[1])].append(u)

    changed = {}
    n_new = 0
    for members in groups.values():
        if len(members) < 2 or not any(u in new_units for u in members):
            continue
        logs = {u: units[u]["df"] for u in members}
        # 較早開始的期優先，同期再依排名（resolve_max_active 以此排序決定誰保留）
        prio = {u: (u[0], units[u]["rank"]) for u in members}
        out, blk = resolve_max_active(
            logs, prio, k=1,
            resimulate=lambda u, bl: resimulate(u, units[u]["blocked"] | bl),
        )
        for u in members:
            if blk[u]:
                changed.setdefault(u, units[u]["df"])
                n_new += len(blk[u] - units[u]["blocked"])
                units[u]["blocked"] |= blk[u]
                units[u]["df"] = out[u]
    return changed, n_new
