# -*- coding: utf-8 -*-
"""resolve_max_active 與「逐日同步模擬」的等價性檢驗（合成資料）。

參考解：每日先處理所有平倉，再依排名讓符合進場條件者補位至 K。
被檢驗解：各配對獨立模擬 + resolve_max_active 以 entry_gate 反覆化解。
兩者逐配對、逐日的持倉必須完全相同。

    python dev/max_active/test_resolver.py
"""
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from strategies.max_active import resolve_max_active  # noqa: E402

ENTRY, T = 1.5, 120
DATES = pd.bdate_range("2020-01-01", periods=T)


def sim_one(z, blocked=frozenset(), start=0):
    """|z| > ENTRY 進場、z 穿越 0 出場；blocked 的日子禁止新開倉。"""
    pos = np.zeros(T, dtype=int)
    p = 0
    for t in range(start, T):
        if p != 0:
            if (p == -1 and z[t] <= 0) or (p == 1 and z[t] >= 0):
                p = 0
        elif abs(z[t]) > ENTRY and DATES[t] not in blocked:
            p = -1 if z[t] > 0 else 1
        pos[t] = p
    df = pd.DataFrame({"Date": DATES[start:], "Position": pos[start:]})
    return df


def lockstep(zs, starts, k):
    n = len(zs)
    pos = np.zeros((n, T), dtype=int)
    cur = np.zeros(n, dtype=int)
    for t in range(T):
        for i in range(n):                       # 先平倉
            if t < starts[i]:
                continue
            if cur[i] == -1 and zs[i][t] <= 0 or cur[i] == 1 and zs[i][t] >= 0:
                cur[i] = 0
        for i in range(n):                       # 再依排名補位（i 即排名）
            if t < starts[i] or cur[i] != 0:
                continue
            if abs(zs[i][t]) > ENTRY and (cur != 0).sum() < k:
                cur[i] = -1 if zs[i][t] > 0 else 1
        for i in range(n):
            pos[i, t] = cur[i] if t >= starts[i] else 0
    return pos


def main():
    rng = np.random.default_rng(0)
    n_ok = 0
    for trial in range(300):
        n = int(rng.integers(3, 12))
        k = int(rng.integers(1, n))
        zs = [np.cumsum(rng.normal(0, 0.5, T)) * 0.6 for _ in range(n)]
        zs = [z - z.mean() for z in zs]
        starts = [int(rng.integers(0, 10)) if rng.random() < 0.3 else 0 for _ in range(n)]
        logs = {i: sim_one(zs[i], start=starts[i]) for i in range(n)}
        res, _ = resolve_max_active(
            logs, ranks={i: i for i in range(n)}, k=k,
            resimulate=lambda i, bl: sim_one(zs[i], frozenset(bl), start=starts[i]))
        got = np.zeros((n, T), dtype=int)
        for i, df in res.items():
            got[i, starts[i]:] = df.Position.to_numpy()
        ref = lockstep(zs, starts, k)
        assert (np.abs(got) != 0).sum(axis=0).max() <= k, f"trial {trial}: 超過上限"
        assert np.array_equal(got, ref), f"trial {trial}: 與逐日同步模擬不同（n={n}, k={k}）"
        n_ok += 1
    print(f"OK：{n_ok} 組合成資料，化解結果與逐日同步模擬逐日相同")


if __name__ == "__main__":
    main()
