# -*- coding: utf-8 -*-
"""
命題 2 主檢定的無母數對照：循環 block bootstrap
======================================================================
`proposition2_daily_hac` 以 Newey-West HAC 標準誤處理重疊部位造成的自相關。
HAC 是**漸近**方法，並假設常態近似；配對交易的日報酬有厚尾與偏態
（大量 0 值日 ＋ 少數大額平倉），小樣本下 t 分布近似可能失準。

本模組以 block bootstrap 建立 H0: E[Δr]=0 的**無母數**虛無分布作為對照：
兩者結論一致 → 該節無懈可擊；不一致 → 以較保守者為準並揭露。

方法
----
循環 block bootstrap（Künsch 1989；circular 版本見 Politis & Romano 1992）：

  1. 將序列去平均 e_t = Δr_t − mean(Δr)，即**強制施加 H0**
  2. 把 e 首尾相接成環，隨機抽 k = n/L 個長度 L 的區塊接成新序列
  3. 重複 B 次，得到 H0 下「樣本平均」的分布
  4. 雙尾 p = P(|bootstrap 平均| ≥ |實際平均|)

區塊長度 L 必須夠長才能保住自相關結構。本專案 FORWARD_DAYS=126、
CONCURRENT_PERIODS=6，故取

    L ∈ {21, 126, 252}  ＝ {一個 rolling_step, 一個完整交易期, 一年}

L=126 是原則上的下限（蓋住一個完整持有期）；L=21 偏短，列出供對照，
若只有 L=21 顯著就代表結論靠切斷長程相關撐起來的，不可採信。

用法：python -m analysis.prop2_block_bootstrap
"""
import os
import sys

import numpy as np
import pandas as pd

from analysis.proposition2_daily_hac import (
    INITIAL_CAPITAL, OUT_DIR, TRADING_DAYS, ew_diff_series, newey_west,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BLOCK_LENGTHS = [21, 126, 252]
N_BOOT = 5000
SEED = 20260729


def circular_block_bootstrap_means(e: np.ndarray, L: int, B: int, rng) -> np.ndarray:
    """
    回傳 B 個 bootstrap 樣本平均。e 應已去平均（H0 已施加）。

    以「環狀累積和」在 O(1) 取得任一區塊和，故整體完全向量化：
    抽 B×k 個區塊起點，一次算完所有區塊和。
    """
    n = len(e)
    k = max(1, n // L)          # 每個 bootstrap 樣本用 k 個區塊，長度 k*L ≈ n
    ext = np.concatenate([e, e[:L]])
    csum = np.concatenate([[0.0], np.cumsum(ext)])
    starts = rng.integers(0, n, size=(B, k))
    block_sums = csum[starts + L] - csum[starts]
    return block_sums.sum(axis=1) / (k * L)


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    series = ew_diff_series()
    if not series:
        print("⚠ 無序列可用")
        return

    rows = []
    for base, d in series.items():
        d = np.asarray(d, dtype=float)
        obs = float(d.mean())
        e = d - obs                              # 施加 H0: E[Δr] = 0
        t_hac, p_hac, lag = newey_west(d)

        rec = {"配對底": base, "交易日": len(d),
               "年化Δ%": round(obs * TRADING_DAYS / INITIAL_CAPITAL * 100, 3),
               "HAC p": round(p_hac, 4)}
        for L in BLOCK_LENGTHS:
            means = circular_block_bootstrap_means(e, L, N_BOOT, rng)
            p = float((np.abs(means) >= abs(obs)).mean())
            rec[f"BB p (L={L})"] = round(p, 4)
        rows.append(rec)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\n" + "=" * 96)
    print(f"Block bootstrap 對照（循環區塊，{N_BOOT} 次重抽，雙尾）")
    print("H0: E[r_DRL − r_ZScore] = 0（以去平均施加）")
    print("=" * 96)
    print(res.to_string(index=False))

    print("\n判讀：L=126 為原則下限（蓋住一個完整持有期）。若 HAC 與 L≥126 的")
    print("      bootstrap 同時顯著，命題 2 的相對宣稱在參數與無母數兩種方法下皆成立。")

    res.to_csv(f"{OUT_DIR}/prop2_block_bootstrap.csv", index=False, encoding="utf-8-sig")
    print(f"\n→ {OUT_DIR}/prop2_block_bootstrap.csv")


if __name__ == "__main__":
    run()
