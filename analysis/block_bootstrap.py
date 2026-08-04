# -*- coding: utf-8 -*-
"""
主檢定的推論引擎：循環 block bootstrap（p 值 + 95% 信賴區間）
======================================================================
本專案兩個命題的主檢定都吃同一種輸入——一條逐日報酬差序列 Δr_t——
並問同一個問題：E[Δr] 是否為 0，以及它的區間估計有多寬。本模組是唯一的答案來源。

為什麼是 bootstrap 而非 Newey-West
----------------------------------
兩者在本專案給出**相同結論**（命題 2 五組全顯著、命題 1 九組全不顯著，
兩法一致），故選擇的依據不是結果而是可辯護性：

  - HAC 是**漸近**方法，要解釋 Bartlett kernel、長期變異數估計、落後階怎麼選、
    以及 t 統計量的漸近常態性。落後階本身還是一個研究者自由度
    （舊版須另跑 {auto, 63, 126, 252} 四種來證明結論不隨它改變）。
  - 循環 block bootstrap 只有一句話：把序列切成長度 L 的區塊、首尾相接成環、
    隨機重抽接成同長度的新序列一萬次，看虛無分布裡有多少次比實際值更極端。
    沒有分布假設，也沒有落後階要選。

配對交易日報酬有厚尾與偏態（大量 0 值日 ＋ 少數大額平倉），無母數方法在此
反而更穩妥。HAC 保留在 `proposition2_daily_hac.newey_west`，供各分析模組
當對照欄使用，但不再是任何一章的主檢定。

區塊長度 L = 126
----------------
L 必須夠長才能保住重疊部位造成的自相關結構。本專案 FORWARD_DAYS=126、
rolling_step=21 → 任一時點有 6 個交易期同時在跑，故取 L = 126＝**一個完整
持有期**：任一區塊內部的持倉相關性完整保留，區塊之間才允許被打散。

這不是掃出來的參數，是由交易期長度直接決定的，故不另做 L 敏感度分析
（舊版曾列 L ∈ {21, 126, 252}：L=21 短於持有期，本就不該用來支持結論）。

p 值與信賴區間出自同一次重抽
----------------------------
    means      = 對原序列重抽的樣本平均        → 取百分位 = 95% CI
    means − obs = 等價於「對去平均序列重抽」    → 虛無分布 = 雙尾 p

第二行是恆等式而非近似：去平均使每個元素平移 −obs，長度 k·L 的重抽序列
其平均恰好平移 −obs。故一次重抽就能同時給出兩者，且兩者必然自洽——
不會出現「p < 0.05 但 CI 涵蓋 0」這種因兩次獨立重抽而產生的矛盾。
"""
import numpy as np

TRADING_DAYS = 252
INITIAL_CAPITAL = 10000.0   # 同 config.INITIAL_CAPITAL，用於把日損益換算成年化 %

BLOCK_L = 126               # 一個完整交易期
N_BOOT = 10000
SEED = 20260804


def circular_block_bootstrap_means(e: np.ndarray, L: int, B: int, rng) -> np.ndarray:
    """
    回傳 B 個 bootstrap 樣本平均（循環區塊；Künsch 1989、Politis & Romano 1992）。

    以「環狀累積和」在 O(1) 取得任一區塊和，故整體完全向量化：
    抽 B×k 個區塊起點，一次算完所有區塊和。
    """
    e = np.asarray(e, dtype=float)
    n = len(e)
    k = max(1, n // L)          # 每個 bootstrap 樣本用 k 個區塊，長度 k*L ≈ n
    ext = np.concatenate([e, e[:L]])
    csum = np.concatenate([[0.0], np.cumsum(ext)])
    starts = rng.integers(0, n, size=(B, k))
    block_sums = csum[starts + L] - csum[starts]
    return block_sums.sum(axis=1) / (k * L)


def to_annual_pct(daily_pnl: float, capital: float = INITIAL_CAPITAL) -> float:
    """日損益 → 年化報酬（%）。capital=1.0 表示輸入已是報酬率而非金額。"""
    return daily_pnl * TRADING_DAYS / capital * 100


def bootstrap_test(d, L: int = BLOCK_L, B: int = N_BOOT, seed: int = SEED,
                   capital: float = INITIAL_CAPITAL) -> dict:
    """
    對逐日序列 d 檢定 H0: E[d] = 0，並給出平均值的 95% 百分位信賴區間。

    `capital` 是 d 的單位：預設 d 為**日損益金額**（$），除以初始資金換算成
    報酬率。若 d 已經是報酬率（如 `regime_cost_dsr_eval.load_daily_returns`
    的輸出，該函式內部已除過一次），必須傳 capital=1.0，否則年化值會小 10^4 倍
    而不報任何錯——這個單位不一致無法由數值本身察覺，只能靠呼叫端聲明。

    回傳欄位皆為**年化百分點**（除 `n`）：
        年化Δ%      觀測到的年化差
        BB p        雙尾 p 值
        CI下界 / CI上界   95% 百分位信賴區間
        顯著        p < 0.05

    區間與 p 值同源（見模組 docstring），故 `CI 涵蓋 0` ⟺ `p ≥ 0.05` 幾乎必然
    成立；兩者並列是為了讓「不顯著」能進一步區分成「效果為零」與「檢定力不足」——
    後者的表徵是區間寬到兩端都具實質意義，這正是命題 1 的情況。
    """
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < L * 2:
        return {"n": n, "年化Δ%": np.nan, "BB p": np.nan,
                "CI下界": np.nan, "CI上界": np.nan, "顯著": "—"}

    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    means = circular_block_bootstrap_means(d, L, B, rng)

    p = float((np.abs(means - obs) >= abs(obs)).mean())
    lo, hi = np.percentile(means, [2.5, 97.5])

    return {
        "n": n,
        "年化Δ%": round(to_annual_pct(obs, capital), 3),
        "BB p": round(p, 4),
        "CI下界": round(to_annual_pct(float(lo), capital), 3),
        "CI上界": round(to_annual_pct(float(hi), capital), 3),
        "顯著": "✔" if p < 0.05 else "✘",
    }


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg 校正後 p 值（step-up，保單調）。"""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out
