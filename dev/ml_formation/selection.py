# -*- coding: utf-8 -*-
"""選取區域（selection region）—— 實際會被交易的那批配對。

領域定義見 CONTEXT.md。此處只講程式的契約。

    selection_region(with_features=False) -> DataFrame
        ADF 通過 + 標籤有效 + 期內 SSD 排名，保證含 `ssd_rank` 欄。

    with_model_scores(region) -> DataFrame
        併入模型分數，並**明確排除暖身 36 期**。

## 為何要有這支

在此之前，九支分析腳本各自重建同一套邏輯——其中七支的排名那行、
五支的篩選那行**逐字相同**。變異出現在不該變異的地方：
排名欄名分裂為 `r` / `ssd_rank`，`dropna` 的子集各寫各的，
而**暖身期的排除是 `dropna(subset=['score_M1', ...])` 的副作用**，
沒有名字也沒有人擁有。2026-08-27 有一支腳本因此崩潰（分數為 NaN 未排除）。

## 為何篩選不可關閉

「選取區域」這個詞的意思就是「ADF 通過、標籤有效、排名在前」。
需要**未篩選的全候選池**時（如 `ladder.py` / `m3_cnn.py` 量全池 AUC），
那是另一個概念，直接 `pd.read_parquet` 即可——不經此 module。

## ⚠ 呼叫端若以「排除法」建特徵清單，新增欄位＝新增特徵

`selection_region()` 會在回傳表上加一欄 `ssd_rank`。對於用
`[c for c in df.columns if c not in drop]` 建特徵清單的呼叫端，
**這一欄會自動變成特徵**——而它是選取用的排名，不是配對的性質。

2026-08-27 遷移時實測：`walkforward.py` 因此由 30 維變 31 維，
模型每期捕獲 +0.005510 → +0.005446，配對 t 檢定 p 由 **0.0004 變 0.0014**。
無任何錯誤訊息，只有數字悄悄改變。已在該檔的 `drop` 集合中明確排除。

**日後若在此 module 新增回傳欄位，必須檢查所有以排除法建特徵的呼叫端。**
目前只有 `dev/action_learn/walkforward.py` 屬此類（`signal_probe.py` 用的是
明列的特徵清單，不受影響）。

## 為何回傳已排名的池，而非前 k 名

`filter_decomp.py` 需要**帶狀**切法（名次 1–20 對 21–40）來分解水準位移，
那是方案 A「模型技巧 +0.0009 對水準位移 −0.0044」這項發現的來源。
回傳前 k 名會讓它用不上。呼叫端自己寫 `region[region.ssd_rank <= 20]`。
"""

import os

import numpy as np
import pandas as pd

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

#: 模型分數的前推暖身期數。`pi <= WARM_UP_PERIODS - 1` 無分數。
#: 由 `dev/ml_formation/ladder.py` 的前推協定決定（暖身 36 期／擴張窗／每 12 期重訓）。
WARM_UP_PERIODS = 36

#: 排名欄名。刻意不用 `r`——`d[d.r <= 20]` 讀不出是哪一種排名，
#: 且日後若加入其他排名（如模型分數排名）會撞名。
RANK_COL = "ssd_rank"

_SCORE_COLS = ("score_M0", "score_M1", "score_M2", "score_M3")


def selection_region(with_features: bool = False) -> pd.DataFrame:
    """回傳選取區域：ADF 通過、標籤有效、期內依 SSD 遞增排名。

    Parameters
    ----------
    with_features
        False（預設）讀 `pool.parquet`（23 欄）；True 讀 `train.parquet`（45 欄，
        多 22 個特徵欄）。實測 train 為 pool 的嚴格超集且列數相同（1,514,281），
        差別僅在特徵欄的有無。全欄讀取，故呼叫端不必知道 schema。

    Returns
    -------
    DataFrame
        含 `ssd_rank` 欄（每期由 1 起連續遞增，SSD 小者在前）。

    Notes
    -----
    **不含模型分數，亦不排除暖身期**——分數與暖身是 `with_model_scores` 的職責。
    只用 pool 欄位（不併分數）的分析本來就不需要排除暖身，此處不強加。
    """
    src = "train.parquet" if with_features else "pool.parquet"
    df = pd.read_parquet(os.path.join(CACHE, src))
    df = df[(df["adf_pass"] == 1) & df["label_valid"]].copy()
    df[RANK_COL] = df.groupby("Period_Start")["SSD"].rank(method="first")
    _assert_region(df, src)
    return df


def with_model_scores(region: pd.DataFrame) -> pd.DataFrame:
    """併入 `scores.parquet` 與 `m3_scores.parquet`，並排除暖身期。

    暖身排除是**顯式**的：以 `pi >= WARM_UP_PERIODS` 篩選，而非依賴
    `dropna(subset=score_*)` 的副作用。兩者在現有快取上等價
    （`score_M1` 恰在 `pi <= 35` 為 NaN），但前者說得出自己在做什麼。

    呼叫端若還需要 `capture_frac` 等欄非空，仍應自行 `dropna`——
    那是各自分析的需求，不是暖身。
    """
    key = ["Period_Start", "Ticker_A", "Ticker_B"]
    s1 = pd.read_parquet(os.path.join(CACHE, "scores.parquet"))
    s3 = pd.read_parquet(os.path.join(CACHE, "m3_scores.parquet"))
    out = region.merge(s1, on=key, how="inner").merge(
        s3.drop(columns=[c for c in ("pi",) if c in s3.columns]), on=key, how="inner")

    if "pi" not in out.columns:
        raise ValueError("scores.parquet 缺少 `pi` 欄，無法判定暖身邊界")
    n_before = len(out)
    out = out[out["pi"] >= WARM_UP_PERIODS].copy()
    if len(out) == n_before:
        raise ValueError(
            f"暖身排除未移除任何列（pi 最小值 {out['pi'].min()}）——"
            f"快取可能已改用不同的前推協定，WARM_UP_PERIODS={WARM_UP_PERIODS} 需複核")
    return out


def _assert_region(df: pd.DataFrame, src: str) -> None:
    """回傳前的靜態斷言。失敗即拋錯，不回傳形狀可疑的表。

    這些斷言就是這支 module 的規格：讀它們比讀 docstring 準確。
    """
    if df.empty:
        raise ValueError(f"{src}：選取區域為空——ADF 或 label_valid 的篩選條件可能有誤")

    if RANK_COL not in df.columns:
        raise ValueError(f"{src}：缺少 {RANK_COL} 欄")

    # 每期的排名必須由 1 起、連續。真正擋得住的失效模式是**先排名、後篩選**——
    # 那會在名次中留下空洞（例如 rank 完才 dropna，前 20 名可能只剩 17 個）。
    #
    # 擋不住的：rank 的 method 由 'first' 改為 'min'/'dense'。SSD 在期內是唯一值
    # （浮點數，實測無平手），三種 method 結果完全相同，斷言無從區分。
    g = df.groupby("Period_Start")[RANK_COL]
    n_per_period = g.size()
    if not np.allclose(g.min(), 1.0):
        raise ValueError(f"{src}：{RANK_COL} 未由 1 起算")
    if not np.allclose(g.max(), n_per_period):
        raise ValueError(
            f"{src}：{RANK_COL} 在某些期別不連續（有空洞）——"
            f"最可能的原因是**先排名、後篩選**：排名須在所有篩選完成之後才算")

    # 期數：形成期共 295 期，篩選後不應剩下極少數期別。
    n_periods = df["Period_Start"].nunique()
    if n_periods < 200:
        raise ValueError(
            f"{src}：篩選後僅剩 {n_periods} 期（預期約 295）——"
            f"快取可能不完整或篩選條件過嚴")
