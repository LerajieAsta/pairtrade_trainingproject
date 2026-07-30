"""
形成期共用基本面工具（與分群、排序無關）
======================================================================
FMP Point-in-Time 基本面的載入、GICS 產業別名正規化、群組中位數插補、
winsorize——供任何需要基本面特徵的策略共用。

從 agglomerative_FMP 抽出以中性化；邏輯逐位元保留（等價測試見
tools/formation_regression.py）。
"""
import os
import numpy as np
import pandas as pd

# ── GICS 產業別名正規化（跨資料源標籤不一致問題） ─────────────────────────
_SECTOR_CANONICAL_MAP = {
    "Healthcare": "Health Care",
    "Technology": "Information Technology",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Financial Services": "Financials",
    "Basic Materials": "Materials",
}

CANONICAL_SECTORS = [
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Financials", "Health Care", "Industrials",
    "Information Technology", "Materials", "Real Estate", "Utilities",
]

UNKNOWN_SECTOR_IDX = len(CANONICAL_SECTORS)  # one-hot 最後一欄


def canonicalize_sector(raw: str) -> str:
    """將原始產業字串正規化為 CANONICAL_SECTORS 之一，無法對應者歸為 Unknown。"""
    if not raw:
        return "Unknown"
    mapped = _SECTOR_CANONICAL_MAP.get(raw, raw)
    return mapped if mapped in CANONICAL_SECTORS else "Unknown"


# 基本面資料快取（每 process 載一次）
_fundamentals_cache: pd.DataFrame = None


def load_pit_fundamentals(parquet_path: str) -> pd.DataFrame:
    """載入 FMP Point-in-Time 基本面 Parquet（快取）。找不到回空 DataFrame。"""
    global _fundamentals_cache
    if _fundamentals_cache is not None:
        return _fundamentals_cache

    if os.path.exists(parquet_path):
        _fundamentals_cache = pd.read_parquet(parquet_path)
    else:
        print(f"  [Formation] 警告：找不到基本面 Parquet 檔案 {parquet_path}，全數標的將回退為全域中位數插補")
        _fundamentals_cache = pd.DataFrame()

    return _fundamentals_cache


def impute_by_group(values: np.ndarray, groups: np.ndarray,
                    use_group_median: bool = True) -> np.ndarray:
    """
    插補缺失值。預設以群組（正規化後產業）中位數，群組無覆蓋則退回全域中位數。

    `use_group_median=False` → 一律用全域中位數。

    為何需要這個開關：產業中位數插補會**把產業資訊寫進特徵向量**。在缺失率高時
    （SEC XBRL 特徵約 20–30% 缺失），相當於為缺資料的股票額外加上一組產業標籤——
    與 `sector_onehot_weight` 是同一種混淆。檢驗「產業先驗是否有價值」的實驗
    （命題 1）必須關閉群組插補，否則對照組的資訊會從後門漏進實驗組。
    """
    values = values.copy()
    nan_mask = np.isnan(values)
    if not nan_mask.any():
        return values

    global_median = float(np.nanmedian(values)) if not np.all(nan_mask) else 0.0

    if not use_group_median:
        values[nan_mask] = global_median
        return values

    for g in np.unique(groups):
        g_mask = groups == g
        g_nan_mask = g_mask & nan_mask
        if not g_nan_mask.any():
            continue
        g_values = values[g_mask & ~nan_mask]
        fill = float(np.median(g_values)) if len(g_values) > 0 else global_median
        values[g_nan_mask] = fill

    values[np.isnan(values)] = global_median
    return values


def winsorize(values: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(values, [lower_pct, upper_pct])
    return np.clip(values, lo, hi)
