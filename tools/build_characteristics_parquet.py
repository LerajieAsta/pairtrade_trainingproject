"""
合併 PIT 基本面與 Green et al. 公司特徵 → formation 用的單一 parquet
======================================================================

    dataset/fundamental/sp500_pit_2000_2025_monthly.parquet   （base，14 欄）
  + dataset/fundamental/sp500_characteristics_monthly.parquet （40 特徵全集）
  → dataset/fundamental/sp500_pit_characteristics_monthly.parquet

只併入「通過覆蓋率門檻」的特徵，且跳過 base 已有的同名欄位——roa/roe/
cash_ratio/capital_intensity 在兩邊是同一套算式，實測數值完全相同，
取哪一邊都一樣，保留 base 的以維持既有欄序。

門檻名單來自 characteristics_coverage.csv 的 passed 欄，由
fetch/fetch_sec_characteristics.py --min-cov 產生；本腳本不自行判定門檻，
避免兩處各有一份閾值而漂移。

用法（系統 python，非 venv——見 fetch/ 的 numpy pickle 相依）：
    python tools/build_characteristics_parquet.py
"""

import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = "dataset/fundamental/sp500_pit_2000_2025_monthly.parquet"
CHARS = "dataset/fundamental/sp500_characteristics_monthly.parquet"
COVER = "dataset/fundamental/characteristics_coverage.csv"
OUT = "dataset/fundamental/sp500_pit_characteristics_monthly.parquet"


def main():
    base = pd.read_parquet(BASE)
    chars = pd.read_parquet(CHARS)
    cover = pd.read_csv(COVER, index_col=0)

    passed = [c for c in cover.index[cover.passed] if c in chars.columns]
    add = [c for c in passed if c not in base.columns]
    skip = [c for c in passed if c in base.columns]

    print(f"通過門檻 {len(passed)} 個：{passed}")
    print(f"  併入 {len(add)} 個：{add}")
    print(f"  跳過 {len(skip)} 個（base 已有同名同值）：{skip}")

    out = base.join(chars[add], how="left")
    assert len(out) == len(base), "join 改變了列數——索引對不齊"
    out.to_parquet(OUT)

    print(f"\n→ {OUT}（{len(out):,} 列 × {out.shape[1]} 欄）")
    print(f"structural_features 應設為：{tuple(passed)}")


if __name__ == "__main__":
    main()
