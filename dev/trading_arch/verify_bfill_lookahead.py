# -*- coding: utf-8 -*-
"""E：價格清洗把首列打成 NaN，再用**第二天**的價回填。

唯讀、不碰 DB。用法：
    python dev/trading_arch/verify_bfill_lookahead.py

zscore_trading.__init__（修正前）：
    _pct = price_df.pct_change().abs()
    price_df = price_df.where(_pct <= 0.50).ffill().bfill()

首列的 pct_change 是 NaN，而 `NaN <= 0.50` → False，
故 where 把整個首列打成 NaN；ffill 無前值可用，bfill 遂以第二列回填。

zscore_window 預設 0 → extended_start_idx == trade_start_idx
（run_trading.py:377）→ 首列就是該期第一個交易日。
"""
import numpy as np
import pandas as pd


def clean_old(df: pd.DataFrame) -> pd.DataFrame:
    pct = df.pct_change(fill_method=None).abs()
    return df.where(pct <= 0.50).ffill().bfill()


def clean_new(df: pd.DataFrame) -> pd.DataFrame:
    """修正：首列（pct_change 為 NaN）不該被判為「超過 50%」。"""
    pct = df.pct_change(fill_method=None).abs()
    return df.where(pct.le(0.50) | pct.isna()).ffill().bfill()


def main() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    df = pd.DataFrame({"A": [100.0, 104.0, 102.0, 103.0, 101.0],
                       "B": [50.0, 49.0, 51.0, 50.5, 52.0]}, index=idx)

    old, new = clean_old(df), clean_new(df)
    print("原始：\n", df, "\n")
    print("修正前（首列被第二列覆蓋）：\n", old, "\n")
    print("修正後：\n", new, "\n")

    print(f"修正前首列 == 原始首列 ? {old.iloc[0].equals(df.iloc[0])}")
    print(f"修正前首列 == 原始次列 ? {old.iloc[0].equals(df.iloc[1])}   ← 前視")
    print(f"修正後首列 == 原始首列 ? {new.iloc[0].equals(df.iloc[0])}")

    # >50% 的真實跳空仍須被遮蔽（修正不得放行這一項）
    gap = pd.DataFrame({"A": [100.0, 100.0, 40.0, 41.0]},
                       index=pd.date_range("2020-01-01", periods=4, freq="B"))
    g = clean_new(gap)
    ok_masked = np.isclose(g["A"].iloc[2], 100.0)     # 跳空日以前值遞補
    ok_kept = np.isclose(g["A"].iloc[3], 41.0)        # 隔日恢復真實價
    print(f"\n>50% 跳空仍被遮蔽 ? {ok_masked}；隔日恢復真實價 ? {ok_kept}")

    assert not old.iloc[0].equals(df.iloc[0]) and old.iloc[0].equals(df.iloc[1])
    assert new.iloc[0].equals(df.iloc[0])
    assert ok_masked and ok_kept
    print("\n全部斷言通過。")


if __name__ == "__main__":
    main()
