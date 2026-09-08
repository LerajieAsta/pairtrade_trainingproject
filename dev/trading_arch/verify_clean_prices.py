# -*- coding: utf-8 -*-
"""E 修正的端到端驗證：`zscore_trading.clean_prices` 的行為與快取正確性。

唯讀、不碰 DB。用法：
    python dev/trading_arch/verify_clean_prices.py

驗五件事：
  1. 首列不再被第二列覆蓋（原本的前視）
  2. 真正的 >50% 跳空仍被遮蔽，且隔日恢復真實價
  3. `Trading.__init__` 拿到的 price_df 首列 == 原始首列
  4. `.attrs` 快取不會讓期間切片拿到全表（pandas 會傳播 attrs）
  5. 快取確實命中（同一物件第二次呼叫回傳同一份結果）
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from strategies.trading.zscore_trading import Trading, clean_prices  # noqa: E402


def main() -> None:
    idx = pd.date_range("2020-01-01", periods=8, freq="B")
    raw = pd.DataFrame({"AAA": [100.0, 104.0, 102.0, 103.0, 101.0, 40.0, 41.0, 42.0],
                        "BBB": [50.0, 49.0, 51.0, 50.5, 52.0, 51.0, 50.0, 49.5]},
                       index=idx)

    out = clean_prices(raw)

    # 1. 首列
    assert out.iloc[0].equals(raw.iloc[0]), "首列仍被覆蓋"
    print(f"1. 首列保持原值           AAA={out['AAA'].iloc[0]:.1f}（原始 {raw['AAA'].iloc[0]:.1f}）  OK")

    # 2. >50% 跳空（AAA 於第 6 列 101 → 40，跌 60%）
    assert np.isclose(out["AAA"].iloc[5], 101.0), "跳空日未被遮蔽"
    assert np.isclose(out["AAA"].iloc[6], 41.0), "跳空隔日未恢復真實價"
    print(f"2. >50% 跳空仍被遮蔽      {raw['AAA'].iloc[5]:.0f} → 遮成 "
          f"{out['AAA'].iloc[5]:.0f}，隔日 {out['AAA'].iloc[6]:.0f}  OK")

    # 3. 走 Trading.__init__ 這條實際路徑
    t = Trading(price_df=raw, trade_dates=idx, selected_pairs=pd.DataFrame(),
                capital_per_pair=1000.0, fee_rate=0.0029, slippage_rate=0.0,
                stop_loss_pct=0.0, entry_z=2.0, exit_z=0.0, zscore_window=0)
    assert t.price_df.iloc[0].equals(raw.iloc[0]), "Trading 內首列被覆蓋"
    print(f"3. Trading.__init__ 首列  AAA={t.price_df['AAA'].iloc[0]:.1f}  OK")

    # 4. attrs 隨切片傳播 → 快取必須驗明正身，否則切片會拿到全表
    sl = raw.iloc[2:6]
    assert "_cleaned_prices" in sl.attrs, "本測試前提不成立（pandas 未傳播 attrs）"
    sl_out = clean_prices(sl)
    assert len(sl_out) == len(sl), f"切片拿到了長度 {len(sl_out)} 的表（應為 {len(sl)}）"
    assert sl_out.index.equals(sl.index), "切片的索引不符"
    print(f"4. 切片不會拿到全表       切片長度 {len(sl_out)}（全表 {len(raw)}）  OK")

    # 5. 快取命中
    big = pd.DataFrame(np.random.default_rng(0).lognormal(0, .02, (6647, 747)).cumprod(axis=0),
                       index=pd.date_range("2000-01-03", periods=6647, freq="B"))
    t0 = time.perf_counter(); clean_prices(big); t1 = time.perf_counter()
    clean_prices(big); t2 = time.perf_counter()
    assert clean_prices(big) is clean_prices(big), "快取未命中"
    print(f"5. 全表快取命中           首次 {(t1 - t0) * 1000:.0f} ms，"
          f"之後 {(t2 - t1) * 1000:.3f} ms  OK")

    print("\n全部通過。")


if __name__ == "__main__":
    main()
