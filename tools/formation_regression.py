#!/usr/bin/env python
"""
形成期數值回歸測試 —— 重構前後 bit-identical 驗證。
======================================================================
用途：把「特徵/分群/排序」共用邏輯抽到中性模組（_features/_clustering/
  _ranking）前後，證明代表策略選出的配對逐列完全相同。

用法（於專案根）：
    python tools/formation_regression.py --save baseline   # 重構前：存基準
    python tools/formation_regression.py --check baseline  # 重構後：比對

機制：用同一份 harness（本檔案不變）載入真實價格資料的固定窗口，
  跑代表策略的 Formation.run()，對輸出配對做雜湊。輸入固定 → 差異
  只可能來自被重構的 formation 模組。
"""
import argparse
import hashlib
import importlib
import inspect
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from strategies.preprocess_equity import DataProcessor
from strategies.config import (
    DB_PATH, TABLE_NAME, INFO_TABLE, TICKER_COL, SECTOR_COL,
    FORMATION_WINDOW, strategies_raw_all,
)

BASELINE_DIR = ROOT / "results" / "regression"

# 代表策略：涵蓋 特徵×分群×排序 的各種組合
REP_STRATEGIES = [
    "HDBSCAN Cluster SSD-DTW-PCA PCA5",        # PCA特徵 + HDBSCAN + ssd_dtw_pca 排序
    "HDBSCAN Cluster SSD-DTW-PCA PCA5 Resid",  # + 因子殘差化
    "Agglomerative Fundamentals (yF)",          # PCA特徵 + Agglomerative + SSD 排序
    "Agglomerative Fundamentals (FMP)",         # + PIT 基本面
]

# 固定測試窗口（取資料中段一個有代表性的 252 日窗）
TEST_WINDOW_END = "2018-06-29"   # 形成期末（含前推 252 日）


def _load_window():
    proc = DataProcessor(db_path=DB_PATH, table_name=TABLE_NAME)
    sector_mapping = proc.load_sector_mapping(INFO_TABLE, TICKER_COL, SECTOR_COL)
    price_pivot, all_dates, _, _ = proc.prepare_backtest_data("2016-01", "2019-12", FORMATION_WINDOW)
    end_ts = pd.to_datetime(TEST_WINDOW_END)
    # 找 <= end_ts 的最後一個交易日索引
    idx = max(i for i, d in enumerate(all_dates) if d <= end_ts)
    form_prices = price_pivot.iloc[idx - FORMATION_WINDOW + 1: idx + 1].dropna(axis=1)
    fs = all_dates[idx - FORMATION_WINDOW + 1].strftime("%Y-%m-%d")
    fe = all_dates[idx].strftime("%Y-%m-%d")
    return form_prices, fs, fe, sector_mapping


def _run_strategy(cfg, form_prices, fs, fe, sector_mapping):
    mod = importlib.import_module(cfg["formation_module"])
    FormationClass = mod.Formation
    valid = set(inspect.signature(FormationClass.__init__).parameters.keys())
    kwargs = {"price_df": form_prices, "form_start": fs, "form_end": fe,
              "top_n": cfg["params"].get("top_n", 20), "sector_mapping": sector_mapping}
    for k, v in cfg["params"].items():
        if k not in kwargs:
            kwargs[k] = v
    kwargs = {k: v for k, v in kwargs.items() if k in valid}
    pairs = FormationClass(**kwargs).run()
    return pairs


def _fingerprint(pairs: pd.DataFrame) -> dict:
    """對配對輸出取穩定指紋：關鍵欄位四捨五入後排序序列化雜湊。"""
    if pairs is None or pairs.empty:
        return {"n_pairs": 0, "hash": "empty"}
    cols = [c for c in ["Ticker_A", "Ticker_B", "Hedge_Ratio", "Spread_Mean",
                        "Spread_Std", "SSD", "DTW_Dist", "OLS_Alpha", "Sector"]
            if c in pairs.columns]
    d = pairs[cols].copy()
    for c in d.select_dtypes("number").columns:
        d[c] = d[c].round(8)
    payload = d.to_csv(index=False)
    return {"n_pairs": int(len(pairs)),
            "cols": cols,
            "hash": hashlib.sha256(payload.encode()).hexdigest(),
            "sample": d.head(5).to_dict("records")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", metavar="TAG", help="跑代表策略並存指紋為基準")
    ap.add_argument("--check", metavar="TAG", help="跑代表策略並與基準比對")
    args = ap.parse_args()
    tag = args.save or args.check
    if not tag:
        ap.error("需指定 --save TAG 或 --check TAG")

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    base_path = BASELINE_DIR / f"{tag}.json"

    print(f"載入測試窗口（形成期末 {TEST_WINDOW_END}）...")
    form_prices, fs, fe, sector_mapping = _load_window()
    print(f"  窗口 {fs} ~ {fe}，{form_prices.shape[1]} 檔股票\n")

    by_name = {s["name"]: s for s in strategies_raw_all}
    fps = {}
    for name in REP_STRATEGIES:
        if name not in by_name:
            print(f"  [略過] {name}（不在現役清單）")
            continue
        print(f"跑 {name} ...")
        try:
            pairs = _run_strategy(by_name[name], form_prices, fs, fe, sector_mapping)
            fps[name] = _fingerprint(pairs)
            print(f"  → {fps[name]['n_pairs']} 對，hash {fps[name]['hash'][:16]}")
        except Exception as e:
            fps[name] = {"error": str(e)}
            print(f"  → 錯誤：{e}")

    if args.save:
        json.dump(fps, open(base_path, "w"), indent=2, ensure_ascii=False)
        print(f"\n✅ 基準已存 {base_path}")
        return

    # check
    if not base_path.exists():
        sys.exit(f"找不到基準 {base_path}，請先 --save {tag}")
    base = json.load(open(base_path))
    print("\n" + "=" * 60)
    all_ok = True
    for name in fps:
        b = base.get(name, {})
        n = fps[name]
        if b.get("hash") == n.get("hash") and "error" not in n:
            print(f"  ✅ {name}: 一致（{n['n_pairs']} 對）")
        else:
            all_ok = False
            print(f"  ❌ {name}: 不一致！")
            print(f"      基準 hash {b.get('hash','?')[:16]} / 現在 {n.get('hash','?')[:16]}")
            if "error" in n:
                print(f"      錯誤：{n['error']}")
    print("=" * 60)
    print("✅ 全部 bit-identical" if all_ok else "❌ 有差異——重構改變了輸出")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
