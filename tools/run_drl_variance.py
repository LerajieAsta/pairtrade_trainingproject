#!/usr/bin/env python
"""
DRL 策略訓練變異數評估：重跑 N 次、報「中位數±範圍」而非單次跑分。
======================================================================

動機：drl_threshold_trading.py 刻意不固定隨機種子（權重初始化、batch
洗牌皆隨機，walk-forward 增量訓練再放大路徑依賴），單次回測的「最佳
年化/最佳 Sharpe」是隨機變數——且為 15 組網格取 max，對雜訊特別敏感。
論文引用單次數字站不住腳；本腳本把 4 個 DRL-THR 策略重跑 N 次，
累積每輪完整網格結果，最後輸出跨輪分布統計。

用法：
    Project/bin/python run_drl_variance.py --runs 5
    （result.db 既有 DRL 結果自動列為第 1 輪，只補跑缺的輪數）

輸出：
    results/drl_variance_runs.csv     每輪 × 每變體的完整 summary 列（原始數據）
    results/drl_variance_summary.csv  跨輪彙總（中位數 / min–max）
    results/logs/drl_variance_runN.log  各輪 run_trading.py 的完整輸出

機制：run_trading.py 的續傳判定 = CSV 存在 AND result.db 有該 _path 列；
每輪開跑前刪除 4 個 DRL METHOD 的 summary 列即強制重算（寫入端
overwrite=True，不會累積重複列）。Z-Score 策略不受影響（決定性，
重跑數字逐位一致，無需納入）。
"""
import argparse
import os
import sqlite3
import subprocess
import sys
import time

import pandas as pd

# 路徑 shim：本工具位於 tools/，將專案根加入 sys.path 並切換 CWD（相對路徑以根為準）
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
import os as _os, sys as _sys
_os.chdir(_ROOT)
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

RESULT_DB = "results/result.db"
# 檔名可用環境變數覆寫，讓不同策略陣容的變異數結果分開存放
#   （舊陣容結果在 results/analysis/drl_variance_runs.csv，勿混寫）
_VAR_TAG = os.environ.get("DRL_VARIANCE_TAG", "").strip()
_suffix = f"_{_VAR_TAG}" if _VAR_TAG else ""
RUNS_CSV = f"results/analysis/drl_variance_runs{_suffix}.csv"
SUMMARY_CSV = f"results/analysis/drl_variance_summary{_suffix}.csv"
LOG_DIR = "results/logs"


def drl_targets(module: str = "drl_threshold_trading"):
    """
    回傳 (config 索引清單, db_method 清單)——現役且用指定交易端的策略。

    module 可換成 rl_threshold_trading，用於 RL-THR 部分回饋對照組。
    後者的隨機性更大（除了權重初始化與 batch 洗牌，還多一層 ε-greedy 探索），
    重跑輪數只會比 DL-THR 更必要，不會更不必要。
    """
    from strategies.config import strategies_raw_all
    target = f"strategies.trading.{module}"
    idx, methods = [], []
    for i, s in enumerate(strategies_raw_all):
        if s["trading_module"] == target and not s.get("formation_only"):
            idx.append(i)
            methods.append(s["db_method"])
    if not idx:
        sys.exit(f"config 中找不到使用 {target} 的現役策略")
    return idx, methods


def fetch_summaries(methods):
    ph = ",".join("?" * len(methods))
    with sqlite3.connect(RESULT_DB) as conn:
        return pd.read_sql_query(
            f"SELECT * FROM strategy_summaries WHERE METHOD IN ({ph})",
            conn, params=methods)


def harvest(run_id, methods):
    df = fetch_summaries(methods)
    if len(df) == 0:
        sys.exit(f"第 {run_id} 輪結束後 result.db 查無 DRL 結果，中止")
    df.insert(0, "run_id", run_id)
    df.to_csv(RUNS_CSV, mode="a", header=not os.path.exists(RUNS_CSV), index=False)
    print(f"  第 {run_id} 輪收成 {len(df)} 列 → {RUNS_CSV}")


def clear_summaries(methods):
    ph = ",".join("?" * len(methods))
    with sqlite3.connect(RESULT_DB) as conn:
        conn.execute(f"DELETE FROM strategy_summaries WHERE METHOD IN ({ph})", methods)
        conn.commit()


def aggregate():
    df = pd.read_csv(RUNS_CSV)
    rows = []
    for method, g in df.groupby("METHOD"):
        per_run = g.groupby("run_id").agg(
            best_ann=("Ann_Ret_Raw", "max"),
            best_sharpe=("Sharpe_Raw", "max"),
            n_pos_sharpe=("Sharpe_Raw", lambda s: int((s > 0).sum())),
            n_variants=("Sharpe_Raw", "size"),
        )
        rows.append({
            "METHOD": method,
            "runs": len(per_run),
            "best_ann_median_%": per_run["best_ann"].median() * 100,
            "best_ann_min_%": per_run["best_ann"].min() * 100,
            "best_ann_max_%": per_run["best_ann"].max() * 100,
            "best_sharpe_median": per_run["best_sharpe"].median(),
            "best_sharpe_min": per_run["best_sharpe"].min(),
            "best_sharpe_max": per_run["best_sharpe"].max(),
            "pos_sharpe_min": f"{per_run['n_pos_sharpe'].min()}/{per_run['n_variants'].iloc[0]}",
        })
    out = pd.DataFrame(rows).sort_values("best_ann_median_%", ascending=False)
    out.to_csv(SUMMARY_CSV, index=False)
    print("\n" + "=" * 100)
    print("DRL 訓練變異數彙總（每輪取網格最佳，跨輪統計）")
    print("=" * 100)
    for _, r in out.iterrows():
        print(f"{r['METHOD']:<48} ({r['runs']} 輪)")
        print(f"    最佳年化   中位 {r['best_ann_median_%']:.2f}%  範圍 [{r['best_ann_min_%']:.2f}%, {r['best_ann_max_%']:.2f}%]")
        print(f"    最佳Sharpe 中位 {r['best_sharpe_median']:.2f}   範圍 [{r['best_sharpe_min']:.2f}, {r['best_sharpe_max']:.2f}]"
              f"   正Sharpe最少 {r['pos_sharpe_min']}")
    print(f"\n彙總已存 {SUMMARY_CSV}；原始逐輪數據在 {RUNS_CSV}")


def main():
    ap = argparse.ArgumentParser(description="DRL 策略重跑 N 次的變異數評估")
    ap.add_argument("--runs", type=int, default=5, help="總輪數（含既有結果的第 1 輪，預設 5）")
    ap.add_argument("--aggregate-only", action="store_true", help="只重算彙總，不跑回測")
    ap.add_argument("--module", default="drl_threshold_trading",
                    help="交易端模組名（drl_threshold_trading 或 rl_threshold_trading）")
    args = ap.parse_args()

    if args.aggregate_only:
        aggregate()
        return

    idx, methods = drl_targets(args.module)
    slice_str = ",".join(str(i) for i in idx)
    print(f"目標策略（STRATEGIES_SLICE={slice_str}）：")
    for m in methods:
        print(f"  - {m}")

    os.makedirs(LOG_DIR, exist_ok=True)
    done_runs = 0
    if os.path.exists(RUNS_CSV):
        done_runs = pd.read_csv(RUNS_CSV)["run_id"].nunique()
        print(f"已有 {done_runs} 輪紀錄，續跑至 {args.runs} 輪")
    elif len(fetch_summaries(methods)) > 0:
        harvest(1, methods)   # result.db 既有結果列為第 1 輪
        done_runs = 1

    for run_id in range(done_runs + 1, args.runs + 1):
        print(f"\n── 第 {run_id}/{args.runs} 輪 ──────────────────────────────")
        clear_summaries(methods)
        # 記檔名同樣帶 tag——否則 RL-THR 的輪次日誌會覆蓋掉既有的 DL-THR 紀錄
        log_path = f"{LOG_DIR}/drl_variance{_suffix}_run{run_id}.log"
        t0 = time.time()
        with open(log_path, "w") as log:
            ret = subprocess.run(
                [sys.executable, "run_trading.py"],
                env={**os.environ, "STRATEGIES_SLICE": slice_str},
                stdout=log, stderr=subprocess.STDOUT,
            ).returncode
        if ret != 0:
            sys.exit(f"run_trading.py 第 {run_id} 輪失敗（exit {ret}），詳見 {log_path}")
        print(f"  回測完成（{(time.time() - t0) / 60:.1f} 分鐘）")
        harvest(run_id, methods)

    aggregate()


if __name__ == "__main__":
    main()
