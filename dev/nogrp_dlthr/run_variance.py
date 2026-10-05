# -*- coding: utf-8 -*-
"""不分組 × DL-THR 的第 2–5 輪重訓（PREREGISTRATION.md §五）。

DL-THR 不固定隨機種子。主檢定只用主跑（result.db，第 1 輪）；本腳本把同一批 45 格
再跑四輪，寫入**獨立的** results/nogrp_dlthr_variance.db，每輪覆寫同一批列。
不沿用 tools/run_drl_variance.py：那支每輪覆寫 result.db，跑完後 result.db 留下的是
最後一輪，主檢定就不再是主跑。

每輪結束、下一輪覆寫之前，立即：
  1. 檢查 45 格齊全、每格列數與主跑相同；
  2. 以與 4.3 節相同的定義（`analysis/proposition2_daily_hac`：每格逐日損益加總、15 格等權、
     DL-THR − Z-Score、年化 Δ = 日均 × 252 ÷ 初始資金）算三組的等權 Δ；
  3. 核對 result.db 的主跑 45 列未被改動。
結果累積在 results/analysis/nogrp_dlthr_variance_{cells,delta}.csv；已完成的輪次會跳過，可中斷續跑。

    python dev/nogrp_dlthr/run_variance.py            # 跑第 2–5 輪（需主跑已完成）
    python dev/nogrp_dlthr/run_variance.py --report   # 只彙總已完成的輪次
"""
import argparse
import hashlib
import os
import sqlite3
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.proposition2_daily_hac import (  # noqa: E402
    INITIAL_CAPITAL, TRADING_DAYS, _grid_cell, baseline_only, load_daily_sids)

MAIN_DB = "results/result.db"
VAR_DB = "results/nogrp_dlthr_variance.db"
CELLS_CSV = "results/analysis/nogrp_dlthr_variance_cells.csv"
DELTA_CSV = "results/analysis/nogrp_dlthr_variance_delta.csv"
RANKS = ("SSD", "DTW", "SDP")
ROUNDS = (2, 3, 4, 5)
N_CELLS = 45


def drl_method(rk):
    return f"Grid (NOGRP-{rk}-DRL)"


def summaries(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
    d = pd.read_sql("SELECT METHOD, _path, Sharpe_Raw, Ann_Ret_Raw, Final_Equity, Entries "
                    "FROM strategy_summaries WHERE METHOD IN (?,?,?)", con,
                    params=[drl_method(r) for r in RANKS])
    con.close()
    d = d[d._path.map(lambda s: bool(baseline_only([s])))]
    return d.sort_values("_path").reset_index(drop=True)


def row_counts(db, paths):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
    out = {p: con.execute("SELECT COUNT(*) FROM trade_logs WHERE strategy_id=?", (p,)).fetchone()[0]
           for p in paths}
    con.close()
    return out


def main_fingerprint():
    """result.db 主跑 45 列的指紋：重訓若意外寫進 result.db，這裡會變。"""
    d = summaries(MAIN_DB)
    return hashlib.sha256(d.to_csv(index=False).encode()).hexdigest(), len(d)


def daily_pnl(db, paths):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
    q = (f"SELECT strategy_id, Date, SUM(Daily_Delta) AS pnl FROM trade_logs "
         f"WHERE strategy_id IN ({','.join('?' * len(paths))}) GROUP BY strategy_id, Date")
    raw = pd.read_sql(q, con, params=list(paths))
    con.close()
    raw["Date"] = pd.to_datetime(raw["Date"])
    return raw.pivot(index="Date", columns="strategy_id", values="pnl")


def ew_delta(drl_db):
    """三組的等權年化 Δ（%），定義同 proposition2_daily_hac.ew_diff_series。"""
    drl = summaries(drl_db)
    con = sqlite3.connect(f"file:{MAIN_DB}?mode=ro", uri=True, timeout=600)
    zmeta = pd.read_sql("SELECT METHOD, _path FROM strategy_summaries WHERE METHOD IN (?,?,?)",
                        con, params=[f"Grid (NOGRP-{r})" for r in RANKS])
    con.close()
    out = {}
    for rk in RANKS:
        d_ids = drl[drl.METHOD == drl_method(rk)]._path.tolist()
        z_ids = baseline_only(zmeta[zmeta.METHOD == f"Grid (NOGRP-{rk})"]._path.tolist())
        dc = {_grid_cell(s): s for s in d_ids}
        zc = {_grid_cell(s): s for s in z_ids}
        cells = sorted(set(dc) & set(zc))
        if not cells:
            continue
        zpx = load_daily_sids([zc[c] for c in cells])          # Z-Score 為決定性，取主庫快取
        dpx = daily_pnl(drl_db, [dc[c] for c in cells])
        idx = zpx.index.union(dpx.index)
        zpx = zpx.reindex(idx).fillna(0.0)
        dpx = dpx.reindex(idx).fillna(0.0)                      # 無損益列 = 當日無部位，補 0
        diff = dpx[[dc[c] for c in cells]].mean(axis=1) - zpx[[zc[c] for c in cells]].mean(axis=1)
        out[rk] = (len(cells), float(diff.mean() * TRADING_DAYS / INITIAL_CAPITAL * 100))
    return out


def record(round_no, db):
    s = summaries(db).assign(round=round_no)
    s.to_csv(CELLS_CSV, mode="a", header=not os.path.exists(CELLS_CSV), index=False)
    dl = ew_delta(db)
    rows = pd.DataFrame([{"round": round_no, "arm": f"NOGRP-{rk}", "cells": n, "ann_delta_pct": v}
                         for rk, (n, v) in dl.items()])
    rows.to_csv(DELTA_CSV, mode="a", header=not os.path.exists(DELTA_CSV), index=False)
    return rows


def done_rounds():
    if not os.path.exists(DELTA_CSV):
        return set()
    return set(pd.read_csv(DELTA_CSV)["round"].unique())


def run_round(r):
    env = {**os.environ, "NOGRP_DLTHR_EXTENSION": "1", "RESULT_DB_PATH": VAR_DB,
           "FORCE_RERUN": "1", "DRL_WORKERS": os.environ.get("DRL_WORKERS", "10"),
           "PYTHONIOENCODING": "utf-8"}
    if os.path.normcase(os.path.abspath(env["RESULT_DB_PATH"])) == os.path.normcase(os.path.abspath(MAIN_DB)):
        raise RuntimeError("重訓不得寫入 result.db")
    log = f"results/logs/nogrp_dlthr_variance_round{r}.log"
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run([sys.executable, "-W", "ignore", "run_trading.py"],
                            env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"  第 {r} 輪 run_trading 結束（exit {rc}，{(time.time() - t0) / 3600:.2f} 小時）→ {log}",
          flush=True)
    if rc != 0:
        raise RuntimeError(f"第 {r} 輪 run_trading 失敗（exit {rc}），見 {log}")


def report():
    if not os.path.exists(DELTA_CSV):
        print("尚無已完成的重訓輪次")
        return
    main = record_main()
    dl = pd.concat([main, pd.read_csv(DELTA_CSV)], ignore_index=True)
    print("\n== 各輪等權年化 Δ（%）：第 1 輪為主跑（result.db）==")
    piv = dl.pivot(index="arm", columns="round", values="ann_delta_pct")
    print(piv.round(3).to_string())
    for arm, row in piv.iterrows():
        m = row.get(1)
        flip = [int(r) for r, v in row.items() if r != 1 and np.sign(v) != np.sign(m)]
        print(f"  {arm}：範圍 {row.min():+.3f} ～ {row.max():+.3f}；與主跑方向相反的輪次 {flip or '無'}")
    cells = pd.concat([summaries(MAIN_DB).assign(round=1), pd.read_csv(CELLS_CSV)], ignore_index=True)
    g = cells.groupby("_path").Sharpe_Raw
    print(f"\n== 每格跨輪 Sharpe（{cells['round'].nunique()} 輪、{g.ngroups} 格）==")
    print(f"  標準差中位 {g.std(ddof=1).median():.4f}；全距中位 {(g.max() - g.min()).median():.4f}")


def record_main():
    return pd.DataFrame([{"round": 1, "arm": f"NOGRP-{rk}", "cells": n, "ann_delta_pct": v}
                         for rk, (n, v) in ew_delta(MAIN_DB).items()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return

    fp0, n_main = main_fingerprint()
    if n_main != N_CELLS:
        raise SystemExit(f"主跑尚未完成：result.db 只有 {n_main}/{N_CELLS} 格，先等主跑結束")
    main_counts = row_counts(MAIN_DB, summaries(MAIN_DB)._path)
    os.makedirs(os.path.dirname(CELLS_CSV), exist_ok=True)

    for r in ROUNDS:
        if r in done_rounds():
            print(f"第 {r} 輪已完成，跳過")
            continue
        print(f"== 第 {r} 輪 ==", flush=True)
        run_round(r)
        s = summaries(VAR_DB)
        if len(s) != N_CELLS:
            raise RuntimeError(f"第 {r} 輪只有 {len(s)}/{N_CELLS} 格")
        vc = row_counts(VAR_DB, s._path)
        bad = {p: (vc[p], main_counts.get(p)) for p in vc if vc[p] != main_counts.get(p)}
        if bad:
            raise RuntimeError(f"第 {r} 輪列數與主跑不符：{list(bad.items())[:3]}")
        fp, _ = main_fingerprint()
        if fp != fp0:
            raise RuntimeError("result.db 的主跑列被改動了——停止")
        rows = record(r, VAR_DB)
        print(rows.to_string(index=False), flush=True)
    report()


if __name__ == "__main__":
    main()
