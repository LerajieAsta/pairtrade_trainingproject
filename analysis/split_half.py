# -*- coding: utf-8 -*-
"""
前後半樣本外分割
======================================================================
論文 4.6.4 與 `thesis/draft/CONTRIBUTIONS.md` §1.2 的來源。原本是一次性腳本、
未進版本庫，於是 2026-08-13 `result.db` 重建後即無法重算，而論文仍在引用
「`GICS-SSD-FW504` 後半期 Sharpe 為 −0.395」「只有 68 個（8.2%）前後半皆正」。
本模組把它補成可重跑的管線成員。

**它回答的問題。** DSR 校正的是「總共看過幾個候選」，**不校正**「以全期指標
挑到一支已在後半期失效的策略」。把樣本對半切開、兩半各算一次 Sharpe，
即可看出某支策略的全期表現是否只是前半期的遺產。

**設計。** 讀 `daily_returns_mainaxis.parquet`（逐日損益，欄＝strategy_id），
只取基準格，以交易日序列的**中點**對半切（不是日曆中點——兩半的樣本數必須相等，
否則兩個 Sharpe 的標準誤不同，比較會偏向樣本較多的那半）。

    Sharpe = sqrt(252) * mean(daily) / std(daily, ddof=1)

**與 `breakeven_dsr.csv` 的 Sharpe 不同口徑。** 該檔的 `Sharpe_Raw` 取自
`strategy_summaries`（複利權益序列）；本模組取自逐日損益（單利）。
兩者不可互相替換，故本模組另輸出全期值供對照——**引用時要引同一欄**。
"""
import os
import re
import sqlite3
import sys
from pathlib import Path as _Path

import numpy as np
import pandas as pd

_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULT_DB = "results/result.db"
OUT_DIR = "results/analysis"
CACHE = f"{OUT_DIR}/daily_returns_mainaxis.parquet"
_BASELINE_CELL = re.compile(r"TradeLogs_Top\d+_SL\d+_ZWin\d+_MSR\d+\.csv$")
TRADING_DAYS = 252
_MIN_DAYS = 504      # 每半至少兩年，否則 Sharpe 的雜訊蓋過訊號


def _sharpe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    sd = x.std(ddof=1)
    return float(np.sqrt(TRADING_DAYS) * x.mean() / sd) if sd > 0 else np.nan


def run():
    if not os.path.exists(CACHE):
        raise SystemExit(f"缺 {CACHE}——先跑 analysis.proposition2_daily_hac")
    px = pd.read_parquet(CACHE)
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    sm = pd.read_sql("SELECT METHOD,_path FROM strategy_summaries", con)
    con.close()

    base = sm[sm._path.map(lambda p: bool(_BASELINE_CELL.search(os.path.basename(p))))]
    have = base[base._path.isin(px.columns)]
    print(f"基準格 {len(base)} 個，其中 {len(have)} 個在逐日快取內")
    if len(have) < len(base):
        miss = sorted(set(base.METHOD) - set(have.METHOD))
        if miss:
            print(f"  ⚠ 下列 METHOD 完全不在快取，未納入：{miss[:6]}"
                  f"{' …' if len(miss) > 6 else ''}")

    rows = []
    for _, r in have.iterrows():
        s = px[r._path].to_numpy(float)
        s = s[np.isfinite(s)]
        if len(s) < 2 * _MIN_DAYS:
            continue
        mid = len(s) // 2          # 中點對半：兩半樣本數相等
        a, b = s[:mid], s[mid:]
        rows.append({"METHOD": r.METHOD, "格": os.path.basename(r._path),
                     "交易日": len(s),
                     "全期": round(_sharpe(s), 3),
                     "前半": round(_sharpe(a), 3),
                     "後半": round(_sharpe(b), 3)})
    d = pd.DataFrame(rows)
    if d.empty:
        raise SystemExit("無可用序列")
    d["兩半皆正"] = (d.前半 > 0) & (d.後半 > 0)
    d["衰減"] = (d.前半 - d.後半).round(3)
    d = d.sort_values("全期", ascending=False)

    n, k = len(d), int(d.兩半皆正.sum())
    print(f"\n可分割的基準格 {n} 個；**兩半皆正者 {k} 個（{k / n * 100:.1f}%）**")
    print("\n--- 全期 Sharpe 前 12 名（注意其後半欄）")
    print(d.head(12).to_string(index=False))

    # 以全期挑選 vs 以前半挑選，看後半的表現差多少
    top_full = d.nlargest(10, "全期")
    top_first = d.nlargest(10, "前半")
    print(f"\n以**全期** Sharpe 取前 10：後半中位 {top_full.後半.median():+.3f}，"
          f"其中為負者 {int((top_full.後半 < 0).sum())}/10")
    print(f"以**前半** Sharpe 取前 10：後半中位 {top_first.後半.median():+.3f}，"
          f"其中為負者 {int((top_first.後半 < 0).sum())}/10")

    os.makedirs(OUT_DIR, exist_ok=True)
    d.to_csv(f"{OUT_DIR}/split_half_ranking.csv", index=False, encoding="utf-8-sig")
    print(f"\n→ {OUT_DIR}/split_half_ranking.csv")


if __name__ == "__main__":
    run()
