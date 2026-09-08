# -*- coding: utf-8 -*-
"""樣本外檢定：配對重現率是不是動量加權增益的條件變數。

設計與判準見 `dev/capital_alloc/PREREG_RECURRENCE.md`（**跑前定稿**）。

⚠ 本檔**不重測 P1**。動量加權未過閘的判定已經作成，不因本檔改變。
本檔只問「增益與重現率的關係」是否在**未參與形成假說的臂**上成立。

只算等權基準、MO、有前史比例三件事；權重定義與 lag 一律沿用 `stage1`。
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_spec = importlib.util.spec_from_file_location(
    "ca_stage1", os.path.join(_ROOT, "dev", "capital_alloc", "stage1.py"))
s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s1)

from strategies.metrics import concurrent_of, metrics_from_pnl  # noqa: E402

#: 形成假說的 5 支臂 —— 本檔一律排除
IN_SAMPLE = {"Grid (NOGRP-DTW)", "Grid (NOGRP-SSD)", "Grid (NOGRP-SDP)",
             "Grid (GICS-SSD)", "Grid (GGR)"}


def oos_cells() -> pd.DataFrame:
    con = __import__("sqlite3").connect(f"file:{s1.RESULT_DB}?mode=ro", uri=True, timeout=300)
    d = pd.read_sql(
        'SELECT _path, METHOD, "TOP N" tn, Sharpe_Raw FROM strategy_summaries '
        "WHERE \"STOP LOSS %\"='0%' AND METHOD NOT LIKE '%DRL%' "
        "AND METHOD NOT LIKE '%RLTHR%'", con)
    con.close()
    d = d[d._path.str.contains(r"TradeLogs_Top\d+_SL\d+_ZWin\d+_MSR\d+\.csv$",
                               regex=True, na=False)].copy()
    d["top_n"] = d.tn.str.extract(r"(\d+)").astype(int)
    d = d[d.top_n.isin(s1.TOPNS) & ~d.METHOD.isin(IN_SAMPLE)]
    return d.sort_values(["METHOD", "top_n"]).reset_index(drop=True)


def main() -> int:
    cl = oos_cells()
    print(f"樣本外：{cl.METHOD.nunique()} 支臂 / {len(cl)} 格\n")

    rows = []
    for _, r in cl.iterrows():
        df = s1.load_logs(r._path)
        if df.empty:
            print(f"  ! {r._path} 無 trade_logs，跳過")
            continue
        lag = max(1, concurrent_of(r._path))
        _, pos, pp, M, V = s1._trailing(df, lag)
        gid = s1.row_gid(df, pp)

        base = metrics_from_pnl(
            pd.Series(df.Daily_Delta.to_numpy(), index=df.Date).groupby(level=0).sum().sort_index())
        sc = s1.build_scales(df, pos, pp, M, V, "MO", lag)
        mo = metrics_from_pnl(s1.apply_scales(df, gid, sc))

        m = np.array([M.get((p, pi), np.nan) for p, pi in zip(pp.pair, pp.pi)])
        hist = float((~np.isnan(m)).mean())

        rows.append({"METHOD": r.METHOD, "arm": r.METHOD.replace("Grid ", ""),
                     "top_n": r.top_n, "conc": lag, "has_history": hist,
                     "Sharpe_EW": base.Sharpe_Raw, "Sharpe_MO": mo.Sharpe_Raw,
                     "dS_MO": mo.Sharpe_Raw - base.Sharpe_Raw,
                     "notional_ratio": s1.notional_ratio(df, gid, sc)})
        print(f"  {rows[-1]['arm']:<22} Top{r.top_n:<3} conc={lag} "
              f"前史={hist:.1%}  EW={base.Sharpe_Raw:+.4f}  "
              f"dS_MO={rows[-1]['dS_MO']:+.4f}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv("results/analysis/capital_alloc_recurrence_oos.csv",
             index=False, encoding="utf-8-sig")

    rho_c, p_c = stats.spearmanr(d.has_history, d.dS_MO)
    a = d.groupby("arm").agg(has_history=("has_history", "median"),
                             dS_MO=("dS_MO", "median"),
                             conc=("conc", "max")).reset_index()
    rho_a, p_a = stats.spearmanr(a.has_history, a.dS_MO)

    print("\n==== 樣本外判定 ====")
    print(f"逐格 (n={len(d)})：Spearman = {rho_c:+.3f}  p={p_c:.4f}")
    print(f"逐臂 (n={len(a)})：Spearman = {rho_a:+.3f}  p={p_a:.4f}")
    verdict = ("支持" if rho_c >= 0.30 else
               "推翻" if rho_c <= -0.30 else "不確定")
    print(f"\n跑前門檻：逐格 rho>=+0.30 支持 / <=-0.30 推翻 / 之間不確定")
    print(f"→ 判定：**{verdict}**（逐臂中位亦 {'>' if rho_a > 0 else '<='} 0）")

    print("\n附帶預測（機制）：並行期數 vs 有前史比例")
    print(a.groupby("conc").has_history.describe()[["count", "min", "50%", "max"]]
          .to_string())
    han = a[a.arm.str.contains("HAN4")]
    if len(han):
        h = float(han.has_history.iloc[0])
        mx6 = float(a[a.conc == 6].has_history.max())
        print(f"\nHAN4-MONTHLY(conc=1) 前史={h:.1%}   conc=6 諸臂最高={mx6:.1%}")
        print(f"→ 附帶預測 {'成立' if h > mx6 else '**不成立**'}")

    print("\n逐臂：")
    print(a.sort_values("has_history", ascending=False)
           .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\n-> results/analysis/capital_alloc_recurrence_oos.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
