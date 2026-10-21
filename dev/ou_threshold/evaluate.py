# -*- coding: utf-8 -*-
"""OU 解析最適門檻的評估（判準見同目錄 PREREGISTRATION.md §四）。

  Q1：OU − Z-Score；Q2：OU − DL-THR。皆為 15 格等權逐日差分（與 4.3 同定義：每格逐日損益加總、
      15 格等權平均後相減、年化 Δ = 日均 × 252 ÷ 初始資金），循環區塊自助法 L = 126、B = 10,000，8 組 BH。
  另報：後半期 Δ、15 格平均 Sharpe 與 rf 超額（5.1.1 衡量方式）、SKIP 比例與動作分布、組合市場 β。

    python dev/ou_threshold/evaluate.py
"""
import os
import sqlite3
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.block_bootstrap import bh_adjust, bootstrap_test  # noqa: E402
from strategies.config import INITIAL_CAPITAL  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DB, OU_DB = "results/result.db", "results/ou_threshold.db"
OUT = "results/analysis/ou_threshold_eval.csv"
HALF = pd.Timestamp("2014-01-02")
SOURCES = ["GICS_SSD", "GICS_SDP", "HDB_SDP", "KM_SSD", "AGG_SSD", "NOGRP_SSD", "NOGRP_DTW", "NOGRP_SDP"]
CELLS = [(t, s) for t in (1, 3, 5, 10, 20) for s in (0, 5, 15)]
SUB = {"z": "", "dl": "_DRL", "ou": "_OUTHR"}


def path(src, kind, tn, sl):
    return f"tiingo/Grid_{src}{SUB[kind]}/TradeLogs_Top{tn}_SL{sl}_ZWin0_MSR0.csv"


def ew_daily(db, src, kind):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
    cols = []
    for tn, sl in CELLS:
        d = pd.read_sql("SELECT Date, SUM(Daily_Delta) d FROM trade_logs WHERE strategy_id=? GROUP BY Date",
                        con, params=(path(src, kind, tn, sl),))
        if d.empty:
            raise RuntimeError(f"缺：{db} {path(src, kind, tn, sl)}")
        cols.append(pd.Series(d.d.values, index=pd.to_datetime(d.Date)))
    con.close()
    return cols


def ew(cols, idx):
    return pd.concat(cols, axis=1).reindex(idx).fillna(0.0).mean(axis=1)


def summ_mean(db, src, kind):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
    ps = [path(src, kind, tn, sl) for tn, sl in CELLS]
    d = pd.read_sql(f"SELECT Sharpe_Raw, Excess_Ret_RF FROM strategy_summaries WHERE _path IN "
                    f"({','.join('?' * len(ps))})", con, params=ps)
    con.close()
    return float(d.Sharpe_Raw.mean()), float(d.Excess_Ret_RF.mean())


def actions(src):
    con = sqlite3.connect(f"file:{OU_DB}?mode=ro", uri=True, timeout=600)
    d = pd.read_sql("SELECT Period_Start, Ticker_A, Ticker_B, MIN(Status='HOLD_CASH (SKIP)') skip, "
                    "AVG(Sel_Entry_Z) e, AVG(Sel_Exit_Z) x FROM trade_logs WHERE strategy_id=? GROUP BY 1,2,3",
                    con, params=(path(src, "ou", 20, 0),))
    con.close()
    return d


def market():
    import strategies.config as C
    from strategies.preprocess_equity import DataProcessor
    pv, *_ = DataProcessor(db_path=C.DB_PATH, table_name=C.TABLE_NAME).prepare_backtest_data(
        "2000-01", "2025-12", 252)
    return pv.pct_change(fill_method=None).clip(-0.5, 0.5).mean(axis=1)


def beta(y, m):
    x = np.column_stack([np.ones(len(m)), m.values])
    return float(np.linalg.lstsq(x, y.values, rcond=None)[0][1])


def main():
    mkt = market()
    rows, d1, d2 = [], {}, {}
    for src in SOURCES:
        z, dl, ou = ew_daily(BASE_DB, src, "z"), ew_daily(BASE_DB, src, "dl"), ew_daily(OU_DB, src, "ou")
        idx = sorted(set().union(*[c.index for c in z + dl + ou]))
        ez, edl, eou = ew(z, idx), ew(dl, idx), ew(ou, idx)
        d1[src], d2[src] = (eou - ez), (eou - edl)
        ann = lambda s: float(s.mean() * 252 / INITIAL_CAPITAL * 100)
        m = mkt.reindex(pd.DatetimeIndex(idx)).fillna(0.0)
        act = actions(src)
        sh = {k: summ_mean(db, src, k) for k, db in (("z", BASE_DB), ("dl", BASE_DB), ("ou", OU_DB))}
        rows.append({"source": src,
                     "dOU_Z": ann(d1[src]), "dOU_Z_H2": ann(d1[src][d1[src].index >= HALF]),
                     "dOU_DL": ann(d2[src]), "dOU_DL_H2": ann(d2[src][d2[src].index >= HALF]),
                     "Sharpe_Z": sh["z"][0], "Sharpe_DL": sh["dl"][0], "Sharpe_OU": sh["ou"][0],
                     "RF_Z": sh["z"][1], "RF_DL": sh["dl"][1], "RF_OU": sh["ou"][1],
                     "beta_Z": beta(ez / INITIAL_CAPITAL, m), "beta_DL": beta(edl / INITIAL_CAPITAL, m),
                     "beta_OU": beta(eou / INITIAL_CAPITAL, m),
                     "skip_OU": float(act.skip.mean()),
                     "act_mode": str(act[act.skip == 0].groupby(["e", "x"]).size().idxmax())})
        r = rows[-1]
        print(f"  {src:<10} OU−Z {r['dOU_Z']:+.3f}%｜OU−DL {r['dOU_DL']:+.3f}%｜SKIP {r['skip_OU']:.1%}｜"
              f"最常選 {r['act_mode']}｜β Z/DL/OU {r['beta_Z']:+.3f}/{r['beta_DL']:+.3f}/{r['beta_OU']:+.3f}", flush=True)
    x = pd.DataFrame(rows)
    for key, dd in (("Q1", d1), ("Q2", d2)):
        bt = [bootstrap_test(dd[s].values) for s in SOURCES]
        x[f"{key}_CI_lo"] = [b["CI下界"] for b in bt]
        x[f"{key}_CI_hi"] = [b["CI上界"] for b in bt]
        x[f"{key}_p"] = [b["BB p"] for b in bt]
        x[f"{key}_pBH"] = bh_adjust(x[f"{key}_p"].values)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    x.to_csv(OUT, index=False, encoding="utf-8-sig")

    print("\n== Q1：OU − Z-Score（PREREGISTRATION.md §四）==")
    pos, med = int((x.dOU_Z > 0).sum()), float(x.dOU_Z.median())
    sig = int(((x.Q1_pBH < 0.05) & (x.dOU_Z > 0)).sum())
    c = [pos >= 6, med > 0, sig >= 2]
    print(f"為正 {pos}/8｜中位 {med:+.3f}%｜BH 顯著為正 {sig}/8 → {['過' if v else '否' for v in c]} → "
          f"{'有效' if all(c) else '不成立'}")
    print("\n== Q2：OU − DL-THR ==")
    med2 = float(x.dOU_DL.median())
    neg = int(((x.Q2_pBH < 0.05) & (x.dOU_DL < 0)).sum())
    if med2 >= 0 and neg == 0:
        v = "解析式不劣於 DL-THR"
    elif med2 < 0 and neg >= 2:
        v = "DL-THR 優於解析式"
    else:
        v = "不確定"
    print(f"中位 {med2:+.3f}%｜BH 顯著為負 {neg}/8 → {v}")
    print("\n== 明細 ==")
    print(x[["source", "dOU_Z", "Q1_CI_lo", "Q1_CI_hi", "Q1_pBH", "dOU_Z_H2", "dOU_DL", "Q2_CI_lo", "Q2_CI_hi",
             "Q2_pBH", "dOU_DL_H2"]].round(4).to_string(index=False))
    print(x[["source", "Sharpe_Z", "Sharpe_DL", "Sharpe_OU", "RF_Z", "RF_DL", "RF_OU", "skip_OU",
             "act_mode", "beta_Z", "beta_DL", "beta_OU"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
