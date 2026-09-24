# -*- coding: utf-8 -*-
"""
延伸研究：核心 20 條策略的兩年形成期版本（504 對 252）
======================================================================
判準寫於結果之前：dev/fw504_all/PLAN.md（commit 080181f）。本模組只執行它。

Q-A　主軸 15 臂：504 是否普遍優於 252（同期等權 Sharpe 差＋後半期條件）
Q-B　DL-THR 5 臂：第三段主結論（DL-THR − Z-Score）在 504 下是否重現

同期：每一對比較只取兩臂**都在交易期內**的日子（504 缺 2001 年的視窗）。
口徑：15 個基準格等權、逐日損益 ÷ 初始資本；檢定為循環區塊拔靴（L=126）。

用法：python -m analysis.fw504_extension
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom

from analysis.block_bootstrap import bh_adjust, bootstrap_test
from analysis.proposition2_daily_hac import (INITIAL_CAPITAL, RESULT_DB, TRADING_DAYS,
                                             _grid_cell, baseline_only, load_daily_sids,
                                             method_paths)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = "results/analysis"
GROUPS = ["NOGRP", "GICS", "HDB", "AGG", "KM"]
RANKS = ["SSD", "DTW", "SDP"]
DLTHR = [("GICS", "SSD"), ("GICS", "SDP"), ("HDB", "SDP"), ("AGG", "SSD"), ("KM", "SSD")]


def _span(sids):
    """每條 strategy_id 的交易期範圍（trade_logs 含未持倉日，最小日期＝首個交易期起點）。"""
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    q = (f"SELECT strategy_id, MIN(Date), MAX(Date) FROM trade_logs "
         f"WHERE strategy_id IN ({','.join('?' * len(sids))}) GROUP BY strategy_id")
    rows = con.execute(q, sids).fetchall()
    con.close()
    return {r[0]: (pd.Timestamp(r[1]), pd.Timestamp(r[2])) for r in rows}


def _cells(meta, method):
    ids = baseline_only(meta.loc[meta.METHOD == method, "_path"].tolist())
    return {_grid_cell(s): s for s in ids}


def _sharpe(r):
    sd = r.std(ddof=1)
    return float(np.sqrt(TRADING_DAYS) * r.mean() / sd) if sd > 0 else np.nan


def _pair(px, span, ca, cb):
    """兩臂在共同格子、共同期間上的等權日報酬（ra, rb）。"""
    cells = sorted(set(ca) & set(cb))
    ids = [ca[c] for c in cells] + [cb[c] for c in cells]
    start = max(span[i][0] for i in ids)
    end = min(span[i][1] for i in ids)
    win = px.loc[(px.index >= start) & (px.index <= end)]
    ra = win[[ca[c] for c in cells]].mean(axis=1) / INITIAL_CAPITAL
    rb = win[[cb[c] for c in cells]].mean(axis=1) / INITIAL_CAPITAL
    return ra, rb, len(cells), start, end


def _halves(r):
    mid = len(r) // 2
    return _sharpe(r.iloc[:mid]), _sharpe(r.iloc[mid:])


def run():
    main252 = [f"Grid ({g}-{r})" for g in GROUPS for r in RANKS]
    main504 = [f"Grid ({g}-{r}-FW504)" for g in GROUPS for r in RANKS]
    dl252 = [f"Grid ({g}-{r}-DRL)" for g, r in DLTHR]
    dl504 = [f"Grid ({g}-{r}-FW504-DRL)" for g, r in DLTHR]
    meta = method_paths(main252 + main504 + dl252 + dl504)
    missing = sorted(set(main504 + dl504) - set(meta.METHOD))
    if missing:
        print(f"⚠ result.db 尚缺：{missing}")
    sids = baseline_only(meta._path.tolist())
    px = load_daily_sids(sids)
    span = _span(sids)

    # ── Q-A ─────────────────────────────────────────────────────────
    rows = []
    for m252, m504 in zip(main252, main504):
        c252, c504 = _cells(meta, m252), _cells(meta, m504)
        if not c252 or not c504:
            continue
        r252, r504, n, s, e = _pair(px, span, c252, c504)
        bt = bootstrap_test((r504 - r252).values * INITIAL_CAPITAL)
        h1_504, h2_504 = _halves(r504)
        h1_252, h2_252 = _halves(r252)
        rows.append({
            "臂": m252.replace("Grid (", "").rstrip(")"), "格數": n,
            "起": s.date(), "迄": e.date(), "交易日": len(r504),
            "Sharpe_252": round(_sharpe(r252), 3), "Sharpe_504": round(_sharpe(r504), 3),
            "ΔSharpe": round(_sharpe(r504) - _sharpe(r252), 3),
            "年化_252%": round(r252.mean() * TRADING_DAYS * 100, 3),
            "年化_504%": round(r504.mean() * TRADING_DAYS * 100, 3),
            "Δ年化pp": bt["年化Δ%"], "CI下界": bt["CI下界"], "CI上界": bt["CI上界"],
            "BB p": bt["BB p"],
            "504前半": round(h1_504, 3), "504後半": round(h2_504, 3),
            "252前半": round(h1_252, 3), "252後半": round(h2_252, 3),
        })
    qa = pd.DataFrame(rows)
    if not qa.empty:
        qa["BH p"] = np.round(bh_adjust(qa["BB p"].values), 4)
        pos = qa[qa["ΔSharpe"] > 0]
        k, n = len(pos), len(qa)
        p_binom = float(binom.sf(k - 1, n, 0.5)) if n else np.nan
        med_back = float(pos["504後半"].median()) if k else np.nan
        cond1 = k >= 11
        cond2 = (med_back >= 0) if k else False
        verdict = ("普遍較優" if cond1 and cond2 else
                   "前半期效果（未稱較優）" if cond1 else "未普遍較優")
        print("\n══ Q-A 主軸 15 臂：504 − 252（同期、15 格等權）")
        print(qa.to_string(index=False))
        print(f"\n  ΔSharpe > 0：{k}/{n}（單尾二項 p = {p_binom:.4f}；判準 ≥ 11）")
        print(f"  這些臂的 504 後半期 Sharpe 中位：{med_back:.3f}（判準 ≥ 0）")
        print(f"  → 裁決：{verdict}")
        qa.attrs["verdict"] = verdict
        qa.to_csv(os.path.join(OUT_DIR, "fw504_qa_mainaxis.csv"), index=False,
                  encoding="utf-8-sig")

    # ── Q-B ─────────────────────────────────────────────────────────
    rows = []
    for (g, r), d252, d504 in zip(DLTHR, dl252, dl504):
        for fw, zs_m, dl_m in [(252, f"Grid ({g}-{r})", d252),
                               (504, f"Grid ({g}-{r}-FW504)", d504)]:
            cz, cd = _cells(meta, zs_m), _cells(meta, dl_m)
            if not cz or not cd:
                continue
            rz, rd, n, s, e = _pair(px, span, cz, cd)
            bt = bootstrap_test((rd - rz).values * INITIAL_CAPITAL)
            rows.append({"配對來源": f"{g}-{r}", "形成期": fw, "格數": n,
                         "起": s.date(), "交易日": len(rd), "年化Δpp": bt["年化Δ%"],
                         "CI下界": bt["CI下界"], "CI上界": bt["CI上界"], "BB p": bt["BB p"]})
    qb = pd.DataFrame(rows)
    if not qb.empty:
        for fw in (252, 504):
            m = qb["形成期"] == fw
            qb.loc[m, "BH p"] = np.round(bh_adjust(qb.loc[m, "BB p"].values), 4)
        print("\n══ Q-B DL-THR − Z-Score（各自全期；504 列為 504 下的重做）")
        print(qb.to_string(index=False))
        q504 = qb[qb["形成期"] == 504]
        npos = int((q504["年化Δpp"] > 0).sum())
        nsig = int((q504["BH p"] < 0.05).sum())
        verdict = ("方向重現" if npos >= 4 else "未重現" if npos <= 2 else "不確定")
        print(f"\n  504 下方向為正：{npos}/{len(q504)}；BH 校正後顯著：{nsig}/{len(q504)}"
              f"  → 裁決：{verdict}")
        qb.to_csv(os.path.join(OUT_DIR, "fw504_qb_dlthr.csv"), index=False,
                  encoding="utf-8-sig")
    return qa, qb


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    run()
