#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回測資料完整性稽核：抓「上層回報成功、底層資料不完整」的缺口
======================================================================

同一類問題在 2026 年 8 月出現了四次，每次都是「摘要層看起來正常、底層資料
其實不完整、而且沒有任何機制會發現」：

  1. `DTW` / `SSD-DTW-PCA` 的形成期只跑了 36 / 64 期（其餘同族 295 期），
     交易期忠實地只跑了已有的那幾期 → 序列僅 861 / 1,449 日，Sharpe 0.82
     卻是那 3 年的產物，一度被當成全期最高值寫進 DSR 門檻。
  2. `Agglomerative (FMP)` 與 `HDBSCAN (殘差)` 各有一格：`strategy_summaries`
     寫成功（Entries 5,869 / 5,686、有 Sharpe），`trade_logs` 卻是 0 列。
     成因是併發寫入的 database is locked，而執行器照樣回報 SUCCESS。
  3. 形成期合併時 `formation_ranked` 插入失敗被 except 吞掉，且共用的
     connection 讓半套資料被下一條策略的 commit 一起送出去。
  4. `Grid (AGG-SSD-NOSEC-GI)` 的 Top3_SL0 遺失，直到統計結果對不上才發現。

本工具把這些型態一次查完。設計原則：**只走索引，不做全表掃描**
（`trade_logs` 有四億列，`SELECT DISTINCT strategy_id` 會跑到天荒地老）。

封存策略的特殊處理
----------------------------------------------------------------------
2026-08-06 起，非現役策略的 `trade_logs` 明細已刻意清除以釋放空間
（見 tools/archive_trade_logs.py），逐日序列先保全於 parquet 快取。
故「封存策略沒有明細」是**預期狀態、不是錯誤**——但若它連快取也沒有，
那就是真的遺失，要報。

用法：
    python -m tools.audit_result_db              # 全部檢查
    python -m tools.audit_result_db --quick      # 跳過逐條探測（僅結構性檢查）
"""
import argparse
import os
import sqlite3
import sys

import pandas as pd

from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

RESULT_DB = "results/result.db"
FORMATION_DB = "formation_data/formation_pairs_sp500_Tiingo.db"
CACHE = "results/analysis/daily_returns_mainaxis.parquet"

OK, WARN, ERR = "✔", "⚠", "✘"
_findings: list[tuple[str, str, str]] = []      # (level, 檢查項, 訊息)


def report(level: str, check: str, msg: str) -> None:
    _findings.append((level, check, msg))
    print(f"  {level} {msg}")


def live_methods() -> set[str]:
    from strategies.config import strategies_raw_all
    return {s["db_method"] for s in strategies_raw_all}


# ── A. 現役策略必須有逐日明細 ────────────────────────────────────────
def check_live_have_logs(con, summaries: pd.DataFrame, live: set[str], quick: bool):
    print("\n[A] 現役策略的逐日明細")
    tgt = summaries[summaries.METHOD.isin(live)]
    if quick:
        print(f"  （--quick：跳過 {len(tgt)} 條逐條探測）")
        return
    missing = []
    for path, entries in zip(tgt._path, tgt.Entries):
        try:
            if float(entries) <= 0:
                continue        # 本來就沒交易，無明細屬正常
        except (TypeError, ValueError):
            pass
        hit = con.execute(
            "SELECT 1 FROM trade_logs WHERE strategy_id=? LIMIT 1", (path,)).fetchone()
        if not hit:
            missing.append((path, entries))
    if missing:
        report(ERR, "A", f"{len(missing)} 條現役策略有摘要卻無明細（併發寫入遺失）：")
        for p, e in missing[:10]:
            print(f"        {p}  (摘要宣稱 Entries={e})")
    else:
        report(OK, "A", f"{len(tgt)} 條現役策略的明細齊全")


# ── B. 封存策略：明細可缺，但逐日序列必須留在快取 ────────────────────
def check_archived_cached(summaries: pd.DataFrame, live: set[str]):
    print("\n[B] 封存策略的逐日序列保全")
    arch = summaries[~summaries.METHOD.isin(live)]
    if not os.path.exists(CACHE):
        report(ERR, "B", f"找不到逐日快取 {CACHE}，封存策略的序列無從驗證")
        return
    cached = set(pd.read_parquet(CACHE).columns)
    lost = [p for p in arch._path if p not in cached]
    if lost:
        report(WARN, "B", f"{len(lost)}/{len(arch)} 條封存策略既無明細也不在快取："
                          f"逐日序列已無法重建")
        for p in lost[:5]:
            print(f"        {p}")
    else:
        report(OK, "B", f"{len(arch)} 條封存策略的逐日序列全部保全於快取")


# ── C. 形成期完整度 ──────────────────────────────────────────────────
def check_formation_complete():
    print("\n[C] 形成期完整度")
    if not os.path.exists(FORMATION_DB):
        report(WARN, "C", f"找不到 {FORMATION_DB}，略過")
        return
    con = sqlite3.connect(f"file:{FORMATION_DB}?mode=ro", uri=True)
    try:
        d = pd.read_sql(
            "SELECT strategy_id, COUNT(DISTINCT Period_Start) n "
            "FROM formation_progress GROUP BY strategy_id", con)
    except Exception as e:
        con.close()
        report(WARN, "C", f"讀取 formation_progress 失敗：{e}")
        return
    con.close()
    if d.empty:
        report(WARN, "C", "formation_progress 無資料")
        return

    # 期望期數依「形成窗 × 交易期 × 滾動步長」而異——Grid HAN4-MONTHLY 是
    # 月頻設定（交易期 21 日），本來就比其餘 29 條多出幾期。若用全體最大值
    # 當期望值，會把 29 條正常策略全報成未跑完。故按滾動組態分群後各自比較。
    from strategies.config import strategies_raw_all
    sig = {}
    for st in strategies_raw_all:
        if st.get("formation_strategy_id_base"):
            continue
        p = st["params"]
        sig[f"{st['name']}_MSR0"] = (p.get("formation_window"),
                                     p.get("trading_window"), p.get("rolling_step"))
    d["sig"] = d.strategy_id.map(sig)
    unknown = d[d.sig.isna()]        # 非現役／舊模組的條目，無 config 可對
    known = d[d.sig.notna()]

    short_all = []
    for g, grp in known.groupby("sig"):
        mx = int(grp.n.max())
        for sid, n in zip(grp.strategy_id, grp.n):
            if n < mx:
                short_all.append((sid, int(n), mx))
    if short_all:
        report(ERR, "C", f"{len(short_all)} 條形成期未跑完（與同組態者相比）："
                         f"其交易序列會遠短於同族，不可與全期數字並列")
        for sid, n, mx in short_all:
            print(f"        {sid:<44} {n}/{mx} 期")
    else:
        report(OK, "C", f"{len(known)} 條現役形成期在各自的滾動組態下皆完整")

    if len(unknown):
        mxu = int(unknown.n.max())
        su = unknown[unknown.n < mxu]
        if len(su):
            report(WARN, "C", f"{len(su)} 條非現役／舊模組條目期數偏少"
                              f"（無 config 可對照，僅供參考）：")
            for sid, n in zip(su.strategy_id, su.n):
                print(f"        {sid:<44} {n}/{mxu} 期")


# ── D. 分階段稽核表 ──────────────────────────────────────────────────
def check_stage_tables():
    print("\n[D] 分階段稽核表（分組／排序／篩選）")
    if not os.path.exists(FORMATION_DB):
        report(WARN, "D", "略過"); return
    con = sqlite3.connect(f"file:{FORMATION_DB}?mode=ro", uri=True)
    have = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    for t in ("formation_groups", "formation_ranked"):
        if t not in have:
            report(WARN, "D", f"{t} 不存在（FORMATION_TRACE=1 才會產生）")
            continue
        n, s = con.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT strategy_id) FROM {t}").fetchone()
        if n == 0:
            report(WARN, "D", f"{t} 是空的——若本次有開 FORMATION_TRACE，"
                              f"代表合併階段把它吞掉了")
        else:
            report(OK, "D", f"{t}: {n:,} 列 / {s} 條策略")
    # 兩張表的策略集合應一致
    if {"formation_groups", "formation_ranked"} <= have:
        g = {r[0] for r in con.execute(
            "SELECT DISTINCT strategy_id FROM formation_groups")}
        r = {r[0] for r in con.execute(
            "SELECT DISTINCT strategy_id FROM formation_ranked")}
        if g and r and g != r:
            report(WARN, "D", f"兩張稽核表的策略集合不一致"
                              f"（僅在 groups: {len(g - r)}、僅在 ranked: {len(r - g)}）"
                              f"——半套合併的徵兆")
    con.close()


# ── E. 摘要欄位自洽 ──────────────────────────────────────────────────
def check_summary_sanity(summaries: pd.DataFrame):
    print("\n[E] 摘要欄位自洽")
    bad = summaries[(summaries.Entries.fillna(0) > 0)
                    & (summaries.Final_Equity.isna())]
    if len(bad):
        report(ERR, "E", f"{len(bad)} 條有交易卻無 Final_Equity")
    dup = summaries._path.duplicated().sum()
    if dup:
        report(ERR, "E", f"{dup} 條 _path 重複（主鍵應唯一）")
    if not len(bad) and not dup:
        report(OK, "E", f"{len(summaries)} 條摘要欄位無明顯矛盾")


def main():
    ap = argparse.ArgumentParser(description="回測資料完整性稽核")
    ap.add_argument("--db", default=RESULT_DB)
    ap.add_argument("--quick", action="store_true", help="跳過逐條探測")
    args = ap.parse_args()

    print("=" * 74)
    print("  回測資料完整性稽核")
    print("=" * 74)

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    summaries = pd.read_sql(
        "SELECT _path, METHOD, Entries, Final_Equity FROM strategy_summaries", con)
    live = live_methods()
    print(f"\n摘要 {len(summaries)} 條 / METHOD {summaries.METHOD.nunique()} 個"
          f"（現役 {summaries.METHOD.isin(live).sum()} 條 / {len(live)} 個）")

    check_live_have_logs(con, summaries, live, args.quick)
    check_archived_cached(summaries, live)
    con.close()
    check_formation_complete()
    check_stage_tables()
    check_summary_sanity(summaries)

    print("\n" + "=" * 74)
    errs = [f for f in _findings if f[0] == ERR]
    warns = [f for f in _findings if f[0] == WARN]
    if errs:
        print(f"  {ERR} {len(errs)} 項錯誤、{len(warns)} 項警告")
        return 1
    if warns:
        print(f"  {WARN} {len(warns)} 項警告，無錯誤")
        return 0
    print(f"  {OK} 全部通過")
    return 0


if __name__ == "__main__":
    sys.exit(main())
