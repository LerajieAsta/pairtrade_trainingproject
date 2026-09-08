# -*- coding: utf-8 -*-
"""
期末強制平倉的配對，之後會回歸嗎
======================================================================
論文 4.4.3 與附錄 C 的來源。原本是一支一次性腳本，**未留在版本庫**，於是
2026-08-13 `result.db` 重建後，論文裡的「4,738 筆、126 日內 39.1% 回歸」
變成無法重算的數字。本模組把它補成可重跑的管線成員。

**問題。** 若期末仍持倉的配對只是「還需要時間」，延長交易期即可解決；
若它們是真發散，延長交易期只會擴大虧損。

**設計。** 取主軸 15 臂（5 分組 × 3 排序）的 Z-Score 基準格 `Top1/SL0%`，
自 `trade_logs` 取出每筆 `PERIOD_END_EXIT`，以**引擎自身的 z 定義**往後
追蹤 126 個交易日（= 一個完整交易期），看 z 是否觸及 0。

    回歸 = z 觸及 0（`exit_z = 0.0`，見 config；zscore_trading 的
           `is_exit_short: z <= exit_z` / `is_exit_long: z >= -exit_z`）

**z 的重建。** 主軸皆為 `ZWin0` 且走路徑 B（標準化 norm_p 空間）：

    norm_p = (log P - Log_Mean) / Log_Std          ← 兩者存在 trade_logs
    spread = norm_p_A - Hedge_Ratio * norm_p_B     ← Hedge_Ratio 存在 trade_logs
    z      = (spread - form_spread_mean) / form_spread_std

`form_spread_mean/std` 未存進 trade_logs，但可由同一筆交易的任兩個
(spread, z) 反解——兩式兩未知數，且該兩常數在一筆交易內固定：

    σ = (s1 - s2) / (z1 - z2) ,  m = s1 - z1 σ

反解後以第三列驗證（`_RESIDUAL_TOL`），不通過者剔除並計數，不靜默略過。

**限制。** (a) 追蹤用未複權收盤價之對數，與回測同源；(b) 只要 z 曾觸及 0
即算回歸，不計是否可實際成交；(c) 僅 Top1/SL0% 一格——該格無停損，
故強平事件不被停損截斷，是觀察「期末仍持倉」最乾淨的一格。
"""
import os
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
PRICE_DB = "dataset/price/sp500_Tiingo.db"
OUT_DIR = "results/analysis"

CELL = "TradeLogs_Top1_SL0_ZWin0_MSR0.csv"
ARMS = [f"Grid ({g}-{s})" for g in ("NOGRP", "GICS", "HDB", "AGG", "KM")
        for s in ("SSD", "DTW", "SDP")]
HORIZONS = [21, 42, 63, 126]
_RESIDUAL_TOL = 1e-3     # 反解後第三列的 |z_hat - z| 容許值
_MIN_Z_GAP = 1e-6        # 反解要求兩列 z 有足夠差距
_CLIP = 10.0             # config.zscore_clip；被夾住的列不可用於反解


def _sids() -> list[str]:
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    df = pd.read_sql(
        f"SELECT METHOD,_path FROM strategy_summaries WHERE METHOD IN "
        f"({','.join('?' * len(ARMS))})", con, params=ARMS)
    con.close()
    return sorted(p for p in df._path if p.endswith(CELL))


def _load_trades(sids: list[str]) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True)
    out = []
    for sid in sids:
        d = pd.read_sql(
            "SELECT Date,Price_A,Price_B,Hedge_Ratio,ZScore,Position,Status,"
            "Period_Start,Period_End,Ticker_A,Ticker_B,"
            "Log_Mean_A,Log_Std_A,Log_Mean_B,Log_Std_B "
            "FROM trade_logs WHERE strategy_id=? AND Position!=0",
            con, params=(sid,))
        if len(d):
            d["sid"] = sid
            out.append(d)
    con.close()
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _prices() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{PRICE_DB}?mode=ro", uri=True)
    px = pd.read_sql("SELECT Date,Symbol,Close FROM Daily_Prices "
                     "WHERE Close IS NOT NULL AND Close>0", con)
    con.close()
    return px.pivot(index="Date", columns="Symbol", values="Close").sort_index()


def _spread(row, pa: float, pb: float) -> float:
    na = (np.log(pa) - row.Log_Mean_A) / row.Log_Std_A
    nb = (np.log(pb) - row.Log_Mean_B) / row.Log_Std_B
    return na - row.Hedge_Ratio * nb


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    sids = _sids()
    print(f"主軸 {len(sids)} 臂 × {CELL}")
    tl = _load_trades(sids)
    if tl.empty:
        raise SystemExit("trade_logs 無明細——該格可能已被 archive_trade_logs 清除")
    print(f"持倉日列數 {len(tl):,}")

    px = _prices()
    dates = px.index.to_numpy()
    pos = {d: i for i, d in enumerate(dates)}
    # trade_logs 的 Date 是 'YYYY-MM-DD 00:00:00'，價格庫是 'YYYY-MM-DD'
    tl["Date"] = tl.Date.str.slice(0, 10)

    key = ["sid", "Ticker_A", "Ticker_B", "Period_Start"]
    recs, bad_solve, no_price, no_room, resids = [], 0, 0, 0, []

    for _, g in tl.groupby(key, sort=False):
        g = g.sort_values("Date")
        if not (g.Status == "PERIOD_END_EXIT").any():
            continue
        r0 = g.iloc[0]
        if pd.isna(r0.Log_Std_A) or pd.isna(r0.Log_Std_B) or r0.Log_Std_A == 0:
            bad_solve += 1
            continue

        s = np.array([_spread(r0, a, b) for a, b in zip(g.Price_A, g.Price_B)])
        z = g.ZScore.to_numpy(float)
        # `zscore_clip = 10.0`：被夾住的列不再滿足 z = (spread − m)/σ 的線性關係，
        # 反解時必須先剔除。強平交易的 z 本來就偏極端，若不剔除，argmin/argmax
        # 幾乎必然挑到被夾住的兩列，反解出的 σ 全錯（實測 744/3,279 筆因此失敗）。
        free = np.abs(z) < _CLIP - 1e-6
        if free.sum() < 3:
            bad_solve += 1
            continue
        zf_, sf_ = z[free], s[free]
        i, j = int(np.argmin(zf_)), int(np.argmax(zf_))
        if abs(zf_[j] - zf_[i]) < _MIN_Z_GAP:
            bad_solve += 1
            continue
        sig = (sf_[j] - sf_[i]) / (zf_[j] - zf_[i])
        m = sf_[i] - zf_[i] * sig
        if not np.isfinite(sig) or sig <= 0:
            bad_solve += 1
            continue
        k = [t for t in range(len(zf_)) if t not in (i, j)]
        resid = float(max(abs((sf_[k] - m) / sig - zf_[k]))) if k else 0.0
        resids.append(resid)
        if resid > _RESIDUAL_TOL:
            bad_solve += 1
            continue

        ex = g[g.Status == "PERIOD_END_EXIT"].iloc[-1]
        d0 = ex.Date
        if d0 not in pos or ex.Ticker_A not in px.columns or ex.Ticker_B not in px.columns:
            no_price += 1
            continue
        i0 = pos[d0]
        if i0 + 1 >= len(dates):
            no_room += 1
            continue
        fwd = slice(i0 + 1, min(i0 + 1 + max(HORIZONS), len(dates)))
        pa = px[ex.Ticker_A].to_numpy()[fwd]
        pb = px[ex.Ticker_B].to_numpy()[fwd]
        ok = np.isfinite(pa) & np.isfinite(pb) & (pa > 0) & (pb > 0)
        if ok.sum() < HORIZONS[0]:
            no_price += 1
            continue
        na = (np.log(np.where(ok, pa, np.nan)) - ex.Log_Mean_A) / ex.Log_Std_A
        nb = (np.log(np.where(ok, pb, np.nan)) - ex.Log_Mean_B) / ex.Log_Std_B
        zf = ((na - ex.Hedge_Ratio * nb) - m) / sig
        zf = np.clip(zf, -10.0, 10.0)

        z_ex = float(ex.ZScore)
        # 回歸 = z 觸及 0（依 exit_z=0.0 的出場條件，以平倉當時的方向判定）
        hit = (zf <= 0) if z_ex > 0 else (zf >= 0)
        hit = hit & np.isfinite(zf)
        first = int(np.argmax(hit)) + 1 if hit.any() else None

        rec = {"sid": ex.sid, "arm": ex.sid.split("/")[1], "z_exit": z_ex,
               "first_hit": first, "n_fwd": int(np.isfinite(zf).sum())}
        for h in HORIZONS:
            rec[f"h{h}"] = bool(first is not None and first <= h)
        tail = zf[:HORIZONS[-1]]
        tail = tail[np.isfinite(tail)]
        rec["z_end"] = float(tail[-1]) if len(tail) else np.nan
        recs.append(rec)

    if resids:
        q = np.percentile(resids, [50, 90, 99])
        print(f"反解殘差分位（中位/90/99）：{q[0]:.2e} / {q[1]:.2e} / {q[2]:.2e}")
    d = pd.DataFrame(recs)
    if d.empty:
        raise SystemExit("無可用強平交易——先看上方的剔除計數，不要當成 0% 回歸")
    print(f"可用強平交易 {len(d):,} 筆"
          f"（反解失敗 {bad_solve:,}、缺價 {no_price:,}、期末無後續 {no_room:,}）")
    d.to_csv(f"{OUT_DIR}/forced_close_followup_trades.csv", index=False)

    rows = [{"追蹤期": f"{h} 日",
             "累計回歸比例%": round(float(d[f"h{h}"].mean()) * 100, 1)}
            for h in HORIZONS]
    summ = pd.DataFrame(rows)
    print("\n" + summ.to_string(index=False))

    rev = d[d[f"h{HORIZONS[-1]}"]]
    nrev = d[~d[f"h{HORIZONS[-1]}"]]
    stat = pd.DataFrame([{
        "強平筆數": len(d),
        "平倉時|z|中位": round(float(d.z_exit.abs().median()), 2),
        "126日回歸%": round(float(d[f"h{HORIZONS[-1]}"].mean()) * 100, 1),
        "回歸者平倉時|z|中位": round(float(rev.z_exit.abs().median()), 2) if len(rev) else np.nan,
        "未回歸者平倉時|z|中位": round(float(nrev.z_exit.abs().median()), 2) if len(nrev) else np.nan,
        "未回歸者126日後|z|中位": round(float(nrev.z_end.abs().median()), 2) if len(nrev) else np.nan,
    }])
    print("\n" + stat.to_string(index=False))

    summ.to_csv(f"{OUT_DIR}/forced_close_followup.csv", index=False)
    stat.to_csv(f"{OUT_DIR}/forced_close_followup_stats.csv", index=False)
    print(f"\n→ {OUT_DIR}/forced_close_followup{{,_stats,_trades}}.csv")


if __name__ == "__main__":
    run()
