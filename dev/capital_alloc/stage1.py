# -*- coding: utf-8 -*-
"""階段一：以既有 trade_logs 做資金配置的精確線性重組。

設計與判準見 `dev/capital_alloc/PREREGISTRATION.md`（**跑前定稿**）。

為何這不是 proxy
----------------
引擎的損益對名目額嚴格線性：`_execute_entry` 中 `v_a + v_b = capital_per_pair`，
毛利與手續費都按名目額比例計。故把某 (配對, 期) 的整條逐日損益乘上 s，
等同於該配對以 s × capital_per_pair 進場——**是恆等式，不是估計**。

唯一的近似是**複利回饋**：引擎的 `capital_per_pair = current_equity / max_pairs`，
權益路徑一變，往後每一筆的名目額都跟著變。本階段忽略這一層。
**故階段一單獨不足以支持任何績效主張**（預先註冊 §四寫死）。

無前視
------
期 k 的權重只用 `Period_Start < k` 的已平倉損益，且分母是**候選集的平均**
而非「當期實際有交易者的總和」——後者是 `dev/utilization/FINDING.md`
記錄的那個當期橫斷面分母陷阱，本檔僅作為對照計算，不列入判準。
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from strategies.metrics import concurrent_of, metrics_from_pnl  # noqa: E402

RESULT_DB = "results/result.db"
OUT_DIR = "results/analysis"
WINDOW = 12      # 落後窗（期），rolling_step=21 → 約 12 個月
WARMUP = 12      # 前 WARMUP+lag 期一律等權（見 _trailing 的 lag 說明）
CAP = 3.0        # 穩健性版本的單配對上限
N_PERM = 200     # ELIM-RND 的重抽次數
EPS = 1e-9

METHODS = ["Grid (NOGRP-DTW)", "Grid (NOGRP-SSD)", "Grid (NOGRP-SDP)",
           "Grid (GICS-SSD)", "Grid (GGR)"]
TOPNS = [5, 10, 20]


def cells() -> pd.DataFrame:
    """13 個基準格（SL0、無任何檔名後綴）。"""
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True, timeout=300)
    d = pd.read_sql(
        'SELECT _path, METHOD, "TOP N" tn, Sharpe_Raw, Ann_Ret_Raw, MDD_Raw '
        "FROM strategy_summaries WHERE METHOD IN (%s) AND \"STOP LOSS %%\"='0%%'"
        % ",".join("?" * len(METHODS)), con, params=METHODS)
    con.close()
    d = d[d._path.str.contains(r"TradeLogs_Top\d+_SL\d+_ZWin\d+_MSR\d+\.csv$",
                               regex=True, na=False)].copy()
    d["top_n"] = d.tn.str.extract(r"(\d+)").astype(int)
    return d[d.top_n.isin(TOPNS)].sort_values(["METHOD", "top_n"]).reset_index(drop=True)


def load_logs(path_key: str) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{RESULT_DB}?mode=ro", uri=True, timeout=300)
    df = pd.read_sql(
        "SELECT Date, Ticker_A, Ticker_B, Period_Start, Daily_Delta, Position "
        "FROM trade_logs WHERE strategy_id = ?", con, params=[path_key])
    con.close()
    df["Date"] = pd.to_datetime(df["Date"])
    df["pair"] = df.Ticker_A + "|" + df.Ticker_B
    return df


def _trailing(df: pd.DataFrame, lag: int):
    """回傳 (periods, pos, pp, M, V)。

    M[(pair, pi)] = 該配對在期 pi 開始**之前即已結束**的最近 WINDOW 期損益總和
    V[(pair, pi)] = 同窗內逐日損益標準差的平均

    ⚠ `lag` 不可省。組合重疊 —— 交易窗 126 日、滾動 21 日 → 同時有
    `lag = 126/21 = 6` 個組合在倉。故期 pi−1 到 pi−5 在期 pi 開始時**尚未平倉**，
    其損益當時不可知。只取 `pj <= pi - lag` 的期，才是配置當下真的看得到的資訊。

    （初版寫成「嚴格早於本期」即可，那是錯的：它讓權重看到五個仍在進行中的期。）
    """
    periods = sorted(df.Period_Start.unique())
    pos = {p: i for i, p in enumerate(periods)}

    pp = (df.groupby(["pair", "Period_Start"], sort=False)
            .agg(pnl=("Daily_Delta", "sum"), sd=("Daily_Delta", "std"))
            .reset_index())
    pp["pi"] = pp.Period_Start.map(pos)
    pp = pp.reset_index(drop=True)
    pp["gid"] = np.arange(len(pp))

    by_pair = {k: g.sort_values("pi") for k, g in pp.groupby("pair", sort=False)}

    M, V = {}, {}
    for pair, g in by_pair.items():
        pis = g.pi.to_numpy()
        pnls = g.pnl.to_numpy()
        sds = g.sd.to_numpy()
        for j, pi in enumerate(pis):
            elig = np.nonzero(pis[:j] <= pi - lag)[0]   # 已結束者
            if len(elig) == 0:
                M[(pair, pi)] = np.nan
                V[(pair, pi)] = np.nan
                continue
            take = elig[-WINDOW:]                        # 最近 WINDOW 期
            M[(pair, pi)] = float(np.nansum(pnls[take]))
            v = sds[take]
            v = v[~np.isnan(v)]
            V[(pair, pi)] = float(np.mean(v)) if len(v) else np.nan
    return periods, pos, pp, M, V


def build_scales(df, pos, pp, M, V, mode: str, lag: int = 6, rng=None, cap=None):
    """回傳長度 len(pp) 的 s 陣列，索引即 pp.gid。等權為 s ≡ 1。

    用 gid 陣列而非 (pair, pi) 字典：ELIM-RND 要重抽 200 次，
    每次都對數十萬列做字典查找會慢兩個數量級。
    """
    scales = np.ones(len(pp), dtype=float)
    for pi, g in pp.groupby("pi", sort=True):
        cands = g.pair.tolist()
        gids = g.gid.to_numpy()
        if pi < WARMUP + lag:
            continue

        if mode == "MO":
            raw = np.array([max(M.get((p, pi), np.nan) or 0.0, 0.0)
                            if not np.isnan(M.get((p, pi), np.nan)) else np.nan
                            for p in cands], dtype=float)
        elif mode == "RI":
            v = np.array([V.get((p, pi), np.nan) for p in cands], dtype=float)
            raw = np.where(np.isnan(v), np.nan, 1.0 / np.maximum(v, EPS))
        elif mode in ("ELIM", "ELIM_RND"):
            m = np.array([M.get((p, pi), np.nan) for p in cands], dtype=float)
            keep = np.where(np.isnan(m), 1.0, (m >= 0).astype(float))
            if mode == "ELIM_RND":
                n_drop = int((keep == 0).sum())
                keep = np.ones(len(cands))
                if n_drop > 0:
                    keep[rng.choice(len(cands), n_drop, replace=False)] = 0.0
            scales[gids] = keep
            continue
        else:
            raise ValueError(mode)

        # 暖身後仍無前史者（該配對首次出現）→ 給等權，不參與正規化的分子偏移
        raw = np.where(np.isnan(raw), np.nanmean(raw) if np.isfinite(np.nanmean(raw)) else 1.0, raw)
        mean = float(np.mean(raw))
        s = np.ones(len(cands)) if mean <= EPS else raw / mean

        if cap is not None:
            s = _waterfill_cap(s, cap)

        scales[gids] = s
    return scales


def _waterfill_cap(s: np.ndarray, cap: float) -> np.ndarray:
    """一次水填：超過 cap 的削平，超額按比例回填給未達上限者，維持平均不變。"""
    s = s.copy()
    over = s > cap
    if not over.any():
        return s
    excess = float((s[over] - cap).sum())
    s[over] = cap
    room = ~over
    if room.any() and s[room].sum() > EPS:
        s[room] = s[room] + excess * (s[room] / s[room].sum())
        s = np.minimum(s, cap)
    return s


def row_gid(df, pp) -> np.ndarray:
    """每一列 trade_logs 對應的 (pair, 期) 群組編號。每格算一次。"""
    key = pd.MultiIndex.from_arrays([df.pair, df.Period_Start])
    idx = pd.MultiIndex.from_arrays([pp.pair, pp.Period_Start])
    return idx.get_indexer(key)


def apply_scales(df, gid, scales) -> pd.Series:
    scaled = df.Daily_Delta.to_numpy() * scales[gid]
    return pd.Series(scaled, index=df.Date).groupby(level=0).sum().sort_index()


def notional_ratio(df, gid, scales) -> float:
    """實際部署名目額比值：有部位的列上 s 的平均。偏離 1 即混有槓桿成分。"""
    m = (df.Position.fillna(0) != 0).to_numpy()
    if not m.any():
        return 1.0
    return float(scales[gid[m]].mean())


def drop_flat_months(pnl: pd.Series, df, gid, scales) -> pd.Series:
    """P4 乙口徑：刪除**加權後**完全沒有部位的月份（原文『NA 補齊、不納入分母』）。

    ⚠ 初版用的是 `df.Position`，也就是**基準**的持倉狀態——而基準在本專案
    幾乎沒有空倉月（6 個組合重疊、top_n≥5），於是甲乙兩口徑逐格完全相同，
    這個檢定變成空的。淘汰規則造成的空倉是「有部位但權重為 0」，
    必須把 s 一起算進去才看得到。
    """
    live = (df.Position.fillna(0).to_numpy() != 0) & (scales[gid] > 0)
    occ = pd.Series(live.astype(int), index=df.Date).groupby(level=0).sum()
    occ.index = pd.to_datetime(occ.index)
    monthly_open = occ.resample("ME").sum()
    live = monthly_open[monthly_open > 0].index
    keep = pnl.index.to_period("M").isin(live.to_period("M"))
    return pnl[keep]


def main() -> int:
    cl = cells()
    print(f"{len(cl)} 格：")
    for _, r in cl.iterrows():
        print(f"  {r.METHOD:<20} Top{r.top_n:<3} Sharpe={r.Sharpe_Raw:+.4f}")

    rows, diag = [], []
    for _, r in cl.iterrows():
        df = load_logs(r._path)
        if df.empty:
            print(f"  ! {r._path} 無 trade_logs，跳過")
            continue
        lag = max(1, concurrent_of(r._path))
        periods, pos, pp, M, V = _trailing(df, lag)
        gid = row_gid(df, pp)

        base_pnl = pd.Series(df.Daily_Delta.to_numpy(), index=df.Date
                             ).groupby(level=0).sum().sort_index()
        base = metrics_from_pnl(base_pnl)

        rec = {"METHOD": r.METHOD, "top_n": r.top_n, "path": r._path,
               "n_periods": len(periods), "lag": lag,
               "Sharpe_EW": base.Sharpe_Raw, "Ann_EW": base.Ann_Ret_Raw,
               "MDD_EW": base.MDD_Raw}

        for mode in ("MO", "RI", "ELIM"):
            sc = build_scales(df, pos, pp, M, V, mode, lag)
            pnl = apply_scales(df, gid, sc)
            m = metrics_from_pnl(pnl)
            rec[f"Sharpe_{mode}"] = m.Sharpe_Raw
            rec[f"Ann_{mode}"] = m.Ann_Ret_Raw
            rec[f"MDD_{mode}"] = m.MDD_Raw
            rec[f"NotRatio_{mode}"] = notional_ratio(df, gid, sc)
            rec[f"MaxS_{mode}"] = float(sc.max())
            if mode == "ELIM":
                # P4：兩種計量口徑
                cut = drop_flat_months(pnl, df, gid, sc)
                m2 = metrics_from_pnl(cut)
                rec["MDD_ELIM_dropflat"] = m2.MDD_Raw
                rec["Ann_ELIM_dropflat"] = m2.Ann_Ret_Raw
            else:
                sc_cap = build_scales(df, pos, pp, M, V, mode, lag, cap=CAP)
                mc = metrics_from_pnl(apply_scales(df, gid, sc_cap))
                rec[f"Sharpe_{mode}_cap"] = mc.Sharpe_Raw

        # ELIM-RND：同率隨機淘汰的分布
        rng = np.random.default_rng(20260903)
        perm = []
        for _ in range(N_PERM):
            sc = build_scales(df, pos, pp, M, V, "ELIM_RND", lag, rng=rng)
            perm.append(metrics_from_pnl(apply_scales(df, gid, sc)).Sharpe_Raw)
        perm = np.array(perm)
        rec["Sharpe_RND_mean"] = float(perm.mean())
        rec["Sharpe_RND_p95"] = float(np.percentile(perm, 95))
        rec["ELIM_beats_RND"] = bool(rec["Sharpe_ELIM"] > rec["Sharpe_RND_p95"])
        rec["ELIM_pctile"] = float((perm < rec["Sharpe_ELIM"]).mean())

        rows.append(rec)
        print(f"  {r.METHOD:<20} Top{r.top_n:<3} "
              f"EW={rec['Sharpe_EW']:+.4f} MO={rec['Sharpe_MO']:+.4f} "
              f"RI={rec['Sharpe_RI']:+.4f} ELIM={rec['Sharpe_ELIM']:+.4f} "
              f"RNDp95={rec['Sharpe_RND_p95']:+.4f} "
              f"NotRatio_MO={rec['NotRatio_MO']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    dst = os.path.join(OUT_DIR, "capital_alloc_stage1.csv")
    out.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"\n-> {dst}  ({len(out)} 格)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
