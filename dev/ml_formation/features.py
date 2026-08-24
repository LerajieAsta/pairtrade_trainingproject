"""M1（分段穩定度）與 M2（圖位置）特徵。

M1 的價差漂移部分直接由快取序列取得：Z_form = (S - mu_s) / sigma_s，故其分段
均值即「該段價差偏離全窗均值幾個 sigma_s」，不必重建 spread。需要重算的只有
分段相關係數。

M2 量的是「這個 rho 有多特殊」而非「rho 有多高」——後者就是 SSD（見
PREREGISTRATION 的仿射恆等式），已由 M0 涵蓋。若 A 與群內幾十檔都高相關，
那是市場 beta；若只與 B 高相關，才是特定經濟連結。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dev.ml_formation.build import CACHE, SEQ_DIR
from dev.ml_formation.pool import (FORMATION_WINDOW, load_groups, load_prices,
                                   roll_indices)
from dev.ml_formation.pipeline_select import normalized

N_SEG = 4
SEG_LEN = FORMATION_WINDOW // N_SEG      # 63


def segment_rho(norm: pd.DataFrame, members: list[str]) -> np.ndarray:
    """(N_SEG, n_pairs) 分段相關係數，配對順序為 triu(k=1) 且 i=B、j=A。"""
    V = norm[members].values.T                       # (N, T)
    iu, ju = np.triu_indices(len(members), k=1)
    out = np.empty((N_SEG, len(iu)))
    for s in range(N_SEG):
        seg = V[:, s * SEG_LEN:(s + 1) * SEG_LEN]
        R = np.corrcoef(seg)
        out[s] = R[iu, ju]
    return out


def graph_feats(R: np.ndarray, iu: np.ndarray, ju: np.ndarray) -> dict:
    """由群內全窗相關矩陣導出「這條連結有多特殊」。"""
    N = R.shape[0]
    Rz = R.copy()
    np.fill_diagonal(Rz, np.nan)
    mean_rho = np.nanmean(Rz, axis=1)                # 每檔對群內同儕的平均相關
    max_rho = np.nanmax(Rz, axis=1)
    # ρ_AB 在各自同儕清單中的名次（0 = 最高）
    order = np.argsort(-np.nan_to_num(Rz, nan=-2.0), axis=1)
    rank_of = np.empty_like(order)
    rows = np.arange(N)[:, None]
    rank_of[rows, order] = np.arange(N)[None, :]

    rho = R[iu, ju]
    denom = max(N - 1, 1)
    # 高相關同儕的共同鄰居：以群內 rho 的第 90 百分位為門檻
    thr = np.nanquantile(Rz, 0.90)
    H = np.nan_to_num(Rz, nan=-2.0) > thr
    common = (H[iu] & H[ju]).sum(axis=1)

    return {
        "rho_excess": rho - 0.5 * (mean_rho[iu] + mean_rho[ju]),
        "mean_rho_A": mean_rho[ju], "mean_rho_B": mean_rho[iu],
        "rank_in_A": rank_of[ju, iu] / denom,
        "rank_in_B": rank_of[iu, ju] / denom,
        "gap_to_best_A": max_rho[ju] - rho,
        "gap_to_best_B": max_rho[iu] - rho,
        "deg_A": H[ju].sum(axis=1) / denom,
        "deg_B": H[iu].sum(axis=1) / denom,
        "common_nb": common / denom,
        "group_size": np.full(len(iu), N),
    }


def build_period_feats(form_prices: pd.DataFrame, group_map: dict,
                       Zform: np.ndarray) -> pd.DataFrame:
    log_px = np.log(form_prices.where(form_prices > 0))
    usable = [t for t in log_px.columns
              if group_map.get(t, "Unknown") != "Unknown" and log_px[t].notna().all()]
    norm = normalized(form_prices, usable)

    by_group: dict[str, list[str]] = {}
    for t in usable:
        by_group.setdefault(group_map[t], []).append(t)

    parts, cursor = [], 0
    for grp, members in by_group.items():
        if len(members) < 2:
            continue
        n_pairs = len(members) * (len(members) - 1) // 2
        iu, ju = np.triu_indices(len(members), k=1)
        V = norm[members].values.T
        R = np.corrcoef(V)

        sr = segment_rho(norm, members)                       # (N_SEG, n_pairs)
        d = graph_feats(R, iu, ju)
        d["rho_seg_std"] = sr.std(axis=0)
        d["rho_seg_min"] = sr.min(axis=0)
        d["rho_seg_drift"] = sr[-1] - sr[0]
        d["rho_seg_range"] = sr.max(axis=0) - sr.min(axis=0)

        # 價差漂移：由快取序列的分段均值取得（單位為 sigma_s）
        Zg = Zform[cursor:cursor + n_pairs]
        segm = np.stack([np.nanmean(Zg[:, s * SEG_LEN:(s + 1) * SEG_LEN], axis=1)
                         for s in range(N_SEG)], axis=1)       # (n_pairs, N_SEG)
        segs = np.stack([np.nanstd(Zg[:, s * SEG_LEN:(s + 1) * SEG_LEN], axis=1)
                         for s in range(N_SEG)], axis=1)
        d["mu_seg_std"] = segm.std(axis=1)
        d["mu_seg_drift"] = segm[:, -1] - segm[:, 0]
        d["mu_seg_absmax"] = np.abs(segm).max(axis=1)
        d["sd_seg_ratio"] = segs[:, -1] / np.where(segs[:, 0] > 1e-9, segs[:, 0], np.nan)
        d["sd_seg_std"] = segs.std(axis=1)
        d["z_form_last"] = np.abs(Zg[:, -1])
        d["z_form_ncross"] = (np.diff(np.sign(np.nan_to_num(Zg)), axis=1) != 0).sum(axis=1)

        df = pd.DataFrame(d)
        df["Ticker_B"] = [members[i] for i in iu]
        df["Ticker_A"] = [members[j] for j in ju]
        parts.append(df)
        cursor += n_pairs

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    pivot, dates, total, first_idx = load_prices()
    idxs = roll_indices(total, first_idx)
    groups = load_groups()

    parts, t0 = [], time.time()
    for k, i in enumerate(idxs, 1):
        ps = dates[i - FORMATION_WINDOW].strftime("%Y-%m-%d")
        gm = groups.get(ps)
        seq_path = os.path.join(SEQ_DIR, f"{ps}.npy")
        if gm is None or not os.path.exists(seq_path):
            continue
        Z = np.load(seq_path)
        df = build_period_feats(pivot.iloc[i - FORMATION_WINDOW:i], gm, Z)
        if df.empty:
            continue
        if len(df) != len(Z):
            raise RuntimeError(f"{ps}: 特徵 {len(df)} 列與序列 {len(Z)} 列不符")
        df.insert(0, "Period_Start", ps)
        parts.append(df)
        if k % 25 == 0 or k == len(idxs):
            el = time.time() - t0
            print(f"  {k}/{len(idxs)}  {ps}  {sum(len(p) for p in parts):,} 列"
                  f"  {el:.0f}s (預估 {el/k*len(idxs)/60:.1f} 分)", flush=True)

    full = pd.concat(parts, ignore_index=True)
    full.to_parquet(os.path.join(CACHE, "feats.parquet"))
    print(f"\n完成：{len(full):,} 列 x {full.shape[1]} 欄 → cache/feats.parquet")


if __name__ == "__main__":
    main()
