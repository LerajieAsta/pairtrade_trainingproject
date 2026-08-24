"""梯子第四階 M3：1D-CNN 序列編碼器，與 M0+M1+M2 表格特徵並接。

前推設定與 ladder.py 完全相同（暖身 36／擴張窗／每 12 期重訓／6 期隔離帶）。
超參數固定不調——梯子是預先註冊的四階而非搜尋。

計算預算：無 GPU（torch 2.2.2+cpu）。每次重訓的訓練集在後期達 118 萬列，
全量跑 3 個 epoch 不切實際，故每次重訓隨機抽樣至多 MAX_TRAIN 列（固定種子）。
此為實作上的算力取捨，不影響梯子的預先註冊結構，但須明載。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dev.ml_formation.ladder import (BURN_IN, CACHE, EMBARGO, F_M0, F_M1, F_M2,
                                     RETRAIN_EVERY, TOP_N, evaluate)

MAX_TRAIN = 400_000
EPOCHS = 3
BATCH = 1024
LR = 1e-3
torch.manual_seed(0)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))


class SeqNet(nn.Module):
    def __init__(self, n_tab: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=2, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, 3, stride=2, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + n_tab, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, seq, tab):
        h = self.conv(seq.unsqueeze(1))
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(torch.cat([h, tab], dim=1)).squeeze(1)


def main():
    D = pd.read_parquet(os.path.join(CACHE, "train.parquet"))
    D["row_id"] = np.arange(len(D))
    D = D[D.label_valid].copy()
    per = np.sort(D.Period_Start.unique())
    D["pi"] = D.Period_Start.map({p: i for i, p in enumerate(per)})
    n_per = len(per)
    feats = F_M0 + F_M1 + F_M2
    X = D[feats].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = D.not_converged.values.astype(np.float32)
    rid = D.row_id.values
    seq = np.load(os.path.join(CACHE, "seq_all.npy"), mmap_mode="r")
    print(f"可標記 {len(D):,} 列，{n_per} 期，表格特徵 {len(feats)} 欄", flush=True)

    pred = np.full(len(D), np.nan, dtype=np.float64)
    pos = {v: i for i, v in enumerate(D.index)}
    rng = np.random.default_rng(0)
    points = list(range(BURN_IN, n_per, RETRAIN_EVERY))
    t0 = time.time()

    for k, T in enumerate(points, 1):
        tr_m = (D.pi <= T - EMBARGO).values
        te_m = ((D.pi >= T) & (D.pi < T + RETRAIN_EVERY)).values
        if tr_m.sum() < 5000 or te_m.sum() == 0 or len(np.unique(y[tr_m])) < 2:
            continue
        tr_idx = np.flatnonzero(tr_m)
        if len(tr_idx) > MAX_TRAIN:
            tr_idx = rng.choice(tr_idx, MAX_TRAIN, replace=False)
        tr_idx.sort()
        mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-8

        net = SeqNet(len(feats))
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        lossf = nn.BCEWithLogitsLoss()
        net.train()
        for _ in range(EPOCHS):
            order = rng.permutation(len(tr_idx))
            for b in range(0, len(order), BATCH):
                sel = tr_idx[order[b:b + BATCH]]
                s = torch.from_numpy(np.asarray(seq[sel], dtype=np.float32))
                t = torch.from_numpy((X[sel] - mu) / sd)
                opt.zero_grad()
                loss = lossf(net(s, t), torch.from_numpy(y[sel]))
                loss.backward()
                opt.step()

        net.eval()
        te_idx = np.flatnonzero(te_m)
        outs = []
        with torch.no_grad():
            for b in range(0, len(te_idx), 8192):
                sel = te_idx[b:b + 8192]
                s = torch.from_numpy(np.asarray(seq[sel], dtype=np.float32))
                t = torch.from_numpy((X[sel] - mu) / sd)
                outs.append(torch.sigmoid(net(s, t)).numpy())
        pred[te_idx] = np.concatenate(outs)
        el = time.time() - t0
        print(f"  重訓 {k}/{len(points)}  T={T}  訓練 {len(tr_idx):,}  測試 {len(te_idx):,}"
              f"  {el:.0f}s (預估總計 {el/k*len(points)/60:.0f} 分)", flush=True)

    D["score_M3"] = pred
    sub = D[(D.pi >= BURN_IN)].dropna(subset=["score_M3"])
    R = evaluate(sub, "score_M3")
    se = R.auc.std() / np.sqrt(len(R))
    print(f"\nM3 (+序列編碼器)  逐期AUC {R.auc.mean():.4f} ± {se:.4f}"
          f"   下界 {R.auc.mean()-se:.4f}   top20未收斂 {R.top_nc.mean():.4f}  ({len(R)} 期)")
    R.assign(rung="M3 (+序列編碼器)").to_parquet(os.path.join(CACHE, "m3_results.parquet"))
    D[["Period_Start", "Ticker_A", "Ticker_B", "pi", "score_M3"]].to_parquet(
        os.path.join(CACHE, "m3_scores.parquet"))
    print("→ cache/m3_results.parquet")


if __name__ == "__main__":
    main()
