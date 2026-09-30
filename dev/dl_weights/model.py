# -*- coding: utf-8 -*-
"""1D-CNN（形成窗序列）+ MLP（表格特徵），兩個輸出頭：log σ̂（RI 用）與 μ̂（MO 用）。"""
import numpy as np
import torch
from torch import nn


class PairNet(nn.Module):
    def __init__(self, n_tab: int, width: int = 32, dropout: float = 0.1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(3, 16, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(16, width, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.tab = nn.Sequential(nn.Linear(n_tab, width), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(2 * width, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, seq, tab):
        return self.head(torch.cat([self.seq(seq), self.tab(tab)], dim=1))


class Scaler:
    """只用訓練集估計的標準化參數（無前視）。"""

    def fit(self, x):
        self.mu = np.nanmean(x, axis=0)
        self.sd = np.nanstd(x, axis=0)
        self.sd[~(self.sd > 1e-12)] = 1.0
        return self

    def __call__(self, x):
        return np.nan_to_num((x - self.mu) / self.sd)


def fit_predict(seq_tr, tab_tr, r_tr, v_tr, seq_te, tab_te, seed: int,
                max_epochs: int = 60, patience: int = 8, batch: int = 256,
                lr: float = 1e-3, wd: float = 1e-4):
    """訓練並回傳測試集的 (σ̂, μ̂)，皆已還原到原尺度。

    驗證集取訓練集中**時間最後**的 20%（輸入已依期排序），早停用；不隨機切分，
    否則同一期的配對會同時落在訓練與驗證兩側。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    lo, hi = np.nanpercentile(r_tr, [1, 99])
    r_c = np.clip(r_tr, lo, hi)
    logv = np.log(v_tr + 1e-6)
    y = np.stack([logv, r_c], axis=1)
    ysc = Scaler().fit(y)
    tsc = Scaler().fit(tab_tr)

    n = len(y)
    n_va = max(1, int(n * 0.2))
    tr, va = slice(0, n - n_va), slice(n - n_va, n)
    T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))  # noqa: E731
    S, X, Y = T(seq_tr), T(tsc(tab_tr)), T(ysc(y))

    net = PairNet(X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    huber = nn.HuberLoss(delta=1.0)

    def loss_fn(out, yy):
        return nn.functional.mse_loss(out[:, 0], yy[:, 0]) + huber(out[:, 1], yy[:, 1])

    best, best_state, bad = np.inf, None, 0
    idx_tr = np.arange(n - n_va)
    for _ in range(max_epochs):
        net.train()
        np.random.shuffle(idx_tr)
        for j in range(0, len(idx_tr), batch):
            b = idx_tr[j:j + batch]
            opt.zero_grad()
            loss_fn(net(S[b], X[b]), Y[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(S[va], X[va]), Y[va]))
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        out = net(T(seq_te), T(tsc(tab_te))).numpy()
    out = out * ysc.sd + ysc.mu
    return np.exp(out[:, 0]), out[:, 1]
