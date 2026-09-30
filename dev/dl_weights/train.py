# -*- coding: utf-8 -*-
"""Walk-forward 訓練，產出每個 (配對, 期) 的 σ̂ 與 μ̂。設計見 PREREGISTRATION.md §三。

    python dev/dl_weights/train.py            # 15 臂
    python dev/dl_weights/train.py "Grid GICS-SSD"
"""
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dev.dl_weights import features as F  # noqa: E402

LAG = 6            # 並行期數：期 k 只看得到 pj ≤ k − 6 的結果
MIN_TRAIN = 48     # 首次訓練需要的已平倉期數
RETRAIN = 12       # 每 12 期重訓
SEEDS = (0, 1, 2)
WORKERS = 4


def pred_file(arm):
    return os.path.join(F.CACHE, f"pred_{arm.replace(' ', '_')}.pkl")


def train_arm(arm: str) -> str:
    import torch
    torch.set_num_threads(2)
    from dev.dl_weights.model import fit_predict

    t0 = time.time()
    meta, seq, tab = F.build(arm, pv=None)     # 主行程已建好快取

    usable = meta.feat_ok.to_numpy()
    labeled = usable & meta.r.notna().to_numpy() & meta.v.notna().to_numpy()
    pi = meta.pi.to_numpy()
    n_per = int(pi.max()) + 1
    k0 = MIN_TRAIN + LAG
    sig = np.full((len(meta), len(SEEDS)), np.nan)
    mu = np.full((len(meta), len(SEEDS)), np.nan)

    for k in range(k0, n_per, RETRAIN):
        tr = labeled & (pi <= k - LAG)
        te = usable & (pi >= k) & (pi < k + RETRAIN)
        if not te.any():
            continue
        for j, s in enumerate(SEEDS):
            sh, mh = fit_predict(seq[tr], tab[tr], meta.r.to_numpy()[tr], meta.v.to_numpy()[tr],
                                 seq[te], tab[te], seed=s)
            sig[te, j], mu[te, j] = sh, mh
        print(f"  {arm} k={k} train={int(tr.sum())} test={int(te.sum())} "
              f"({time.time() - t0:.0f}s)", flush=True)

    out = meta[["Period_Start", "Trade_Start", "Ticker_A", "Ticker_B", "Pair_Rank", "pi", "r", "v"]].copy()
    out["sigma_hat"] = np.nanmean(sig, axis=1)
    out["mu_hat"] = np.nanmean(mu, axis=1)
    for j, s in enumerate(SEEDS):
        out[f"sigma_s{s}"], out[f"mu_s{s}"] = sig[:, j], mu[:, j]
    out.attrs["k0"] = k0
    with open(pred_file(arm), "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return f"{arm}: done in {time.time() - t0:.0f}s"


def main():
    arms = sys.argv[1:] or F.ARMS
    need = [a for a in arms if not os.path.exists(
        os.path.join(F.CACHE, f"feat_{a.replace(' ', '_')}.pkl"))]
    if need:
        pv = F.price_pivot()
        for a in need:
            t = time.time()
            meta, _, _ = F.build(a, pv)
            print(f"features {a}: {len(meta)} 列，可用 {int(meta.feat_ok.sum())}，"
                  f"有標的 {int(meta.r.notna().sum())}（{time.time() - t:.0f}s）", flush=True)
    todo = [a for a in arms if not os.path.exists(pred_file(a))]
    with ProcessPoolExecutor(max_workers=min(WORKERS, max(1, len(todo)))) as ex:
        for msg in ex.map(train_arm, todo):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
