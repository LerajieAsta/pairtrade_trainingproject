"""消融梯子 M0 → M2 的淨化前推（XGBoost 階）。M3 的 CNN 另跑。

前推設定（PREREGISTRATION 決策 9）
    暖身 36 期／擴張窗／每 12 期重訓／6 期隔離帶

隔離帶的來源：形成期 p 的標籤要到 p 的交易期結束才知道，而交易期 126 日、
滾動步長 21 日，故落後 6 期。於重訓點 T 只能用 <= T-6 期的標籤，該模型用於
預測 T .. T+11 期。

超參數固定不調——梯子是預先註冊的四階，不是搜尋。任何調參都會把診斷階段
變成一次隱性的試驗宇宙擴張。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
BURN_IN = 36
EMBARGO = 6
RETRAIN_EVERY = 12
TOP_N = 20

F_M0 = ["SSD"]
F_M1 = ["rho_seg_std", "rho_seg_min", "rho_seg_drift", "rho_seg_range",
        "mu_seg_std", "mu_seg_drift", "mu_seg_absmax",
        "sd_seg_ratio", "sd_seg_std", "z_form_last", "z_form_ncross"]
F_M2 = ["rho_excess", "mean_rho_A", "mean_rho_B", "rank_in_A", "rank_in_B",
        "gap_to_best_A", "gap_to_best_B", "deg_A", "deg_B", "common_nb", "group_size"]

XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                  tree_method="hist", eval_metric="logloss",
                  reg_lambda=1.0, n_jobs=-1, random_state=0)


def load():
    D = pd.read_parquet(os.path.join(CACHE, "train.parquet"))
    D = D[D.label_valid].copy()
    per = np.sort(D.Period_Start.unique())
    D["pi"] = D.Period_Start.map({p: i for i, p in enumerate(per)})
    return D, len(per)


def evaluate(D: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """逐期回報全池 AUC 與 ADF 通過者中 top-20 的未收斂率。分數越大越可能不收斂。"""
    rows = []
    for pi, g in D.groupby("pi"):
        if g.not_converged.nunique() < 2:
            continue
        auc = roc_auc_score(g.not_converged.values, g[score_col].values)
        sub = g[g.adf_pass]
        top = sub.nsmallest(TOP_N, score_col) if len(sub) >= TOP_N else sub
        rows.append({"pi": pi, "n": len(g), "auc": auc,
                     "top_nc": top.not_converged.mean() if len(top) else np.nan,
                     "n_adf": len(sub)})
    return pd.DataFrame(rows)


def walk_forward(D: pd.DataFrame, n_per: int, feats: list[str], tag: str) -> pd.Series:
    from xgboost import XGBClassifier
    pred = pd.Series(np.nan, index=D.index)
    t0 = time.time()
    points = list(range(BURN_IN, n_per, RETRAIN_EVERY))
    for k, T in enumerate(points, 1):
        tr = D[D.pi <= T - EMBARGO]
        te = D[(D.pi >= T) & (D.pi < T + RETRAIN_EVERY)]
        if len(tr) < 5000 or te.empty or tr.not_converged.nunique() < 2:
            continue
        m = XGBClassifier(**XGB_PARAMS)
        m.fit(tr[feats].values, tr.not_converged.values)
        pred.loc[te.index] = m.predict_proba(te[feats].values)[:, 1]
        if k % 5 == 0 or k == len(points):
            print(f"    [{tag}] 重訓 {k}/{len(points)}  T={T}  "
                  f"訓練 {len(tr):,}  {time.time()-t0:.0f}s", flush=True)
    return pred


def main():
    D, n_per = load()
    print(f"可標記 {len(D):,} 列，{n_per} 期；暖身 {BURN_IN} 期後評估\n")

    rungs = {"M0 (僅SSD)": None,
             "M1 (+分段穩定度)": F_M0 + F_M1,
             "M2 (+圖位置)": F_M0 + F_M1 + F_M2}
    results = {}
    for tag, feats in rungs.items():
        if feats is None:
            D["score_M0"] = D.SSD
            col = "score_M0"
        else:
            col = f"score_{tag[:2]}"
            D[col] = walk_forward(D, n_per, feats, tag[:2])
        sub = D[D.pi >= BURN_IN].dropna(subset=[col])
        R = evaluate(sub, col)
        results[tag] = R
        se = R.auc.std() / np.sqrt(len(R))
        print(f"  {tag:<20} 逐期AUC {R.auc.mean():.4f} ± {se:.4f}   "
              f"top20未收斂 {R.top_nc.mean():.4f}   ({len(R)} 期)\n")

    out = os.path.join(CACHE, "ladder_results.parquet")
    pd.concat([r.assign(rung=k) for k, r in results.items()]).to_parquet(out)
    D[["Period_Start", "Ticker_A", "Ticker_B", "pi"] +
      [c for c in D.columns if c.startswith("score_")]].to_parquet(
        os.path.join(CACHE, "scores.parquet"))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
