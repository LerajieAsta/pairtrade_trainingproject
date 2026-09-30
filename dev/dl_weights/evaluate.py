# -*- coding: utf-8 -*-
"""DL-RI／DL-MO 的評估：60 格精確線性重組、隨機置換對照、預測技巧。判準見 PREREGISTRATION.md §四。

    python dev/dl_weights/evaluate.py
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import sparse, stats

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dev.capital_alloc import stage1 as S1  # noqa: E402
from dev.dl_weights import features as F  # noqa: E402
from dev.dl_weights.train import pred_file  # noqa: E402
from strategies.config import INITIAL_CAPITAL  # noqa: E402
from strategies.metrics import concurrent_of  # noqa: E402

TOPNS = (3, 5, 10, 20)
N_PERM = 200
HALF = pd.Timestamp("2014-01-02")
OUT = "results/analysis/dl_weights_eval.csv"
EPS = 1e-12


def daily_returns(pnl: pd.Series) -> pd.Series:
    """與 metrics_from_pnl 同口徑的日報酬（分母 = 前一日權益，全期路徑）。"""
    eq = INITIAL_CAPITAL + pnl.cumsum()
    return pnl / eq.shift(1).fillna(INITIAL_CAPITAL)


def sharpe(ret: pd.Series) -> float:
    sd = ret.std()
    return float(np.sqrt(252) * ret.mean() / sd) if sd > 0 else 0.0


def dl_scales(pp: pd.DataFrame, pred: pd.DataFrame, mode: str, first_pi_form: int):
    """DL 權重：候選集內正規化為平均 1；缺預測者補候選均值；MO 全 ≤ 0 時退回等權。"""
    # pp.Period_Start 來自 trade_logs，其值是交易期起日 → 以 Trade_Start 對接
    key = pred.Ticker_A + "|" + pred.Ticker_B + "#" + pred.Trade_Start
    col = "sigma_hat" if mode == "RI" else "mu_hat"
    val = pd.Series(pred[col].to_numpy(), index=key)
    active = pd.Series(pred.pi.to_numpy() >= first_pi_form, index=key)
    k = pp.pair + "#" + pp.Period_Start
    x = val.reindex(k).to_numpy()
    on = active.reindex(k).fillna(False).to_numpy()
    s = np.ones(len(pp))
    for _, g in pp.groupby("pi", sort=False):
        ids = g.gid.to_numpy()
        if not on[ids].any():
            continue
        v = x[ids]
        raw = 1.0 / np.maximum(v, EPS) if mode == "RI" else np.maximum(v, 0.0)
        raw = np.where(np.isfinite(raw), raw, np.nan)
        fill = np.nanmean(raw) if np.isfinite(np.nanmean(raw)) else 1.0
        raw = np.where(np.isnan(raw), fill, raw)
        m = raw.mean()
        s[ids] = raw / m if m > EPS else 1.0
    return s


def permute_within(pp, scales, rng):
    s = scales.copy()
    for _, g in pp.groupby("pi", sort=False):
        ids = g.gid.to_numpy()
        s[ids] = rng.permutation(s[ids])
    return s


def skill(pred: pd.DataFrame, k0: int) -> dict:
    """每期橫斷面 Spearman 的平均與 t 值（Top 20 全部候選、有標的者）。"""
    d = pred[(pred.pi >= k0) & pred.r.notna() & pred.mu_hat.notna()]
    out = {}
    for name, a, b in (("IC_mu", "mu_hat", "r"), ("IC_sigma", "sigma_hat", "v")):
        ics = [stats.spearmanr(g[a], g[b])[0] for _, g in d.groupby("pi") if len(g) >= 5]
        ics = np.array([x for x in ics if np.isfinite(x)])
        out[name] = float(ics.mean()) if len(ics) else np.nan
        out[name + "_t"] = float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))) if len(ics) > 2 else np.nan
    return out


def main():
    rows = []
    rng = np.random.default_rng(20260929)
    for arm in F.ARMS:
        if not os.path.exists(pred_file(arm)):
            print(f"  ! {arm} 尚無預測，跳過")
            continue
        with open(pred_file(arm), "rb") as fh:
            pred = pickle.load(fh)
        k0 = pred.attrs.get("k0", 54)
        sk = skill(pred, k0)
        first_ps = sorted(pred.loc[pred.pi >= k0, "Trade_Start"].unique())[0]

        for tn in TOPNS:
            path = F.cell_path(arm, tn)
            df = S1.load_logs(path)
            if df.empty:
                continue
            lag = max(1, concurrent_of(path))
            periods, pos, pp, M, V = S1._trailing(df, lag)
            gid = S1.row_gid(df, pp)
            start = df.loc[df.Period_Start == first_ps, "Date"].min()
            # 日期 × (配對, 期) 的稀疏損益矩陣：重組 = A @ s，與 S1.apply_scales 逐位相同但快兩個數量級
            codes, uniq = pd.factorize(df.Date, sort=True)
            A = sparse.csr_matrix((df.Daily_Delta.to_numpy(), (codes, gid)),
                                  shape=(len(uniq), len(pp)))
            didx = pd.DatetimeIndex(uniq)

            def evaluate(sc):
                ret = daily_returns(pd.Series(A @ sc, index=didx))
                w = ret[ret.index >= start]
                return sharpe(w), sharpe(w[w.index >= HALF])

            rec = {"arm": arm, "top_n": tn, "eval_start": str(start.date()), **sk}
            base = np.ones(len(pp))
            rec["Sharpe_EW"], rec["H2_EW"] = evaluate(base)
            for mode in ("RI", "MO"):
                sc_h = S1.build_scales(df, pos, pp, M, V, mode, lag)
                rec[f"Sharpe_{mode}_H"], rec[f"H2_{mode}_H"] = evaluate(sc_h)
                sc = dl_scales(pp, pred, mode, k0)
                rec[f"Sharpe_{mode}_DL"], rec[f"H2_{mode}_DL"] = evaluate(sc)
                rec[f"MaxS_{mode}_DL"] = float(sc.max())
                rec[f"NotRatio_{mode}_DL"] = S1.notional_ratio(df, gid, sc)
                perm = np.array([evaluate(permute_within(pp, sc, rng))[0] for _ in range(N_PERM)])
                rec[f"Perm95_{mode}_DL"] = float(np.percentile(perm, 95))
                rec[f"PermPct_{mode}_DL"] = float((perm < rec[f"Sharpe_{mode}_DL"]).mean())
            rows.append(rec)
            print(f"  {arm:<16} Top{tn:<2} EW={rec['Sharpe_EW']:+.3f} "
                  f"RI_H={rec['Sharpe_RI_H']:+.3f} RI_DL={rec['Sharpe_RI_DL']:+.3f} "
                  f"MO_H={rec['Sharpe_MO_H']:+.3f} MO_DL={rec['Sharpe_MO_DL']:+.3f} "
                  f"(perm95 RI {rec['Perm95_RI_DL']:+.3f} MO {rec['Perm95_MO_DL']:+.3f})", flush=True)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n-> {OUT}（{len(out)} 格）\n")

    print("== 判準（PREREGISTRATION.md §四）==")
    for mode in ("RI", "MO"):
        d = out[f"Sharpe_{mode}_DL"] - out.Sharpe_EW
        d2 = out[f"H2_{mode}_DL"] - out.H2_EW
        beat = (out[f"Sharpe_{mode}_DL"] > out[f"Perm95_{mode}_DL"]).sum()
        dh = out[f"Sharpe_{mode}_H"] - out.Sharpe_EW
        c = [d.median() >= 0.05, (d > 0).sum() >= 40, beat >= 40, d2.median() > 0]
        print(f"DL-{mode}: 中位 ΔSharpe {d.median():+.4f} [{'過' if c[0] else '否'}]｜"
              f"為正 {(d > 0).sum()}/{len(d)} [{'過' if c[1] else '否'}]｜"
              f"勝置換 p95 {beat}/{len(d)} [{'過' if c[2] else '否'}]｜"
              f"後半期中位 {d2.median():+.4f} [{'過' if c[3] else '否'}]"
              f" → {'有效' if all(c) else '未過閘'}")
        print(f"   對照 歷史統計版 {mode}_H：中位 ΔSharpe {dh.median():+.4f}，為正 {(dh > 0).sum()}/{len(dh)}")
    sk = out.drop_duplicates("arm")
    print(f"預測技巧（15 臂平均）：IC_mu {sk.IC_mu.mean():+.4f}（t 中位 {sk.IC_mu_t.median():.2f}）｜"
          f"IC_sigma {sk.IC_sigma.mean():+.4f}（t 中位 {sk.IC_sigma_t.median():.2f}）")


if __name__ == "__main__":
    main()
