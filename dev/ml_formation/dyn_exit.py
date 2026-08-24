"""動態出場檢定：交易期的量價衝擊能否用來提早停損。

決策點的正確設定
----------------
不是「進場時這對會不會收斂」（那是形成期問題，今天已否證），而是
「**已在部位、|z| 還在擴大時，這次衝擊代表關係破裂還是雜訊**」。

評價方式：不看 AUC。今天已證明 AUC 會誤導（M1 全域 AUC +0.07 卻換到 top-20
零增益；ADF 統計量全域 AUC 更高但 top-20 顯著更差）。此處直接比較**淨捕獲**：

    基準規則   進場後持有至 z 穿越 0 或期末強平
    衝擊規則   同上，但衝擊觸發時立即平倉（在當下的 z 認賠，換取避開更深的發散）

衝擊觸發能賺錢的條件：被砍掉的部位若續抱會虧更多。若續抱其實會回來，
提早出場就是把贏面砍掉。這兩者的淨額就是規則的價值。
"""
from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dev.ml_formation.adf import adf_pass_batch
from dev.ml_formation.build import CACHE, FORWARD_DAYS
from dev.ml_formation.pool import FORMATION_WINDOW, load_groups, load_prices, roll_indices, window_pairs
from dev.ml_formation.pipeline_select import normalized, spreads_for

ENTRY_Z, COST, LOOK = 2.0, 0.0058, 60
THRESH = [np.inf, 6.0, 5.0, 4.0, 3.0, 2.0]        # inf = 不啟用規則（基準）


def shock_panels():
    c = sqlite3.connect("file:dataset/price/sp500_Tiingo.db?mode=ro", uri=True)
    d = pd.read_sql_query("select Date,Symbol,Open,COALESCE(Adj_Close,Close) C,Volume "
                          "from Daily_Prices where COALESCE(Adj_Close,Close)>0", c)
    d["Date"] = pd.to_datetime(d.Date)
    piv = lambda col: d.pivot_table(index="Date", columns="Symbol", values=col,
                                    aggfunc="last").sort_index()
    O, C_, V = piv("Open"), piv("C"), piv("Volume")
    lv = np.log(V.replace(0, np.nan))
    med = lv.rolling(LOOK, min_periods=20).median()
    mad = (lv - med).abs().rolling(LOOK, min_periods=20).median()
    ABN = ((lv - med) / (mad.replace(0, np.nan) * 1.4826)).fillna(0.0)
    GAP = (np.log(O) - np.log(C_.shift(1))).abs().fillna(0.0)
    return ABN, GAP


def main():
    t0 = time.time()
    pivot, dates, total, first_idx = load_prices()
    groups = load_groups()
    ABN, GAP = shock_panels()
    print(f"面板就緒 ({time.time()-t0:.0f}s)", flush=True)

    acc = {f"{t}": [] for t in THRESH}
    accG = {f"{t}": [] for t in THRESH}
    for k, i in enumerate(roll_indices(total, first_idx), 1):
        ps = dates[i - FORMATION_WINDOW].strftime("%Y-%m-%d")
        gm = groups.get(ps)
        if gm is None:
            continue
        fp = pivot.iloc[i - FORMATION_WINDOW:i]
        tp = pivot.iloc[i:min(i + FORWARD_DAYS, total)]
        pool = window_pairs(fp, gm)
        if pool.empty:
            continue
        us = sorted(set(pool.Ticker_A) | set(pool.Ticker_B))
        norm = normalized(fp, us)
        S = spreads_for(pool, norm)
        pa, _s, _ = adf_pass_batch(S)
        mu, sd = S.mean(1), S.std(1, ddof=1)
        lpf = np.log(fp[us].where(fp[us] > 0)); ml, sl = lpf.mean(), lpf.std()
        nt = (np.log(tp[us].where(tp[us] > 0)) - ml) / (sl + 1e-12)
        A = nt[pool.Ticker_A.tolist()].values.T
        B = nt[pool.Ticker_B.tolist()].values.T
        Z = np.nan_to_num((A - pool.Hedge_Ratio.values[:, None] * B - mu[:, None]) /
                          np.where(sd[:, None] > 1e-12, sd[:, None], np.nan), nan=0.0)
        n, T = Z.shape
        if T < 5:
            continue
        # 衝擊面板對齊本交易窗；不對稱＝單腳衝擊
        idx = ABN.index.searchsorted(tp.index[0])
        sa = ABN.reindex(columns=pool.Ticker_A.tolist()).values[idx:idx + T].T
        sb = ABN.reindex(columns=pool.Ticker_B.tolist()).values[idx:idx + T].T
        asym = np.abs(np.nan_to_num(sa) - np.nan_to_num(sb))          # (n, T)

        r = np.arange(n)
        hit = np.abs(Z) >= ENTRY_Z
        en = np.where(hit.any(1), hit.argmax(1), -1)
        has = en >= 0
        zi = Z[r, np.clip(en, 0, T - 1)]
        sg = np.sign(zi)
        after = np.arange(T)[None, :] > en[:, None]
        conv = np.where(sg[:, None] > 0, Z <= 0, Z >= 0) & after
        cday = np.where(conv.any(1), conv.argmax(1), T - 1)
        ba = pool.Hedge_Ratio.abs().values
        unit = sd * (sl[pool.Ticker_A.tolist()].values + ba * sl[pool.Ticker_B.tolist()].values) / (1 + ba)

        for th in THRESH:
            if np.isinf(th):
                xd = np.where(conv.any(1), cday, T - 1)
            else:
                trig = (asym >= th) & after
                td = np.where(trig.any(1), trig.argmax(1), T)      # 觸發日；無則 T
                cd = np.where(conv.any(1), cday, T - 1)
                xd = np.minimum(cd, np.where(td < T, td, T - 1))
            zo = Z[r, np.clip(xd, 0, T - 1)]
            cap = np.where(has, sg * (zi - zo) * unit, np.nan) - COST
            d = pd.DataFrame({"SSD": pool.SSD.values, "p": pa, "cap": cap})
            v = d[d.p]
            acc[f"{th}"].append(np.nansum(v.nsmallest(20, "SSD").cap))
            accG[f"{th}"].append(np.nansum(v.nsmallest(5, "SSD").cap))
        if k % 60 == 0:
            print(f"  {k}/295  {time.time()-t0:.0f}s", flush=True)

    print(f"\n{'不對稱門檻':<12}{'top20每期':>12}{'top20比值':>11}{'top5每期':>12}{'top5比值':>11}")
    for th in THRESH:
        a = np.array(acc[f"{th}"]); b = np.array(accG[f"{th}"])
        lab = "不啟用(基準)" if np.isinf(th) else f"asym>={th:.0f}"
        print(f"{lab:<12}{a.mean():>12.5f}{a.mean()/a.std():>11.3f}"
              f"{b.mean():>12.5f}{b.mean()/b.std():>11.3f}")
    print("\n每期＝該期選中配對的淨捕獲總和（已扣 0.58% 成本）；比值＝跨期均值/標準差")


if __name__ == "__main__":
    main()
