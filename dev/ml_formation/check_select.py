"""第二階對帳：重現的 top-20 名單是否與 formation_pairs 完全相同。"""
import sys, sqlite3, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from dev.ml_formation.pool import (load_prices, roll_indices, load_groups,
                                   window_pairs, FORMATION_WINDOW, FORM_DB, STRAT_FORM)
from dev.ml_formation.pipeline_select import normalized, select_topn

pivot, dates, total, first_idx = load_prices()
idxs = roll_indices(total, first_idx)
groups = load_groups()
with sqlite3.connect(f"file:{FORM_DB}?mode=ro", uri=True) as c:
    fp = pd.read_sql_query(
        "SELECT Period_Start, Ticker_A, Ticker_B, Pair_Rank FROM formation_pairs "
        "WHERE strategy_id = ?", c, params=(STRAT_FORM,))

rng = np.random.default_rng(0)
sample = sorted(rng.choice(len(idxs), size=12, replace=False))
tot_exact = tot_n = 0
for k in sample:
    i = idxs[k]
    ps = dates[i - FORMATION_WINDOW].strftime("%Y-%m-%d")
    gm = groups.get(ps)
    fw = pivot.iloc[i - FORMATION_WINDOW:i]
    pool = window_pairs(fw, gm)
    if pool.empty:
        continue
    usable = sorted(set(pool.Ticker_A) | set(pool.Ticker_B))
    norm = normalized(fw, usable)
    sel = select_topn(pool, norm, top_n=20)
    db = fp[fp.Period_Start == ps].sort_values("Pair_Rank")
    mine = list(zip(sel.Ticker_A, sel.Ticker_B))
    theirs = list(zip(db.Ticker_A, db.Ticker_B))
    same_set = set(mine) == set(theirs)
    same_order = mine == theirs
    inter = len(set(mine) & set(theirs))
    tot_exact += inter; tot_n += len(theirs)
    flag = "完全相同" if same_order else ("集合相同順序不同" if same_set else f"交集 {inter}/{len(theirs)}")
    print(f"{ps}  池 {len(pool):>5}  ADF通過 {int(sel.shape[0]) if len(sel)<20 else '≥20':>3}"
          f"  我的 {len(mine):>2}  儲存 {len(theirs):>2}  → {flag}")

print(f"\n合計 {tot_exact}/{tot_n} 組相符（{100*tot_exact/max(tot_n,1):.1f}%）")
