"""第一階對帳：重建的 SSD / beta 是否與 formation_pairs 儲存值相符。"""
import sys, sqlite3, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from dev.ml_formation.pool import (load_prices, roll_indices, load_groups,
                                   window_pairs, FORMATION_WINDOW, FORM_DB, STRAT_FORM)

pivot, dates, total, first_idx = load_prices()
idxs = roll_indices(total, first_idx)
groups = load_groups()
print(f"價格表 {pivot.shape[0]} 日 x {pivot.shape[1]} 檔 | 視窗 {len(idxs)} 個 | 分組期數 {len(groups)}")

with sqlite3.connect(f"file:{FORM_DB}?mode=ro", uri=True) as c:
    fp = pd.read_sql_query(
        "SELECT Period_Start, Ticker_A, Ticker_B, Pair_Rank, Formation_Params "
        "FROM formation_pairs WHERE strategy_id = ?", c, params=(STRAT_FORM,))
prm = fp.Formation_Params.apply(json.loads).apply(pd.Series)
fp = pd.concat([fp.drop(columns=["Formation_Params"]), prm], axis=1)

rng = np.random.default_rng(0)
sample = sorted(rng.choice(len(idxs), size=12, replace=False))
rows = []
for k in sample:
    i = idxs[k]
    ps = dates[i - FORMATION_WINDOW].strftime("%Y-%m-%d")
    gm = groups.get(ps)
    if gm is None:
        print(f"  {ps}: formation_groups 無此期，跳過"); continue
    pool = window_pairs(pivot.iloc[i - FORMATION_WINDOW:i], gm)
    stored = fp[fp.Period_Start == ps]
    if pool.empty or stored.empty:
        print(f"  {ps}: 池或儲存為空"); continue
    m = stored.merge(pool, on=["Ticker_A", "Ticker_B"], suffixes=("_db", "_re"))
    rows.append({
        "period": ps, "池大小": len(pool), "儲存": len(stored), "對上": len(m),
        "SSD最大絕對差": (m.SSD_db - m.SSD_re).abs().max() if len(m) else np.nan,
        "beta最大絕對差": (m.Hedge_Ratio_db - m.Hedge_Ratio_re).abs().max() if len(m) else np.nan,
        "sstd最大絕對差": (m.Spread_Std_db - m.Spread_Std_re).abs().max() if len(m) else np.nan,
    })
R = pd.DataFrame(rows)
print(R.to_string(index=False))
print()
print(f"全部 {R['對上'].sum()}/{R['儲存'].sum()} 組對上")
print(f"SSD  最大絕對差 {R['SSD最大絕對差'].max():.6f}   （儲存為 round(.,6)）")
print(f"beta 最大絕對差 {R['beta最大絕對差'].max():.6f}   （儲存為 round(.,4) → 容差 5e-5）")
print(f"sstd 最大絕對差 {R['sstd最大絕對差'].max():.6f}   （由 round 後的 beta 推得，容差較寬）")
