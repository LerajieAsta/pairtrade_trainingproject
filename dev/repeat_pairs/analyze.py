# -*- coding: utf-8 -*-
"""重複入選配對的績效：同一配對在同時進行的前 5 期內也被選過，是否表現較好。

資料：tmp/dl_weights/pred_*.pkl（15 支主軸臂 Top20 的每個 (配對, 期) 報酬 r 與波動 v，
由 dev/dl_weights/features.py 從 result.db 建出）。結論見同目錄 FINDING.md。

    python dev/repeat_pairs/analyze.py
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
import pandas as pd, glob, numpy as np
import statsmodels.api as sm
rows=[]
for f in sorted(glob.glob('tmp/dl_weights/pred_*.pkl')):
    d=pd.read_pickle(f)[['Trade_Start','Ticker_A','Ticker_B','Pair_Rank','r','v']].copy()
    d['arm']=f.split('pred_')[1][:-4]
    # 不分順序：DTW／SDP 臂內同一配對會以 A|B 與 B|A 兩向出現，經濟上是同一條價差
    d['pair']=[a+'|'+b if a<b else b+'|'+a for a,b in zip(d.Ticker_A,d.Ticker_B)]
    per=sorted(d.Trade_Start.unique()); d['i']=d.Trade_Start.map({x:k for k,x in enumerate(per)})
    s=set(zip(d.pair,d.i))
    d['nrep']=[sum((p,i-j) in s for j in range(1,6)) for p,i in zip(d.pair,d.i)]
    # 連續被選幾期（含本期）
    streak=[]
    for p,i in zip(d.pair,d.i):
        k=1
        while (p,i-k) in s: k+=1
        streak.append(k)
    d['streak']=streak
    d=d[d.i>=5]  # 前 5 期無法判斷
    rows.append(d)
d=pd.concat(rows); d=d[d.r.notna()]
d['rep']=d.nrep>0
d['h2']=pd.to_datetime(d.Trade_Start)>='2014-01-02'
d['rk']=pd.cut(d.Pair_Rank,[0,3,5,10,20],labels=['1-3','4-5','6-10','11-20'])
pct=lambda s:f"{s.mean():+.3%}"
def nw(x,lag=6):
    x=x.dropna(); m=sm.OLS(x.values,np.ones(len(x))).fit(cov_type='HAC',cov_kwds={'maxlags':lag}); return m.params[0],m.tvalues[0],len(x)
def diff_test(x):
    # 逐 (臂,期) 在排名區間內差分，再對每期跨臂平均
    g=x.groupby(['arm','Trade_Start','rk','rep'],observed=True).r.mean().unstack('rep').dropna()
    g=(g[True]-g[False]).groupby(['arm','Trade_Start']).mean()
    per=g.groupby('Trade_Start').mean().sort_index()
    arm=g.groupby('arm').mean()
    return nw(per), (arm>0).sum(), len(arm)
print("== 1. 重複 vs 新配對（未控制排名）")
for nm,x in (('全期',d),('2014 前',d[~d.h2]),('2014 後',d[d.h2])):
    print(f"{nm}: 重複占比 {x.rep.mean():.1%} | 新配對 r {pct(x[~x.rep].r)} 勝率 {(x[~x.rep].r>0).mean():.1%} | 重複 r {pct(x[x.rep].r)} 勝率 {(x[x.rep].r>0).mean():.1%}")
print("\n== 2. 重複配對的排名分布（平均 Pair_Rank）", d.groupby('rep').Pair_Rank.mean().round(2).to_dict())
print("\n== 3. 同排名區間內：平均 r（新 / 重複）")
t=d.groupby(['rk','rep'],observed=True).r.agg(['mean','size']).unstack('rep')
print((t['mean']*100).round(3).assign(n_new=t['size'][False],n_rep=t['size'][True]).to_string())
print("\n== 4. 控制排名後的差分（重複 − 新），逐期平均、Newey-West t")
for nm,x in (('全期',d),('2014 前',d[~d.h2]),('2014 後',d[d.h2])):
    (m,tv,n),pos,na=diff_test(x); print(f"{nm}: 差 {m:+.3%}  t={tv:+.2f}  期數 {n}  臂為正 {pos}/{na}")
print("\n== 5. 依前 5 期內被選過幾次")
print(d.groupby('nrep').r.agg(n='size',mean=lambda s:f"{s.mean():+.3%}",win=lambda s:f"{(s>0).mean():.1%}",rank=lambda s:0).drop(columns='rank').join(d.groupby('nrep').Pair_Rank.mean().round(1)).to_string())
print("\n== 6. 依連續被選期數（含本期）")
d['st']=d.streak.clip(upper=8)
print(d.groupby('st').r.agg(n='size',mean=lambda s:f"{s.mean():+.3%}",win=lambda s:f"{(s>0).mean():.1%}").join(d.groupby('st').Pair_Rank.mean().round(1)).to_string())
print("\n== 7. 逐臂（全期、控制排名後差分）")
g=d.groupby(['arm','Trade_Start','rk','rep'],observed=True).r.mean().unstack('rep').dropna()
g=(g[True]-g[False]).groupby(['arm','Trade_Start']).mean()
print((g.groupby('arm').mean()*100).round(3).to_string())
print("\n== 8. 波動（v）新 / 重複", (d.groupby('rep').v.mean()*100).round(3).to_dict())
