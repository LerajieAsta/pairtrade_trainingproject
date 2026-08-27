# -*- coding: utf-8 -*-
"""探針：最佳動作是否與可觀測特徵相關（決定是否投入完整前推）。"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
from dev.ml_formation.selection import selection_region, with_model_scores
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import spearmanr
C=r'C:\Clark\YZU\Papper\Code\dev\ml_formation\cache'
K=['Period_Start','Ticker_A','Ticker_B']
es=pd.read_parquet(f'{C}/exit_scan.parquet')
tr=selection_region(with_features=True)
caps=[c for c in es.columns if c.startswith('cap_e')]
es=es[K+caps]
d=tr.merge(es,on=K,how='inner')
d=d[d.ssd_rank<=5]
M=np.where(np.isnan(d[caps].to_numpy(dtype=float)),0.0,d[caps].to_numpy(dtype=float))
best=np.argmax(M,axis=1)
d['best_ez']=[float(caps[i].split('_')[1][1:]) for i in best]
d['best_xz']=[float(caps[i].split('_')[2][1:]) for i in best]
print('樣本 %d 列、%d 期（Top5）'%(len(d),d.Period_Start.nunique()))
print('最佳 entry_z 分布:', d.best_ez.value_counts(normalize=True).sort_index().round(3).to_dict())
print()
feats=['rho','SSD','Spread_Std','adf_stat','rho_excess','rho_seg_std','rho_seg_drift',
       'rho_seg_range','sd_seg_ratio','sd_seg_std','z_form_last','z_form_ncross',
       'mu_seg_drift','mu_seg_absmax','deg_A','common_nb','group_size']
feats=[f for f in feats if f in d.columns]
print('=== Spearman(特徵, 最佳 entry_z) ——逐期算再平均 ===')
rows=[]
for f in feats:
    rs=[]
    for _,g in d.groupby('Period_Start'):
        if g[f].nunique()>2 and g.best_ez.nunique()>1:
            r=spearmanr(g[f],g.best_ez).statistic
            if np.isfinite(r): rs.append(r)
    if len(rs)>50:
        m=np.mean(rs); se=np.std(rs,ddof=1)/np.sqrt(len(rs))
        rows.append((f,m,se,m/se))
rows.sort(key=lambda x:-abs(x[3]))
for f,m,se,t in rows[:10]:
    print('  %-16s rho=%+.4f ± %.4f   t=%+6.2f  %s'%(f,m,se,t,'***' if abs(t)>3 else ''))
print()
print('=== 對照：最佳動作的 capture 對「最佳常數」的每期優勢 ===')
cm=M.mean(axis=0); bc=int(np.argmax(cm))
g=pd.DataFrame({'ps':d.Period_Start.values,'const':M[:,bc],'oracle':M.max(axis=1)}).groupby('ps').mean()
print('  常數 %s %+.6f   完美 %+.6f   比 %.1f 倍'%(caps[bc],g['const'].mean(),g.oracle.mean(),
      g.oracle.mean()/g['const'].mean()))
