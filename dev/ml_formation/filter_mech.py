# -*- coding: utf-8 -*-
"""機制檢驗：模型是否偏好「發散幅度小」的配對？"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
from dev.ml_formation.selection import selection_region, with_model_scores
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import spearmanr
C=r'C:\Clark\YZU\Papper\Code\dev\ml_formation\cache'
K=['Period_Start','Ticker_A','Ticker_B']
d=with_model_scores(selection_region())
t=d[d.ssd_rank<=40].copy()
t['abs_z_entry']=t.z_entry.abs()
print('SSD 前 40 名、%d 期、%d 列'%(t.Period_Start.nunique(),len(t)))
print()
print('分數越低＝模型認為越會收斂（否決算子取分數最低的 20 個）')
print()
print('逐期 Spearman(模型分數, X) 的 259 期平均：')
print('  %-22s %-10s %-10s'%('X','M1','M3'))
for col,lab in [('abs_z_entry','|z_進場|（發散幅度）'),
                ('capture_frac','capture_frac'),
                ('Spread_Std','價差標準差'),
                ('not_converged','not_converged')]:
    row=[]
    for m in ['score_M1','score_M3']:
        r=[spearmanr(g[m],g[col]).statistic for _,g in t.groupby('Period_Start')
           if g[col].nunique()>2]
        r=np.array(r); row.append('%+.4f'%np.nanmean(r))
    print('  %-22s %-10s %-10s'%(lab,row[0],row[1]))
print()
print('=== 直接對照：模型選中 vs 未選中（N=40 取 20）===')
for m in ['score_M1','score_M3']:
    sel=t.sort_values(['Period_Start',m]).groupby('Period_Start').head(20)
    key=set(map(tuple,sel[K].values))
    t['_in']=[tuple(x) in key for x in t[K].values]
    a=t[t._in]; b=t[~t._in]
    print('  %s'%m)
    print('     %-14s 選中 %+.5f   未選中 %+.5f'%('|z_進場|',a.abs_z_entry.mean(),b.abs_z_entry.mean()))
    print('     %-14s 選中 %+.5f   未選中 %+.5f'%('capture_frac',a.capture_frac.mean(),b.capture_frac.mean()))
    print('     %-14s 選中 %+.4f   未選中 %+.4f'%('未收斂率',a.not_converged.mean(),b.not_converged.mean()))
