# -*- coding: utf-8 -*-
"""門檻二：Kalman vs 靜態 beta，逐日抽樣 + block bootstrap。依 dev/kalman/PREREGISTRATION.md。"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from strategies.returns import daily_returns
from analysis.block_bootstrap import bootstrap_test
import statsmodels.api as sm
RDB=r'C:\Clark\YZU\Papper\Code\results\result.db'
_T='tiingo/Grid_%s/TradeLogs_%s_ZWin0_MSR0.csv'
TOPS=['Top1','Top3','Top5','Top10','Top20']; SLS=['SL0','SL5','SL15']
ARMS={'B1 KAL（僅換 beta）':'NOGRP_DTW_KAL','B2 KALINN（創新版）':'NOGRP_DTW_KALINN'}
cells=[f'{t}_{s}' for t in TOPS for s in SLS]
ids={}
for c in cells:
    ids[('base',c)]=_T%('NOGRP_DTW',c)
    for lab,sd in ARMS.items(): ids[(lab,c)]=_T%(sd,c)
R=daily_returns(sorted(set(ids.values())), result_db=RDB)
print('逐日報酬矩陣', R.shape)

def nw(d):
    d=np.asarray(d,float); d=d[~np.isnan(d)]
    m=sm.OLS(d,np.ones(len(d))).fit(cov_type='HAC',cov_kwds={'maxlags':21})
    return float(m.pvalues[0])

print()
print('=== 門檻二：年化報酬差（Kalman − 靜態），四檔停損等權聚合、逐日抽樣 ===')
print('    block bootstrap L=126 B=10000；Bonferroni 門檻 0.025')
print()
for lab in ARMS:
    # 每個 top_n 內對三檔停損等權，再對 top_n 等權（比照論文交易層主檢定）
    d_all=[]
    for t in TOPS:
        bb=pd.concat([R[ids[('base',f'{t}_{s}')]] for s in SLS],axis=1).mean(axis=1)
        kk=pd.concat([R[ids[(lab,f'{t}_{s}')]] for s in SLS],axis=1).mean(axis=1)
        d_all.append(kk-bb)
    d=pd.concat(d_all,axis=1).mean(axis=1).dropna()
    r=bootstrap_test(d.values, capital=1.0)
    p_nw=nw(d.values)
    ok='✔' if (r['BB p']<0.025 and r['年化Δ%']>0) else '✘'
    print('  %-22s  年化Δ %+.3f pp   BB p %.4f   NW p %.4f   CI [%+.3f, %+.3f]   %s'%(
        lab,r['年化Δ%'],r['BB p'],p_nw,r['CI下界'],r['CI上界'],ok))

print()
print('=== 逐格 Sharpe 對照（參考，非判準）===')
def sh(s):
    s=s.dropna(); return s.mean()/s.std(ddof=1)*np.sqrt(252) if len(s)>20 and s.std()>0 else np.nan
rows=[]
for c in cells:
    row={'格':c,'靜態':round(sh(R[ids[('base',c)]]),3)}
    for lab,_ in ARMS.items(): row[lab.split('（')[0]]=round(sh(R[ids[(lab,c)]]),3)
    rows.append(row)
t=pd.DataFrame(rows)
t['B1−靜態']=(t['B1 KAL']-t['靜態']).round(3); t['B2−靜態']=(t['B2 KALINN']-t['靜態']).round(3)
print(t.to_string(index=False))
print()
print('B1 勝出 %d/15 格，中位差 %+.4f'%((t['B1−靜態']>0).sum(),t['B1−靜態'].median()))
print('B2 勝出 %d/15 格，中位差 %+.4f'%((t['B2−靜態']>0).sum(),t['B2−靜態'].median()))
