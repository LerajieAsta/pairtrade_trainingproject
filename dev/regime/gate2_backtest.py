# -*- coding: utf-8 -*-
"""門檻二：regime 閘實跑回測 vs 基準 + 循環 block 置換。依 dev/regime/PREREGISTRATION.md。"""
import sys, sqlite3, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from strategies.returns import daily_returns
from analysis.block_bootstrap import bootstrap_test
import statsmodels.api as sm
TD=252; RDB=r'C:\Clark\YZU\Papper\Code\results\result.db'
_T='tiingo/Grid_%s/TradeLogs_%s_ZWin0_MSR0.csv'
TOPS=['Top1','Top3','Top5','Top10','Top20']; SLS=['SL0','SL5','SL15']
ARMS={'基準':'NOGRP_DTW','VG50':'NOGRP_DTW_VG50','VG67':'NOGRP_DTW_VG67'}
ids=[_T%(sd,f'{t}_{s}') for sd in ARMS.values() for t in TOPS for s in SLS]
R=daily_returns(ids, result_db=RDB)

def agg(sd):
    return pd.concat([pd.concat([R[_T%(sd,f'{t}_{s}')] for s in SLS],axis=1).mean(axis=1)
                      for t in TOPS],axis=1).mean(axis=1)
A={k:agg(v) for k,v in ARMS.items()}
def nw(d):
    d=np.asarray(d,float); d=d[~np.isnan(d)]
    return float(sm.OLS(d,np.ones(len(d))).fit(cov_type='HAC',cov_kwds={'maxlags':21}).pvalues[0])
def ann(x):
    x=np.asarray(x,float); return ((1+x).prod())**(TD/len(x))-1

print('=== 門檻二第 1 項：對基準的年化報酬差 ===')
for k in ['VG50','VG67']:
    d=(A[k]-A['基準']).dropna()
    r=bootstrap_test(d.values,capital=1.0)
    print('  %s  年化Δ %+.3f pp   BB p %.4f   NW p %.4f   CI [%+.3f, %+.3f]   %s'%(
        k,r['年化Δ%'],r['BB p'],nw(d.values),r['CI下界'],r['CI上界'],
        '✔' if (r['BB p']<0.025 and r['年化Δ%']>0) else '✘'))

# ── 第 2 項：對「同跳過率的循環 block 置換」──────────────────────────
con=sqlite3.connect(r'C:\Clark\YZU\Papper\Code\dataset\price\sp500_Tiingo.db')
px=pd.read_sql("SELECT Date,Symbol,COALESCE(Adj_Close,Close) p FROM Daily_Prices",con); con.close()
px['Date']=pd.to_datetime(px.Date)
w=px.pivot_table(index='Date',columns='Symbol',values='p').replace(0.0,np.nan)
mkt=np.log(w).diff().mean(axis=1)
vol=(mkt.rolling(63,min_periods=40).std()*np.sqrt(TD)).shift(1)
pct=vol.expanding(min_periods=504).rank()/vol.expanding(min_periods=504).count()
def circ(g,L,B,rng):
    n=len(g); nb=int(np.ceil(n/L)); out=np.empty((B,n),bool)
    for b in range(B):
        st=rng.integers(0,n,nb)
        out[b]=g[np.concatenate([(np.arange(s,s+L)%n) for s in st])[:n]]
    return out
print()
print('=== 門檻二第 2 項：對同跳過率的循環 block 置換（L=63、1,000 次）===')
print('    置換以「基準逐日報酬 × 閘」近似（無成本上界，與門檻一同口徑）')
rng=np.random.default_rng(20260826)
base=A['基準'].dropna()
for k,p_ in [('VG50',50),('VG67',67)]:
    gate=(pct>=p_/100.0).where(pct.notna(),True).reindex(base.index).fillna(True).values.astype(bool)
    obs_real=ann(A[k].dropna().reindex(base.index).fillna(0).values)
    P=circ(gate,63,1000,rng)
    nul=np.array([ann(np.where(P[b],base.values,0.0)) for b in range(1000)])
    p=float((nul>=obs_real).mean())
    print('  %s  實跑年化 %+.3f%%   置換中位 %+.3f%%   p %.4f   %s'%(
        k,obs_real*100,np.median(nul)*100,p,'✔' if p<0.025 else '✘'))

print()
print('=== 逐格 Sharpe（參考）===')
def sh(s):
    s=s.dropna(); return s.mean()/s.std(ddof=1)*np.sqrt(TD) if len(s)>20 and s.std()>0 else np.nan
rows=[]
for t in TOPS:
    for s_ in SLS:
        c=f'{t}_{s_}'
        rows.append({'格':c,'基準':round(sh(R[_T%('NOGRP_DTW',c)]),3),
                     'VG50':round(sh(R[_T%('NOGRP_DTW_VG50',c)]),3),
                     'VG67':round(sh(R[_T%('NOGRP_DTW_VG67',c)]),3)})
t=pd.DataFrame(rows); t['VG50−基準']=(t.VG50-t.基準).round(3); t['VG67−基準']=(t.VG67-t.基準).round(3)
print(t.to_string(index=False))
print()
print('VG50 勝 %d/15 中位 %+.4f   VG67 勝 %d/15 中位 %+.4f'%(
    (t['VG50−基準']>0).sum(),t['VG50−基準'].median(),
    (t['VG67−基準']>0).sum(),t['VG67−基準'].median()))
