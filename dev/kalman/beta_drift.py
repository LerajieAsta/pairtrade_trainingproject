# -*- coding: utf-8 -*-
"""方案 B 診斷：形成期 beta 與交易期實現 beta 的漂移。"""
import sys, sqlite3, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from strategies.config import (DB_PATH, TABLE_NAME, BACKTEST_START, BACKTEST_END,
                               FORMATION_WINDOW, FORWARD_DAYS)
from strategies.preprocess_equity import DataProcessor
C=r'C:\Clark\YZU\Papper\Code\dev\ml_formation\cache'

proc=DataProcessor(db_path=DB_PATH, table_name=TABLE_NAME)
pp, ad, td, fi = proc.prepare_backtest_data(BACKTEST_START, BACKTEST_END, FORMATION_WINDOW)
dstr=np.array(pd.DatetimeIndex(ad).strftime('%Y-%m-%d'))
pos={s:i for i,s in enumerate(dstr)}

d=pd.read_parquet(f'{C}/pool.parquet',
   columns=['Period_Start','Ticker_A','Ticker_B','SSD','adf_pass','label_valid',
            'not_converged','capture_frac','Hedge_Ratio'])
d=d[(d.adf_pass==1)&d.label_valid].copy()
d['r']=d.groupby('Period_Start').SSD.rank(method='first')
t=d[d.r<=20]
print('樣本：SSD top-20、%d 期、%d 對'%(t.Period_Start.nunique(),len(t)))

def zb(P):                       # 管線的口徑：形成窗內 z 標準化 log 價，再 OLS
    L=np.log(np.maximum(P,1e-8)); return (L-L.mean(0))/(L.std(0)+1e-12)
rows=[]
for ps,g in t.groupby('Period_Start'):
    if ps not in pos: continue
    fs=pos[ps]; idx=fs+FORMATION_WINDOW; te=min(idx+FORWARD_DAYS, len(dstr))
    F=pp.iloc[fs:idx]; T=pp.iloc[idx:te]
    if len(T)<40: continue
    for _,r in g.iterrows():
        a,b=r.Ticker_A,r.Ticker_B
        if a not in F.columns or b not in F.columns: continue
        Pf=F[[a,b]].dropna(); Pt=T[[a,b]].dropna()
        if len(Pf)<100 or len(Pt)<40: continue
        Zf=zb(Pf.values); mu=np.log(np.maximum(Pf.values,1e-8)).mean(0); sd=np.log(np.maximum(Pf.values,1e-8)).std(0)
        Zt=(np.log(np.maximum(Pt.values,1e-8))-mu)/(sd+1e-12)   # 用形成期參數標準化（管線口徑）
        bf=np.cov(Zf[:,0],Zf[:,1])[0,1]/np.var(Zf[:,1],ddof=1)
        bt=np.cov(Zt[:,0],Zt[:,1])[0,1]/np.var(Zt[:,1],ddof=1)
        rows.append((ps,a,b,bf,bt,r.not_converged,r.capture_frac))
D=pd.DataFrame(rows,columns=['ps','A','B','b_form','b_trade','nc','cap'])
D['drift']=D.b_trade-D.b_form
D['adrift']=D.drift.abs()
print('可算 %d 對'%len(D))
print()
print('形成期 beta:  平均 %.4f  標準差 %.4f'%(D.b_form.mean(),D.b_form.std()))
print('交易期 beta:  平均 %.4f  標準差 %.4f'%(D.b_trade.mean(),D.b_trade.std()))
print('|漂移| 分位: ', '  '.join('%d%%:%.3f'%(q,np.percentile(D.adrift,q)) for q in [25,50,75,90,95]))
print('相對漂移 |drift/b_form| 中位: %.1f%%'%(100*np.median(D.adrift/D.b_form.abs().clip(1e-3))))
print()
print('=== 漂移大小 vs 結果 ===')
print('  |漂移|帶        n     未收斂率   平均capture')
q=D.adrift.quantile([0,.2,.4,.6,.8,1.0]).values
for i in range(5):
    b=D[(D.adrift>=q[i])&(D.adrift<=q[i+1])]
    print('  %.3f–%.3f  %5d    %.4f    %+.6f'%(q[i],q[i+1],len(b),b.nc.mean(),b.cap.mean()))
from scipy.stats import spearmanr
print()
print('  Spearman(|漂移|, capture) = %+.4f   Spearman(|漂移|, not_converged) = %+.4f'%(
    spearmanr(D.adrift,D.cap).statistic, spearmanr(D.adrift,D.nc).statistic))
