# -*- coding: utf-8 -*-
"""門檻一：三腿 vs 兩腿的樣本外經濟捕獲。依 dev/multileg/PREREGISTRATION.md。"""
import sys, sqlite3, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import ttest_rel
from strategies.config import (DB_PATH, TABLE_NAME, BACKTEST_START, BACKTEST_END,
                               FORMATION_WINDOW, FORWARD_DAYS)
from strategies.preprocess_equity import DataProcessor
ENTRY_Z=2.0

proc=DataProcessor(db_path=DB_PATH, table_name=TABLE_NAME)
pp, ad, td, fi = proc.prepare_backtest_data(BACKTEST_START, BACKTEST_END, FORMATION_WINDOW)
dstr=np.array(pd.DatetimeIndex(ad).strftime('%Y-%m-%d')); pos={s:i for i,s in enumerate(dstr)}
mem=pd.read_sql("select Symbol,start_date,end_date from index_memberships", sqlite3.connect(DB_PATH))
fc=sqlite3.connect(r'C:\Clark\YZU\Papper\Code\formation_data\formation_pairs_sp500_Tiingo.db')
sel=pd.read_sql('select Period_Start,Ticker_A,Ticker_B from formation_pairs '
                'where strategy_id=? order by Period_Start,Pair_Rank',fc,
                params=('Grid NOGRP-DTW_MSR0',))
print('基準配對 %d 組、%d 期'%(len(sel),sel.Period_Start.nunique()))

def capture(z, sd_s, unit):
    """同一進場/出場規則下的近似經濟捕獲；無進場回傳 nan。"""
    hit=np.flatnonzero(np.abs(z)>ENTRY_Z)
    if len(hit)==0: return np.nan, np.nan
    e=hit[0]; s=np.sign(z[e])
    after=z[e+1:]
    cross=np.flatnonzero(np.sign(after)!=s) if len(after) else np.array([],int)
    x=(e+1+cross[0]) if len(cross) else len(z)-1
    return (abs(z[e])-abs(z[x]))*unit, (1.0 if len(cross) else 0.0)

rows=[]
for ps,g in sel.groupby('Period_Start'):
    if ps not in pos: continue
    fs=pos[ps]; idx=fs+FORMATION_WINDOW; te=min(idx+FORWARD_DAYS,len(dstr))
    fe=dstr[idx-1]
    F=pp.iloc[fs:idx]; T=pp.iloc[idx:te]
    if len(T)<40: continue
    act=set(mem[(mem.start_date<=fe)&((mem.end_date.isna())|(mem.end_date>=fe))].Symbol.unique())
    cols=[c for c in F.columns if c in act]
    Fv=F[cols].dropna(axis=1); Tv=T[Fv.columns]
    if Fv.shape[1]<50: continue
    L=np.log(np.maximum(Fv.values,1e-8)); mu=L.mean(0); sd=L.std(0)+1e-12
    Z=(L-mu)/sd                                   # 形成期標準化
    Zt=(np.log(np.maximum(Tv.values,1e-8))-mu)/sd  # 交易期，用形成期參數
    ci={c:i for i,c in enumerate(Fv.columns)}
    Zc=Z-Z.mean(0)                                 # 置中，供最小平方
    for _,r in g.iterrows():
        a,b=r.Ticker_A,r.Ticker_B
        if a not in ci or b not in ci: continue
        ia,ib=ci[a],ci[b]
        ya=Zc[:,ia]; xb=Zc[:,ib]
        # 兩腿
        b2=float(xb@ya/(xb@xb)) if xb@xb>1e-12 else 0.0
        r2=ya-b2*xb; s2=r2.std(ddof=1)
        # 三腿：對所有 C 一次算完（2 變數正規方程的封閉解）
        Xc=Zc; n=Xc.shape[1]
        sbb=float(xb@xb); sby=float(xb@ya)
        sbc=xb@Xc; scc=(Xc*Xc).sum(0); scy=ya@Xc
        det=sbb*scc-sbc**2
        ok=np.abs(det)>1e-9
        B2=np.zeros(n); B3=np.zeros(n)
        B2[ok]=(scc[ok]*sby-sbc[ok]*scy[ok])/det[ok]
        B3[ok]=(sbb*scy[ok]-sbc[ok]*sby)/det[ok]
        res=ya[:,None]-B2[None,:]*xb[:,None]-B3[None,:]*Xc
        sd3=res.std(axis=0,ddof=1); sd3[~ok]=np.inf; sd3[[ia,ib]]=np.inf
        k=int(np.argmin(sd3))
        if not np.isfinite(sd3[k]): continue
        b2n,b3n,s3=float(B2[k]),float(B3[k]),float(sd3[k])
        c=Fv.columns[k]
        # 樣本外：以形成期係數建構價差，形成期 mu/sigma 標準化
        m2=float(np.mean(Z[:,ia]-b2*Z[:,ib]))
        m3=float(np.mean(Z[:,ia]-b2n*Z[:,ib]-b3n*Z[:,k]))
        z2=(Zt[:,ia]-b2*Zt[:,ib]-m2)/max(s2,1e-9)
        z3=(Zt[:,ia]-b2n*Zt[:,ib]-b3n*Zt[:,k]-m3)/max(s3,1e-9)
        u2=s2*(sd[ia]+abs(b2)*sd[ib])/(1+abs(b2))
        u3=s3*(sd[ia]+abs(b2n)*sd[ib]+abs(b3n)*sd[k])/(1+abs(b2n)+abs(b3n))
        c2,cv2=capture(z2,s2,u2); c3,cv3=capture(z3,s3,u3)
        rows.append((ps,c2,c3,cv2,cv3,s2,s3,
                     np.std(Zt[:,ia]-b2*Zt[:,ib],ddof=1),
                     np.std(Zt[:,ia]-b2n*Zt[:,ib]-b3n*Zt[:,k],ddof=1)))
D=pd.DataFrame(rows,columns=['ps','c2','c3','cv2','cv3','s2f','s3f','s2o','s3o'])
print('可比 %d 組、%d 期'%(len(D),D.ps.nunique()))
V=D.dropna(subset=['c2','c3'])
print('兩者皆有進場 %d 組'%len(V))
print()
print('=== 門檻一：樣本外每期平均 capture（三腿 − 兩腿）===')
per=V.groupby('ps')[['c2','c3']].mean().dropna()
t,p=ttest_rel(per.c3,per.c2)
print('  兩腿 %+.6f   三腿 %+.6f   差 %+.6f   配對t p %.4f   %s'%(
    per.c2.mean(),per.c3.mean(),(per.c3-per.c2).mean(),p,
    '✔' if (p<0.05 and (per.c3-per.c2).mean()>0) else '✘'))
print()
print('=== 附帶（不作判準）===')
print('  形成期殘差標準差比 sigma3/sigma2  中位 %.4f  ← 必然 <1，過擬合的直接量度'%
      (D.s3f/D.s2f).median())
print('  交易期實現標準差比 sigma3/sigma2  中位 %.4f'%(D.s3o/D.s2o).median())
print('  樣本外標準差膨脹  兩腿 %.3f 倍   三腿 %.3f 倍'%(
      (D.s2o/D.s2f).median(),(D.s3o/D.s3f).median()))
print('  收斂率  兩腿 %.4f   三腿 %.4f'%(V.cv2.mean(),V.cv3.mean()))
