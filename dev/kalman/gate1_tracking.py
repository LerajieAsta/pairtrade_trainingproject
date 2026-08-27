# -*- coding: utf-8 -*-
"""門檻一：Kalman beta_t 對實現 beta 的追蹤誤差，是否顯著低於靜態 beta_form。
依 dev/kalman/PREREGISTRATION.md。濾波僅用 t 之前資料。"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
from dev.ml_formation.selection import selection_region, with_model_scores
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import ttest_rel
from strategies.config import (DB_PATH, TABLE_NAME, BACKTEST_START, BACKTEST_END,
                               FORMATION_WINDOW, FORWARD_DAYS)
from strategies.preprocess_equity import DataProcessor
C=r'C:\Clark\YZU\Papper\Code\dev\ml_formation\cache'

proc=DataProcessor(db_path=DB_PATH, table_name=TABLE_NAME)
pp, ad, td, fi = proc.prepare_backtest_data(BACKTEST_START, BACKTEST_END, FORMATION_WINDOW)
dstr=np.array(pd.DatetimeIndex(ad).strftime('%Y-%m-%d')); pos={s:i for i,s in enumerate(dstr)}

d=selection_region()
t=d[d.ssd_rank<=20]

def kalman(za, zb, a0, b0, R, Q):
    """回傳 beta_t 序列（每個 t 的事前估計，只用 t 之前的觀測）。"""
    x=np.array([a0,b0]); P=np.eye(2)*Q*10.0
    out=np.empty(len(za))
    for i in range(len(za)):
        P=P+Q                                  # 預測
        out[i]=x[1]                            # 事前 beta（用於 t 期決策，無前視）
        H=np.array([1.0, zb[i]])
        S=H@P@H+R
        K=(P@H)/S
        x=x+K*(za[i]-H@x)                      # 更新（用 t 期觀測，供 t+1）
        P=P-np.outer(K,H@P)
    return out

RW=41                                          # 實現 beta 的滾動窗（僅供評分）
CFG={'K-EST':None,'K-5':1e-5,'K-4':1e-4,'K-3':1e-3}
rec={k:[] for k in CFG}; rec['static']=[]
np_ok=0
for ps,g in t.groupby('Period_Start'):
    if ps not in pos: continue
    fs=pos[ps]; idx=fs+FORMATION_WINDOW; te=min(idx+FORWARD_DAYS,len(dstr))
    F=pp.iloc[fs:idx]; T=pp.iloc[idx:te]
    if len(T)<RW+10: continue
    for _,r in g.iterrows():
        a,b=r.Ticker_A,r.Ticker_B
        if a not in F.columns or b not in F.columns: continue
        Pf=F[[a,b]].dropna(); Pt=T[[a,b]].dropna()
        if len(Pf)<200 or len(Pt)<RW+10: continue
        Lf=np.log(np.maximum(Pf.values,1e-8)); mu=Lf.mean(0); sd=Lf.std(0)+1e-12
        Zf=(Lf-mu)/sd
        Zt=(np.log(np.maximum(Pt.values,1e-8))-mu)/sd
        # 形成期 OLS
        bf=np.cov(Zf[:,0],Zf[:,1])[0,1]/np.var(Zf[:,1],ddof=1)
        af=Zf[:,0].mean()-bf*Zf[:,1].mean()
        Rv=np.var(Zf[:,0]-af-bf*Zf[:,1],ddof=1)
        # Q_est：形成窗切 6 個 42 日子窗
        sub=[]
        for k in range(6):
            s0,s1=k*42,(k+1)*42
            if s1>len(Zf): break
            v=np.var(Zf[s0:s1,1],ddof=1)
            if v>1e-10: sub.append(np.cov(Zf[s0:s1,0],Zf[s0:s1,1])[0,1]/v)
        q_est=np.var(np.diff(sub),ddof=1) if len(sub)>=3 else 1e-5
        # 實現 beta（滾動窗，評分用）
        za,zb=Zt[:,0],Zt[:,1]
        real=np.full(len(za),np.nan)
        for i in range(RW,len(za)):
            w=slice(i-RW,i); v=np.var(zb[w],ddof=1)
            if v>1e-10: real[i]=np.cov(za[w],zb[w])[0,1]/v
        m=~np.isnan(real)
        if m.sum()<20: continue
        np_ok+=1
        rec['static'].append(np.abs(bf-real[m]).mean())
        for name,delta in CFG.items():
            Q=np.eye(2)*(q_est if delta is None else delta*Rv)
            bk=kalman(za,zb,af,bf,Rv,Q)
            rec[name].append(np.abs(bk[m]-real[m]).mean())

S=np.array(rec['static'])
print('可評分配對數: %d'%np_ok)
print('實現 beta 定義：交易期 %d 日滾動 OLS（僅供評分，非交易用）'%RW)
print()
print('平均絕對追蹤誤差（越低越好），對靜態配對比較')
print('  配置      平均MAE    對靜態差      配對t p        Bonferroni(0.0125)')
print('  static    %.4f     (基準)'%S.mean())
for name in CFG:
    K=np.array(rec[name]); diff=K-S
    tt,p=ttest_rel(K,S)
    print('  %-8s  %.4f    %+.4f      %.3e     %s'%(
        name,K.mean(),diff.mean(),p,'✔' if (p<0.0125 and diff.mean()<0) else '✘'))
