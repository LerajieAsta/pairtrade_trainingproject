# -*- coding: utf-8 -*-
"""可行性診斷（未預先登記、不入試驗宇宙）：進場當下的發散原因能否預測收斂。
對 GICS-SSD-FW504 Top20/SL0 的 6,915 筆交易，取進場前 5 日（含當日）的單腳異常報酬、異常成交量、
市場變動，對「期內收斂（EXIT）」算 AUC 與五分位收斂率。Engelberg et al. (2009)、Jacobs & Weber (2015)。

    python dev/formation_vs_trading/entry_cause.py
"""
import sqlite3, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
P="tiingo/Grid_GICS_SSD_FW504/TradeLogs_Top20_SL0_ZWin0_MSR0.csv"
c=sqlite3.connect("file:results/result.db?mode=ro",uri=True)
t=pd.read_sql("select Period_Start,Ticker_A,Ticker_B,Date,Status,Trade_PnL from trade_logs where strategy_id=? and (Status like 'ENTER%' or Trade_PnL!=0) order by Period_Start,Ticker_A,Ticker_B,Date",c,params=(P,))
ent=t[t.Status.str.startswith("ENTER")].copy(); cls=t[t.Trade_PnL!=0].copy()
k=["Period_Start","Ticker_A","Ticker_B"]
ent["n"]=ent.groupby(k).cumcount(); cls["n"]=cls.groupby(k).cumcount()
tr=ent.merge(cls[k+["n","Status","Trade_PnL"]],on=k+["n"],suffixes=("","_x"))
tr["conv"]=(tr.Status_x=="EXIT").astype(int)
print("交易數",len(tr),"收斂率",tr.conv.mean().round(3))
p=sqlite3.connect("file:dataset/price/sp500_Tiingo.db?mode=ro",uri=True)
d=pd.read_sql("select Date,Symbol,COALESCE(Adj_Close,Close) C,Volume from Daily_Prices where COALESCE(Adj_Close,Close)>0",p)
d["Date"]=pd.to_datetime(d.Date)
C=d.pivot(index="Date",columns="Symbol",values="C").sort_index(); V=d.pivot(index="Date",columns="Symbol",values="Volume").reindex(C.index)
R=np.log(C).diff(); mkt=R.mean(axis=1)
ex=R.sub(mkt,axis=0)                      # 市場調整報酬
sig=ex.rolling(60,min_periods=40).std().shift(1)
zr=ex/sig                                 # 標準化異常報酬
lv=np.log(V.replace(0,np.nan)); av=(lv-lv.rolling(60,min_periods=40).median().shift(1))
msig=mkt.rolling(60,min_periods=40).std().shift(1); zm=(mkt/msig).abs()
W=5; idx=C.index
rows=[]
for r in tr.itertuples(index=False):
    dt=pd.Timestamp(r.Date); i=idx.get_loc(dt); s=slice(max(0,i-W+1),i+1)
    za=zr[r.Ticker_A].iloc[s].abs(); zb=zr[r.Ticker_B].iloc[s].abs()
    va=av[r.Ticker_A].iloc[s]; vb=av[r.Ticker_B].iloc[s]
    jump=max(za.max(),zb.max())
    asym=abs(za.max()-zb.max())
    vol=max(va.max(),vb.max()); vasym=abs(va.max()-vb.max())
    sumsq_idio=float((zr[r.Ticker_A].iloc[s]**2).sum()+(zr[r.Ticker_B].iloc[s]**2).sum())
    rows.append((jump,asym,vol,vasym,float(zm.iloc[s].max()),sumsq_idio))
F=pd.DataFrame(rows,columns=["jump","asym","abnvol","vol_asym","mkt_move","idio_sumsq"],index=tr.index)
X=pd.concat([tr,F],axis=1).dropna(subset=list(F.columns))
print("有效",len(X))
for f in F.columns:
    auc=roc_auc_score(1-X.conv,X[f])     # 預測「不收斂」
    q=pd.qcut(X[f].rank(method="first"),5,labels=False)
    g=X.groupby(q).agg(conv=("conv","mean"),pnl=("Trade_PnL","mean"))
    print(f"{f:<11} AUC(不收斂) {auc:.3f}｜五分位收斂率 {g.conv.round(3).tolist()}｜平均損益 {g.pnl.round(2).tolist()}")
X["half"]=pd.to_datetime(X.Date)>=pd.Timestamp("2014-01-02")
for h,g in X.groupby("half"):
    print("後半" if h else "前半", {f: round(roc_auc_score(1-g.conv,g[f]),3) for f in ["jump","asym","abnvol","mkt_move"]})
X.to_pickle("results/analysis/entry_cause.pkl")
