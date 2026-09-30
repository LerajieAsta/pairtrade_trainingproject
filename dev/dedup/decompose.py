# -*- coding: utf-8 -*-
"""跨期去重的機制拆解（事後分析，見 dev/dedup/RESULTS.md「機制」一節）。

以 baseline（不去重）逐日紀錄，把「較晚那份」重複部位在較早那份離場後的損益，
依較早那份的平倉方式（正常收斂／期末強制平倉／停損／下市）拆開。
含平倉日：部位歸零那天的列仍記著收斂時實現的損益。

    python dev/dedup/decompose.py
"""
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
import sqlite3,pandas as pd,numpy as np
from strategies.dedup import pair_key
R=sqlite3.connect('file:results/result.db?mode=ro',uri=True,timeout=600)
Q="select Period_Start,Date,Ticker_A,Ticker_B,Position,Daily_Delta,Status from trade_logs where strategy_id=?"
EXITS={'EXIT':'正常收斂','PERIOD_END_EXIT':'期末強制平倉','STOP_LOSS_TRIGGERED':'停損','FORCED_CLOSE_DELISTED':'下市'}
for arm,tn in (('GICS_SSD',20),('NOGRP_DTW',20),('GICS_SSD',3),('NOGRP_SSD',5)):
    b=pd.read_sql(Q,R,params=(f"tiingo/Grid_{arm}/TradeLogs_Top{tn}_SL0_ZWin0_MSR0.csv",))
    b['k']=[pair_key((x,y)) for x,y in zip(b.Ticker_A,b.Ticker_B)]
    b['Date']=pd.to_datetime(b.Date)
    per=sorted(b.Period_Start.unique()); b['pi']=b.Period_Start.map({p:i for i,p in enumerate(per)})
    live=b[b.Position!=0]
    first=live.groupby(['Date','k']).pi.transform('min')
    later_units=set(zip(live[live.pi>first].pi,live[live.pi>first].k))
    held=live.groupby(['Date','k']).pi.apply(set)
    # 各單位的平倉事件（部位歸零那列的 Status）
    ex=b[b.Status.isin(EXITS)][['k','pi','Date','Status']].sort_values('Date')
    ex_by_k={k:g for k,g in ex.groupby('k')}
    rows=b[[ (p,k) in later_units for p,k in zip(b.pi,b.k)]].copy()
    hs=held.reindex(list(zip(rows.Date,rows.k)))
    rows['older_holds']=[isinstance(s,set) and any(x<p for x in s) for s,p in zip(hs.values,rows.pi)]
    after=rows[~rows.older_holds].copy()
    tag=[]
    for d,k,p in zip(after.Date,after.k,after.pi):
        g=ex_by_k.get(k)
        if g is None: tag.append('無'); continue
        g=g[(g.pi<p)&(g.Date<=d)]
        tag.append(EXITS[g.Status.iloc[-1]] if len(g) else '無（尚未重疊前）')
    after['older_exit']=tag
    s=after.groupby('older_exit').Daily_Delta.sum().round(0)
    print(f"{arm:<9} Top{tn:<2}｜較早那份離場後的損益，依較早那份的平倉方式： {s.to_dict()}")
