# -*- coding: utf-8 -*-
"""自洽檢驗：把費率調到宣稱的 break-even，淨利應歸零。"""
import sys, sqlite3, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from strategies.config import INITIAL_CAPITAL
c=sqlite3.connect(r'file:C:\Clark\YZU\Papper\Code\results\result.db?mode=ro',uri=True)
FEE=0.0029; CONC=6
for sub,tn,lab,claim in [('Grid_NOGRP_DTW',1,'NOGRP-DTW Top1',1.219),('Grid_GGR',20,'GGR Top20',0.411)]:
    sid='tiingo/%s/TradeLogs_Top%d_SL0_ZWin0_MSR0.csv'%(sub,tn)
    d=pd.read_sql('select Date,Status,Daily_Delta from trade_logs where strategy_id=?',c,params=(sid,))
    d['Date']=pd.to_datetime(d.Date)
    pnl=d.groupby('Date').Daily_Delta.sum().sort_index()
    eq=INITIAL_CAPITAL+pnl.cumsum()
    evd=d[d.Status.str.contains('ENTER',na=False)].groupby('Date').size()
    cap_t=(eq/(tn*CONC)).reindex(evd.index).ffill()
    base_fee=(evd*cap_t*2).reindex(pnl.index).fillna(0.0)     # 每次進場含一進一出
    print('=== %s（宣稱往返 BE %.3f%%）==='%(lab,claim))
    for rt in [0.580, claim-0.1, claim, claim+0.1]:
        side=rt/200.0
        adj=base_fee*(FEE-side)
        p=(pnl+adj).cumsum()
        print('   往返 %.3f%%  →  期末淨利 $%s'%(rt,f'{p.iloc[-1]:,.0f}'))
    print()
