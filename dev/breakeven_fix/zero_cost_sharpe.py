# -*- coding: utf-8 -*-
"""零成本情境的 Sharpe：把手續費逐日加回權益曲線。"""
import sys, sqlite3, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
from strategies.metrics import metrics_from_pnl
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from strategies.config import INITIAL_CAPITAL
TD=252; RDB=r'C:\Clark\YZU\Papper\Code\results\result.db'
c=sqlite3.connect(f'file:{RDB}?mode=ro',uri=True)

def run(sid, top_n, lab, fee_rates=(0.0029,0.0015,0.0010,0.0005,0.0)):
    d=pd.read_sql('select Date,Period_Start,Ticker_A,Ticker_B,Position,Hedge_Ratio,'
                  'Daily_Delta,Status from trade_logs where strategy_id=?',c,params=(sid,))
    d['Date']=pd.to_datetime(d.Date)
    cap=INITIAL_CAPITAL/(top_n*(126//21))
    # 每次 ENTER 或平倉事件的名目額 = cap（v_a+v_b=cap）
    ev=d.Status.str.contains('ENTER',na=False) | d.Status.str.contains('CLOSE|STOP|FORCE|EXIT',na=False,regex=True)
    fee_events=d[ev].groupby('Date').size()
    pnl=d.groupby('Date').Daily_Delta.sum().sort_index()
    print('=== %s ==='%lab)
    print('  費用事件 %d 次'%fee_events.sum())
    print('  單邊成本    年化%      Sharpe     MDD%')
    base=None
    for fr in fee_rates:
        add=(fee_events*cap*(0.0029-fr)).reindex(pnl.index).fillna(0.0)
        p=pnl+add
        _m=metrics_from_pnl(p)
        ann,sh,mdd=_m['Ann_Ret_Raw'],_m['Sharpe_Raw'],_m['MDD_Raw']
        if base is None: base=sh
        print('   %.2f%%      %+7.3f    %6.3f     %6.2f'%(fr*100,ann*100,sh,mdd*100))
    print()

run('tiingo/Grid_NOGRP_DTW/TradeLogs_Top1_SL0_ZWin0_MSR0.csv',1,'NOGRP-DTW Top1/SL0（旗艦）')
run('tiingo/Grid_GGR/TradeLogs_Top20_SL0_ZWin0_MSR0.csv',20,'GGR Top20/SL0（經典錨點）')
