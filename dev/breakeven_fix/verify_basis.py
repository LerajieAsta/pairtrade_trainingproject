# -*- coding: utf-8 -*-
"""驗證 break-even 的名目額口徑。"""
import sys, sqlite3, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from strategies.config import INITIAL_CAPITAL
RDB=r'C:\Clark\YZU\Papper\Code\results\result.db'
c=sqlite3.connect(f'file:{RDB}?mode=ro',uri=True)
FEE=0.0029

for sid,tn,lab in [('tiingo/Grid_NOGRP_DTW/TradeLogs_Top1_SL0_ZWin0_MSR0.csv',1,'NOGRP-DTW Top1/SL0'),
                   ('tiingo/Grid_NOGRP_DTW/TradeLogs_Top20_SL0_ZWin0_MSR0.csv',20,'NOGRP-DTW Top20/SL0')]:
    d=pd.read_sql('select Date,Period_Start,Ticker_A,Ticker_B,Status,Cumulative_PnL,Daily_Delta '
                  'from trade_logs where strategy_id=?',c,params=(sid,))
    d['Date']=pd.to_datetime(d.Date)
    s=pd.read_sql('select Entries,Exits,Final_Equity from strategy_summaries where _path=?',c,params=(sid,)).iloc[0]
    ev=int(s.Entries)+int(s.Exits)
    # 逐日組合權益 = 初始 + 累計已實現＋未實現（用 Daily_Delta 累加）
    eq=INITIAL_CAPITAL+d.groupby('Date').Daily_Delta.sum().sort_index().cumsum()
    conc=126//21
    # 事件發生當日的 capital_per_pair = 當日權益 / (top_n*conc)
    evd=d[d.Status.str.contains('ENTER|CLOSE|STOP|FORCE|EXIT',na=False,regex=True)].groupby('Date').size()
    cap_t=(eq/(tn*conc)).reindex(evd.index).ffill()
    notion_true=float((evd*cap_t).sum())
    notion_paper=(INITIAL_CAPITAL/tn)*ev
    notion_fixed=(INITIAL_CAPITAL/(tn*conc))*ev
    pnet=float(s.Final_Equity)-INITIAL_CAPITAL
    print('=== %s ==='%lab)
    print('  事件數 %d（Entries %d + Exits %d）'%(ev,s.Entries,s.Exits))
    print('  名目額：論文口徑 $%s ｜ 固定初始/6 $%s ｜ 逐日權益/6（正確）$%s'%(
        f'{notion_paper:,.0f}',f'{notion_fixed:,.0f}',f'{notion_true:,.0f}'))
    for lab2,N in [('論文口徑',notion_paper),('逐日權益/6（正確）',notion_true)]:
        be=2*(FEE+pnet/N)
        print('    %-18s 往返 break-even %.3f%%   （對成本 0.580%%，餘裕 %+.3f pp）'%(lab2,be*100,(be-0.0058)*100))
    print('  實付手續費（正確口徑）$%s   對淨利 %s 的 %.2f 倍'%(
        f'{notion_true*FEE:,.0f}',f'{pnet:,.0f}',notion_true*FEE/pnet if pnet else float('nan')))
    print()
