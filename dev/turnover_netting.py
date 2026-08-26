# -*- coding: utf-8 -*-
"""方案 F：毛周轉 vs 淨周轉（正確處理空檔——不在倉的日子曝險為 0）。"""
import sys, sqlite3, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from strategies.config import INITIAL_CAPITAL
RDB=r'C:\Clark\YZU\Papper\Code\results\result.db'
c=sqlite3.connect(f'file:{RDB}?mode=ro',uri=True)
CONC=126//21; FEE=0.0029

def tv(series_by_key, cal):
    """各 key 的曝險序列在完整日曆上（缺席＝0）的總變異量之和。"""
    tot=0.0
    for _,s in series_by_key:
        v=s.set_index('Date').expo
        v=v[~v.index.duplicated()].reindex(cal, fill_value=0.0)
        a=v.values
        tot+=np.abs(np.diff(np.concatenate(([0.0],a,[0.0])))).sum()
    return tot

def run(sid, top_n, lab):
    d=pd.read_sql('select Date,Period_Start,Ticker_A,Ticker_B,Position '
                  'from trade_logs where strategy_id=?',c,params=(sid,))
    if d.empty: print(lab,'無資料'); return None
    d['Date']=pd.to_datetime(d.Date)
    sw=d.Ticker_A>d.Ticker_B
    d['P1']=np.where(sw,d.Ticker_B,d.Ticker_A); d['P2']=np.where(sw,d.Ticker_A,d.Ticker_B)
    cap=INITIAL_CAPITAL/(top_n*CONC)
    d['expo']=np.where(sw,-d.Position,d.Position).astype(float)*cap
    cal=pd.DatetimeIndex(sorted(d.Date.unique()))
    gross=tv(d.groupby(['Period_Start','P1','P2']), cal)
    netdf=d.groupby(['Date','P1','P2'],as_index=False).expo.sum()
    net=tv(netdf.groupby(['P1','P2']), cal)
    saved=gross-net
    print('=== %s ==='%lab)
    print('  毛周轉 $%s'%f'{gross:,.0f}')
    print('  淨周轉 $%s'%f'{net:,.0f}')
    print('  可省   $%s = %.2f%%   → 手續費 $%s'%(f'{saved:,.0f}',
          saved/gross*100 if gross else 0, f'{saved*FEE:,.0f}'))
    return saved*FEE

r=[]
r.append(('NOGRP-DTW Top1', 1, run('tiingo/Grid_NOGRP_DTW/TradeLogs_Top1_SL0_ZWin0_MSR0.csv',1,'NOGRP-DTW Top1/SL0')))
print()
r.append(('NOGRP-DTW Top20',20, run('tiingo/Grid_NOGRP_DTW/TradeLogs_Top20_SL0_ZWin0_MSR0.csv',20,'NOGRP-DTW Top20/SL0')))
print()
r.append(('GGR Top20',20, run('tiingo/Grid_GGR/TradeLogs_Top20_SL0_ZWin0_MSR0.csv',20,'GGR Top20/SL0')))
print()
s=pd.read_sql('select METHOD,"TOP N" t,Final_Equity from strategy_summaries '
              'where METHOD in ("Grid (NOGRP-DTW)","Grid (GGR)") and "STOP LOSS %"="0%"',c)
print('=== 對照淨利 ===')
for lab,tn,fee in r:
    m='Grid (GGR)' if lab.startswith('GGR') else 'Grid (NOGRP-DTW)'
    row=s[(s.METHOD==m)&(s.t=='Top %d'%tn)]
    if row.empty or fee is None: continue
    net=float(row.Final_Equity.iloc[0])-INITIAL_CAPITAL
    print('  %-16s 淨利 $%s   可省手續費 $%s = 淨利的 %.1f%%'%(
        lab,f'{net:,.0f}',f'{fee:,.0f}',fee/net*100 if net else 0))
