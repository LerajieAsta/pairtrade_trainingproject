# -*- coding: utf-8 -*-
import sys, sqlite3, re, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
c=sqlite3.connect(r'file:C:\Clark\YZU\Papper\Code\results\result.db?mode=ro',uri=True)
ARMS=["Grid (NOGRP-SDP)","Grid (NOGRP-SSD)","Grid (GICS-SSD)","Grid (NOGRP-DTW)"]
_q=('select _path,METHOD,"TOP N" t,"STOP LOSS %" s,"ENTRY Z" ez,Ann_Ret_Raw,'
    'Sharpe_Raw,MDD_Raw,Entries,Avg_Utilization,Profit_Factor,Forced_Closes,Stop_Losses '
    'from strategy_summaries where METHOD in (' + ','.join('?'*len(ARMS)) + ')')
d=pd.read_sql(_q,c,params=ARMS)
d['dsz']=[float(m.group(1))/10 if (m:=re.search(r'_DSZ(\d+)',p)) else 0.0 for p in d._path]
d['mhd']=[int(m.group(1)) if (m:=re.search(r'_MHD(\d+)',p)) else 0 for p in d._path]
d['tn']=d.t.str.replace('Top','').str.strip().astype(int)
d=d[(d.ez==2.0)]                                   # 單變因：entry_z 固定 2.0
big=d[d.tn>=5]
print('=== z 停損（dynamic_stop_z）｜top_n>=5 的年化%% ===')
z=big[big.mhd==0]
print(z.pivot_table(index='METHOD',columns='dsz',values='Ann_Ret_Raw',aggfunc='max').mul(100).round(3).to_string())
print('--- Sharpe ---')
print(z.pivot_table(index='METHOD',columns='dsz',values='Sharpe_Raw',aggfunc='max').round(3).to_string())
print()
print('=== 時間停損（max_holding_days）｜top_n>=5 的年化%% ===')
m_=big[big.dsz==0]
print(m_.pivot_table(index='METHOD',columns='mhd',values='Ann_Ret_Raw',aggfunc='max').mul(100).round(3).to_string())
print('--- Sharpe ---')
print(m_.pivot_table(index='METHOD',columns='mhd',values='Sharpe_Raw',aggfunc='max').round(3).to_string())
print()
print('=== 機制：NOGRP-SDP Top5/SL0 ===')
x=d[(d.METHOD=='Grid (NOGRP-SDP)')&(d.t=='Top 5')&(d.s=='0%')]
o=x[['dsz','mhd','Ann_Ret_Raw','Sharpe_Raw','Entries','Avg_Utilization','Profit_Factor','Stop_Losses','Forced_Closes']].copy()
o['年化%']=(o.Ann_Ret_Raw*100).round(3); o['利用率%']=(o.Avg_Utilization*100).round(1)
print(o[['dsz','mhd','年化%','Sharpe_Raw','Entries','利用率%','Profit_Factor','Stop_Losses','Forced_Closes']].round(3).sort_values(['dsz','mhd']).to_string(index=False))
print()
print('top_n>=5 且年化>2%% 的配置數:',(big.Ann_Ret_Raw>0.02).sum())
