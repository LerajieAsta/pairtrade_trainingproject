# -*- coding: utf-8 -*-
import sys, sqlite3, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
c=sqlite3.connect(r'file:C:\Clark\YZU\Papper\Code\results\result.db?mode=ro',uri=True)
d=pd.read_sql('''select METHOD,"TOP N" t,"STOP LOSS %" s,"ENTRY Z" ez,Ann_Ret_Raw,Sharpe_Raw,
 MDD_Raw,Ann_Ret_Employed,Avg_Utilization,Entries,Profit_Factor from strategy_summaries
 where METHOD in ("Grid (NOGRP-SDP)","Grid (NOGRP-SSD)","Grid (GICS-SSD)","Grid (NOGRP-DTW)")''',c)
d['tn']=d.t.str.replace('Top','').str.strip().astype(int)
print('各方法 × entry_z 的格數:')
print(d.pivot_table(index='METHOD',columns='ez',values='Ann_Ret_Raw',aggfunc='size').fillna(0).astype(int).to_string())
print()
for lab,sel in [('全部 top_n',d),('top_n >= 5（常規）',d[d.tn>=5])]:
    print('=== %s：年化%% ==='%lab)
    p=sel.pivot_table(index='METHOD',columns='ez',values='Ann_Ret_Raw',aggfunc='max')*100
    print(p.round(3).to_string())
    print('--- Sharpe ---')
    q=sel.pivot_table(index='METHOD',columns='ez',values='Sharpe_Raw',aggfunc='max')
    print(q.round(3).to_string()); print()
print('=== top_n>=5 且年化 > 2%% 的配置 ===')
w=d[(d.tn>=5)&(d.Ann_Ret_Raw>0.02)].sort_values('Ann_Ret_Raw',ascending=False)
if len(w):
    o=pd.DataFrame({'METHOD':w.METHOD,'格':w.t+'/SL'+w.s,'entry_z':w.ez,
      '年化%':(w.Ann_Ret_Raw*100).round(3),'Sharpe':w.Sharpe_Raw.round(3),
      'MDD%':(w.MDD_Raw*100).round(2),'利用率%':(w.Avg_Utilization*100).round(1),
      '進場':w.Entries.astype(int),'PF':w.Profit_Factor.round(3)})
    print(o.to_string(index=False))
else:
    print('  （無）')
