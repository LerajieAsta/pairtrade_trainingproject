# -*- coding: utf-8 -*-
import sys, sqlite3, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from analysis.regime_cost_dsr_eval import INCOMPLETE_RUNS, TRADING_DAYS
c=sqlite3.connect(r'file:C:\Clark\YZU\Papper\Code\results\result.db?mode=ro',uri=True)
s=pd.read_sql('select METHOD,Sharpe_Raw from strategy_summaries',c)
live=s[~s.METHOD.isin(INCOMPLETE_RUNS)]
NEW={'Grid (GGR)','Grid (GGR-DTW)','Grid (GGR-SDP)','Grid (NOGRP-DTW-KAL)',
     'Grid (NOGRP-DTW-KALINN)','Grid (NOGRP-DTW-VG50)','Grid (NOGRP-DTW-VG67)',
     'Grid (GICS-SSD-FW504)','Grid (NOGRP-DTW-TW63)'}
old=live[~live.METHOD.isin(NEW)]
def v(x): return float(np.var(pd.Series(x).dropna(),ddof=1))/TRADING_DAYS
print('釘死值  method 44 / var 0.00008584088699455   config 819 / var 0.00026687963033353')
print()
print('=== 逆推：排除本次 9 條新臂後 ===')
print('  method 數 %d   config 數 %d'%(old.METHOD.nunique(),len(old)))
for lab,val in [('每 METHOD 取最佳 Sharpe', v(old.groupby('METHOD').Sharpe_Raw.max())),
                ('每 METHOD 取平均 Sharpe', v(old.groupby('METHOD').Sharpe_Raw.mean()))]:
    print('  %-22s var=%.17f  %s'%(lab,val,'← 相符' if abs(val-0.00008584088699455)<1e-9 else ''))
print('  config 全部列          var=%.17f  %s'%(v(old.Sharpe_Raw),
      '← 相符' if abs(v(old.Sharpe_Raw)-0.00026687963033353)<1e-9 else ''))
print()
print('=== 新宇宙（全部現役）===')
print('  method 數 %d   config 數 %d'%(live.METHOD.nunique(),len(live)))
print('  method 取最佳 var=%.17f'%v(live.groupby('METHOD').Sharpe_Raw.max()))
print('  method 取平均 var=%.17f'%v(live.groupby('METHOD').Sharpe_Raw.mean()))
print('  config 全部列 var=%.17f'%v(live.Sharpe_Raw))
