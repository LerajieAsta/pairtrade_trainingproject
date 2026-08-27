# -*- coding: utf-8 -*-
import sys, sqlite3, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
from dev.ml_formation.selection import selection_region, with_model_scores
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from strategies.config import DB_PATH, TABLE_NAME
TD=252
# ── 事實一：regime 標籤在「全樣本分位」vs「擴張窗分位」下差多少 ──
con=sqlite3.connect(DB_PATH)
px=pd.read_sql(f"SELECT Date,Symbol,COALESCE(Adj_Close,Close) p FROM {TABLE_NAME}",con); con.close()
px['Date']=pd.to_datetime(px.Date)
w=px.pivot_table(index='Date',columns='Symbol',values='p').replace(0.0,np.nan)
mkt=np.log(w).diff().mean(axis=1).dropna()
vol=(mkt.rolling(63).std()*np.sqrt(TD)).dropna()
q1f,q2f=vol.quantile([1/3,2/3])
full=np.where(vol<=q1f,'Calm',np.where(vol<=q2f,'Normal','Turbulent'))
# 擴張窗：只用截至當日的歷史（暖身 504 日）
e1=vol.expanding(504).quantile(1/3); e2=vol.expanding(504).quantile(2/3)
m=e1.notna()&e2.notna()
exp=np.where(vol[m]<=e1[m],'Calm',np.where(vol[m]<=e2[m],'Normal','Turbulent'))
agree=(full[m.values]==exp).mean()
print('=== 事實一：regime 標籤的前視程度 ===')
print('  可比日數 %d（暖身 504 日後）'%m.sum())
print('  全樣本分位 vs 擴張窗分位  標籤一致率 %.1f%%'%(agree*100))
print('  全樣本 Turbulent 佔比 %.1f%%   擴張窗 Turbulent 佔比 %.1f%%'%(
    (full[m.values]=='Turbulent').mean()*100,(exp=='Turbulent').mean()*100))
print()
# ── 事實二：選中配對的價差波動離散度（等額配置是否等於等風險）──
C=r'C:\Clark\YZU\Papper\Code\dev\ml_formation\cache'
d=selection_region()
t=d[d.ssd_rank<=20]
g=t.groupby('Period_Start').Spread_Std_exact
ratio=(g.max()/g.min().clip(1e-9))
print('=== 事實二：top-20 內價差波動的離散度（等額配置 ≠ 等風險？）===')
print('  每期 max/min 價差標準差比  中位 %.2f  分位 25%%:%.2f 75%%:%.2f 90%%:%.2f'%(
    ratio.median(),ratio.quantile(.25),ratio.quantile(.75),ratio.quantile(.90)))
print('  每期組內變異係數 中位 %.3f'%(t.groupby('Period_Start').Spread_Std_exact.std()/
                                     t.groupby('Period_Start').Spread_Std_exact.mean()).median())
print()
print('  價差波動五分位 vs capture：')
q=t.Spread_Std_exact.quantile([0,.2,.4,.6,.8,1.0]).values
for i in range(5):
    b=t[(t.Spread_Std_exact>=q[i])&(t.Spread_Std_exact<=q[i+1])]
    print('    %.4f–%.4f  n=%4d  平均capture %+.6f  |capture| 平均 %.6f'%(
        q[i],q[i+1],len(b),b.capture_frac.mean(),b.capture_frac.abs().mean()))
