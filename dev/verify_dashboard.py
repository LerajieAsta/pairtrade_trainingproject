# -*- coding: utf-8 -*-
"""離線驗證：解析器修正 + 彙總表邏輯。"""
import sys, sqlite3, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
src=open(r'C:\Clark\YZU\Papper\Code\dashboard.py',encoding='utf-8').read()
a=src.index('def _canonical_method'); b=src.index('@st.cache_data', a)
ns={'re':__import__('re')}; exec(src[a:b], ns)
f=ns['extract_features_from_path']
c=sqlite3.connect(r'file:C:\Clark\YZU\Papper\Code\results\result.db?mode=ro',uri=True)
d=pd.read_sql('select * from strategy_summaries',c)
d['parsed']=[f(p)[3] for p in d._path]
print('=== 解析器 ===')
print('  Unknown 比例  修正前 34.0%%  →  修正後 %.1f%%'%((d.parsed=='Unknown').mean()*100))
print('  與 DB METHOD 相符  %d / %d'%((d.parsed==d.METHOD).sum(),len(d)))
print()
# 彙總表邏輯
_sum_src=d
_RANK={'Sharpe':'Sharpe_Raw','Ann. Return':'Ann_Ret_Raw','Final Equity':'Final_Equity',
       'Calmar':'Calmar_Raw','Ann. Ret Employed':'Ann_Ret_Employed','Profit Factor':'Profit_Factor'}
for lab,col in _RANK.items():
    v=pd.to_numeric(_sum_src[col],errors='coerce')
    idx=v.groupby(_sum_src['METHOD']).idxmax().dropna()
    b_=_sum_src.loc[idx]
    print('  以 %-18s 選代表格 → %d 個方法，最佳 %s = %.4f (%s)'%(
        lab,len(b_),lab,pd.to_numeric(b_[col],errors='coerce').max(),
        b_.loc[pd.to_numeric(b_[col],errors='coerce').idxmax(),'METHOD']))
print()
v=pd.to_numeric(_sum_src['Sharpe_Raw'],errors='coerce')
b_=_sum_src.loc[v.groupby(_sum_src['METHOD']).idxmax().dropna()].copy()
def _pc(col): return pd.to_numeric(b_.get(col,np.nan),errors='coerce')*100
out=pd.DataFrame({'METHOD':b_['METHOD'],
    'BEST CELL':(b_.get('TOP N','').astype(str)+' / SL'+b_.get('STOP LOSS %','').astype(str)),
    'TRADE':b_.get('TRADE_METHOD',''),'ANN%':_pc('Ann_Ret_Raw').round(3),
    'SHARPE':pd.to_numeric(b_['Sharpe_Raw'],errors='coerce').round(3),
    'MAXDD%':_pc('MDD_Raw').round(2),'ROIC%':_pc('Ann_Ret_Employed').round(3),
    'UTIL%':_pc('Avg_Utilization').round(1),'ENTRIES':pd.to_numeric(b_['Entries'],errors='coerce'),
    'EQUITY':pd.to_numeric(b_['Final_Equity'],errors='coerce').round(0)}).sort_values('SHARPE',ascending=False)
print('=== 彙總表前 10 列（以 Sharpe 選代表格）===')
print(out.head(10).to_string(index=False))
print()
print('總方法數 %d ；ENTRIES 或 UTIL 為 0 的可疑列：'%len(out))
print(out[(out.ENTRIES<1000)&(out['ROIC%']>50)].to_string(index=False))
