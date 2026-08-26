# -*- coding: utf-8 -*-
import sys, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
C=r'C:\Clark\YZU\Papper\Code\dev\ml_formation\cache'
pool=pd.read_parquet(f'{C}/pool.parquet',
  columns=['Period_Start','Ticker_A','Ticker_B','SSD','adf_pass','label_valid',
           'not_converged','capture_frac'])
K=['Period_Start','Ticker_A','Ticker_B']
d=(pool.merge(pd.read_parquet(f'{C}/scores.parquet'),on=K)
        .merge(pd.read_parquet(f'{C}/m3_scores.parquet'),on=K))
d=d[(d.adf_pass==1)&d.label_valid].dropna(subset=['score_M1','score_M3','capture_frac'])
d['r']=d.groupby('Period_Start').SSD.rank(method='first')
print('=== SSD 名次帶的水準（259 期）===')
print('  名次帶      平均capture   未收斂率')
for lo,hi in [(1,20),(21,40),(41,60),(61,100)]:
    b=d[(d.r>=lo)&(d.r<=hi)]
    print('  %3d–%-3d    %+.6f     %.4f'%(lo,hi,b.capture_frac.mean(),b.not_converged.mean()))
print()
print('=== 分解 N=40 的否決結果（對 SSD top-20 的差）===')
base=d[d.r<=20].groupby('Period_Start').capture_frac.mean()
for m in ['score_M1','score_M3']:
    sub=d[d.r<=40]
    sel=sub.sort_values(['Period_Start',m]).groupby('Period_Start').head(20)
    got=sel.groupby('Period_Start').capture_frac.mean()
    pool40=sub.groupby('Period_Start').capture_frac.mean()   # top-40 全體平均＝隨機取 20 的期望
    idx=base.index.intersection(got.index)
    print('  %s'%m)
    print('    SSD top-20            %+.6f   (基準)'%base[idx].mean())
    print('    top-40 全體（＝隨機選）  %+.6f   水準位移 %+.6f'%(
        pool40[idx].mean(), pool40[idx].mean()-base[idx].mean()))
    print('    模型自 top-40 選 20     %+.6f   模型相對隨機 %+.6f'%(
        got[idx].mean(), got[idx].mean()-pool40[idx].mean()))
    print('    → 淨效果              %+.6f'%(got[idx].mean()-base[idx].mean()))
