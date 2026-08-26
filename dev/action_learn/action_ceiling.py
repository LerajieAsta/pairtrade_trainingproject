# -*- coding: utf-8 -*-
"""任何門檻選擇演算法的天花板：完美預知的逐對動作 vs 最佳常數動作。"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
C=r'C:\Clark\YZU\Papper\Code\dev\ml_formation\cache'
d=pd.read_parquet(f'{C}/exit_scan.parquet')
p=pd.read_parquet(f'{C}/pool.parquet',columns=['Period_Start','Ticker_A','Ticker_B','SSD','adf_pass','label_valid'])
K=['Period_Start','Ticker_A','Ticker_B']
d=d.drop(columns=[c for c in ('SSD','adf_pass') if c in d.columns]).merge(p,on=K,how='inner')
d=d[(d.adf_pass==1)&d.label_valid]
d['r']=d.groupby('Period_Start').SSD.rank(method='first')
caps=[c for c in d.columns if c.startswith('cap_e')]
print('動作數（entry_z × exit_z 組合）: %d'%len(caps))
print('候選: %s ...'%caps[:4])
for TOPN in [1,5,20]:
    sub=d[d.r<=TOPN].dropna(subset=caps)
    if sub.empty: continue
    M=sub[caps].to_numpy()
    per=sub.Period_Start.values
    # 最佳常數動作：全樣本平均最高的那一個
    mean_by_action=M.mean(axis=0)
    best_const=int(np.argmax(mean_by_action))
    # 完美預知：逐列取最大
    oracle=M.max(axis=1)
    # 逐期彙總
    df=pd.DataFrame({'ps':per,'const':M[:,best_const],'oracle':oracle})
    g=df.groupby('ps').mean()
    print()
    print('=== Top %d （%d 列、%d 期）==='%(TOPN,len(sub),g.shape[0]))
    print('  最佳常數動作: %s   每期平均 capture %+.6f'%(caps[best_const],g['const'].mean()))
    print('  完美預知逐對: 每期平均 capture %+.6f'%g['oracle'].mean())
    print('  → 天花板增益 %+.6f  （%.1f 倍）'%(
        g['oracle'].mean()-g['const'].mean(),
        g['oracle'].mean()/g['const'].mean() if g['const'].mean()!=0 else np.nan))
    # 各常數動作的排名，看最佳與次佳差多少
    order=np.argsort(-mean_by_action)[:5]
    print('  前 5 個常數動作: %s'%', '.join('%s %+.6f'%(caps[i],mean_by_action[i]) for i in order))
