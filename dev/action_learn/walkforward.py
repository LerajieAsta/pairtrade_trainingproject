# -*- coding: utf-8 -*-
"""以監督回歸學習逐對門檻。依 dev/action_learn/PREREGISTRATION.md。"""
import sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
from scipy.stats import ttest_rel
from sklearn.ensemble import HistGradientBoostingRegressor
C=r'C:\Clark\YZU\Papper\Code\dev\ml_formation\cache'
K=['Period_Start','Ticker_A','Ticker_B']
es=pd.read_parquet(f'{C}/exit_scan.parquet'); caps=[c for c in es.columns if c.startswith('cap_e')]
tr=pd.read_parquet(f'{C}/train.parquet')
d=tr.merge(es[K+caps],on=K,how='inner')
d=d[(d.adf_pass==1)&d.label_valid].copy()
d['r']=d.groupby('Period_Start').SSD.rank(method='first')
per=sorted(d.Period_Start.unique()); pidx={p:i for i,p in enumerate(per)}
d['pi']=d.Period_Start.map(pidx)
Y=np.where(np.isnan(d[caps].to_numpy(dtype=float)),0.0,d[caps].to_numpy(dtype=float))
drop=set(K+caps+['Trade_Start','Group','r','pi','label_valid','not_converged','capture_frac',
                 'entry_day','conv_day','days_to_conv','n_cross_trade','z_end','z_entry','valid_days'])
feats=[c for c in d.columns if c not in drop and pd.api.types.is_numeric_dtype(d[c])]
X=d[feats].to_numpy(dtype=np.float32)
print('特徵 %d 維、動作 %d 個、樣本 %d 列、%d 期'%(len(feats),len(caps),len(d),len(per)))
WARM,RETRAIN,PURGE=36,12,6
pi=d.pi.values; top5=(d.r<=5).values
rows=[]; models=None; last=-999
for t in range(WARM,len(per)):
    if models is None or t-last>=RETRAIN:
        m=pi<=t-1-PURGE
        if m.sum()<500: continue
        models=[]
        for j in range(len(caps)):
            g=HistGradientBoostingRegressor(max_iter=120,max_depth=4,learning_rate=0.06,
                                            min_samples_leaf=40,random_state=0)
            g.fit(X[m],Y[m,j]); models.append(g)
        last=t
    sel=(pi==t)&top5
    if sel.sum()==0: continue
    P=np.column_stack([g.predict(X[sel]) for g in models])
    pick=np.argmax(P,axis=1)
    got=Y[sel][np.arange(sel.sum()),pick]
    # 擴張窗最佳常數（只用 t 之前）
    hist=pi<=t-1
    bc=int(np.argmax(Y[hist].mean(axis=0))) if hist.sum()>0 else 0
    rows.append({'pi':t,'model':got.mean(),'const_exp':Y[sel][:,bc].mean(),
                 'cur':Y[sel][:,caps.index('cap_e2.0_x0.0')].mean(),
                 'oracle':Y[sel].max(axis=1).mean(),'bc':caps[bc]})
R=pd.DataFrame(rows)
print('評估期數 %d'%len(R))
print()
print('=== 門檻檢定 ===')
t_,p_=ttest_rel(R.model,R.const_exp)
print('  模型            每期 %+.6f'%R.model.mean())
print('  擴張窗最佳常數  每期 %+.6f   （最常選中: %s）'%(R.const_exp.mean(),R.bc.mode().iloc[0]))
print('  → 差 %+.6f   配對t p %.4f   %s'%(R.model.mean()-R.const_exp.mean(),p_,
      '✔ 過閘' if (p_<0.05 and R.model.mean()>R.const_exp.mean()) else '✘ 未過'))
print()
print('  附帶：現行 e2.0_x0.0 每期 %+.6f'%R.cur.mean())
print('        完美預知          每期 %+.6f'%R.oracle.mean())
cap=(R.model.mean()-R.const_exp.mean())/(R.oracle.mean()-R.const_exp.mean())
print('        模型吃到天花板的 %.1f%%'%(cap*100))
