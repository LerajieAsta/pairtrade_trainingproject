# -*- coding: utf-8 -*-
"""方案 A：模型當否決權。依 dev/ml_formation/PREREGISTRATION_FILTER.md 執行。"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score
C=r'C:\Clark\YZU\Papper\Code\dev\ml_formation\cache'

pool=pd.read_parquet(f'{C}/pool.parquet',
      columns=['Period_Start','Ticker_A','Ticker_B','SSD','adf_pass',
               'label_valid','not_converged','capture_frac'])
s1=pd.read_parquet(f'{C}/scores.parquet')
s3=pd.read_parquet(f'{C}/m3_scores.parquet')
K=['Period_Start','Ticker_A','Ticker_B']
d=pool.merge(s1,on=K,how='inner').merge(s3,on=K,how='inner')
print('合併後 %d 列，%d 期'%(len(d), d.Period_Start.nunique()))

# 暖身 36 期無分數（前推協定所致，與梯子相同）：一律排除
d=d[(d.adf_pass==1)&(d.label_valid)].copy()
_n0=len(d)
d=d.dropna(subset=['score_M1','score_M3','capture_frac'])
print('去除暖身期／無效捕獲後：%d 列（-%d），%d 期'%(len(d),_n0-len(d),d.Period_Start.nunique()))
print('ADF 通過且標籤有效：%d 列，%d 期'%(len(d), d.Period_Start.nunique()))
d['ssd_rank']=d.groupby('Period_Start').SSD.rank(method='first')

# ── 門檻一：模型在 SSD 前 40 名內的 AUC ─────────────────────────
print()
print('=== 門檻一：SSD 前 40 名內的逐期 AUC（平均 − 1SE > 0.50）===')
top40=d[d.ssd_rank<=40]
g1={}
for m in ['score_M1','score_M3']:
    a=[]
    for p,g in top40.groupby('Period_Start'):
        y=g.not_converged.values
        if 0<y.sum()<len(y): a.append(roc_auc_score(y,g[m].values))
    a=np.array(a); se=a.std(ddof=1)/np.sqrt(len(a))
    g1[m]=(a.mean(),se,a.mean()-se,len(a))
    print('  %-9s  期數 %3d  AUC %.4f ± %.4f   下界 %.4f   %s'%(
        m,len(a),a.mean(),se,a.mean()-se,'通過' if a.mean()-se>0.50 else '未過'))

# ── 門檻二：否決後 top-20 的 capture_frac ────────────────────────
print()
print('=== 門檻二：否決後前 20 對的每期平均 capture_frac ===')
NS=[20,30,40,60,100]
res={}
for m in ['score_M1','score_M3']:
    per={}
    for N in NS:
        sub=d[d.ssd_rank<=N]
        if N==20:
            sel=sub
        else:
            sel=sub.sort_values([ 'Period_Start',m]).groupby('Period_Start').head(20)
        per[N]=sel.groupby('Period_Start').capture_frac.mean()
    per=pd.DataFrame(per).dropna()
    res[m]=per
    base=per[20]
    print('  --- %s ---  共同期數 %d'%(m,len(per)))
    print('     N   平均capture   對N=20的差    配對t p      Bonferroni(0.0125)')
    for N in NS:
        if N==20:
            print('    %3d   %+.6f      (基準)'%(N,base.mean())); continue
        diff=per[N]-base
        t,p=ttest_rel(per[N],base)
        print('    %3d   %+.6f     %+.6f      %.4f        %s'%(
            N,per[N].mean(),diff.mean(),p,'✔' if (p<0.0125 and diff.mean()>0) else '✘'))

# ── 附帶：未收斂率（不作判準）────────────────────────────────
print()
print('=== 附帶報告：否決後 top-20 未收斂率（梯子基準 0.4266，不作判準）===')
for m in ['score_M1','score_M3']:
    line=[]
    for N in NS:
        sub=d[d.ssd_rank<=N]
        sel=sub if N==20 else sub.sort_values(['Period_Start',m]).groupby('Period_Start').head(20)
        line.append('N=%d %.4f'%(N,sel.groupby('Period_Start').not_converged.mean().mean()))
    print('  %-9s %s'%(m,'  '.join(line)))
