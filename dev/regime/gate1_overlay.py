# -*- coding: utf-8 -*-
"""門檻一：regime 曝險疊加 vs 循環 block 置換。依 dev/regime/PREREGISTRATION.md。"""
import sys, sqlite3, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
from strategies.metrics import metrics_from_returns
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from strategies.config import DB_PATH, TABLE_NAME
from strategies.returns import daily_returns
TD=252; RDB=r'C:\Clark\YZU\Papper\Code\results\result.db'

# ── 無前視的 regime gate（比照 run_trading._build_dispersion_gate 的寫法）──
con=sqlite3.connect(DB_PATH)
px=pd.read_sql(f"SELECT Date,Symbol,COALESCE(Adj_Close,Close) p FROM {TABLE_NAME}",con); con.close()
px['Date']=pd.to_datetime(px.Date)
w=px.pivot_table(index='Date',columns='Symbol',values='p').replace(0.0,np.nan)
mkt=np.log(w).diff().mean(axis=1)
vol=(mkt.rolling(63,min_periods=40).std()*np.sqrt(TD)).shift(1)      # 只用昨日以前
r=vol.expanding(min_periods=504).rank(); n=vol.expanding(min_periods=504).count()
pct=(r/n)
print('regime 訊號：%d 日，暖身後 %d 日'%(len(pct),pct.notna().sum()))

_T='tiingo/Grid_%s/TradeLogs_%s_ZWin0_MSR0.csv'
TOPS=['Top1','Top3','Top5','Top10','Top20']; SLS=['SL0','SL5','SL15']
ids=[_T%('NOGRP_DTW',f'{t}_{s}') for t in TOPS for s in SLS]
R=daily_returns(ids, result_db=RDB)
# 三檔停損等權 → 再對 top_n 等權（比照論文交易層主檢定）
cols=[]
for t in TOPS:
    cols.append(pd.concat([R[_T%('NOGRP_DTW',f'{t}_{s}')] for s in SLS],axis=1).mean(axis=1))
ret=pd.concat(cols,axis=1).mean(axis=1).dropna()
print('基準逐日報酬 %d 日'%len(ret))

def ann(x):
    return metrics_from_returns(pd.Series(np.asarray(x,float)))['Ann_Ret_Raw']

def circ_perm(g, L, B, rng):
    """循環 block 置換：保留跳過率與叢集結構，只打亂時點。"""
    n=len(g); nb=int(np.ceil(n/L))
    out=np.empty((B,n),dtype=bool)
    for b in range(B):
        starts=rng.integers(0,n,nb)
        idx=np.concatenate([(np.arange(s,s+L)%n) for s in starts])[:n]
        out[b]=g[idx]
    return out

print()
print('=== 門檻一：曝險疊加 vs 循環 block 置換（L=63、1,000 次）===')
print('  pctl  阻擋日%   實際年化%   置換中位%   置換p     判定')
rng=np.random.default_rng(20260826)
for pctl in [50,67]:
    gate=(pct>=pctl/100.0)
    gate=gate.where(pct.notna(),True)                    # 暖身期一律允許
    g=gate.reindex(ret.index).fillna(True).values.astype(bool)
    obs=ann(np.where(g,ret.values,0.0))
    P=circ_perm(g,63,1000,rng)
    nul=np.array([ann(np.where(P[b],ret.values,0.0)) for b in range(1000)])
    p=float((nul>=obs).mean())
    print('  %4d  %6.1f%%   %+8.3f   %+8.3f   %.4f   %s'%(
        pctl,(~g).mean()*100,obs*100,np.median(nul)*100,p,'✔' if p<0.025 else '✘'))
print()
print('  對照：不疊加（全額）年化 %+.3f%%'%(ann(ret.values)*100))
