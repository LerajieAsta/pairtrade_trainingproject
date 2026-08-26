# -*- coding: utf-8 -*-
"""前視檢驗：用「當日占用」定規模等於偷看。改用落後資訊重測。"""
import sys, sqlite3, numpy as np, pandas as pd
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from strategies.config import INITIAL_CAPITAL
TD=252
c=sqlite3.connect(r'file:C:\Clark\YZU\Papper\Code\results\result.db?mode=ro',uri=True)
for sub,tn,lab in [('Grid_NOGRP_SDP',5,'NOGRP-SDP Top5'),('Grid_NOGRP_SSD',5,'NOGRP-SSD Top5')]:
    sid='tiingo/%s/TradeLogs_Top%d_SL0_ZWin0_MSR0.csv'%(sub,tn)
    d=pd.read_sql('select Date,Position,Daily_Delta from trade_logs where strategy_id=?',c,params=(sid,))
    d['Date']=pd.to_datetime(d.Date)
    occ=d.assign(on=(d.Position!=0).astype(int)).groupby('Date').on.sum().sort_index()
    pnl=d.groupby('Date').Daily_Delta.sum().sort_index()
    cap=INITIAL_CAPITAL/(tn*6)
    df=pd.DataFrame({'occ':occ,'pnl':pnl}).dropna(); df=df[df.occ>0]
    print('=== %s ==='%lab)
    base=INITIAL_CAPITAL+df.pnl.cumsum(); rb=base.pct_change().dropna()
    print('  【現行】               年化 %+.3f%%  Sharpe %.3f  MDD %.2f%%'%(
        ((base.iloc[-1]/INITIAL_CAPITAL)**(TD/len(rb))-1)*100,
        rb.mean()/rb.std(ddof=1)*np.sqrt(TD),(base/base.cummax()-1).min()*100))
    for lagname, basis in [('當日占用（有前視）', df.occ),
                           ('落後 1 日', df.occ.shift(1)),
                           ('落後 21 日', df.occ.shift(21)),
                           ('落後 63 日均', df.occ.shift(1).rolling(63).mean()),
                           ('擴張窗歷史均（無前視）', df.occ.shift(1).expanding(252).mean())]:
        b=basis.reindex(df.index).ffill().bfill().clip(lower=1)
        scale=(0.85*INITIAL_CAPITAL/(b*cap)).clip(upper=3.0)
        p2=df.pnl*scale
        eq=INITIAL_CAPITAL+p2.cumsum(); r=eq.pct_change().dropna()
        ann=(eq.iloc[-1]/INITIAL_CAPITAL)**(TD/len(r))-1
        sh=r.mean()/r.std(ddof=1)*np.sqrt(TD)
        mdd=(eq/eq.cummax()-1).min()
        print('  85%% 曝險 / %-22s 年化 %+.3f%%  Sharpe %.3f  MDD %.2f%%  均倍數 %.2fx'%(
            lagname,ann*100,sh,mdd*100,scale.mean()))
    print()
