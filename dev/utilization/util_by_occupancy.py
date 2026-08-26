# -*- coding: utf-8 -*-
"""低占用日加碼的風險面：每部位-日報酬的均值與波動。"""
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
    occ=d.assign(on=(d.Position!=0).astype(int)).groupby('Date').on.sum()
    pnl=d.groupby('Date').Daily_Delta.sum()
    df=pd.DataFrame({'occ':occ,'pnl':pnl}).dropna()
    df=df[df.occ>0].copy()
    cap=INITIAL_CAPITAL/(tn*6)                 # 每部位名目額（近似，以初始資金）
    df['r_pos']=df.pnl/(df.occ*cap)            # 每部位-日報酬率
    print('=== %s ==='%lab)
    print('  占用帶      日數   每部位日報酬均值    std      年化Sharpe(單部位)')
    q=df.occ.quantile([0,.2,.4,.6,.8,1.0]).values
    for i in range(5):
        b=df[(df.occ>=q[i])&(df.occ<=q[i+1])]
        if len(b)<50: continue
        m,s=b.r_pos.mean(),b.r_pos.std(ddof=1)
        print('  %5.0f–%-5.0f  %5d    %+.6f   %.6f      %6.3f'%(
            q[i],q[i+1],len(b),m,s,m/s*np.sqrt(TD) if s>0 else np.nan))
    print()
    # 反事實：把每日部位規模改為 equity*目標曝險/當日占用數（即固定總曝險）
    for target in [0.7,0.85,1.0]:
        scale=(target*INITIAL_CAPITAL/(df.occ*cap)).clip(upper=3.0)   # 上限 3x 防爆
        p2=df.pnl*scale
        eq=INITIAL_CAPITAL+p2.cumsum()
        r=eq.pct_change().dropna()
        ann=(eq.iloc[-1]/INITIAL_CAPITAL)**(TD/len(r))-1
        sh=r.mean()/r.std(ddof=1)*np.sqrt(TD)
        mdd=(eq/eq.cummax()-1).min()
        print('  固定總曝險 %.0f%%：年化 %+.3f%%  Sharpe %.3f  MDD %.2f%%  平均倍數 %.2fx'%(
            target*100,ann*100,sh,mdd*100,scale.mean()))
    base=INITIAL_CAPITAL+df.pnl.cumsum(); rb=base.pct_change().dropna()
    print('  【現行】            年化 %+.3f%%  Sharpe %.3f  MDD %.2f%%'%(
        ((base.iloc[-1]/INITIAL_CAPITAL)**(TD/len(rb))-1)*100,
        rb.mean()/rb.std(ddof=1)*np.sqrt(TD),(base/base.cummax()-1).min()*100))
    print()
