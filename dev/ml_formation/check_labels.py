"""驗證向量化標籤與交易引擎的實際結果是否一致。

引擎一期內可多次進出，PERIOD_END_EXIT 指最後一筆未平倉；本標籤講的是
第一次進場是否收斂。故比對對象是該配對-期的**首筆**交易的 Status。
"""
import sys, sqlite3; sys.path.insert(0, ".")
import numpy as np, pandas as pd

SID_T = 'tiingo/Grid_GICS_SSD/TradeLogs_Top20_SL0_ZWin0_MSR0.csv'
P = pd.read_parquet("dev/ml_formation/cache/pool.parquet")
print(f"池 {len(P):,} 列   label_valid {P.label_valid.mean():.4f}   "
      f"其中 not_converged {P[P.label_valid].not_converged.mean():.4f}")

rc = sqlite3.connect('file:results/result.db?mode=ro', uri=True)
tl = pd.read_sql_query(
    "SELECT Date, Ticker_A, Ticker_B, Period_Start, Status, Trade_PnL "
    "FROM trade_logs WHERE strategy_id=? AND Status IN ('EXIT','PERIOD_END_EXIT') "
    "AND Trade_PnL<>0 ORDER BY Ticker_A, Ticker_B, Period_Start, Date", rc, params=(SID_T,))
first = tl.groupby(['Ticker_A', 'Ticker_B', 'Period_Start'], as_index=False).first()
first['Trade_Start'] = first.Period_Start          # trade_logs 的 Period_Start 即交易期起日
first['eng_not_conv'] = (first.Status == 'PERIOD_END_EXIT').astype(int)
print(f"引擎首筆交易 {len(first):,} 筆   強平比例 {first.eng_not_conv.mean():.4f}")

M = P[P.label_valid].merge(
    first[['Ticker_A', 'Ticker_B', 'Trade_Start', 'eng_not_conv', 'Status']],
    on=['Ticker_A', 'Ticker_B', 'Trade_Start'], how='inner')
print(f"\n對上 {len(M):,} 筆")
agree = (M.not_converged == M.eng_not_conv).mean()
print(f"標籤一致率 {100*agree:.2f}%")
print(pd.crosstab(M.not_converged, M.eng_not_conv,
                  rownames=['我的 not_converged'], colnames=['引擎強平']).to_string())

bad = M[M.not_converged != M.eng_not_conv]
if len(bad):
    print(f"\n不一致 {len(bad)} 筆，抽樣：")
    print(bad[['Period_Start','Ticker_A','Ticker_B','entry_day','conv_day',
               'days_to_conv','z_entry','z_end','Status']].head(10).to_string(index=False))
