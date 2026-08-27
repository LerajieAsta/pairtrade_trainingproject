# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'C:\Clark\YZU\Papper\Code')
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from dev.ml_formation.selection import (selection_region, with_model_scores,
                                        RANK_COL, WARM_UP_PERIODS, _assert_region)
ok=[]
def chk(n,c,d=''):
    ok.append(c); print(('  PASS  ' if c else '  FAIL  ')+n+(('  '+d) if d else ''))

r=selection_region()
chk('selection_region() 非空', len(r)>0, f'{len(r):,} 列 / {r.Period_Start.nunique()} 期')
chk(f'含 {RANK_COL} 欄', RANK_COL in r.columns)
g=r.groupby('Period_Start')[RANK_COL]
chk('每期由 1 起連續', np.allclose(g.min(),1.0) and np.allclose(g.max(),g.size()))
chk('等價於舊寫法', len(r)==len(pd.read_parquet(r'C:\Clark\YZU\Papper\Code\dev\ml_formation\cache\pool.parquet')
      .query('adf_pass==1 and label_valid')))

rf=selection_region(with_features=True)
chk('with_features=True 列數相同', len(rf)==len(r), f'{len(rf):,}')
chk('with_features=True 欄較多', len(rf.columns)>len(r.columns), f'{len(rf.columns)} vs {len(r.columns)}')

s=with_model_scores(r)
chk('with_model_scores 排除暖身', s.pi.min()>=WARM_UP_PERIODS, f'pi 最小 {s.pi.min()}')
chk('分數無 NaN', s.score_M1.notna().all() and s.score_M3.notna().all())
chk('等價於舊的 dropna 副作用',
    len(s)==len(r.merge(pd.read_parquet(r'C:\Clark\YZU\Papper\Code\dev\ml_formation\cache\scores.parquet'),
                        on=['Period_Start','Ticker_A','Ticker_B'])
                .merge(pd.read_parquet(r'C:\Clark\YZU\Papper\Code\dev\ml_formation\cache\m3_scores.parquet').drop(columns=['pi']),
                       on=['Period_Start','Ticker_A','Ticker_B'])
                .dropna(subset=['score_M1','score_M3'])),
    f'{len(s):,} 列')

print('\n斷言是否真的擋得住：')
for name, mutate in [
    ('空表',            lambda d: d.iloc[0:0]),
    ('缺 rank 欄',      lambda d: d.drop(columns=[RANK_COL])),
        ('排名後刪掉第 1 名（min 變 2）', lambda d: d[d[RANK_COL] != 1]),
    ('排名後刪掉第 2 名（留空洞）',   lambda d: d[d[RANK_COL] != 2]),
    ('只剩少數期',       lambda d: d[d.Period_Start.isin(sorted(d.Period_Start.unique())[:10])]),
]:
    try:
        _assert_region(mutate(r.head(200000).copy()), 'test'); chk(name+' → 應拋錯', False, '沒擋住')
    except ValueError as e:
        chk(name+' → 擋下', True, str(e)[:44])
print('\n%d / %d'%(sum(ok),len(ok)))
sys.exit(0 if all(ok) else 1)
