"""
研究框架次步 #6：Regime 分層評估 + Break-even 成本表 + Deflated Sharpe Ratio
======================================================================

對 results/result.db 內已回測的策略做三項穩健性評估（純分析層，不重跑回測）：

1. Break-even 成本表（Do & Faff 2012 精神）
   每策略在往返成本升到多少時淨利歸零。成本模型精確可解：進出場手續費 =
   friction × 名目額，且 _execute_entry 中 v_a+v_b = capital_per_pair（名目額
   恰等於每配對資金），故總費用 F0 = friction × capital_per_pair × (進場+出場次數)。
   單邊 break-even c* = 現行單邊費(0.29%) + 淨利/Σ名目額；往返 = 2c*。
   c* 越高 = 策略越能承受摩擦成本 = 越穩健。

2. Regime 分層 Sharpe
   以等權市場（price DB 全體日報酬均值）的滾動波動率三分位（Calm/Normal/
   Turbulent）與 126 日趨勢（Bull/Bear）標記每個交易日，分層計算各策略年化
   Sharpe，檢驗績效是否集中在特定 regime（如僅高波動期獲利）。

3. Deflated Sharpe Ratio（Bailey & López de Prado 2014）
   我們在每策略族挑 15 個配置（TopN×停損）中選最佳 → 選擇偏誤使 Sharpe 虛高。
   DSR 在給定「試驗次數 N、試驗間 Sharpe 變異、報酬偏態/峰態、樣本長度」下，
   給出「真實 Sharpe > 0」的機率，校正多重檢定。DSR<0.95 = 無法在選擇偏誤下
   宣稱顯著為正。

用法：python -m analysis.regime_cost_dsr_eval  [輸出報表 + results/analysis/*.csv]
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import norm

from strategies import returns as strategy_returns
from strategies.config import DB_PATH, TABLE_NAME, INITIAL_CAPITAL

RESULT_DB = "results/result.db"
OUT_DIR = "results/analysis"
CURRENT_FEE_SIDE = 0.0029          # 現行單邊摩擦（0.29%）
TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


# ── 市場 regime 標記 ────────────────────────────────────────────────
def build_market_regimes(price_db: str = DB_PATH, table: str = TABLE_NAME) -> pd.DataFrame:
    """等權市場日報酬 → 波動率三分位 + 126 日趨勢，回傳 DataFrame(index=Date)。"""
    con = sqlite3.connect(price_db)
    # Daily_Prices 寬表或長表？先探測欄位
    # 長表的識別欄名依資料源而異（Tiingo 用 Symbol，其他用 Ticker），
    # 兩者皆須支援——否則長表會被誤判為寬表，set_index("Date") 保留重複日期，
    # 後續 rolling 與 join 全部失效（regime 標記變成逐筆列而非逐日）。
    cols = pd.read_sql(f"SELECT * FROM {table} LIMIT 1", con).columns.tolist()
    id_col = next((c for c in ("Ticker", "Symbol") if c in cols), None)
    px_col = next((c for c in ("Adj_Close", "Close") if c in cols), None)
    if id_col and px_col and "Date" in cols:                               # 長表
        px = pd.read_sql(f"SELECT Date, {id_col}, {px_col} FROM {table}", con)
        con.close()
        px["Date"] = pd.to_datetime(px["Date"])
        wide = px.pivot_table(index="Date", columns=id_col, values=px_col)
    else:                                                                  # 寬表（Date + 各 ticker 欄）
        px = pd.read_sql(f"SELECT * FROM {table}", con)
        con.close()
        px["Date"] = pd.to_datetime(px["Date"])
        wide = px.set_index("Date").select_dtypes("number")

    if not wide.index.is_unique:
        raise ValueError(f"regime 日期索引不唯一（{len(wide)} 列 / "
                         f"{wide.index.nunique()} 個日期）——價格表格式判定有誤")

    wide = wide.replace(0.0, np.nan)                                       # 防 log(0)=-inf
    rets = np.log(wide).diff()
    mkt = rets.mean(axis=1).dropna()                                       # 等權市場日報酬
    vol = mkt.rolling(63).std() * np.sqrt(TRADING_DAYS)                    # 年化滾動波動
    trend = mkt.rolling(126).sum()                                         # 126 日累積報酬

    df = pd.DataFrame({"mkt_ret": mkt, "vol": vol, "trend": trend}).dropna()
    q1, q2 = df["vol"].quantile([1/3, 2/3])
    df["vol_regime"] = np.where(df["vol"] <= q1, "Calm",
                        np.where(df["vol"] <= q2, "Normal", "Turbulent"))
    df["trend_regime"] = np.where(df["trend"] >= 0, "Bull", "Bear")
    return df


# ── 每策略最佳配置的日報酬序列 ──────────────────────────────────────
def load_daily_returns(strategy_ids: list[str], result_db: str = RESULT_DB) -> dict:
    """
    回傳 {strategy_id: pd.Series(日報酬, index=Date)}。

    2026-08-04：改由 strategies.returns 供應，口徑自「ΣDaily_Delta / 初始資金」
    （單利）改為「Daily_Delta / 前一日權益」（複利）。

    原本這張表在同一列裡混用兩種口徑：「Sharpe」欄取自 strategy_summaries
    （db_utils 算的複利值），而「門檻SR0」與「DSR」由本函式的單利序列算出，
    連 _trial_specs 的 var_sr 都取自複利的 Sharpe_Raw 橫斷面變異——等於拿複利
    離散度導出的門檻去比單利的 Sharpe。改用複利後三者落在同一個定義上。

    複利亦是引擎實際的行為：portfolio_manager.allocate_capital 以
    current_equity / max_pairs 決定部位規模，故承擔風險的資本本來就隨權益走。

    序列的生命期由該策略自己的首末交易日界定（見 strategies.returns），
    上線前不補零——舊實作對晚上線的策略族（如 2009 才有資料的 F09）會被
    上游的聯集補零稀釋 |Sharpe|。此處 dropna 後的形狀與舊版相同：
    index 只含該策略存在的交易日。
    """
    daily = strategy_returns.daily_returns(strategy_ids, result_db=result_db)
    return {sid: daily[sid].dropna() for sid in daily.columns}


# ── Deflated Sharpe Ratio ───────────────────────────────────────────
def deflated_sharpe(daily_ret: pd.Series, n_trials: int, var_sr_trials: float) -> dict:
    """
    Bailey & López de Prado (2014)。daily_ret：日報酬序列。
    n_trials：選擇時試過的獨立配置數。var_sr_trials：試驗間（每期）Sharpe 變異。
    回傳 dict(SR_ann, SR0_ann, DSR, T)。
    """
    r = daily_ret.dropna().values
    T = len(r)
    if T < 20 or r.std(ddof=1) == 0:
        return {"SR_ann": np.nan, "SR0_ann": np.nan, "DSR": np.nan, "T": T}

    sr = r.mean() / r.std(ddof=1)                                          # 每日 Sharpe（非年化）
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurt()) + 3.0                                # pandas kurt 為超額，還原

    # 選擇偏誤下的期望最大 Sharpe（每日尺度）
    if n_trials < 2 or var_sr_trials <= 0:
        sr0 = 0.0
    else:
        std_sr = np.sqrt(var_sr_trials)
        z1 = norm.ppf(1.0 - 1.0 / n_trials)
        z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        sr0 = std_sr * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)

    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2))
    dsr = float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom))
    return {"SR_ann": sr * np.sqrt(TRADING_DAYS),
            "SR0_ann": sr0 * np.sqrt(TRADING_DAYS),
            "DSR": dsr, "T": T}


# ── 試驗宇宙：DSR 的 N 與 var_sr 必須描述同一組試驗 ──────────────────
# 2026-07-29 修正：原本 n_trials = 該 METHOD 底下的列數（＝15 個參數格），
# 大幅低估選擇偏誤——result.db 實際有 87 個相異 METHOD、1,435 個回測配置，
# 其中一大票是試過後封存的負面結果（MST、PCA-Loadings-ResidFDR、F09 結構性
# 財報、SEC-PIT-Beta、動量特徵、adf/pca 掃描…）。這些全部都是
# researcher degrees of freedom，依 Bailey & López de Prado (2014) 必須計入 N。
#
# 三種口徑各自內部自洽（N 與 var_sr 取自同一集合），主表用 SPEC_MAIN：
#   cells  N=15    ：僅該策略的 15 個參數格，var 取格間（舊行為，保留供對照）
#   method N=87    ：相異 METHOD 數，var 取「各 METHOD 最佳 Sharpe」之間
#   config N=1,435 ：全部回測配置，var 取所有配置 Sharpe 之間
#
# N=87 為主口徑：一個 METHOD 內的 15 格高度相關（共用配對、僅組合設定不同），
# 不宜各算一次試驗；但每個相異 METHOD 代表一次真正獨立的建模決策。
SPEC_MAIN = "method"


def _trial_specs(summ: pd.DataFrame, method: str) -> dict:
    """
    回傳 {spec: (n_trials, var_sr_daily)}。變異數換算到每日尺度（÷252）。

    var_sr 取自 Sharpe_Raw 的橫斷面變異，而 Sharpe_Raw 是複利口徑；自
    2026-08-04 起 load_daily_returns 也是複利，兩者才落在同一尺度上
    （在此之前門檻由複利離散度導出、卻套在單利的 Sharpe 上）。
    """
    def _var(vals) -> float:
        v = pd.Series(vals).dropna()
        return float(np.var(v, ddof=1)) / TRADING_DAYS if len(v) > 1 else 0.0

    g = summ[summ.METHOD == method]
    per_method_best = summ.groupby("METHOD").Sharpe_Raw.max()
    return {
        "cells":  (int(len(g)), _var(g.Sharpe_Raw)),
        "method": (int(summ.METHOD.nunique()), _var(per_method_best)),
        "config": (int(len(summ)), _var(summ.Sharpe_Raw)),
    }


# ── 主流程 ──────────────────────────────────────────────────────────
def _top_n_int(v) -> int:
    return int(str(v).replace("Top", "").strip())


def run(methods: list[str] = None):
    os.makedirs(OUT_DIR, exist_ok=True)
    con = sqlite3.connect(RESULT_DB)
    summ = pd.read_sql("SELECT * FROM strategy_summaries", con)
    con.close()

    if methods is None:
        # 預設：所有 Z-Score 誠實策略（排除 formation-only 與純 DRL 對照）
        methods = [m for m in summ.METHOD.unique() if "DRL" not in m]

    # 每策略最佳配置（Sharpe 最高）。試驗間 Sharpe 變異改由 _trial_specs 依
    # 各口徑同步計算，確保 N 與 var_sr 永遠取自同一組試驗。
    best_rows = {}
    for m in methods:
        g = summ[summ.METHOD == m]
        if g.empty:
            continue
        best_rows[m] = g.loc[g.Sharpe_Raw.idxmax()]

    regimes = build_market_regimes()
    sids = [r["_path"] for r in best_rows.values()]
    daily = load_daily_returns(sids)

    # ── 表 1：Break-even 成本 + DSR ──
    rows1 = []
    for m, br in best_rows.items():
        top_n = _top_n_int(br["TOP N"])
        cap_per_pair = INITIAL_CAPITAL / top_n
        entries, exits = float(br["Entries"]), float(br["Exits"])
        sigma_notional = cap_per_pair * (entries + exits)
        p_net = float(br["Final_Equity"]) - INITIAL_CAPITAL
        c_side_be = CURRENT_FEE_SIDE + p_net / sigma_notional if sigma_notional > 0 else np.nan
        rt_be = 2.0 * c_side_be                                            # 往返 break-even

        ret = daily.get(br["_path"], pd.Series(dtype=float))
        specs = _trial_specs(summ, m)
        dsrs = {k: deflated_sharpe(ret, n, v) for k, (n, v) in specs.items()}
        main = dsrs[SPEC_MAIN]
        rows1.append({
            "策略": m, "最佳配置": f"Top{top_n}/SL{br['STOP LOSS %']}",
            "Sharpe": round(float(br["Sharpe_Raw"]), 3),
            "淨利$": round(p_net, 0),
            "往返break-even%": round(rt_be * 100, 3),
            "成本餘裕(vs0.58%)": round((rt_be - 0.0058) * 100, 3),
            "門檻SR0": round(main["SR0_ann"], 3),
            "DSR": round(main["DSR"], 3),
            "DSR顯著(>0.95)": "✔" if main["DSR"] >= 0.95 else "✘",
            # 敏感度三欄：N 的選擇是判斷，故三種口徑並列，讓讀者自行檢視
            "DSR@N15": round(dsrs["cells"]["DSR"], 3),
            "DSR@N87": round(dsrs["method"]["DSR"], 3),
            "DSR@N1435": round(dsrs["config"]["DSR"], 3),
            "N試驗": specs[SPEC_MAIN][0], "T日": main["T"],
        })
    tbl1 = pd.DataFrame(rows1).sort_values("Sharpe", ascending=False)

    # ── 表 2：Regime 分層 Sharpe ──
    rows2 = []
    for m, br in best_rows.items():
        s = daily.get(br["_path"])
        if s is None or s.empty:
            continue
        j = pd.DataFrame({"ret": s}).join(regimes[["vol_regime", "trend_regime"]], how="inner")
        rec = {"策略": m}
        for reg_col, labels in [("vol_regime", ["Calm", "Normal", "Turbulent"]),
                                ("trend_regime", ["Bull", "Bear"])]:
            for lab in labels:
                sub = j[j[reg_col] == lab]["ret"]
                if len(sub) > 20 and sub.std(ddof=1) > 0:
                    rec[lab] = round(sub.mean() / sub.std(ddof=1) * np.sqrt(TRADING_DAYS), 2)
                else:
                    rec[lab] = np.nan
        rows2.append(rec)
    tbl2 = pd.DataFrame(rows2)

    # ── 輸出 ──
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n" + "=" * 90)
    print("表 1：Break-even 成本 + Deflated Sharpe（每策略最佳配置）")
    print("=" * 90)
    print(tbl1.to_string(index=False))
    print("\n" + "=" * 90)
    print("表 2：Regime 分層年化 Sharpe（波動率三分位 | 趨勢）")
    print("=" * 90)
    print(tbl2.to_string(index=False))

    tbl1.to_csv(f"{OUT_DIR}/breakeven_dsr.csv", index=False, encoding="utf-8-sig")
    tbl2.to_csv(f"{OUT_DIR}/regime_sharpe.csv", index=False, encoding="utf-8-sig")
    print(f"\n[已存] {OUT_DIR}/breakeven_dsr.csv, regime_sharpe.csv")
    return tbl1, tbl2


if __name__ == "__main__":
    run()
