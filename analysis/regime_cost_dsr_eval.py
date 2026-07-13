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
    cols = pd.read_sql(f"SELECT * FROM {table} LIMIT 1", con).columns.tolist()
    if "Ticker" in cols and "Close" in cols and "Date" in cols:            # 長表
        px = pd.read_sql(f"SELECT Date, Ticker, Close FROM {table}", con)
        con.close()
        px["Date"] = pd.to_datetime(px["Date"])
        wide = px.pivot_table(index="Date", columns="Ticker", values="Close")
    else:                                                                  # 寬表（Date + 各 ticker 欄）
        px = pd.read_sql(f"SELECT * FROM {table}", con)
        con.close()
        px["Date"] = pd.to_datetime(px["Date"])
        wide = px.set_index("Date").select_dtypes("number")

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
    """回傳 {strategy_id: pd.Series(日報酬, index=Date)}；報酬 = ΣDaily_Delta / 初始資金。"""
    con = sqlite3.connect(result_db)
    placeholders = ",".join("?" * len(strategy_ids))
    q = (f"SELECT strategy_id, Date, SUM(Daily_Delta) AS pnl "
         f"FROM trade_logs WHERE strategy_id IN ({placeholders}) "
         f"GROUP BY strategy_id, Date")
    df = pd.read_sql(q, con, params=strategy_ids)
    con.close()
    df["Date"] = pd.to_datetime(df["Date"])
    out = {}
    for sid, g in df.groupby("strategy_id"):
        s = g.set_index("Date")["pnl"].sort_index() / INITIAL_CAPITAL
        out[sid] = s
    return out


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

    # 每策略最佳配置（Sharpe 最高）＋ 試驗間 Sharpe 變異（該族 15 配置）
    best_rows, var_sr = {}, {}
    for m in methods:
        g = summ[summ.METHOD == m]
        if g.empty:
            continue
        best_rows[m] = g.loc[g.Sharpe_Raw.idxmax()]
        # 試驗間 Sharpe 變異需與 deflated_sharpe 內的「每日」Sharpe 同尺度：
        # summaries 存年化 Sharpe，SR_daily = SR_ann/sqrt(252) → var 除以 252。
        var_ann = float(np.var(g.Sharpe_Raw.dropna(), ddof=1)) if len(g) > 1 else 0.0
        var_sr[m] = var_ann / TRADING_DAYS

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

        n_trials = int((summ.METHOD == m).sum())
        dsr = deflated_sharpe(daily.get(br["_path"], pd.Series(dtype=float)),
                              n_trials, var_sr[m])
        rows1.append({
            "策略": m, "最佳配置": f"Top{top_n}/SL{br['STOP LOSS %']}",
            "Sharpe": round(float(br["Sharpe_Raw"]), 3),
            "淨利$": round(p_net, 0),
            "往返break-even%": round(rt_be * 100, 3),
            "成本餘裕(vs0.58%)": round((rt_be - 0.0058) * 100, 3),
            "DSR": round(dsr["DSR"], 3),
            "DSR顯著(>0.95)": "✔" if dsr["DSR"] >= 0.95 else "✘",
            "N試驗": n_trials, "T日": dsr["T"],
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
