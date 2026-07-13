"""
敏感性分析報表（OFAT）
======================================================================
讀 results/result.db，自動辨識由 config.make_sensitivity_variants 產生的
「db_method 含 [param=value]」列，依 (基準策略, 參數) 分組，列出關鍵指標隨
參數值的變化曲線——供口試委員檢視每個參數的邊際敏感度與穩健區間。

同時涵蓋交易端 _list 網格掃描的參數（top_n、stop_loss、entry_z）：這些不會
產生 [param=value] 的 db_method，而是同一 METHOD 下多個配置列，本報表也一併
輸出其對 Sharpe 的敏感度。

用法：python -m analysis.sensitivity_report
"""

import re
import sqlite3
import numpy as np
import pandas as pd

RESULT_DB = "results/result.db"
_TAG = re.compile(r"^(?P<base>.+?)\s*\[(?P<param>[A-Za-z_]+)=(?P<val>[^\]]+)\]\s*$")


def _load(con):
    return pd.read_sql("SELECT * FROM strategy_summaries", con)


def _best(g: pd.DataFrame) -> pd.Series:
    """該群組（同 METHOD 全配置）Sharpe 最高的一列。"""
    return g.loc[g.Sharpe_Raw.idxmax()]


def formation_param_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """formation 參數敏感性（由 [param=value] 變體構成）。"""
    rows = []
    for method in df.METHOD.unique():
        m = _TAG.match(method)
        if not m:
            continue
        g = df[df.METHOD == method]
        best = _best(g)
        fc = (g.Forced_Closes / g.Entries.replace(0, np.nan)).median()
        rows.append({
            "基準策略": m.group("base"),
            "參數": m.group("param"),
            "值": m.group("val"),
            "最佳Sharpe": round(float(best.Sharpe_Raw), 3),
            "中位強平率": round(float(fc), 3) if pd.notna(fc) else np.nan,
            "最佳PF": round(float(best.Profit_Factor), 3),
            "最佳年化動用": round(float(best.Ann_Ret_Employed), 4),
            "最佳配置": f"Top{str(best['TOP N']).replace('Top ','')}/SL{best['STOP LOSS %']}",
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # 值轉數字排序（失敗則字串序）
    def _numify(s):
        try:
            return float(s)
        except ValueError:
            return s
    out["_v"] = out["值"].map(_numify)
    return out.sort_values(["基準策略", "參數", "_v"]).drop(columns="_v").reset_index(drop=True)


def trading_param_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """交易端 _list 網格參數（top_n、stop_loss）對 Sharpe 的敏感度，逐 METHOD。"""
    base = df[~df.METHOD.str.contains(r"\[", regex=True)]
    rows = []
    for method in base.METHOD.unique():
        g = base[base.METHOD == method]
        rec = {"策略": method}
        # top_n 敏感度（固定 SL=0%）
        gt = g[g["STOP LOSS %"] == "0%"]
        for tn in ["Top 1", "Top 3", "Top 5", "Top 10", "Top 20"]:
            sub = gt[gt["TOP N"] == tn]
            rec[tn.replace("Top ", "TopN=")] = round(float(sub.Sharpe_Raw.iloc[0]), 3) if len(sub) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def run():
    con = sqlite3.connect(RESULT_DB)
    df = _load(con)
    con.close()

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)

    fps = formation_param_sensitivity(df)
    print("\n" + "=" * 100)
    print("表 A：Formation 參數敏感性（OFAT 變體；每個值 = 一次全 295 期重跑）")
    print("=" * 100)
    if fps.empty:
        print("（尚無 [param=value] 變體結果——先跑 "
              "SENSITIVITY_PARAM=... run_formation.py + run_trading.py）")
    else:
        print(fps.to_string(index=False))
        fps.to_csv("results/analysis/sensitivity_formation.csv", index=False, encoding="utf-8-sig")

    tps = trading_param_sensitivity(df)
    print("\n" + "=" * 100)
    print("表 B：交易端 top_n 敏感性（既有網格，SL=0%；Sharpe）")
    print("=" * 100)
    print(tps.to_string(index=False))
    tps.to_csv("results/analysis/sensitivity_trading_topn.csv", index=False, encoding="utf-8-sig")
    print("\n[已存] results/analysis/sensitivity_formation.csv, sensitivity_trading_topn.csv")
    return fps, tps


if __name__ == "__main__":
    run()
