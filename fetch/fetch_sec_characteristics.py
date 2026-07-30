# -*- coding: utf-8 -*-
"""
擴充公司特徵擷取（Green et al. 2017 / Han, He & Toh 2021 特徵集的可得子集）
======================================================================
命題 1 的動機來源 Han, He & Toh (2021) 以 **48 動量因子 + 78 公司特徵**分群；
本研究原僅有 2 個連續基本面特徵（市值、盈餘殖利率）＋ 10 個結構性比率。
本模組把可得的公司特徵補齊，以檢驗「特徵維度不足」是否為命題 1 失敗的原因。

資料來源與成本
--------------
**不需要任何新的網路請求**：SEC `companyfacts` 端點一次回傳該公司全部 XBRL 概念，
而 `dataset/fundamental/sec_cache/CIK*.json` 已快取 622 檔（843 檔成分股中）。
本模組只是從既有快取中多解出一些概念。PIT 對齊沿用 `fetch_sec_fundamentals`
的 `_pit_instant` / `_pit_ttm` / `_asof_value`——皆以**申報日 filed** 為準，無前視。

已知限制（必須寫進論文）
------------------------
1. **XBRL 強制申報自 ~2009 起**，2000–2008 無資料。故使用本特徵集的實驗必須以
   `BACKTEST_START=2009-01` 執行，且 GICS 基準須在同一期間重跑才可比。
2. 78 個特徵中約 11 個**永遠取不到**（員工數、Compustat 上市年資、sin 股分類、
   可轉債／擔保債旗標、財報公布日 EPS 意外），因 XBRL 未標記或需 Compustat。
3. **刻意排除產業調整（`*_ia`）變體。** Green 特徵集含 bm_ia / cfp_ia / mve_ia /
   chatoia / chpmia 等「減去產業中位數」的欄位。本研究正在檢驗「產業先驗的價值」，
   納入它們會把 GICS 資訊從後門塞回特徵向量——與 sector one-hot 同一個混淆。
4. 同理，下游 `_features.impute_by_group` 以**產業中位數**插補缺失值。在缺失率高
   時等同注入產業資訊，故本模組輸出覆蓋率供篩選（預設只保留 >70%）。

用法：
    python -m fetch.fetch_sec_characteristics            # 建檔並報告覆蓋率
    python -m fetch.fetch_sec_characteristics --min-cov 0.7
"""
import argparse
import glob
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

from fetch.fetch_sec_fundamentals import (
    CACHE_DIR, _asof_value, _load_ticker_cik_map, _pit_instant, _pit_ttm,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_PARQUET = "dataset/fundamental/sp500_pit_2000_2025_monthly.parquet"
OUT_PARQUET = "dataset/fundamental/sp500_characteristics_monthly.parquet"
TIINGO_DB = "dataset/price/sp500_Tiingo.db"

# ── 追加的 XBRL 概念（別名依覆蓋筆數擇優，見 _pit_* 實作）──────────────
NEW_INSTANT = {
    "current_assets": ["AssetsCurrent"],
    "current_liab":   ["LiabilitiesCurrent"],
    "receivables":    ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "inventory":      ["InventoryNet"],
    "lt_debt":        ["LongTermDebtNoncurrent", "LongTermDebt"],
    # 既有 10 概念亦一併重解，使本檔自成一致（避免與舊 parquet 的口徑漂移）
    "assets":         ["Assets"],
    "liabilities":    ["Liabilities"],
    "equity":         ["StockholdersEquity",
                       "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash":           ["CashAndCashEquivalentsAtCarryingValue"],
    "ppe":            ["PropertyPlantAndEquipmentNet"],
}
NEW_FLOW = {
    "capex":         ["PaymentsToAcquirePropertyPlantAndEquipment",
                      "PaymentsToAcquireProductiveAssets"],
    "depreciation":  ["DepreciationDepletionAndAmortization",
                      "DepreciationAndAmortization", "Depreciation"],
    "dividends":     ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "income_tax":    ["IncomeTaxExpenseBenefit"],
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "revenue":       ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                      "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "net_income":    ["NetIncomeLoss"],
    "gross_profit":  ["GrossProfit"],
    "op_income":     ["OperatingIncomeLoss"],
    "ocf":           ["NetCashProvidedByUsedInOperatingActivities"],
}


def _div(a, b):
    """安全除法：分母為 0 或任一為非有限值 → NaN。"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out = np.full(a.shape, np.nan)
    ok = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[ok] = a[ok] / b[ok]
    return out


def extract_levels(symbols: list, month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """自快取 companyfacts 解出各概念的 PIT 月度水準值（filed <= 月底，無前視）。"""
    # ticker → CIK 由 SEC 官方對照表取得（單一小請求，免金鑰；沿用既有實作）
    cmap = {k.upper(): v for k, v in _load_ticker_cik_map().items()}
    cached = {os.path.basename(f).replace(".json", "")
              for f in glob.glob(os.path.join(CACHE_DIR, "CIK*.json"))}
    rows, n_hit = [], 0

    for i, sym in enumerate(symbols):
        # _load_ticker_cik_map 回傳 10 碼零填充（無前綴）；快取檔名為 CIK{10碼}.json
        cik = cmap.get(sym.upper())
        key = f"CIK{cik}" if cik else None
        facts = {}
        if key and key in cached:
            try:
                facts = json.load(open(os.path.join(CACHE_DIR, f"{key}.json"),
                                      encoding="utf-8"))
            except Exception:
                facts = {}
        if facts:
            n_hit += 1
            inst = {k: _pit_instant(facts, c) for k, c in NEW_INSTANT.items()}
            flow = {k: _pit_ttm(facts, c) for k, c in NEW_FLOW.items()}
        else:
            inst = {k: pd.DataFrame() for k in NEW_INSTANT}
            flow = {k: pd.DataFrame() for k in NEW_FLOW}

        for d in month_ends:
            rec = {"date": d, "ticker": sym}
            for k, s in {**inst, **flow}.items():
                rec[k] = _asof_value(s, d, val_col="val") if not s.empty else np.nan
            rows.append(rec)

        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(symbols)} 檔（快取命中 {n_hit}）", flush=True)

    print(f"  完成：{len(symbols)} 檔，companyfacts 命中 {n_hit} 檔")
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


def build_characteristics(lv: pd.DataFrame, mcap: pd.Series) -> pd.DataFrame:
    """
    由水準值組出 Green et al. 風格的比率/成長特徵。

    刻意不含 `*_ia`（產業調整）變體——見模組 docstring 限制 3。
    成長類以同一 ticker 的 12 個月前值計算（兩端皆為 filed<=t 的 PIT 值，無前視）。
    """
    d = lv.copy()
    d["mcap"] = mcap.reindex(d.index)
    g = d.groupby(level="ticker")
    lag = g.shift(12)                      # 12 個月前的同一概念

    c = pd.DataFrame(index=d.index)
    # ── 價值 / 規模 ──
    c["bm"]        = _div(d.equity, d.mcap)
    c["ep"]        = _div(d.net_income, d.mcap)
    c["cfp"]       = _div(d.ocf, d.mcap)
    c["sp"]        = _div(d.revenue, d.mcap)
    c["dy"]        = _div(d.dividends, d.mcap)
    c["lev"]       = _div(d.liabilities, d.mcap)
    # ── 獲利能力 ──
    c["roe"]       = _div(d.net_income, d.equity)
    c["roa"]       = _div(d.net_income, d.assets)
    c["roic"]      = _div(d.op_income - d.income_tax, d.equity + d.lt_debt - d.cash)
    c["gma"]       = _div(d.gross_profit, d.assets)
    c["operprof"]  = _div(d.op_income, d.equity)
    c["gross_margin"] = _div(d.gross_profit, d.revenue)
    c["op_margin"] = _div(d.op_income, d.revenue)
    # ── 財務結構 / 流動性 ──
    c["leverage"]  = _div(d.liabilities, d.assets)
    c["currat"]    = _div(d.current_assets, d.current_liab)
    c["quick"]     = _div(d.current_assets - d.inventory, d.current_liab)
    c["cash_ratio"] = _div(d.cash, d.assets)
    c["cashdebt"]  = _div(d.ocf, d.liabilities)
    c["cashpr"]    = _div(d.mcap + d.lt_debt - d.assets, d.cash)
    # Almeida & Campello (2007) 有形資產可抵押比例
    c["tang"]      = _div(d.cash + 0.715 * d.receivables
                          + 0.547 * d.inventory + 0.535 * d.ppe, d.assets)
    # ── 營運效率 ──
    c["asset_turnover"] = _div(d.revenue, d.assets)
    c["salerec"]   = _div(d.revenue, d.receivables)
    c["salecash"]  = _div(d.revenue, d.cash)
    c["capital_intensity"] = _div(d.ppe, d.assets)
    c["depr"]      = _div(d.depreciation, d.ppe)
    c["tb"]        = _div(d.income_tax, d.pretax_income)
    # ── 應計項目 ──
    c["accruals"]  = _div(d.net_income - d.ocf, d.assets)
    c["absacc"]    = np.abs(c["accruals"])
    c["pctacc"]    = _div(d.net_income - d.ocf, np.abs(d.net_income))
    # ── 投資 ──
    c["invest"]    = _div(d.capex, d.assets)
    # ── 成長（12 個月變化）──
    c["agr"]       = _div(d.assets - lag.assets, lag.assets)
    c["sgr"]       = _div(d.revenue - lag.revenue, lag.revenue)
    c["egr"]       = _div(d.equity - lag.equity, lag.equity)
    c["lgr"]       = _div(d.lt_debt - lag.lt_debt, lag.lt_debt)
    c["chinv"]     = _div(d.inventory - lag.inventory, d.assets)
    c["chtx"]      = _div(d.income_tax - lag.income_tax, d.assets)
    c["pchcapx"]   = _div(d.capex - lag.capex, lag.capex)
    c["pchdepr"]   = _div(c.depr - _div(lag.depreciation, lag.ppe),
                          _div(lag.depreciation, lag.ppe))
    c["pchcurrat"] = _div(c.currat - _div(lag.current_assets, lag.current_liab),
                          _div(lag.current_assets, lag.current_liab))
    c["pchquick"]  = _div(c.quick - _div(lag.current_assets - lag.inventory, lag.current_liab),
                          _div(lag.current_assets - lag.inventory, lag.current_liab))
    return c.replace([np.inf, -np.inf], np.nan)


def _membership_mask(index: pd.MultiIndex) -> pd.Series:
    """
    (date, ticker) → 該股於該月底是否真的在 S&P 500 內。

    基準面板是 843 檔 × 312 月的**完全交叉積**，其中僅約 60% 對應真實成分股身分
    （其餘是該股尚未納入或已剔除的期間，必然無財報值）。覆蓋率若用交叉積當分母，
    上限就只有 60%，門檻判斷會失真——故一律以成分股身分為分母。
    """
    con = sqlite3.connect(f"file:{TIINGO_DB}?mode=ro", uri=True)
    m = pd.read_sql("SELECT Symbol,start_date,end_date FROM index_memberships", con)
    con.close()
    m["start_date"] = pd.to_datetime(m.start_date)
    m["end_date"] = pd.to_datetime(m.end_date).fillna(pd.Timestamp("2100-01-01"))
    spans = {s: g[["start_date", "end_date"]].values
             for s, g in m.groupby("Symbol")}
    dates = index.get_level_values("date")
    tickers = index.get_level_values("ticker")
    out = np.zeros(len(index), dtype=bool)
    for i, (d, t) in enumerate(zip(dates, tickers)):
        for s, e in spans.get(t, ()):
            if s <= d <= e:
                out[i] = True
                break
    return pd.Series(out, index=index)


def run(min_cov: float, cov_start: str):
    base = pd.read_parquet(BASE_PARQUET)
    idx = base.index
    symbols = sorted(idx.get_level_values("ticker").unique())
    month_ends = pd.DatetimeIndex(sorted(idx.get_level_values("date").unique()))
    print(f"基準面板：{len(symbols)} 檔 × {len(month_ends)} 月（完全交叉積）")

    print("解析快取 companyfacts…")
    lv = extract_levels(symbols, month_ends)
    ch = build_characteristics(lv, base["market_cap"])

    # 覆蓋率分母 = 「該期間內真實在指數內」的 ticker-month（見 _membership_mask）
    print("計算成分股身分遮罩…")
    inmemb = _membership_mask(ch.index)
    sel = inmemb & (ch.index.get_level_values("date") >= cov_start)
    n = int(sel.sum())
    cov = ch[sel].notna().mean().sort_values(ascending=False)
    keep = cov[cov > min_cov].index.tolist()

    print(f"\n{'='*74}")
    print(f"各特徵覆蓋率（分母＝{cov_start} 後真實成分股身分 {n:,} 個 ticker-month）")
    print(f"門檻 {min_cov:.0%}")
    print(f"{'='*74}")
    for k, v in cov.items():
        print(f"  {'✔' if v > min_cov else '✘'} {k:20s} {v*100:5.1f}%")
    print(f"\n通過門檻 {len(keep)}/{len(cov)} 個特徵：{keep}")

    # 全集一律落檔（含未通過者），篩選交由下游決定——避免此處破壞性丟棄
    ch.to_parquet(OUT_PARQUET)
    pd.DataFrame({"coverage": cov, "passed": cov > min_cov}).to_csv(
        "dataset/fundamental/characteristics_coverage.csv", encoding="utf-8-sig")
    print(f"→ {OUT_PARQUET}（{ch.shape[0]:,} 列 × {ch.shape[1]} 特徵，全集）")
    print(f"→ dataset/fundamental/characteristics_coverage.csv（含 passed 欄）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cov", type=float, default=0.70)
    ap.add_argument("--cov-start", type=str, default="2012-01-01",
                    help="覆蓋率計算起點；XBRL 2009 起但覆蓋率至 2012 才穩定")
    a = ap.parse_args()
    run(a.min_cov, a.cov_start)
