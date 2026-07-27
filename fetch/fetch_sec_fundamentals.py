"""
SEC EDGAR PIT 基本面抓取（免費、官方、無需 API key）
======================================================================

背景：FMP /api/v3 已停用、且低階方案歷史深度受限，無法建 2000–2025 PIT 基本面。
本模組改以 SEC EDGAR XBRL companyfacts（免費、官方、以申報日 filed 做 PIT 對齊）
取得每股盈餘與流通股數，配合 Tiingo API 的「原始（未調整）收盤價」計算：
    market_cap = 原始收盤價 × 流通股數        （兩者皆當期未調整 → 直接相乘，無需拆股因子）
    pe_ratio   = 原始收盤價 / TTM EPS
    industry   = Tiingo Constituents 的 GICS_Sector

為何用 Tiingo 原始價而非本地 DB 或 yfinance：
  - 本地 sp500_Tiingo.db 的 Close 已「拆股調整」，與 SEC 當期股數基準不一致；
  - yfinance 對已下市股（如 TIE）常抓不到；
  - Tiingo API 的 close 欄為原始價、且涵蓋已下市股，配 SEC 當期股數可直接算市值/PE。

輸出 parquet 的 schema 與 fetch_fmp_fundamentals.py 完全相同（index=[date,ticker]，
欄位 close/market_cap/pe_ratio/industry），故 agglomerative_FMP.py 不需改動即可沿用。

覆蓋限制：XBRL 強制申報自 ~2009 起，2000–2008 多無資料（下游會以產業中位數插補，
或可將 Agglomerative Fundamentals 回測起點設在 2009）。已下市個股若不在 SEC 現行
ticker→CIK 對照中則缺該檔（同樣插補）。

用法（SEC 免金鑰，僅需 User-Agent；Tiingo 原始股價需金鑰）：
    PowerShell:
      $env:SEC_USER_AGENT="你的名字 email@校.edu.tw"
      $env:TIINGO_API_KEY="你的 Tiingo 金鑰"
      python fetch/fetch_sec_fundamentals.py
"""

import os
import time
import json
import sqlite3
import requests
import numpy as np
import pandas as pd

# SEC 要求所有請求帶可識別的 User-Agent（請改成你的 email）
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "PairsTradingThesis research@example.com")
SEC_HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
SEC_RATE_DELAY = 0.15   # ~6–7 req/s，SEC 上限 10 req/s

TIINGO_DB = "dataset/price/sp500_Tiingo.db"
CACHE_DIR = "dataset/fundamental/sec_cache"
OUTPUT_PATH = "dataset/fundamental/sp500_pit_2000_2025_monthly.parquet"
# Tiingo API 金鑰（原始股價用）：改由環境變數提供，勿硬編碼。
TIINGO_API_KEY = os.environ.get("TIINGO_API_KEY", "").strip()

# 診斷用欄位對應（XBRL 概念名）
EPS_CONCEPTS = ["EarningsPerShareDiluted", "EarningsPerShareBasic"]
SHARES_CONCEPTS_DEI = ["EntityCommonStockSharesOutstanding"]
SHARES_CONCEPTS_GAAP = ["CommonStockSharesOutstanding",
                        "WeightedAverageNumberOfDilutedSharesOutstanding",
                        "WeightedAverageNumberOfSharesOutstandingBasic"]

# ── 結構性財報特徵（Sanders 2021 / Green et al. 2017 路線）─────────────────
# 每個特徵給多個 XBRL 概念別名，取第一個有資料者（跨公司/年度的標籤不一致）。
# 快取抽樣覆蓋率：Assets/NetIncomeLoss/營運現金流/固定資產 100%、股東權益 95%、
# 現金 95%、營收 82%、營業利益 80%、總負債 72%、毛利 70%。
BALANCE_CONCEPTS = {   # 時點量（instantaneous）
    "assets":       ["Assets"],
    "liabilities":  ["Liabilities"],
    "equity":       ["StockholdersEquity",
                     "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash":         ["CashAndCashEquivalentsAtCarryingValue"],
    "ppe":          ["PropertyPlantAndEquipmentNet"],
}
FLOW_CONCEPTS = {      # 期間量（duration）→ 取 TTM
    # 營收概念隨會計準則演進（ASC 606, 2018）而改名，且同一公司可能兩者都有
    # 但舊概念早已停用（如 AAPL 的 Revenues 僅存 1 筆）→ 取「筆數最多」者而非第一個
    "revenue":      ["RevenueFromContractWithCustomerExcludingAssessedTax",
                     "Revenues", "SalesRevenueNet",
                     "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "net_income":   ["NetIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "op_income":    ["OperatingIncomeLoss"],
    "ocf":          ["NetCashProvidedByUsedInOperatingActivities"],
}


def _load_constituents():
    """自 Tiingo DB 取得成分股清單與 GICS 產業。"""
    con = sqlite3.connect(TIINGO_DB)
    df = pd.read_sql("SELECT DISTINCT Symbol, GICS_Sector FROM Constituents", con)
    con.close()
    df["Symbol"] = df["Symbol"].str.upper()
    return df


def _load_ticker_cik_map():
    """SEC 官方 ticker→CIK 對照（10 碼零填充）。"""
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in r.json().values()}


def _fetch_companyfacts(cik: str) -> dict:
    """單一 CIK 的全部 XBRL facts（含快取）；一檔一次請求。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"CIK{cik}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    for attempt in range(4):
        try:
            r = requests.get(url, headers=SEC_HEADERS, timeout=30)
            if r.status_code == 200:
                data = r.json()
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                time.sleep(SEC_RATE_DELAY)
                return data
            if r.status_code == 404:
                return {}
            time.sleep(1.0 * (attempt + 1))
        except requests.exceptions.RequestException:
            time.sleep(1.0 * (attempt + 1))
    return {}


def _concept_entries(facts: dict, taxonomy: str, concept: str) -> list:
    try:
        units = facts["facts"][taxonomy][concept]["units"]
    except KeyError:
        return []
    # 取第一個單位（EPS 為 'USD/shares'，股數為 'shares'）
    key = next(iter(units))
    return units[key]


def _pit_eps_ttm(facts: dict) -> pd.DataFrame:
    """
    以申報日 filed 建立 PIT TTM EPS 逐步序列。
    取季度 EPS（期間長度 ~60–100 天）最近 4 筆加總；無季度時退回年度 EPS（10-K）。
    回傳 columns=[filed, ttm_eps]，依 filed 排序。
    """
    entries = []
    for concept in EPS_CONCEPTS:
        rows = _concept_entries(facts, "us-gaap", concept)
        if rows:
            entries = rows
            break
    if not entries:
        return pd.DataFrame(columns=["filed", "ttm_eps"])

    recs = []
    for e in entries:
        if not e.get("filed") or e.get("val") is None or not e.get("start") or not e.get("end"):
            continue
        dur = (pd.Timestamp(e["end"]) - pd.Timestamp(e["start"])).days
        recs.append({"start": pd.Timestamp(e["start"]), "end": pd.Timestamp(e["end"]),
                     "filed": pd.Timestamp(e["filed"]), "val": float(e["val"]),
                     "kind": "Q" if 60 <= dur <= 100 else ("A" if dur >= 300 else "O")})
    if not recs:
        return pd.DataFrame(columns=["filed", "ttm_eps"])
    df = pd.DataFrame(recs).drop_duplicates(subset=["end", "kind"]).sort_values("end")

    q = df[df.kind == "Q"].copy()
    a = df[df.kind == "A"].copy()
    out = []
    if len(q) >= 4:
        for i in range(3, len(q)):
            window = q.iloc[i - 3:i + 1]
            out.append({"filed": window["filed"].max(), "ttm_eps": float(window["val"].sum())})
    # 以年度 EPS 補足（季度不足或更早期）
    for _, row in a.iterrows():
        out.append({"filed": row["filed"], "ttm_eps": float(row["val"])})
    if not out:
        return pd.DataFrame(columns=["filed", "ttm_eps"])
    return (pd.DataFrame(out).sort_values("filed")
            .drop_duplicates(subset="filed", keep="last").reset_index(drop=True))


def _pit_instant(facts: dict, concepts: list) -> pd.DataFrame:
    """
    時點量（資產負債表項目）的 PIT 序列：columns=[filed, val]，依 filed 排序。
    同一 filed 有多筆時取期末日最新者（避免比較年度與季度殘留）。
    """
    best = pd.DataFrame(columns=["filed", "val"])
    for c in concepts:
        rows = _concept_entries(facts, "us-gaap", c)
        recs = []
        for e in rows:
            if not e.get("filed") or e.get("val") is None or not e.get("end"):
                continue
            recs.append({"filed": pd.Timestamp(e["filed"]),
                         "end": pd.Timestamp(e["end"]), "val": float(e["val"])})
        if recs:
            df = pd.DataFrame(recs).sort_values(["filed", "end"])
            df = df.groupby("filed", as_index=False).last()[["filed", "val"]]
            # 概念別名可能新舊並存但舊者早已停用 → 取覆蓋筆數最多者
            if len(df) > len(best):
                best = df
    return best


def _pit_ttm(facts: dict, concepts: list) -> pd.DataFrame:
    """
    期間量（損益/現金流項目）的 PIT TTM 序列：columns=[filed, val]。
    取季度資料（期間 60–100 天）最近 4 筆加總；無季度時退回年度（300–400 天）。
    與 _pit_eps_ttm 同一邏輯，泛化為任意概念。
    """
    best = pd.DataFrame(columns=["filed", "val"])
    for c in concepts:
        rows = _concept_entries(facts, "us-gaap", c)
        recs = []
        for e in rows:
            if not all(e.get(k) for k in ("filed", "start", "end")) or e.get("val") is None:
                continue
            dur = (pd.Timestamp(e["end"]) - pd.Timestamp(e["start"])).days
            recs.append({"filed": pd.Timestamp(e["filed"]), "end": pd.Timestamp(e["end"]),
                         "dur": dur, "val": float(e["val"])})
        if not recs:
            continue
        df = pd.DataFrame(recs)
        q = df[(df["dur"] >= 60) & (df["dur"] <= 100)].sort_values(["filed", "end"])
        out = []
        if not q.empty:
            q = q.drop_duplicates(subset=["end"], keep="last").sort_values("end")
            for i in range(3, len(q)):
                w = q.iloc[i - 3:i + 1]
                out.append({"filed": w["filed"].max(), "val": float(w["val"].sum())})
        if not out:   # 退回年度
            a = df[(df["dur"] >= 300) & (df["dur"] <= 400)].sort_values(["filed", "end"])
            out = [{"filed": r["filed"], "val": r["val"]} for _, r in a.iterrows()]
        if out:
            res = pd.DataFrame(out).sort_values("filed")
            res = res.groupby("filed", as_index=False).last()
            if len(res) > len(best):
                best = res
    return best


def _pit_shares(facts: dict) -> pd.DataFrame:
    """以申報日 filed 建立 PIT 流通股數逐步序列（cover-page 優先，退回 GAAP）。"""
    entries = []
    for c in SHARES_CONCEPTS_DEI:
        rows = _concept_entries(facts, "dei", c)
        if rows:
            entries = rows
            break
    if not entries:
        for c in SHARES_CONCEPTS_GAAP:
            rows = _concept_entries(facts, "us-gaap", c)
            if rows:
                entries = rows
                break
    recs = []
    for e in entries:
        if not e.get("filed") or e.get("val") is None:
            continue
        recs.append({"filed": pd.Timestamp(e["filed"]), "shares": float(e["val"])})
    if not recs:
        return pd.DataFrame(columns=["filed", "shares"])
    return (pd.DataFrame(recs).sort_values("filed")
            .drop_duplicates(subset="filed", keep="last").reset_index(drop=True))


def _tiingo_raw_monthly_close(symbol: str, start: str, end: str) -> pd.Series:
    """
    自 Tiingo API 取「原始（未調整）」日收盤價，月底重取樣。含已下市股（如 TIE）。
    回傳 Series[date(月底) -> raw_close]；快取於 CACHE_DIR。
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{symbol}_tiingo_raw.pkl")
    if os.path.exists(cache):
        try:
            return pd.read_pickle(cache)
        except Exception:
            pass
    if not TIINGO_API_KEY:
        raise RuntimeError("未設定環境變數 TIINGO_API_KEY（Tiingo 原始股價需要）。")
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    params = {"startDate": start, "endDate": end, "format": "json"}
    headers = {"Content-Type": "application/json", "Authorization": f"Token {TIINGO_API_KEY}"}
    rows = []
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 200:
                rows = r.json()
                break
            if r.status_code == 404:
                break                       # 該 ticker Tiingo 無資料
            time.sleep(1.0 * (attempt + 1))
        except requests.exceptions.RequestException:
            time.sleep(1.0 * (attempt + 1))
    if not rows:
        s = pd.Series(dtype=float)
        pd.to_pickle(s, cache)
        return s
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").set_index("date")
    monthly = df["close"].resample("ME").last().dropna()   # close = 原始未調整價
    pd.to_pickle(monthly, cache)
    time.sleep(0.05)
    return monthly


def _asof_value(pit_df: pd.DataFrame, on_date: pd.Timestamp, filed_col="filed", val_col=None):
    """回傳 filed <= on_date 的最新一筆值（PIT，無前視）。"""
    if pit_df.empty:
        return np.nan
    sub = pit_df[pit_df[filed_col] <= on_date]
    if sub.empty:
        return np.nan
    return float(sub.iloc[-1][val_col])


def build_dataset(start="2000-01-01", end="2025-12-31"):
    os.makedirs(CACHE_DIR, exist_ok=True)
    const = _load_constituents()
    industry_map = dict(zip(const["Symbol"], const["GICS_Sector"]))
    symbols = const["Symbol"].tolist()
    cik_map = _load_ticker_cik_map()
    month_ends = pd.date_range(start=start, end=end, freq="ME")

    print(f"成分股 {len(symbols)} 檔 | SEC CIK 命中 {sum(s in cik_map for s in symbols)} 檔 | "
          f"月份 {len(month_ends)}")

    all_rows = []
    n_ok = 0
    for i, sym in enumerate(symbols):
        cik = cik_map.get(sym)
        eps_pit = shares_pit = pd.DataFrame()
        bal_pit, flow_pit = {}, {}
        if cik:
            facts = _fetch_companyfacts(cik)
            if facts:
                eps_pit = _pit_eps_ttm(facts)
                shares_pit = _pit_shares(facts)
                # 結構性財報特徵（Sanders/Green 路線）
                bal_pit = {k: _pit_instant(facts, c) for k, c in BALANCE_CONCEPTS.items()}
                flow_pit = {k: _pit_ttm(facts, c) for k, c in FLOW_CONCEPTS.items()}
                if not eps_pit.empty or not shares_pit.empty:
                    n_ok += 1
        try:
            px = _tiingo_raw_monthly_close(sym, start, end)   # 原始（未調整）月底價
        except RuntimeError as e:
            raise SystemExit(str(e))
        for d in month_ends:
            close = float(px.get(d, np.nan)) if len(px) else np.nan
            shares = _asof_value(shares_pit, d, val_col="shares") if not shares_pit.empty else np.nan
            ttm_eps = _asof_value(eps_pit, d, val_col="ttm_eps") if not eps_pit.empty else np.nan
            # 原始價 × 當期股數 → 市值；原始價 / TTM EPS → PE（皆當期未調整，直接相乘）
            mcap = close * shares if (np.isfinite(close) and np.isfinite(shares)) else np.nan
            pe = close / ttm_eps if (np.isfinite(close) and np.isfinite(ttm_eps) and ttm_eps > 0) else np.nan

            # 結構性特徵（皆為 filed <= d 的最新值，無前視）
            # 無 CIK / companyfacts 抓不到時 bal_pit/flow_pit 為空 dict → 全部補 NaN
            _b = {k: (_asof_value(bal_pit[k], d, val_col="val")
                      if k in bal_pit and not bal_pit[k].empty else np.nan)
                  for k in BALANCE_CONCEPTS}
            _f = {k: (_asof_value(flow_pit[k], d, val_col="val")
                      if k in flow_pit and not flow_pit[k].empty else np.nan)
                  for k in FLOW_CONCEPTS}

            def _safe_div(a, b):
                return a / b if (np.isfinite(a) and np.isfinite(b) and b != 0) else np.nan

            book_to_market = _safe_div(_b["equity"], mcap)          # 帳面市值比（價值因子）
            roe            = _safe_div(_f["net_income"], _b["equity"])   # 股東權益報酬率
            roa            = _safe_div(_f["net_income"], _b["assets"])   # 資產報酬率
            leverage       = _safe_div(_b["liabilities"], _b["assets"])  # 負債比
            gross_margin   = _safe_div(_f["gross_profit"], _f["revenue"])  # 毛利率
            op_margin      = _safe_div(_f["op_income"], _f["revenue"])     # 營業利益率
            asset_turnover = _safe_div(_f["revenue"], _b["assets"])        # 資產週轉率
            cash_ratio     = _safe_div(_b["cash"], _b["assets"])           # 現金占比
            capital_int    = _safe_div(_b["ppe"], _b["assets"])            # 資本密集度
            accruals       = _safe_div(_f["net_income"] - _f["ocf"], _b["assets"])  # 應計項目

            all_rows.append((d, sym, close, mcap, pe, industry_map.get(sym, "Unknown"),
                             book_to_market, roe, roa, leverage, gross_margin,
                             op_margin, asset_turnover, cash_ratio, capital_int, accruals))
        if (i + 1) % 25 == 0:
            print(f"  進度 {i+1}/{len(symbols)} | 有基本面 {n_ok} 檔")

    df = pd.DataFrame(all_rows, columns=[
        "date", "ticker", "close", "market_cap", "pe_ratio", "industry",
        "book_to_market", "roe", "roa", "leverage", "gross_margin",
        "op_margin", "asset_turnover", "cash_ratio", "capital_intensity", "accruals"])
    df = df.set_index(["date", "ticker"])
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH)
    nn_mc = df["market_cap"].notna().sum()
    nn_pe = df["pe_ratio"].notna().sum()
    print(f"\n✅ 輸出 {OUTPUT_PATH}")
    print(f"   shape={df.shape} | market_cap 非空 {nn_mc:,}（{nn_mc/len(df):.1%}）| "
          f"pe_ratio 非空 {nn_pe:,}（{nn_pe/len(df):.1%}）")
    return df


if __name__ == "__main__":
    build_dataset()
