# -*- coding: utf-8 -*-
"""
補抓 Tiingo 未調整（raw）月底收盤價 —— 可續傳、不快取失敗
======================================================================
為什麼需要：市值 = 未調整價 × 當期申報股數。本地 `sp500_Tiingo.db` 的 Close 已
拆股調整（實測 AAPL 2020-08-24 為 $122，未調整實價約 $500），與 SEC 的當期股數
基準不一致，故市值必須另取未調整價。

為什麼現有快取不能用：`fetch_sec_fundamentals._tiingo_raw_monthly_close` 在請求
失敗時**會把空序列寫入快取**，且下次看到快取就直接回傳、永不重試。843 檔中
198 檔成功、576 檔（2012–2025 成分股）留下**被永久快取的失敗**——多半是速率／
符號數上限所致，而非該 ticker 真的無資料。受影響的是所有需要市值的估值比率
（bm / ep / cfp / sp / dy / lev / cashpr / mve，覆蓋率僅 15–22%）。

本模組的差異：
  1. 只重抓「2012–2025 期間為成分股且快取為空」者
  2. **絕不快取失敗**——失敗就留著，下次可續傳
  3. 明確區分 404（該 ticker 真的無資料，寫入標記避免無限重試）與
     429/5xx（速率限制／伺服器問題，不寫任何東西）
  4. 撞到速率限制即優雅停止並回報，供分批跨小時／跨日續跑

金鑰：放在專案根目錄 `.env`（已在 .gitignore 內），格式
    TIINGO_API_KEY=你的金鑰

用法：
    python -m fetch.refetch_tiingo_raw --limit 60     # 先試 60 檔，觀察上限
    python -m fetch.refetch_tiingo_raw                # 全部（會自動在撞限時停下）
"""
import argparse
import glob
import os
import sqlite3
import sys
import time

import pandas as pd
import requests

CACHE_DIR = "dataset/fundamental/sec_cache"
TIINGO_DB = "dataset/price/sp500_Tiingo.db"
NODATA_MARK = "__NODATA__"        # 404 標記檔內容，與「失敗」區分

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _load_env_key() -> str:
    """自 .env 讀 TIINGO_API_KEY（環境變數優先）。"""
    k = os.environ.get("TIINGO_API_KEY", "").strip()
    if k:
        return k
    for p in (".env", "../.env"):
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if line.startswith("TIINGO_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _targets(start="2012-01-01", end="2025-12-31") -> list:
    """2012–2025 期間為成分股、且尚無有效原始價快取者。"""
    con = sqlite3.connect(f"file:{TIINGO_DB}?mode=ro", uri=True)
    m = pd.read_sql("SELECT Symbol,start_date,end_date FROM index_memberships", con)
    con.close()
    m["start_date"] = pd.to_datetime(m.start_date)
    m["end_date"] = pd.to_datetime(m.end_date).fillna(pd.Timestamp("2100-01-01"))
    need = set(m[(m.end_date >= start) & (m.start_date <= end)].Symbol)

    # 有效性以「實際載入後長度 > 0」判定，不用檔案大小——短序列（如僅 60 個月的
    # 早期下市股）的 pickle 可能小於任何合理的大小門檻，會被誤判成空快取而重複抓取。
    ok, nodata = set(), set()
    for f in glob.glob(os.path.join(CACHE_DIR, "*_tiingo_raw.pkl")):
        sym = os.path.basename(f).replace("_tiingo_raw.pkl", "")
        try:
            if len(pd.read_pickle(f)) > 0:
                ok.add(sym)
        except Exception:
            pass
    for f in glob.glob(os.path.join(CACHE_DIR, "*_tiingo_nodata.txt")):
        nodata.add(os.path.basename(f).replace("_tiingo_nodata.txt", ""))
    return sorted(need - ok - nodata)


def fetch_one(symbol: str, key: str, start: str, end: str):
    """
    回傳 (status, series)。status ∈ {"ok", "nodata", "ratelimit", "error"}。
    失敗一律不寫快取——留待續傳。
    """
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    try:
        r = requests.get(url,
                         params={"startDate": start, "endDate": end, "format": "json"},
                         headers={"Content-Type": "application/json",
                                  "Authorization": f"Token {key}"},
                         timeout=30)
    except requests.exceptions.RequestException as e:
        return "error", str(e)

    if r.status_code == 404:
        return "nodata", None
    if r.status_code in (429, 403):
        return "ratelimit", r.text[:300]
    if r.status_code != 200:
        return "error", f"HTTP {r.status_code}: {r.text[:200]}"

    # Tiingo 偶爾以 200 回傳非 JSON（錯誤頁／截斷回應）。若不接住，
    # JSONDecodeError 會直接中斷整個批次——實測曾因此損失數批的額度。
    try:
        rows = r.json()
    except ValueError:
        return "error", f"回應非 JSON（前 120 字）：{r.text[:120]!r}"
    if not rows:
        return "nodata", None
    df = pd.DataFrame(rows)
    if "close" not in df or "date" not in df:
        return "error", f"未預期的回應欄位：{list(df.columns)[:8]}"
    s = (df.assign(date=pd.to_datetime(df["date"]).dt.tz_localize(None))
           .set_index("date")["close"].astype(float)
           .resample("ME").last().dropna())
    return "ok", s


def run(limit: int, delay: float, start: str, end: str):
    key = _load_env_key()
    if not key:
        raise SystemExit(
            "未找到 TIINGO_API_KEY。請在專案根目錄建立 .env（已在 .gitignore 內）：\n"
            "    TIINGO_API_KEY=你的金鑰")

    todo = _targets(start, end)
    if limit:
        todo = todo[:limit]
    print(f"待補抓 {len(todo)} 檔（延遲 {delay}s/檔）\n")

    n = {"ok": 0, "nodata": 0, "ratelimit": 0, "error": 0}
    for i, sym in enumerate(todo, 1):
        status, payload = fetch_one(sym, key, start, end)
        n[status] += 1
        if status == "ok":
            pd.to_pickle(payload, os.path.join(CACHE_DIR, f"{sym}_tiingo_raw.pkl"))
            print(f"  [{i:4d}/{len(todo)}] {sym:6s} ✔ {len(payload)} 個月")
        elif status == "nodata":
            # 真的無資料 → 寫標記，避免下次重試（與「失敗」明確區分）
            open(os.path.join(CACHE_DIR, f"{sym}_tiingo_nodata.txt"), "w").close()
            print(f"  [{i:4d}/{len(todo)}] {sym:6s} － Tiingo 無此資料（已標記）")
        elif status == "ratelimit":
            print(f"\n⚠ 撞到速率／方案上限（{sym}）。API 回應：\n  {payload}\n")
            print(f"已完成 {n['ok']} 檔，未寫入任何失敗快取——稍後重跑本指令即可續傳。")
            break
        else:
            print(f"  [{i:4d}/{len(todo)}] {sym:6s} ✘ {payload}")
        time.sleep(delay)

    print(f"\n本次：成功 {n['ok']}、無資料 {n['nodata']}、"
          f"錯誤 {n['error']}、撞限 {n['ratelimit']}")
    rest = len(_targets(start, end))
    print(f"仍待補抓 {rest} 檔")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="本次最多抓幾檔（0=全部）")
    ap.add_argument("--delay", type=float, default=1.0, help="每檔間隔秒數")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default="2025-12-31")
    a = ap.parse_args()
    run(a.limit, a.delay, a.start, a.end)
