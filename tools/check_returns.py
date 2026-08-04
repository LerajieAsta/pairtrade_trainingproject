# -*- coding: utf-8 -*-
"""
strategies/returns.py 的行為檢查（合成資料，不碰資料庫）
======================================================================
以手寫的迷你 trade_logs 驗證報酬序列的定義。全部斷言都跑在純函式核心上，
不需要 result.db、不需要價格庫、不需要 pytest，秒級完成。

涵蓋本模組存在的理由 —— 那三個曾經各自作答的問題：

  · 生命期規則：晚上線的策略不會被補出上線前的零報酬
    （F09 系列 2009 才上線，舊的聯集補 0 替它捏造了 2001–2008 共八年的零，
      使 |Sharpe| 被稀釋 21%）
  · 複利分母：報酬除以**前一日權益**而非固定初始資本
    （引擎的部位規模本身跟著權益走，見 portfolio_manager.allocate_capital）
  · 交集對齊：生命期不重疊時拋錯，不靜默補 0

用法：python -m tools.check_returns
"""
import sys

import numpy as np
import pandas as pd

from strategies.returns import (
    align, equity_from_pnl, pnl_from_log, returns_from_pnl,
)

# 20 天的假日曆（週一~週五，跳過週末）
CAL = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=20))

_fails = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  \033[92mPASS\033[0m  {name}")
    else:
        print(f"  \033[91mFAIL\033[0m  {name}" + (f"\n        {detail}" if detail else ""))
        _fails.append(name)


def log(dates, deltas) -> pd.DataFrame:
    """組一份最小的交易明細（欄位同 trade_logs 的相關子集）。"""
    return pd.DataFrame({"Date": pd.to_datetime(list(dates)),
                         "Daily_Delta": list(deltas)})


# ── 1. 生命期：內補 0、外留 NaN ──────────────────────────────────────
def t_lifetime():
    print("\n[1] 生命期規則")
    # 第 5~9 天才有交易紀錄，且第 7 天空手（無列）
    d = log([CAL[5], CAL[6], CAL[8], CAL[9]], [10.0, -5.0, 3.0, 2.0])
    s = pnl_from_log(d, CAL)

    check("上線前留 NaN（不捏造零報酬）", bool(s.iloc[:5].isna().all()),
          f"實得 {s.iloc[:5].tolist()}")
    check("下線後留 NaN", bool(s.iloc[10:].isna().all()),
          f"實得 {s.iloc[10:].tolist()}")
    check("生命期內的空手日補 0", s.iloc[7] == 0.0, f"實得 {s.iloc[7]}")
    check("生命期內的損益原樣保留",
          s.iloc[5] == 10.0 and s.iloc[6] == -5.0 and s.iloc[9] == 2.0)
    check("生命期長度 = 首末交易日之間的日曆日數", int(s.notna().sum()) == 5,
          f"實得 {int(s.notna().sum())}")

    # 同一天多列（多個配對）應加總
    d2 = log([CAL[5], CAL[5], CAL[6]], [4.0, 6.0, 1.0])
    check("同日多配對加總", pnl_from_log(d2, CAL).iloc[5] == 10.0)

    check("空明細回傳全 NaN", bool(pnl_from_log(log([], []), CAL).isna().all()))


# ── 2. 複利分母 ──────────────────────────────────────────────────────
def t_compounding():
    print("\n[2] 複利報酬")
    d = log([CAL[0], CAL[1], CAL[2]], [1000.0, 1000.0, 1000.0])
    pnl = pnl_from_log(d, CAL)
    eq = equity_from_pnl(pnl, initial_capital=10000.0)
    r = returns_from_pnl(pnl, initial_capital=10000.0)

    check("權益 = 初始 + 累計損益",
          [eq.iloc[0], eq.iloc[1], eq.iloc[2]] == [11000.0, 12000.0, 13000.0],
          f"實得 {eq.iloc[:3].tolist()}")
    check("首日分母 = 初始資本", abs(r.iloc[0] - 0.1) < 1e-12, f"實得 {r.iloc[0]}")
    check("次日分母 = 前一日權益（非初始資本）",
          abs(r.iloc[1] - 1000.0 / 11000.0) < 1e-12,
          f"實得 {r.iloc[1]}，若為單利口徑會是 0.1")
    check("第三日分母 = 前一日權益",
          abs(r.iloc[2] - 1000.0 / 12000.0) < 1e-12, f"實得 {r.iloc[2]}")
    check("生命期外的報酬為 NaN", bool(r.iloc[3:].isna().all()))

    # 空手日：權益不變、報酬為 0
    d2 = log([CAL[0], CAL[2]], [1000.0, 500.0])
    eq2 = equity_from_pnl(pnl_from_log(d2, CAL), 10000.0)
    r2 = returns_from_pnl(pnl_from_log(d2, CAL), 10000.0)
    check("空手日權益不變", eq2.iloc[1] == 11000.0, f"實得 {eq2.iloc[1]}")
    check("空手日報酬為 0（非 NaN，該日策略確實存在）", r2.iloc[1] == 0.0)


# ── 3. 複利 vs 單利：確認兩者真的不同 ────────────────────────────────
def t_basis_differs():
    print("\n[3] 口徑確實有別（若這裡 PASS 表示分母選擇是實質決策）")
    rng = np.random.default_rng(42)
    deltas = rng.normal(50, 300, 20)
    pnl = pnl_from_log(log(CAL, deltas), CAL)

    r_comp = returns_from_pnl(pnl, 10000.0)
    r_simple = pnl / 10000.0
    sr = lambda x: np.sqrt(252) * x.mean() / x.std(ddof=1)
    diff = abs(sr(r_comp) / sr(r_simple) - 1)
    check("複利與單利的 Sharpe 不同", diff > 1e-6,
          f"複利 {sr(r_comp):.4f} vs 單利 {sr(r_simple):.4f}")


# ── 4. 交集對齊 ──────────────────────────────────────────────────────
def t_align():
    print("\n[4] 跨策略對齊")
    early = pnl_from_log(log(CAL[0:15], [10.0] * 15), CAL).rename("early")
    late = pnl_from_log(log(CAL[10:20], [10.0] * 10), CAL).rename("late")
    df = pd.concat([early, late], axis=1)

    out = align(df, how="intersect", min_overlap=3)
    check("交集裁到共同區間", len(out) == 5, f"實得 {len(out)} 日")
    check("交集內無 NaN", bool(out.notna().all().all()))
    check("交集起訖正確",
          out.index[0] == CAL[10] and out.index[-1] == CAL[14],
          f"實得 {out.index[0].date()}~{out.index[-1].date()}")

    # 重疊不足 → 拋錯（這是 F09 陷阱的防線）
    raised = False
    try:
        align(df, how="intersect", min_overlap=252)
    except ValueError as e:
        raised = "生命期交集" in str(e)
    check("重疊不足時拋錯而非靜默回傳", raised)

    # 完全不重疊
    a = pnl_from_log(log(CAL[0:5], [1.0] * 5), CAL).rename("a")
    b = pnl_from_log(log(CAL[10:15], [1.0] * 5), CAL).rename("b")
    raised2 = False
    try:
        align(pd.concat([a, b], axis=1), how="intersect", min_overlap=1)
    except ValueError:
        raised2 = True
    check("完全不重疊時拋錯", raised2)

    # union 是明確的選擇，不是預設
    u = align(df, how="union")
    check("union 補 0（僅供確知同期時使用）", u.notna().all().all() and len(u) == 20)


# ── 5. F09 情境重現：舊的聯集補 0 會稀釋 Sharpe ──────────────────────
def t_f09_scenario():
    print("\n[5] F09 情境（本模組存在的直接理由）")
    rng = np.random.default_rng(7)
    # 只在後半段上線的策略
    late_dates = CAL[10:]
    deltas = rng.normal(-20, 200, len(late_dates))
    pnl_correct = pnl_from_log(log(late_dates, deltas), CAL)

    # 舊行為：聯集補 0，把上線前也算進去
    pnl_padded = pnl_correct.fillna(0.0)

    sr = lambda x: np.sqrt(252) * x.mean() / x.std(ddof=1)
    s_correct = sr(pnl_correct.dropna())
    s_padded = sr(pnl_padded)

    check("上線前補 0 會稀釋 |Sharpe|", abs(s_padded) < abs(s_correct),
          f"正確 {s_correct:.4f} vs 補零 {s_padded:.4f}")
    check("稀釋幅度 ≈ sqrt(生命期/全期)",
          abs(abs(s_padded / s_correct) - np.sqrt(len(late_dates) / len(CAL))) < 0.05,
          f"實際比 {abs(s_padded / s_correct):.4f}，"
          f"預期 {np.sqrt(len(late_dates) / len(CAL)):.4f}")
    check("本模組預設不會發生此稀釋", bool(pnl_correct.iloc[:10].isna().all()))


def main() -> int:
    print("=" * 68)
    print("  strategies/returns.py 行為檢查（合成資料）")
    print("=" * 68)
    for t in (t_lifetime, t_compounding, t_basis_differs, t_align, t_f09_scenario):
        t()
    print("\n" + "=" * 68)
    if _fails:
        print(f"  \033[91m{len(_fails)} 項失敗\033[0m：" + "、".join(_fails))
        return 1
    print("  \033[92m全部通過\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
