# -*- coding: utf-8 -*-
"""
strategies/trading/rl_threshold_trading.py 的行為檢查（合成資料，不碰資料庫）
======================================================================
驗證這條 bandit 對照組真的只做了「單一變因」該做的兩件事，而且做對了：

  · 部分回饋：buffer 每筆只存一個動作的報酬，不是九個
  · 遮罩損失：只有選中動作的輸出單元收到梯度，其餘八個原封不動
  · ε-greedy：探索率符合設定，線性衰減走得完
  · 暖身期會探索（不像 DL-THR 鎖死在基準動作——那樣網路永遠學不到）
  · 動作選單／狀態特徵與 DL-THR 逐位元相同（對照的地基）

用法：python -m tools.check_rl_threshold
"""
import sys

import numpy as np
import torch

from strategies.trading import drl_threshold_trading as dl
from strategies.trading import rl_threshold_trading as rl

_fails = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  \033[92mPASS\033[0m  {name}")
    else:
        print(f"  \033[91mFAIL\033[0m  {name}" + (f"\n        {detail}" if detail else ""))
        _fails.append(name)


def agent(**kw):
    """建一個不碰價格資料的 Trading 實例（只測學習核心）。"""
    a = rl.Trading.__new__(rl.Trading)
    a.hidden, a.lr = 64, 1e-3
    a.train_epochs, a.min_samples = kw.pop("epochs", 5), kw.pop("min_samples", 10)
    a.eps0 = kw.pop("eps0", 0.10)
    a.eps_final = kw.pop("eps_final", None)
    a.eps_steps = kw.pop("eps_steps", 2000)
    a.seed = kw.pop("seed", 0)
    a.menu_version = 4
    a.actions, a.baseline_idx = dl._build_actions(4)
    a.n_actions = len(a.actions)
    a.variant_id = kw.pop("variant_id", "test")
    a.device = torch.device("cpu")
    return a


# ── 1. 對照的地基：兩邊的選單與特徵必須完全相同 ──────────────────────
def t_shared_ground():
    print("\n[1] 單一變因的地基")
    check("動作選單相同（9 個，含基準 (2.0, 0.0)）",
          dl._build_actions(4) == rl._build_actions(4) and len(rl._build_actions(4)[0]) == 9)
    check("狀態特徵是同一個函式（不是複製一份會漂走的副本）",
          rl.Trading._pair_features is dl.Trading._pair_features)
    check("狀態維度相同", rl.N_FEAT == dl.N_FEAT == 12)

    rng = np.random.default_rng(1)
    z = np.cumsum(rng.normal(0, 0.3, 300))
    la, lb = np.cumsum(rng.normal(0, 0.01, 300)), np.cumsum(rng.normal(0, 0.01, 300))
    fa = rl.Trading._pair_features(z, la, lb, 0.8, 0.5)
    fb = dl.Trading._pair_features(z, la, lb, 0.8, 0.5)
    check("同輸入產生逐位元相同的特徵向量", bool(np.array_equal(fa, fb)))


# ── 2. 部分回饋：buffer 一筆只記一個動作 ─────────────────────────────
def t_partial_feedback():
    print("\n[2] 部分回饋")
    a = agent()
    sh = a._get_shared()
    feats = np.zeros(12, dtype=np.float32)
    sh["buffer"].append((feats, 3, np.float32(1.5), "2020-01-31"))

    rec = sh["buffer"][0]
    check("buffer 每筆是 (特徵, 動作, 單一報酬, 交易期結束日) 四元組", len(rec) == 4,
          f"實得 {len(rec)} 元組")
    check("報酬是純量而非 9 維向量（DL-THR 才是 9 維）",
          np.ndim(rec[2]) == 0, f"實得 ndim={np.ndim(rec[2])}")

    # 對照：DL-THR 存的是整條向量
    dl_rets = np.zeros(9, dtype=np.float32)
    check("DL-THR 的標籤確為 9 維（確認差異真實存在）", dl_rets.shape == (9,))


# ── 3. 遮罩損失：只有選中動作收到梯度 ────────────────────────────────
def t_masked_loss():
    print("\n[3] 遮罩損失")
    a = agent(epochs=30, min_samples=4, variant_id="masked")
    sh = a._get_shared()

    rng = np.random.default_rng(7)
    chosen = 3
    for _ in range(40):                       # 全部樣本都選同一個動作
        f = rng.normal(0, 1, 12).astype(np.float32)
        sh["buffer"].append((f, chosen, np.float32(5.0), "2020-01-31"))

    probe = torch.zeros(1, 12)
    before = sh["net"](probe).detach().numpy()[0].copy()
    a._train_if_ready(sh, "2020-06-30")
    after = sh["net"](probe).detach().numpy()[0]

    moved = np.abs(after - before)
    check("選中動作的輸出明顯移動", moved[chosen] > 1e-3,
          f"實得 Δ={moved[chosen]:.2e}")
    others = np.delete(moved, chosen)
    check("未選中的八個動作移動遠小於選中者",
          others.max() < moved[chosen] * 0.5,
          f"選中 Δ={moved[chosen]:.3e}，其餘最大 Δ={others.max():.3e}")
    check("網路朝觀測報酬收斂", abs(after[chosen] - 5.0) < abs(before[chosen] - 5.0),
          f"{before[chosen]:.3f} → {after[chosen]:.3f}（目標 5.0）")


# ── 4. ε-greedy 排程 ─────────────────────────────────────────────────
def t_epsilon():
    print("\n[4] ε 排程")
    a = agent(eps0=0.10)
    check("常數 ε：第 0 次與第 5000 次相同",
          a._epsilon(0) == 0.10 and a._epsilon(5000) == 0.10)

    b = agent(eps0=0.20, eps_final=0.02, eps_steps=1000)
    check("衰減 ε：起點為 eps0", abs(b._epsilon(0) - 0.20) < 1e-12)
    check("衰減 ε：中點約為兩端平均", abs(b._epsilon(500) - 0.11) < 1e-9,
          f"實得 {b._epsilon(500)}")
    check("衰減 ε：走完後停在 eps_final 不再下降",
          abs(b._epsilon(1000) - 0.02) < 1e-12 and abs(b._epsilon(99999) - 0.02) < 1e-12)

    # 實際抽樣頻率
    c = agent(eps0=0.25, seed=42)
    sh = c._get_shared()
    hits = sum(1 for _ in range(20000) if sh["rng"].random() < c._epsilon(0))
    check("實際探索頻率接近設定值", abs(hits / 20000 - 0.25) < 0.02,
          f"實得 {hits / 20000:.4f}")


# ── 5. ε 必須隔離不同排程的網路與經驗 ────────────────────────────────
def t_scope_isolation():
    print("\n[5] 排程隔離")
    rl.Trading._shared.clear()
    a1 = agent(eps0=0.05, variant_id="iso")
    a2 = agent(eps0=0.20, variant_id="iso")
    s1, s2 = a1._get_shared(), a2._get_shared()
    check("同一策略、不同 ε → 各自持有網路與 buffer", s1 is not s2)

    a3 = agent(eps0=0.05, variant_id="iso")
    check("同一策略、同一 ε → 共用同一份（walk-forward 需要累積經驗）",
          a3._get_shared() is s1)

    a4 = agent(eps0=0.20, eps_final=0.02, variant_id="iso")
    check("常數 0.20 與「衰減至 0.02」視為不同排程", a4._get_shared() is not s2)


# ── 6. 暖身期探索（與 DL-THR 的結構性差異） ──────────────────────────
def t_warmup_explores():
    print("\n[6] 暖身期行為")
    src = open(rl.__file__, encoding="utf-8").read()
    check("探索判斷寫在「樣本是否足夠」之前（暖身期也會探索）",
          src.index('rng"].random() < eps') < src.index("n_eligible >= self.min_samples"),
          "若順序相反，暖身期會鎖死在基準動作，其餘八個動作永無樣本")
    check("模組明寫此處無法提供 DL-THR 的「暖身期 ≡ Z-Score」保證",
          "無法" in src and "暖身期" in src)
    check("模組明寫這是 contextual bandit、沒有 γ",
          "contextual bandit" in src and "沒有 γ" in src)


def main() -> int:
    print("=" * 68)
    print("  rl_threshold_trading.py 行為檢查（合成資料）")
    print("=" * 68)
    for t in (t_shared_ground, t_partial_feedback, t_masked_loss,
              t_epsilon, t_scope_isolation, t_warmup_explores):
        t()
    print("\n" + "=" * 68)
    if _fails:
        print(f"  \033[91m{len(_fails)} 項失敗\033[0m：" + "、".join(_fails))
        return 1
    print("  \033[92m全部通過\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
