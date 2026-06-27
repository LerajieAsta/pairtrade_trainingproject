"""
Ensemble Formation：取多個子策略的配對交集，交集優先排名，不足時以聯集補足。

排名邏輯：
  Tier 1（交集）— 兩個子策略都選到的配對，依兩者平均 Rank 升序排列
  Tier 2（補足）— 只有一個子策略選到，依該策略的 Rank 升序補至 top_n

支援任意兩個 Formation 子策略組合，透過 sub_strategies 參數傳入。
"""

import importlib
import inspect
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class Formation:
    def __init__(
        self,
        price_df:        pd.DataFrame,
        form_start:      str,
        form_end:        str,
        top_n:           int  = 20,
        sector_mapping:  dict = None,
        sub_strategies:  list = None,
        sub_top_n_multiplier: int = 3,
        **kwargs,
    ):
        """
        參數
        ----
        sub_strategies : list of dict
            每個 dict 須包含：
              - "module"  : str，formation module 路徑，如 "strategies.formation.HDBSCAN_UMAP"
              - "params"  : dict，傳給該策略 __init__ 的額外參數（不含 price_df/form_start/form_end/top_n/sector_mapping）
        sub_top_n_multiplier : int
            子策略候選數 = top_n × multiplier，候選池越大交集機率越高，預設 3
        """
        self.price_df       = price_df.copy()
        self.form_start     = form_start
        self.form_end       = form_end
        self.top_n          = top_n
        self.sector_mapping = sector_mapping or {}
        self.sub_strategies = sub_strategies or []
        self.sub_top_n      = top_n * sub_top_n_multiplier
        self.extra_kwargs   = kwargs
        self.selected_pairs = pd.DataFrame()

    # ── 內部工具 ──────────────────────────────────────────────────────────

    def _canonical_pair(self, a: str, b: str) -> tuple:
        """統一配對順序，確保 (A,B) == (B,A)"""
        return (min(a, b), max(a, b))

    def _run_sub(self, sub_cfg: dict) -> pd.DataFrame:
        """執行單一子策略，回傳 DataFrame（含 Ticker_A, Ticker_B, Rank）"""
        module = importlib.import_module(sub_cfg["module"])
        FormationClass = module.Formation

        sig = inspect.signature(FormationClass.__init__)
        valid_params = set(sig.parameters.keys())

        base = {
            "price_df":       self.price_df,
            "form_start":     self.form_start,
            "form_end":       self.form_end,
            "top_n":          self.sub_top_n,
            "sector_mapping": self.sector_mapping,
        }
        merged = {**base, **self.extra_kwargs, **sub_cfg.get("params", {})}
        valid_kwargs = {k: v for k, v in merged.items() if k in valid_params}

        instance = FormationClass(**valid_kwargs)
        result = instance.run()
        if result is None:
            return pd.DataFrame()
        return result

    # ── 主流程 ────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        if len(self.sub_strategies) < 2:
            print("[Ensemble] 警告：sub_strategies 少於 2 個，無法取交集，回傳空結果")
            return pd.DataFrame()

        sub_results = []
        for i, cfg in enumerate(self.sub_strategies):
            name = cfg.get("name", cfg["module"].split(".")[-1])
            print(f"  [Ensemble] 執行子策略 {i+1}/{len(self.sub_strategies)}: {name}")
            df = self._run_sub(cfg)
            print(f"  [Ensemble] {name} 選出 {len(df)} 對")
            sub_results.append((name, df))

        # 每個子策略建立 pair_key → Rank 映射
        rank_maps = []
        for name, df in sub_results:
            if df.empty:
                rank_maps.append({})
                continue
            rmap = {}
            for _, row in df.iterrows():
                key = self._canonical_pair(str(row["Ticker_A"]), str(row["Ticker_B"]))
                rmap[key] = int(row.get("Rank", row.name + 1))
            rank_maps.append(rmap)

        all_keys = set()
        for rmap in rank_maps:
            all_keys |= set(rmap.keys())

        # 分 Tier
        intersection_keys = all_keys.copy()
        for rmap in rank_maps:
            intersection_keys &= set(rmap.keys())

        union_only_keys = all_keys - intersection_keys

        print(f"  [Ensemble] 交集配對: {len(intersection_keys)} 對 | 僅單策略: {len(union_only_keys)} 對")

        # 建立候選列表（含原始 DataFrame 的欄位，取第一個子策略的資料為主）
        # 建立 key → row 的映射（優先取 Quality_Score 較高的子策略的資料）
        key_to_row = {}
        for name, df in sub_results:
            if df.empty:
                continue
            for _, row in df.iterrows():
                key = self._canonical_pair(str(row["Ticker_A"]), str(row["Ticker_B"]))
                if key not in key_to_row:
                    key_to_row[key] = row.to_dict()
                else:
                    # 若兩個策略都有，取 Quality_Score 較高者的欄位
                    existing_score = key_to_row[key].get("Quality_Score", 0)
                    new_score = row.get("Quality_Score", 0)
                    if new_score > existing_score:
                        key_to_row[key] = row.to_dict()

        def avg_rank(key):
            ranks = [rmap[key] for rmap in rank_maps if key in rmap]
            return sum(ranks) / len(ranks)

        # Tier 1：交集，依平均 Rank 排序
        tier1 = sorted(intersection_keys, key=avg_rank)
        # Tier 2：聯集補足，依最小 Rank 排序
        tier2 = sorted(union_only_keys, key=lambda k: min(rmap[k] for rmap in rank_maps if k in rmap))

        selected_keys = (tier1 + tier2)[:self.top_n]

        if not selected_keys:
            return pd.DataFrame()

        records = []
        for rank, key in enumerate(selected_keys, start=1):
            row = key_to_row[key].copy()
            row["Rank"]          = rank
            row["Ticker_A"]      = key[0]
            row["Ticker_B"]      = key[1]
            row["Form_Start"]    = self.form_start
            row["Form_End"]      = self.form_end
            row["In_Consensus"]  = key in intersection_keys   # 標記是否為交集配對
            avg_r = avg_rank(key)
            row["Avg_Sub_Rank"]  = round(avg_r, 2)
            records.append(row)

        self.selected_pairs = pd.DataFrame(records).reset_index(drop=True)
        return self.selected_pairs
