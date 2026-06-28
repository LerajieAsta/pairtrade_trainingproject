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

        # 單次遍歷同時建立 rank_maps 和 key_to_row（合併原本的兩遍 iterrows）
        rank_maps = []
        key_to_row: dict = {}
        for name, df in sub_results:
            if df.empty:
                rank_maps.append({})
                continue
            rmap: dict = {}
            for _, row in df.iterrows():
                key = self._canonical_pair(str(row["Ticker_A"]), str(row["Ticker_B"]))
                rmap[key] = int(row.get("Rank", row.name + 1))
                if key not in key_to_row:
                    key_to_row[key] = row.to_dict()
                else:
                    # 若兩個策略都有，取 Quality_Score 較高者；相同時以 Pair_Rank 較小者優先
                    existing_score = key_to_row[key].get("Quality_Score", 0)
                    new_score = row.get("Quality_Score", 0)
                    existing_rank = key_to_row[key].get("Pair_Rank", key_to_row[key].get("Rank", 9999))
                    new_rank = row.get("Pair_Rank", row.get("Rank", 9999))
                    if new_score > existing_score or (new_score == existing_score and new_rank < existing_rank):
                        key_to_row[key] = row.to_dict()
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

            # 若原始方向與正則 key 不同，翻轉 OLS 相關欄位以確保 Y=key[0]、X=key[1]
            orig_a = str(row.get("Ticker_A", ""))
            if orig_a != key[0]:
                hr = float(row.get("Hedge_Ratio", 1.0))
                safe_hr = hr if abs(hr) > 1e-8 else 1.0
                row["Hedge_Ratio"] = round(1.0 / safe_hr, 4)
                if row.get("OLS_Alpha") is not None:
                    row["OLS_Alpha"] = round(-float(row["OLS_Alpha"]) / safe_hr, 6)
                if "Spread_Mean" in row:
                    # 翻轉後 E[spread_new] = -OLS_Alpha / Hedge_Ratio（OLS 殘差均值=0 時 ≈ 0）
                    ols_alpha_val = float(row.get("OLS_Alpha") or 0.0)
                    row["Spread_Mean"] = round(-ols_alpha_val / safe_hr, 6)
                if "Spread_Std" in row:
                    row["Spread_Std"] = round(float(row["Spread_Std"]) / abs(safe_hr), 6)
                # 同步交換 Sector / Log_Mean / Log_Std / First_Price
                for col_a, col_b in [("Sector_A", "Sector_B"), ("Log_Mean_A", "Log_Mean_B"),
                                      ("Log_Std_A", "Log_Std_B"), ("First_Price_A", "First_Price_B")]:
                    if col_a in row and col_b in row:
                        row[col_a], row[col_b] = row[col_b], row[col_a]

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
