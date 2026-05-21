import json
from pathlib import Path

def main():
    notebook_path = Path("notebooks/HDBSCAN.ipynb")
    if not notebook_path.exists():
        print(f"Error: {notebook_path} does not exist.")
        return
        
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    patched_count = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
            
        source = cell.get("source", [])
        source_str = "".join(source)
        
        modified = False
        
        # 1. Formation init signature
        target_sig = '        # UMAP 參數（S&P500 規模下常開）\n'
        if target_sig in source_str and 'reduce_method: str = "umap"' not in source_str:
            source_str = source_str.replace(
                target_sig,
                '        # 降維方法：\'umap\' 或 \'pca\'\n        reduce_method: str = "umap",\n        # UMAP 參數（S&P500 規模下常開）\n'
            )
            modified = True
            
        # 2. Formation init body
        target_body = (
            '        # UMAP 常開（S&P500 規模下效果最佳）\n'
            '        if not UMAP_AVAILABLE:\n'
            '            raise RuntimeError("umap-learn 未安裝，請執行：pip install umap-learn")\n'
            '        self.umap_n_components = umap_n_components\n'
        )
        if target_body in source_str and 'self.reduce_method = reduce_method.lower()' not in source_str:
            source_str = source_str.replace(
                target_body,
                '        self.reduce_method = reduce_method.lower()\n'
                '        if self.reduce_method == "umap" and not UMAP_AVAILABLE:\n'
                '            raise RuntimeError("umap-learn 未安裝，請執行：pip install umap-learn")\n'
                '        self.umap_n_components = umap_n_components\n'
            )
            modified = True
            
        # 3. Insert _pca_reduce before _hdbscan_cluster
        target_hdbscan = '    # ── Step 3：HDBSCAN 分群 ────────────────────────────────────────────────\n'
        if target_hdbscan in source_str and 'def _pca_reduce' not in source_str:
            pca_code = (
                '    # ── Step 2.5：PCA 降維（穩健性對照）─────────────────────────────────────\n'
                '    def _pca_reduce(self, X: np.ndarray) -> np.ndarray:\n'
                '        from sklearn.decomposition import PCA\n'
                '        n_stocks = X.shape[0]\n'
                '        n_comp   = min(self.umap_n_components, n_stocks - 1)\n'
                '        if n_comp < 1:\n'
                '            return X\n'
                '        pca = PCA(n_components=n_comp, random_state=self.umap_random_state)\n'
                '        return pca.fit_transform(X)\n\n'
            )
            source_str = source_str.replace(target_hdbscan, pca_code + target_hdbscan)
            modified = True
            
        # 4. Formation run reduce method call
        target_run_umap = (
            '        # Step 2：UMAP 降維（常開）\n'
            '        X_embed = self._umap_reduce(X)\n'
        )
        if target_run_umap in source_str and 'if self.reduce_method == "pca":' not in source_str:
            source_str = source_str.replace(
                target_run_umap,
                '        # Step 2：降維 (UMAP 或 PCA)\n'
                '        if self.reduce_method == "pca":\n'
                '            X_embed = self._pca_reduce(X)\n'
                '        else:\n'
                '            X_embed = self._umap_reduce(X)\n'
            )
            modified = True
            
        # 5. RollingBacktester init
        target_backtester_init = (
            '        # EG + ADF p 值參數\n'
            '        adf_max_lags: int,\n'
            '        adf_pvalue_threshold: float,   # 0.01=保守 / 0.05=積極\n'
            '        output_dir: Path,\n'
            '    ):\n'
        )
        if target_backtester_init in source_str and 'reduce_method: str = "umap"' not in source_str:
            source_str = source_str.replace(
                target_backtester_init,
                '        # EG + ADF p 值參數\n'
                '        adf_max_lags: int,\n'
                '        adf_pvalue_threshold: float,   # 0.01=保守 / 0.05=積極\n'
                '        output_dir: Path,\n'
                '        reduce_method: str = "umap",\n'
                '    ):\n'
            )
            modified = True
            
        # 6. RollingBacktester Formation initialization call
        target_formation_call = (
            '            formation = Formation(\n'
            '                price_df=form_data,\n'
            '                form_start=fs_str, form_end=fe_str,\n'
            '                top_n=max(self.top_n_list),\n'
            '                sector_mapping=sector_mapping,\n'
            '                min_tickers_for_pairing=self.min_tickers_for_pairing,\n'
            '                hdbscan_min_cluster_size=self.hdbscan_min_cluster_size,\n'
            '                hdbscan_min_samples=self.hdbscan_min_samples,\n'
            '                hdbscan_metric=self.hdbscan_metric,\n'
            '                umap_n_components=self.umap_n_components,\n'
            '                umap_n_neighbors=self.umap_n_neighbors,\n'
            '                umap_min_dist=self.umap_min_dist,\n'
            '                umap_random_state=self.umap_random_state,\n'
            '                adf_max_lags=self.adf_max_lags,\n'
            '                adf_pvalue_threshold=self.adf_pvalue_threshold,\n'
            '            )\n'
        )
        if target_formation_call in source_str and 'reduce_method=getattr(self, "reduce_method", "umap")' not in source_str:
            source_str = source_str.replace(
                target_formation_call,
                '            formation = Formation(\n'
                '                price_df=form_data,\n'
                '                form_start=fs_str, form_end=fe_str,\n'
                '                top_n=max(self.top_n_list),\n'
                '                sector_mapping=sector_mapping,\n'
                '                min_tickers_for_pairing=self.min_tickers_for_pairing,\n'
                '                hdbscan_min_cluster_size=self.hdbscan_min_cluster_size,\n'
                '                hdbscan_min_samples=self.hdbscan_min_samples,\n'
                '                hdbscan_metric=self.hdbscan_metric,\n'
                '                umap_n_components=self.umap_n_components,\n'
                '                umap_n_neighbors=self.umap_n_neighbors,\n'
                '                umap_min_dist=self.umap_min_dist,\n'
                '                umap_random_state=self.umap_random_state,\n'
                '                adf_max_lags=self.adf_max_lags,\n'
                '                adf_pvalue_threshold=self.adf_pvalue_threshold,\n'
                '                reduce_method=getattr(self, "reduce_method", "umap"),\n'
                '            )\n'
            )
            modified = True
            
        # 7. Filename exporting
        target_filename = (
            '                sl_str   = f"SL{int(sl*100)}" if sl > 0 else "SL0"\n'
            '                filename = f"HDBSCAN_TradeLogs_Top{n}_{sl_str}_ZWin{z_win}.csv"\n'
        )
        if target_filename in source_str and 'rm_str' not in source_str:
            source_str = source_str.replace(
                target_filename,
                '                sl_str   = f"SL{int(sl*100)}" if sl > 0 else "SL0"\n'
                '                rm_str   = getattr(self, "reduce_method", "umap").upper()\n'
                '                filename = f"HDBSCAN_{rm_str}_TradeLogs_Top{n}_{sl_str}_ZWin{z_win}.csv"\n'
            )
            modified = True
            
        if modified:
            # 將 source_str 分解回 lines 串列，保持原本 JSON 結構
            # 每行需包含 \n，並注意分割方式
            lines = []
            parts = source_str.split("\n")
            for i, p in enumerate(parts):
                if i < len(parts) - 1:
                    lines.append(p + "\n")
                else:
                    if p:
                        lines.append(p)
            cell["source"] = lines
            patched_count += 1
            
    if patched_count > 0:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Successfully patched {patched_count} cells in {notebook_path}.")
    else:
        print("No cells matched patching targets. They may already be patched.")

if __name__ == "__main__":
    main()
