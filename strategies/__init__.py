"""
Strategies 模組
================

本模組包含用於配對交易 (Pairs Trading) 的回測策略與相關工具：
1. db_utils.py: 資料庫連線、指標計算與交易紀錄 (CSV) 匯入工具。
2. ssd_basic.py: 基於 SSD 與 Z-Score 的基本滾動配對交易回測系統 (單因子)。
3. ssd.py: 完整版 SSD 配對交易滾動回測系統。
4. HDBSCAN.py: 基於機器學習/降維 (PCA/UMAP) 與密度聚類 (HDBSCAN) 的配對交易回測系統。
5. HDBSCAN_MultiFactor.py: 結合多因子特徵進行 HDBSCAN 聚類的進階配對交易回測系統。
"""
