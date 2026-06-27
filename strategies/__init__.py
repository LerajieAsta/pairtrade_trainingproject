"""
Strategies 模組
================

Formation 策略（strategies/formation/）：
  ssd_basic.py           — SSD Basic：靜態單期 SSD 配對篩選
  ssd_rolling.py         — SSD Rolling：滾動多期 SSD 配對篩選
  HDBSCAN_MultiScale.py  — HDBSCAN MultiScale：多時間尺度跨期穩定性聚類配對
  HDBSCAN_UMAP.py        — HDBSCAN UMAP：雙層特徵（單股10維 + 配對層10維）聚類配對
  DTW_Cointegration_Paper.py — DTW Paper：DTW / SSD-DTW-PCA 論文對齊版

舊版 HDBSCAN 策略（SS-UMAP/SS-PCA/MacroCluster/CrossSector 及對應 DRL LSTM）
已於 2026-06-27 移至 archive/11506/formation/。
"""
