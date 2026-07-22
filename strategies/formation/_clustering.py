"""
形成期共用分群層（與特徵、排序無關）
======================================================================
把「特徵矩陣 → 群標籤」抽成中性 dispatcher，讓分群方法可插拔：
    labels = cluster("hdbscan"|"agglomerative"|"kmeans", X, **params)

各 backend 只負責分群，不碰特徵萃取或群內排序。新增分群法
（如 K-means、Spectral）只需在此加一個 backend + 在 config 指定
`cluster_method`，不必更動任何既有策略模組。

噪音 / 未分群點以 label = -1 表示（下游配對流程映射為 "Unknown" 排除）。
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# HDBSCAN 後端偵測（優先用獨立 hdbscan 套件，否則退回 sklearn 內建）
try:
    import hdbscan as _hdbscan_pkg
    _HDBSCAN_LIB = "hdbscan"
except ImportError:
    try:
        from sklearn.cluster import HDBSCAN as _sklearn_HDBSCAN
        _HDBSCAN_LIB = "sklearn"
    except ImportError:
        _HDBSCAN_LIB = None


def cluster_hdbscan(X: np.ndarray, min_cluster_size: int = 5,
                    min_samples: int = 2, metric: str = "euclidean") -> np.ndarray:
    """
    HDBSCAN 密度聚類。群大小 / min_samples 找不到群落時逐步減半重試；
    仍失敗回全噪音（-1）。與 HDBSCAN_UMAP._hdbscan_cluster 逐位元等價。
    """
    min_cs = min_cluster_size
    min_samp = min_samples
    while min_cs >= 2:
        cs = min(min_cs, max(2, X.shape[0] // 5))
        samp = max(1, min(min_samp, cs - 1))

        if _HDBSCAN_LIB == "hdbscan":
            clf = _hdbscan_pkg.HDBSCAN(
                min_cluster_size=cs, min_samples=samp,
                metric=metric, core_dist_n_jobs=-1)
        else:
            clf = _sklearn_HDBSCAN(
                min_cluster_size=cs, min_samples=samp,
                metric=metric, n_jobs=-1)

        labels = clf.fit_predict(X)
        if len(set(labels) - {-1}) >= 1:
            return labels

        min_cs //= 2
        min_samp = max(1, min_samp // 2)

    print("  [Formation] HDBSCAN 未找到任何群落。")
    return np.full(X.shape[0], -1)


def cluster_agglomerative(X: np.ndarray, linkage: str = "average",
                          threshold_percentile: float = 75.0) -> np.ndarray:
    """
    Agglomerative 階層聚類，distance_threshold 依合併距離分位數自動校準。
    每點皆分配群組（無噪音概念）。與 agglomerative_*._cluster 逐位元等價。
    """
    n = X.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)

    probe = AgglomerativeClustering(
        n_clusters=1, linkage=linkage, compute_distances=True).fit(X)
    distances = probe.distances_
    threshold = float(np.percentile(distances, threshold_percentile)) if len(distances) > 0 else 0.0
    if threshold <= 0.0:
        threshold = float(np.max(distances)) if len(distances) > 0 else 1e-6
        if threshold <= 0.0:
            threshold = 1e-6

    final = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, linkage=linkage).fit(X)
    return final.labels_


def cluster_kmeans(X: np.ndarray, n_clusters: int = 10,
                   random_state: int = 42) -> np.ndarray:
    """
    K-means 分群（新 backend）。需預先指定群數；每點皆分配群組。
    注意：K-means 假設等向球狀群且對特徵尺度敏感——因子載荷已以
    sqrt(特徵值) 加權，尺度有意義，但群數需由使用者/網格決定。
    """
    from sklearn.cluster import KMeans
    n = X.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)
    k = max(1, min(n_clusters, n - 1))
    return KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X)


_BACKENDS = {
    "hdbscan": cluster_hdbscan,
    "agglomerative": cluster_agglomerative,
    "kmeans": cluster_kmeans,
}


def cluster(method: str, X: np.ndarray, **params) -> np.ndarray:
    """
    分群 dispatcher。
        method: "hdbscan" | "agglomerative" | "kmeans"
        params: 傳給對應 backend 的參數（多餘的鍵會被過濾）
    回傳長度 N 的整數標籤（-1 = 噪音，僅 HDBSCAN 會產生）。
    """
    fn = _BACKENDS.get(method)
    if fn is None:
        raise ValueError(f"未知分群方法 '{method}'，可選：{list(_BACKENDS)}")
    import inspect
    valid = set(inspect.signature(fn).parameters) - {"X"}
    return fn(X, **{k: v for k, v in params.items() if k in valid})
