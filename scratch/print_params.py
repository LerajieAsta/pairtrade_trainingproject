import sys
import os

# Set arguments
sys.argv = ["main.py", "--db", "sp500_Current"]

from main import parse_args, resolve_paths

args = parse_args()
DB_PATH, OUTPUT_ROOT = resolve_paths(args)
allow_reentry = args.allow_reentry
use_vol_adjust = args.use_vol_adjust
reentry_suffix = "ReEntry" if allow_reentry else "NoReEntry"

FORMATION_WINDOW = 252
base_params = {
    "entry_z":                      2.0,
    "exit_z":                       0.0,
    "formation_window":             FORMATION_WINDOW,
    "trading_window":               126,
    "rolling_step":                 21,
    "fee_rate":                     0.001,
    "slippage_rate":                0.001,
    "initial_capital":              10000,
    "allow_reentry":                allow_reentry,
    "zscore_clip":                  10.0,
    "min_spread_std":               1e-6,
    "min_tickers_for_pairing":      2,
    "use_vol_adjust":               use_vol_adjust,
    "max_holding_days":             30,
    "top_n_list":                   [5, 10, 20],
    "stop_loss_list":               [0, 0.05, 0.15],
    "zscore_window_list":           [0],
    "use_vol_adjust_list":          [use_vol_adjust],
    "portfolio_stop_loss_pct_list": [0.0],
    "max_sector_ratio_list":        [0.0, 0.30, 0.50],
    "dynamic_stop_z_list":          [0.0],
}

hdbscan_common = {
    "use_dynamic_stop":         True,
    "hdbscan_min_cluster_size": 5,
    "hdbscan_min_samples":      2,
    "hdbscan_metric":           "euclidean",
    "adf_max_lags":             1,
    "adf_pvalue_threshold":     0.01,
    "min_corr":                 0.50,
    "min_zero_crossings":       5,
}

strategies_raw = [
    {
        "name":      "SSD Basic (基本配對距離)",
        "db_method": "SSD (Basic)",
        "module":    "strategies.ssd_basic",
        "sub_dir":   f"SSD_Basic_{reentry_suffix}",
        "params":    base_params,
    },
    {
        "name":      "SSD Rolling (優化殘差配對)",
        "db_method": "SSD",
        "module":    "strategies.ssd",
        "sub_dir":   f"SSD_{reentry_suffix}",
        "params":    base_params,
    },
    {
        "name":      "HDBSCAN Clustering + UMAP",
        "db_method": "HDBSCAN (UMAP)",
        "module":    "strategies.HDBSCAN_UMAP",
        "sub_dir":   f"HDBSCAN_UMAP_{reentry_suffix}",
        "params":  {
            **base_params,
            **hdbscan_common,
            "reduce_method":     "umap",
            "umap_n_components": 5,
            "umap_n_neighbors":  40,
            "umap_min_dist":     0.01,
            "umap_random_state": 42,
        },
    },
]

print("UMAP strategy raw params keys:")
print(list(strategies_raw[2]["params"].keys()))
print("UMAP strategy raw params:")
print(strategies_raw[2]["params"])
