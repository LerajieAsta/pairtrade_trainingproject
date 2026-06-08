import sys
import os
import importlib
import time
import pandas as pd

sys.path.append(os.getcwd())

from main import parse_args, resolve_paths
from strategies.HDBSCAN import DataProcessor

# Set up dummy args
sys.argv = ["main.py", "--db", "sp500_Current", "--dry-run"]
args = parse_args()
DB_PATH, OUTPUT_ROOT = resolve_paths(args)
allow_reentry = args.allow_reentry
use_vol_adjust = args.use_vol_adjust
reentry_suffix = "ReEntry" if allow_reentry else "NoReEntry"

print("Loading data...")
processor = DataProcessor(db_path=DB_PATH, table_name="Daily_Prices")
sector_mapping = processor.load_sector_mapping("Constituents", "Symbol", "GICS_Sector")
price_pivot, all_dates, total_days, local_first_trade_idx = processor.prepare_backtest_data(
    "2000-01", "2025-12", 252
)

# Build UMAP config
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

base_params = {
    "entry_z":                      2.0,
    "exit_z":                       0.0,
    "formation_window":             252,
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

strategy_config = {
    "name":         "HDBSCAN Clustering + UMAP",
    "db_method":    "HDBSCAN (UMAP)",
    "dataset_name": "Current",
    "module":       "strategies.HDBSCAN_UMAP",
    "output_dir":   f"{OUTPUT_ROOT}/HDBSCAN_UMAP_{reentry_suffix}",
    "log_path":     f"{OUTPUT_ROOT}/logs/HDBSCAN_Clustering___UMAP.log",
    "params":  {
        **base_params,
        **hdbscan_common,
        "reduce_method":     "umap",
        "umap_n_components": 5,
        "umap_n_neighbors":  40,
        "umap_min_dist":     0.01,
        "umap_random_state": 42,
    },
    "db_path":      DB_PATH,
}

name = strategy_config["name"]
module_path = strategy_config["module"]
params = strategy_config["params"]
output_dir = strategy_config["output_dir"]

module = importlib.import_module(module_path)

print("Running strategy run_strategy directly in test script...")
try:
    module.run_strategy(
        price_pivot=price_pivot,
        all_dates=all_dates,
        total_days=total_days,
        local_first_trade_idx=local_first_trade_idx,
        sector_mapping=sector_mapping,
        params=params,
        output_dir=output_dir,
        db_method=strategy_config["db_method"],
        dataset_name=strategy_config["dataset_name"],
        db_path=strategy_config.get("db_path")
    )
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
