import sys
import os
import importlib
import pandas as pd

sys.path.append(os.getcwd())

from main import parse_args, resolve_paths
from strategies.ssd import DataProcessor

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
    "2015-06", "2016-12", 252
)

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
    "max_sector_ratio_list":        [0.0],
    "dynamic_stop_z_list":          [0.0],
}

strategy_config = {
    "name":         "Pure DTW (Notebook Ver)",
    "db_method":    "Pure_DTW",
    "dataset_name": "Current",
    "module":       "strategies.DTW_Pure_Notebook",
    "output_dir":   f"{OUTPUT_ROOT}/Pure_DTW_{reentry_suffix}_Test",
    "log_path":     f"{OUTPUT_ROOT}/logs/Pure_DTW_Test.log",
    "params":  {
        **base_params,
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
