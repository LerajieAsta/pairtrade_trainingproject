import inspect
from pathlib import Path
from strategies.HDBSCAN_UMAP import RollingBacktester

params = {
    "top_n_list": [5, 10, 20],
    "stop_loss_list": [0, 0.05, 0.15],
    "zscore_window_list": [0],
    "entry_z": 2.0,
    "exit_z": 0.0,
    "formation_window": 120,
    "trading_window": 126,
    "rolling_step": 21,
    "fee_rate": 0.001,
    "slippage_rate": 0.001,
    "initial_capital": 10000,
    "allow_reentry": False,
    "zscore_clip": 10.0,
    "min_spread_std": 1e-6,
    "min_tickers_for_pairing": 2,
    "use_vol_adjust": False,
    "max_holding_days": 30,
    "use_dynamic_stop": True,
    "hdbscan_min_cluster_size": 5,
    "hdbscan_min_samples": 2,
    "hdbscan_metric": "euclidean",
    "adf_max_lags": 1,
    "adf_pvalue_threshold": 0.01,
    "min_corr": 0.50,
    "min_zero_crossings": 5,
    "reduce_method": "umap",
    "umap_n_components": 5,
    "umap_n_neighbors": 40,
    "umap_min_dist": 0.01,
    "umap_random_state": 42,
}

init_sig = inspect.signature(RollingBacktester.__init__)
valid_params = {}
for param_name, param in init_sig.parameters.items():
    if param_name in ('self', 'output_dir', 'db_method', 'dataset_name', 'db_path'):
        continue
    if param_name in params:
        valid_params[param_name] = params[param_name]
    elif param.default is not inspect.Parameter.empty:
        valid_params[param_name] = param.default

print("valid_params:", valid_params)

try:
    engine = RollingBacktester(
        output_dir=Path("results/test"),
        db_method="HDBSCAN (UMAP)",
        dataset_name="TestDataset",
        db_path="results/test.db",
        **valid_params
    )
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
