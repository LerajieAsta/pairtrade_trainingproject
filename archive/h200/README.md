# H200 GPU 伺服器相關檔案（2026-07-06 封存）

專案已不再使用 H200 雲端 GPU 伺服器，以下檔案封存留檔：

| 檔案 | 原位置 | 用途 |
|------|--------|------|
| `benchmark_h200.py` | repo 根目錄 | H200 GPU 效能壓力測試（GEMM TFLOPS、VRAM 佔用、燒機） |
| `pack_results.py` | repo 根目錄 | 大檔傳輸打包工具（zstd 壓縮 + 切塊 + SHA256），用於從 GPU 伺服器抓回 results/ |
| `setup.sh` | repo 根目錄 | Linux 伺服器環境初始化腳本（Git LFS + venv + requirements） |

本機（Windows）環境初始化請用 `setup.bat`。
DRL 策略在本機為 CPU-only 執行，並行數由 `strategies/config.py` 的
`DRL_MAX_WORKERS` 控制（torch.set_num_threads(1)，每 worker 佔 1 核）。
