import json
from pathlib import Path

def main():
    src_path = Path("notebooks/HDBSCAN.ipynb")
    dst_path = Path("notebooks/HDBSCAN_Autoencoder.ipynb")
    
    if not src_path.exists():
        print(f"Error: {src_path} does not exist.")
        return
        
    with open(src_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    # 修改 metadata 與 Markdown
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "raw" and "非監督式學習配對交易策略實作與回測" in "".join(cell.get("source", [])):
            cell["source"] = [
                "---\n",
                "title: \"深度表徵學習配對交易策略實作與回測\"\n",
                "subtitle: \"以 S&P 500 為例\"\n",
                "author: \"李伯修\"\n",
                "date: \"2026-05-20\"\n",
                "format:\n",
                "  revealjs:\n",
                "    navigation-mode: default\n",
                "    slide-number: h.v\n",
                "    scrollable: true\n",
                "    width: 1600\n",
                "    height: 900\n",
                "    controls: true\n",
                "    progress: true\n",
                "execute:\n",
                "  echo: true\n",
                "  output-location: slide\n",
                "---"
            ]
        elif cell.get("cell_type") == "markdown" and "HDBSCAN 分群配對交易系統" in "".join(cell.get("source", [])):
            cell["source"] = [
                "# 🧠 深度學習 Autoencoder-HDBSCAN 配對交易系統\n",
                "\n",
                "本文件詳細說明系統中核心的深度自編碼器 (Autoencoder) 無監督特徵選股、HDBSCAN 分群，以及共整合交易邏輯。"
            ]
            
    # 修改 code cell 中的程式碼
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
            
        source_str = "".join(cell.get("source", []))
        
        # 1. 替換開頭的說明文字和引入 PyTorch
        if "HDBSCAN 分群配對交易滾動回測系統" in source_str:
            target_import = "from sklearn.preprocessing import StandardScaler\n"
            replacement_import = (
                "from sklearn.preprocessing import StandardScaler\n"
                "import torch\n"
                "import torch.nn as nn\n"
                "import torch.optim as optim\n"
            )
            source_str = source_str.replace("HDBSCAN 分群配對交易滾動回測系統", "Autoencoder-HDBSCAN 深度表徵配對交易滾動回測系統")
            source_str = source_str.replace(target_import, replacement_import)
            
            # 刪除 _extract_features 函數，改為 MLPAutoencoder 與 train_autoencoder
            target_feat_fn = (
                "def _extract_features(log_price: np.ndarray) -> np.ndarray:\n"
                "    \"\"\"\n"
                "    從對數價格序列萃取多維特徵向量，用於 HDBSCAN 分群。\n"
            )
            # 我們需要尋找 _extract_features 的整段代碼
            # 在 python 中，我們可以直接用 replace 替換 _extract_features 整個部分
            # 讓我們先看看 source_str 裡有沒有這個函數
            if "def _extract_features" in source_str:
                # 找到 _extract_features 到 return features 結尾
                fn_start = source_str.find("def _extract_features")
                fn_end = source_str.find("return features") + len("return features")
                
                # 新的 Autoencoder 定義
                ae_code = (
                    "class MLPAutoencoder(nn.Module):\n"
                    "    def __init__(self, input_dim, latent_dim=8):\n"
                    "        super(MLPAutoencoder, self).__init__()\n"
                    "        self.encoder = nn.Sequential(\n"
                    "            nn.Linear(input_dim, 64),\n"
                    "            nn.Tanh(),\n"
                    "            nn.Linear(64, latent_dim)\n"
                    "        )\n"
                    "        self.decoder = nn.Sequential(\n"
                    "            nn.Linear(latent_dim, 64),\n"
                    "            nn.Tanh(),\n"
                    "            nn.Linear(64, input_dim)\n"
                    "        )\n"
                    "    def forward(self, x):\n"
                    "        latent = self.encoder(x)\n"
                    "        decoded = self.decoder(latent)\n"
                    "        return latent, decoded\n\n"
                    "def train_autoencoder(X_train, latent_dim=8, epochs=100, lr=0.01):\n"
                    "    tensor_x = torch.tensor(X_train, dtype=torch.float32)\n"
                    "    input_dim = X_train.shape[1]\n"
                    "    model = MLPAutoencoder(input_dim, latent_dim)\n"
                    "    optimizer = optim.Adam(model.parameters(), lr=lr)\n"
                    "    criterion = nn.MSELoss()\n"
                    "    \n"
                    "    model.train()\n"
                    "    for epoch in range(epochs):\n"
                    "        optimizer.zero_grad()\n"
                    "        latent, decoded = model(tensor_x)\n"
                    "        loss = criterion(decoded, tensor_x)\n"
                    "        loss.backward()\n"
                    "        optimizer.step()\n"
                    "        \n"
                    "    model.eval()\n"
                    "    with torch.no_grad():\n"
                    "        latent_features, _ = model(tensor_x)\n"
                    "    return latent_features.numpy()"
                )
                
                source_str = source_str[:fn_start] + ae_code + source_str[fn_end:]
                
        # 2. 修改 Formation 類別的 _build_feature_matrix，使其提取原始對數報酬序列
        if "def _build_feature_matrix(self) -> tuple[np.ndarray, list[str]]:" in source_str:
            new_build_feat = (
                "    def _build_feature_matrix(self) -> tuple[np.ndarray, list[str]]:\n"
                "        log_prices = np.log(self.price_df)\n"
                "        valid_tickers = []\n"
                "        ret_rows = []\n"
                "        for ticker in log_prices.columns:\n"
                "            series = log_prices[ticker].values\n"
                "            if len(series) < 30 or not np.all(np.isfinite(series)):\n"
                "                continue\n"
                "            ret = np.diff(series)\n"
                "            ret_rows.append(ret)\n"
                "            valid_tickers.append(ticker)\n\n"
                "        if not ret_rows:\n"
                "            return np.empty((0, 0)), []\n\n"
                "        X = np.vstack(ret_rows)                    # (n_stocks, seq_len)\n"
                "        X = StandardScaler().fit_transform(X)        # 標準化\n"
                "        return X, valid_tickers"
            )
            # 尋找 _build_feature_matrix 函數體
            fn_start = source_str.find("    def _build_feature_matrix(self)")
            fn_end = source_str.find("        return X, valid_tickers") + len("        return X, valid_tickers")
            source_str = source_str[:fn_start] + new_build_feat + source_str[fn_end:]
            
        # 3. 修改 Formation.run，調用 Autoencoder 提取特徵
        if "X_embed = self._pca_reduce(X)" in source_str:
            source_str = source_str.replace("X_embed = self._pca_reduce(X)", "X_embed = self._pca_reduce(latent_features)")
            source_str = source_str.replace("X_embed = self._umap_reduce(X)", "X_embed = self._umap_reduce(latent_features)")
            
            target_run_step1 = (
                "        # Step 1：特徵萃取\n"
                "        X, valid_tickers = self._build_feature_matrix()\n"
            )
            replacement_run_step1 = (
                "        # Step 1：提取報酬率序列矩陣，並利用 Autoencoder 訓練深度特徵\n"
                "        X_raw, valid_tickers = self._build_feature_matrix()\n"
                "        if len(valid_tickers) < self.min_tickers_for_pairing:\n"
                "            self.selected_pairs = pd.DataFrame()\n"
                "            return self.selected_pairs\n\n"
                "        # 訓練自編碼器，壓縮為 8 維深度特徵\n"
                "        latent_features = train_autoencoder(X_raw, latent_dim=8, epochs=100, lr=0.01)\n"
            )
            source_str = source_str.replace(target_run_step1, replacement_run_step1)
            # 移除多餘的 check
            source_str = source_str.replace(
                "        if len(valid_tickers) < self.min_tickers_for_pairing:\n"
                "            self.selected_pairs = pd.DataFrame()\n"
                "            return self.selected_pairs\n"
                "\n"
                "        # Step 2",
                "        # Step 2"
            )
            
        # 4. 修改 RollingBacktester 的 _export_results，使其輸出含 HDBSCAN_AE 的檔名
        if "filename = f\"HDBSCAN_{rm_str}_TradeLogs" in source_str:
            source_str = source_str.replace(
                "filename = f\"HDBSCAN_{rm_str}_TradeLogs",
                "filename = f\"HDBSCAN_AE_{rm_str}_TradeLogs"
            )
            
        # 5. 修改 main 區塊的輸出路徑
        if "OUTPUT_DIR    = Path(r\"../results/current/HDBSCAN_NoReEntry\")" in source_str:
            source_str = source_str.replace(
                "OUTPUT_DIR    = Path(r\"../results/current/HDBSCAN_NoReEntry\")",
                "OUTPUT_DIR    = Path(r\"../results/current/HDBSCAN_AE_NoReEntry\")"
            )
            
        # 將 source_str 分解回 lines 串列，保持原本 JSON 結構
        lines = []
        parts = source_str.split("\n")
        for i, p in enumerate(parts):
            if i < len(parts) - 1:
                lines.append(p + "\n")
            else:
                if p:
                    lines.append(p)
        cell["source"] = lines

    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Successfully generated {dst_path} based on HDBSCAN.ipynb.")

if __name__ == "__main__":
    main()
