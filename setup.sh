#!/bin/bash
set -e

echo "============================================================"
echo "     Pairs Trading Project - Environment Setup Wizard (Bash)"
echo "============================================================"
echo

# 1. Git LFS
echo "[INFO] Step 1: Checking Git LFS and fetching files..."
if ! command -v git &> /dev/null; then
    echo "[WARNING] Git command not found!"
else
    # Auto-install git-lfs if not available
    if ! git lfs version &> /dev/null; then
        echo "[INFO] git-lfs not found. Installing to ~/bin ..."
        mkdir -p "$HOME/bin"
        LFS_TAR="$HOME/git-lfs.tar.gz"
        LFS_DIR="$HOME/git-lfs-3.5.1"
        curl -L https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz -o "$LFS_TAR"
        tar -xzf "$LFS_TAR" -C "$HOME"
        cp "$LFS_DIR/git-lfs" "$HOME/bin/git-lfs"
        rm -f "$LFS_TAR"
        export PATH="$HOME/bin:$PATH"
        echo "[SUCCESS] git-lfs installed to ~/bin"
        echo "[INFO] Add the following line to your ~/.bashrc to make it permanent:"
        echo "       export PATH=\"\$HOME/bin:\$PATH\""
    fi
    echo "[INFO] Initializing Git LFS..."
    git lfs install
    echo "[INFO] Fetching Git LFS files..."
    git lfs pull
fi
echo

# 2. Check Python
echo "[INFO] Step 2: Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "[ERROR] Python not found! Please install Python 3.8+"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

$PYTHON_CMD --version
echo

# 3. Create Virtual Environment
echo "[INFO] Step 3: Setting up virtual environment..."
VENV_DIR="Project"
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    echo "[INFO] Virtual environment '$VENV_DIR' already exists. Skipping creation."
else
    echo "[INFO] Creating virtual environment '$VENV_DIR'..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo "[SUCCESS] Virtual environment '$VENV_DIR' created."
fi
echo

# 4. Install Dependencies
echo "[INFO] Step 4: Installing dependencies..."
source "$VENV_DIR/bin/activate"

echo "[INFO] Upgrading pip..."
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo "[INFO] Installing requirements..."
    pip install -r requirements.txt
else
    echo "[WARNING] requirements.txt not found!"
fi

if [ -f "pyproject.toml" ]; then
    echo "[INFO] Installing local project in editable mode..."
    pip install -e .
fi
echo

echo "============================================================"
echo "[SUCCESS] Project environment setup completed successfully!"
echo "============================================================"
