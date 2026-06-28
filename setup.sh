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
