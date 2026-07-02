@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo      Pairs Trading Project - Environment Setup Wizard
echo ============================================================
echo.

rem 1. Check Git LFS and fetch files
echo [INFO] Step 1: Checking Git LFS and fetching database/large files...
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Git command not found! Please ensure Git is installed and in your PATH.
) else (
    echo [INFO] Initializing Git LFS...
    git lfs install
    echo [INFO] Fetching Git LFS files...
    git lfs pull
    if !errorlevel! neq 0 (
        echo [WARNING] Git LFS pull failed or LFS is not installed. 
        echo [WARNING] Large data files might be missing. Please make sure git-lfs is installed.
    ) else (
        echo [SUCCESS] Git LFS files fetched successfully.
    )
)
echo.

rem 2. Check Python installation
echo [INFO] Step 2: Checking Python installation...
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.8+ and add it to your PATH.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%I in ('python --version') do set PY_VER=%%I
echo [INFO] Found Python version: !PY_VER!
echo.

rem 3. Create Virtual Environment
echo [INFO] Step 3: Setting up virtual environment...
if exist "Project\Scripts\activate.bat" (
    echo [INFO] Virtual environment 'Project' already exists. Skipping creation.
) else (
    echo [INFO] Creating virtual environment 'Project'...
    python -m venv Project
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment 'Project' created.
)
echo.

rem 4. Install Dependencies
echo [INFO] Step 4: Installing dependencies...
echo [INFO] Activating virtual environment...
call Project\Scripts\activate.bat

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing requirements from requirements.txt...
if exist "requirements.txt" (
    pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to install dependencies from requirements.txt!
        pause
        exit /b 1
    )
) else (
    echo [WARNING] requirements.txt not found! Skipping dependency installation.
)

echo [INFO] Installing local project in editable mode...
if exist "pyproject.toml" (
    pip install -e .
    if !errorlevel! neq 0 (
        echo [WARNING] Editable install failed!
    ) else (
        echo [SUCCESS] Editable install completed.
    )
) else (
    echo [INFO] pyproject.toml not found. Skipping editable install.
)
echo.

echo ============================================================
echo [SUCCESS] Project environment setup completed successfully!
echo [INFO] You can now start the Streamlit Dashboard using run.bat
echo ============================================================
echo.
pause
