@echo off

echo ============================================================
echo      Pairs Trading Project - Environment Setup Tool
echo ============================================================
echo.

rem 1. Check if Python is installed
where python >nul 2>&1
if not errorlevel 1 goto :PYTHON_FOUND
echo [ERROR] Python not found. Please install Python (>= 3.8) and add it to PATH.
pause
exit /b 1

:PYTHON_FOUND
rem 2. Check and create virtual environment
if exist "Project\Scripts\activate.bat" goto :VENV_EXISTS
echo [INFO] Virtual environment 'Project' not found. Creating one...
python -m venv Project
if not errorlevel 1 goto :VENV_CREATED
echo [ERROR] Failed to create virtual environment.
pause
exit /b 1

:VENV_EXISTS
echo [INFO] Existing virtual environment 'Project' detected.
goto :VENV_OK

:VENV_CREATED
echo [SUCCESS] Virtual environment 'Project' created successfully!

:VENV_OK
echo.
echo [INFO] Activating virtual environment and installing dependencies...
echo [INFO] Loading Project\Scripts\activate.bat...
call Project\Scripts\activate.bat

echo.
echo [1/3] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [2/3] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if not errorlevel 1 goto :REQ_OK
echo [WARNING] Some dependencies failed to install. Please check network/logs.
goto :LOCAL_INSTALL

:REQ_OK
echo [SUCCESS] Requirements installed successfully!

:LOCAL_INSTALL
echo.
echo [3/3] Installing local src module in editable mode...
pip install -e .
if not errorlevel 1 goto :LOCAL_OK
echo [WARNING] Failed to install local src module.
goto :FINISH

:LOCAL_OK
echo [SUCCESS] Local src module installed successfully!

:FINISH
echo.
echo ============================================================
echo  Setup completed successfully! Run run.bat to start Dashboard.
echo ============================================================
pause
