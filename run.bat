@echo off

echo ============================================================
echo      Pairs Trading Project - Dashboard Launcher
echo ============================================================
echo.

rem 1. Check if virtual environment exists
if exist "Project\Scripts\activate.bat" goto :VENV_OK

echo [WARNING] Virtual environment 'Project' is missing or incomplete!
echo [INFO] Please initialize the environment before starting the Dashboard.
echo.
set /p choice="Do you want to run setup.bat now? (Y/N): "
if /i "%choice%"=="Y" goto :RUN_SETUP
echo [INFO] Startup canceled.
exit /b 0

:RUN_SETUP
call setup.bat
if exist "Project\Scripts\activate.bat" goto :VENV_OK
echo [ERROR] Virtual environment still missing. Aborting.
pause
exit /b 1

:VENV_OK
rem 2. Activate virtual environment and run Streamlit
echo [INFO] Loading virtual environment (Project)...
call Project\Scripts\activate.bat

echo [INFO] Starting Pairs Trading Comparison Dashboard...
echo [INFO] If browser does not open automatically, go to: http://localhost:8501
echo.

streamlit run app.py

if not errorlevel 1 goto :EOF
echo.
echo [ERROR] Streamlit failed to run. Please ensure all dependencies are installed.
echo [TIP] You can try running setup.bat again.
echo.
pause
