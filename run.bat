@echo off

echo ============================================================
echo      Pairs Trading Project - Dashboard Launcher
echo ============================================================
echo.

rem 1. Check if virtual environment exists
if exist "Project\Scripts\activate.bat" goto :VENV_OK

echo [WARNING] Virtual environment 'Project' is missing or incomplete!
pause
exit /b 1

:VENV_OK
rem 2. Activate virtual environment and run Streamlit
echo [INFO] Loading virtual environment (Project)...
call Project\Scripts\activate.bat

echo [INFO] Starting Pairs Trading Comparison Dashboard...
echo [INFO] If browser does not open automatically, go to: http://localhost:8501
echo.

Project\Scripts\python.exe -m streamlit run dashboard.py

if not errorlevel 1 goto :EOF
echo.
echo [ERROR] Streamlit failed to run. Please ensure all dependencies are installed.
echo [TIP] You can try running setup.bat again.
echo.
pause
