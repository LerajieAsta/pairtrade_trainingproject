@echo off
rem Adaptive workspace directory switching
if exist "preprocess_equity.py" (
    if exist "..\Project" (
        cd ..
        echo [INFO] Auto-Detection: Running inside tohtml/. Switching to parent root...
    )
)

echo ============================================================
echo      Pairs Trading Presentation Slide Compiler
===========================================================
echo.

set PYTHON_EXE=Project\Scripts\python.exe

if not exist %PYTHON_EXE% (
    echo [ERROR] Virtual environment Python not found!
    echo Please run setup.bat first to initialize.
    pause
    exit /b 1
)

echo [1/3] Compiling optimal equity curves and injecting metrics...
%PYTHON_EXE% tohtml/preprocess_equity.py
if errorlevel 1 (
    echo.
    echo [ERROR] Preprocessing failed!
    pause
    exit /b 1
)

echo.
echo [2/3] Generating interactive Plotly charts...
%PYTHON_EXE% tohtml/generate_plotly_iframe.py
if errorlevel 1 (
    echo.
    echo [ERROR] Plotly chart generation failed!
    pause
    exit /b 1
)

rem Read Minguo date default name from temp file
set DEFAULT_NAME=report
if exist tmp\default_name.txt (
    set /p DEFAULT_NAME=<tmp\default_name.txt
)

echo.
set /p OUT_NAME="Enter output HTML filename [default: %DEFAULT_NAME%]: "
if "%OUT_NAME%"=="" set OUT_NAME=%DEFAULT_NAME%

rem Safety: Block 'index' to prevent overwriting Documentation Hub homepage
if /i "%OUT_NAME%"=="index" (
    echo.
    echo [WARNING] 'index' is reserved for the Documentation Hub homepage!
    echo Automatic fallback to default name: %DEFAULT_NAME%
    set OUT_NAME=%DEFAULT_NAME%
)

rem Safeguard: Strip any spaces to prevent filename errors
set OUT_NAME=%OUT_NAME: =%

echo.
echo [3/3] Calling Quarto to render RevealJS presentation slides...
echo Output target: docs/%OUT_NAME%.html
echo.

rem Quarto outputs to notebooks/ directory (relative to input notebook)
quarto render notebooks/analysis.ipynb --to revealjs --output "%OUT_NAME%.html"
if errorlevel 1 (
    echo.
    echo [ERROR] Quarto render failed! Please ensure Quarto is installed and in PATH.
    pause
    exit /b 1
)

if not exist docs mkdir docs

rem Clean up existing files in docs if they exist
if exist "docs\%OUT_NAME%.html" del /f /q "docs\%OUT_NAME%.html"

rem Move generated HTML from project root to docs/
if exist "%OUT_NAME%.html" (
    move "%OUT_NAME%.html" "docs\%OUT_NAME%.html" >nul
) else (
    echo [WARNING] Generated file %OUT_NAME%.html not found.
)

echo.
echo [INFO] Syncing RevealJS library and Plotly iframes to docs/...
if exist "notebooks\analysis_files" (
    xcopy "notebooks\analysis_files" "docs\analysis_files" /E /I /Y /Q >nul
)
if exist "notebooks\iframe_figures" (
    xcopy "notebooks\iframe_figures" "docs\iframe_figures" /E /I /Y /Q >nul
)

echo.
echo ============================================================
echo  [SUCCESS] Slides rendered and published to docs/ folder!
echo  Saved to: docs/%OUT_NAME%.html
echo ============================================================
echo.

if exist "docs\%OUT_NAME%.html" (
    start "" "docs\%OUT_NAME%.html"
)
pause
