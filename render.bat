@echo off
echo ============================================================
echo      Pairs Trading Presentation Slide Compiler
echo ============================================================
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

echo.
set /p OUT_NAME="Enter output HTML filename [default: index]: "
if "%OUT_NAME%"=="" set OUT_NAME=index

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
if exist "docs\%OUT_NAME%_files" rmdir /s /q "docs\%OUT_NAME%_files"

rem Move generated files from project root to docs/
if exist "%OUT_NAME%.html" (
    move "%OUT_NAME%.html" "docs\%OUT_NAME%.html" >nul
) else (
    echo [WARNING] Generated file %OUT_NAME%.html not found.
)

if exist "%OUT_NAME%_files" (
    move "%OUT_NAME%_files" "docs\%OUT_NAME%_files" >nul
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
