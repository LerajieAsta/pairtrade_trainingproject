@echo off
rem ====================================================================
rem  pt.bat - S&P 500 Pairs Trading unified entry
rem  Usage: pt <command>   (no command = help)
rem ====================================================================
setlocal
cd /d "%~dp0"

set "PY=Project\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

if "%~1"=="" goto help
if /i "%~1"=="help" goto help
if /i "%~1"=="setup" goto setup
if /i "%~1"=="status" goto status
if /i "%~1"=="formation" goto formation
if /i "%~1"=="trading" goto trading
if /i "%~1"=="all" goto all
if /i "%~1"=="dashboard" goto dashboard
if /i "%~1"=="slides" goto slides
if /i "%~1"=="variance" goto variance
if /i "%~1"=="snapshot" goto snapshot
if /i "%~1"=="fetch-price" goto fetchprice
if /i "%~1"=="fetch-fund" goto fetchfund
if /i "%~1"=="fetch-fmp" goto fetchfmp
echo [pt] unknown command: %1
echo.
goto help

:setup
call setup.bat
goto :eof

:status
"%PY%" tools\status.py %2
goto :eof

:formation
"%PY%" run_formation.py
goto :eof

:trading
"%PY%" run_trading.py
goto :eof

:all
"%PY%" run_formation.py
if errorlevel 1 goto :eof
"%PY%" run_trading.py
goto :eof

:dashboard
"%PY%" -m streamlit run dashboard.py
goto :eof

:slides
pushd notebooks
quarto render
popd
goto :eof

:variance
"%PY%" tools\run_drl_variance.py %2 %3
goto :eof

:snapshot
"%PY%" tools\snapshot_run.py %2 %3 %4
goto :eof

:fetchprice
"%PY%" fetch\SP500_Tiingo.py
goto :eof

:fetchfund
"%PY%" fetch\fundamentals_yfinance.py
goto :eof

:fetchfmp
"%PY%" fetch\fetch_fmp_fundamentals.py
goto :eof

:help
echo ============================================================
echo   pt ^<command^>  --  Pairs Trading workflow
echo ============================================================
echo   [status]
echo     pt status         專案狀態總覽(資料/形成期/交易期/投影片+建議動作)
echo     pt status --brief 只列缺項
echo   [env / data]
echo     pt setup          建立虛擬環境並安裝套件(= setup.bat)
echo     pt fetch-price    下載 Tiingo 價格資料
echo     pt fetch-fund     下載 yFinance 基本面快照
echo     pt fetch-fmp      下載 FMP Point-in-Time 基本面
echo   [backtest]
echo     pt formation      形成期(篩選配對)
echo     pt trading        交易期(逐日模擬)
echo     pt all            形成期+交易期連跑
echo   [results]
echo     pt dashboard      Streamlit 績效儀表板(= run.bat)
echo     pt slides         渲染 Quarto 投影片至 docs/slides/
echo   [tools]
echo     pt variance N     DRL 訓練變異數評估(N 輪)
echo     pt snapshot tag   歸檔 result.db(重跑前保留舊版)
echo ============================================================
goto :eof
