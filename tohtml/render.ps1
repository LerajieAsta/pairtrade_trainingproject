Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host " [*] Pairs Trading Slide Deck - One-Click Quarto Compile Suite" -ForegroundColor Cyan
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1/3
Write-Host "[Step 1/3] Scanning local backtest CSV data..." -ForegroundColor Yellow
$env:PYTHONIOENCODING="utf-8"
python preprocess_equity.py
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Data preprocessing failed!" -ForegroundColor Red
    Write-Host "Please check:"
    Write-Host "1. Is python and pandas library installed on this PC?"
    Write-Host "2. Does the 'results/' directory exist under project root?"
    Write-Host ""
    Read-Host "Press Enter to exit..."
    exit $LASTEXITCODE
}

# Step 2/3
Write-Host ""
Write-Host "[Step 2/3] Generating offline interactive Plotly chart..." -ForegroundColor Yellow
python generate_plotly_iframe.py
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Offline Plotly HTML generation failed!" -ForegroundColor Red
    Write-Host "Please check if plotly library is installed?"
    Write-Host ""
    Read-Host "Press Enter to exit..."
    exit $LASTEXITCODE
}

# Step 3/3
Write-Host ""
Write-Host "[Step 3/3] Calling Quarto to render Revealjs presentation slides..." -ForegroundColor Yellow
quarto render notebooks/analysis.ipynb --to revealjs
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Quarto presentation render failed!" -ForegroundColor Red
    Write-Host "Please ensure Quarto is installed and added to your PATH environment variable."
    Write-Host ""
    Read-Host "Press Enter to exit..."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "=======================================================================" -ForegroundColor Green
Write-Host " [SUCCESS] Congratulations! Slide deck rendered successfully!" -ForegroundColor Green
Write-Host " Launching notebooks/analysis.html in your default browser..." -ForegroundColor Green
Write-Host "=======================================================================" -ForegroundColor Green
Write-Host ""

if (Test-Path "notebooks/analysis.html") {
    Start-Process "notebooks/analysis.html"
}
Read-Host "Press Enter to exit..."
