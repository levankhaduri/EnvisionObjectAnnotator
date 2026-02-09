# EnvisionObjectAnnotator Run Script (Windows)
# Run: .\run.ps1

Write-Host "`nStarting EnvisionObjectAnnotator...`n" -ForegroundColor Cyan

# Check if setup was done
if (-not (Test-Path "backend\.venv")) {
    Write-Host "ERROR: Backend not set up. Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "frontend\node_modules")) {
    Write-Host "ERROR: Frontend not set up. Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

# Start backend in new window
Write-Host "Starting backend API server..." -ForegroundColor Yellow
$backendScript = @"
cd '$PWD\backend'
.\.venv\Scripts\uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"@
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendScript

# Wait for backend to start
Start-Sleep -Seconds 3

# Start frontend in new window
Write-Host "Starting frontend dev server..." -ForegroundColor Yellow
$frontendScript = @"
cd '$PWD\frontend'
npm run dev
"@
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendScript

# Wait for frontend to start
Start-Sleep -Seconds 3

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  EnvisionObjectAnnotator Running!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nOpen in browser: http://localhost:5173" -ForegroundColor White
Write-Host "`nBackend API: http://localhost:8000" -ForegroundColor White
Write-Host "`nClose the PowerShell windows to stop the servers.`n" -ForegroundColor Yellow

# Try to open browser
Start-Process "http://localhost:5173"
