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

# Poll /health until the backend responds. Torch + SAM2 import is usually
# slower than the old hardcoded 3s wait, which would leave the browser
# pointed at a dead port.
$backendReady = $false
$timeoutSeconds = 60
Write-Host "  Waiting for backend to be ready (up to ${timeoutSeconds}s)..." -ForegroundColor Cyan
for ($i = 0; $i -lt $timeoutSeconds; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            $backendReady = $true
            Write-Host "  Backend ready after $($i + 1)s" -ForegroundColor Green
            break
        }
    } catch {
        # not up yet, keep polling
    }
    Start-Sleep -Seconds 1
}
if (-not $backendReady) {
    Write-Host "  WARNING: Backend did not respond within ${timeoutSeconds}s." -ForegroundColor Yellow
    Write-Host "  Check the backend PowerShell window for errors." -ForegroundColor Yellow
}

# Start frontend in new window
Write-Host "Starting frontend dev server..." -ForegroundColor Yellow
$frontendScript = @"
cd '$PWD\frontend'
npm run dev
"@
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendScript

# Poll the Vite dev server until it serves something. Same rationale as above.
$frontendReady = $false
Write-Host "  Waiting for frontend to be ready (up to 30s)..." -ForegroundColor Cyan
for ($i = 0; $i -lt 30; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:5173" -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            $frontendReady = $true
            Write-Host "  Frontend ready after $($i + 1)s" -ForegroundColor Green
            break
        }
    } catch {
        # not up yet
    }
    Start-Sleep -Seconds 1
}
if (-not $frontendReady) {
    Write-Host "  WARNING: Frontend did not respond within 30s." -ForegroundColor Yellow
    Write-Host "  Check the frontend PowerShell window for errors." -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  EnvisionObjectAnnotator Running!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nOpen in browser: http://localhost:5173" -ForegroundColor White
Write-Host "`nBackend API: http://localhost:8000" -ForegroundColor White
Write-Host "`nClose the PowerShell windows to stop the servers.`n" -ForegroundColor Yellow

# Try to open browser
Start-Process "http://localhost:5173"
