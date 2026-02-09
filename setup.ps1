# EnvisionObjectAnnotator Setup Script (Windows)
# Run: .\setup.ps1

# Don't use Stop - pip outputs notices to stderr which breaks the script
$ErrorActionPreference = "Continue"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  EnvisionObjectAnnotator Setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check prerequisites
Write-Host "[1/6] Checking prerequisites..." -ForegroundColor Yellow

# Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "ERROR: Python not found. Install Python 3.10+ from python.org" -ForegroundColor Red
    exit 1
}
$pyVersion = python --version
Write-Host "  Found: $pyVersion" -ForegroundColor Green

# Node.js
$node = Get-Command node -ErrorAction SilentlyContinue
if (-not $node) {
    Write-Host "ERROR: Node.js not found. Install Node.js 18+ from nodejs.org" -ForegroundColor Red
    exit 1
}
$nodeVersion = node --version
Write-Host "  Found: Node.js $nodeVersion" -ForegroundColor Green

# ffmpeg
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    Write-Host "WARNING: ffmpeg not found. Video processing may fail." -ForegroundColor Yellow
    Write-Host "  Install from https://ffmpeg.org/download.html and add to PATH" -ForegroundColor Yellow
} else {
    Write-Host "  Found: ffmpeg" -ForegroundColor Green
}

# Git
$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) {
    Write-Host "ERROR: Git not found. Install Git from git-scm.com" -ForegroundColor Red
    exit 1
}
Write-Host "  Found: git" -ForegroundColor Green

# Create backend venv and install dependencies
Write-Host "`n[2/6] Setting up backend (Python venv + dependencies)..." -ForegroundColor Yellow
Push-Location backend

if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "  Created virtual environment" -ForegroundColor Green
} else {
    Write-Host "  Virtual environment already exists" -ForegroundColor Green
}

$pip = ".\.venv\Scripts\pip.exe"
$python_venv = ".\.venv\Scripts\python.exe"

Write-Host "  Installing backend requirements..." -ForegroundColor Cyan
& $python_venv -m pip install --upgrade pip 2>&1 | Out-Null
& $pip install -r requirements.txt 2>&1 | Where-Object { $_ -notmatch "notice|WARNING" }
& $pip install numpy matplotlib tqdm opencv-python psutil 2>&1 | Where-Object { $_ -notmatch "notice|WARNING" }

# Install PyTorch (with CUDA if available)
Write-Host "  Installing PyTorch (this may take a few minutes)..." -ForegroundColor Cyan
& $pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  CUDA install failed, trying CPU version..." -ForegroundColor Yellow
    & $pip install torch torchvision torchaudio 2>&1 | Where-Object { $_ -notmatch "notice|WARNING" }
}

Pop-Location
Write-Host "  Backend setup complete" -ForegroundColor Green

# Install SAM2
Write-Host "`n[3/6] Installing SAM2..." -ForegroundColor Yellow
if (-not (Test-Path "sam2")) {
    git clone https://github.com/facebookresearch/sam2.git --quiet
}
Push-Location sam2
& "..\backend\.venv\Scripts\pip.exe" install -e . 2>&1 | Out-Null
Pop-Location
Write-Host "  SAM2 installed" -ForegroundColor Green

# Install EdgeTAM (optional but recommended)
Write-Host "`n[4/6] Installing EdgeTAM..." -ForegroundColor Yellow
if (-not (Test-Path "EdgeTAM")) {
    git clone https://github.com/facebookresearch/EdgeTAM.git --quiet
}
Push-Location EdgeTAM
& "..\backend\.venv\Scripts\pip.exe" install -e . 2>&1 | Out-Null
& "..\backend\.venv\Scripts\pip.exe" install timm 2>&1 | Out-Null
Pop-Location
Write-Host "  EdgeTAM installed" -ForegroundColor Green

# Download checkpoints
Write-Host "`n[5/6] Downloading model checkpoints..." -ForegroundColor Yellow

# SAM2 checkpoints
if (-not (Test-Path "checkpoints")) {
    New-Item -ItemType Directory -Path "checkpoints" | Out-Null
}

$sam2_checkpoints = @(
    @{name="sam2.1_hiera_tiny.pt"; url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"},
    @{name="sam2.1_hiera_small.pt"; url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"},
    @{name="sam2.1_hiera_base_plus.pt"; url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"},
    @{name="sam2.1_hiera_large.pt"; url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"}
)

foreach ($ckpt in $sam2_checkpoints) {
    $path = "checkpoints\$($ckpt.name)"
    if (-not (Test-Path $path)) {
        Write-Host "  Downloading $($ckpt.name)..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $ckpt.url -OutFile $path -UseBasicParsing
    } else {
        Write-Host "  $($ckpt.name) already exists" -ForegroundColor Green
    }
}

# EdgeTAM checkpoint
if (-not (Test-Path "EdgeTAM\checkpoints")) {
    New-Item -ItemType Directory -Path "EdgeTAM\checkpoints" | Out-Null
}
$edgetam_path = "EdgeTAM\checkpoints\edgetam.pt"
if (-not (Test-Path $edgetam_path)) {
    Write-Host "  Downloading edgetam.pt..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri "https://huggingface.co/Arnav0400/EdgeTAM/resolve/main/edgetam.pt" -OutFile $edgetam_path -UseBasicParsing
} else {
    Write-Host "  edgetam.pt already exists" -ForegroundColor Green
}

Write-Host "  Checkpoints downloaded" -ForegroundColor Green

# Install frontend
Write-Host "`n[6/6] Setting up frontend (Node.js)..." -ForegroundColor Yellow
Push-Location frontend
npm install --silent 2>&1 | Out-Null
Pop-Location
Write-Host "  Frontend setup complete" -ForegroundColor Green

# Done
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nTo run the application:" -ForegroundColor Yellow
Write-Host "  .\run.ps1" -ForegroundColor White
Write-Host "`nOr manually:" -ForegroundColor Yellow
Write-Host "  Terminal 1: cd backend && .\.venv\Scripts\uvicorn app.main:app --reload" -ForegroundColor White
Write-Host "  Terminal 2: cd frontend && npm run dev" -ForegroundColor White
Write-Host "`nThen open: http://localhost:5173`n" -ForegroundColor Cyan
