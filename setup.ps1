# EnvisionObjectAnnotator Setup Script (Windows)
# Run: .\setup.ps1

# PSScriptAnalyzer does not track variable usage inside native-command
# argument lists (e.g. `git checkout $SAM2_REF`), so it falsely flags
# $SAM2_REF / $EDGETAM_REF as unused. Suppress that one rule.
[Diagnostics.CodeAnalysis.SuppressMessageAttribute(
    'PSUseDeclaredVarsMoreThanAssignments', '',
    Justification = 'Refs are passed to git checkout as native-command args.'
)]
param()

# Don't use Stop - pip writes deprecation notices to stderr by design and
# that would terminate the script prematurely. We rely on $LASTEXITCODE
# checks (via Assert-LastExit) after each native command for real failures.
$ErrorActionPreference = "Continue"

# Pin upstream SAM2 / EdgeTAM refs here. Leaving these at "main" means a
# force-push or layout change upstream can break a fresh install with no
# warning. Replace with a specific tag or SHA before cutting a release.
# Example: $SAM2_REF = "v2.1.0"; $EDGETAM_REF = "abc1234"
$SAM2_REF = "main"
$EDGETAM_REF = "main"

function Assert-LastExit {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: $Step failed (exit $LASTEXITCODE)." -ForegroundColor Red
        Write-Host "Re-run with output visible to diagnose:" -ForegroundColor Yellow
        Write-Host "  .\setup.ps1 2>&1 | Tee-Object setup.log" -ForegroundColor Yellow
        exit 1
    }
}

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
Assert-LastExit "pip upgrade"
& $pip install -r requirements.txt 2>&1 | Where-Object { $_ -notmatch "notice|WARNING" }
Assert-LastExit "pip install -r requirements.txt"
& $pip install numpy matplotlib tqdm opencv-python psutil 2>&1 | Where-Object { $_ -notmatch "notice|WARNING" }
Assert-LastExit "pip install numpy/matplotlib/tqdm/opencv-python/psutil"

# Install PyTorch (with CUDA if available)
Write-Host "  Installing PyTorch (this may take a few minutes)..." -ForegroundColor Cyan
& $pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  CUDA wheel install failed, trying CPU version..." -ForegroundColor Yellow
    & $pip install torch torchvision torchaudio 2>&1 | Where-Object { $_ -notmatch "notice|WARNING" }
    Assert-LastExit "pip install torch (CPU fallback)"
}

# Verify GPU availability
$cudaCheck = & $python_venv -c "import torch; print(torch.cuda.is_available())" 2>&1
if ($cudaCheck -match "True") {
    $gpuName = & $python_venv -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
    Write-Host "  GPU detected: $gpuName" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "  WARNING: No GPU detected! Processing will be very slow." -ForegroundColor Red
    Write-Host "  To enable GPU acceleration:" -ForegroundColor Yellow
    Write-Host "    1. Install NVIDIA drivers: https://www.nvidia.com/drivers" -ForegroundColor Yellow
    Write-Host "    2. Install CUDA 12.1+: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    Write-Host "    3. Re-run this setup script" -ForegroundColor Yellow
    Write-Host ""
}

Pop-Location
Write-Host "  Backend setup complete" -ForegroundColor Green

# Install SAM2
Write-Host "`n[3/6] Installing SAM2 (ref: $SAM2_REF)..." -ForegroundColor Yellow
if (-not (Test-Path "sam2")) {
    git clone https://github.com/facebookresearch/sam2.git --quiet
    Assert-LastExit "git clone sam2"
}
git -C sam2 fetch --quiet --tags
git -C sam2 checkout --quiet $SAM2_REF
Assert-LastExit "git checkout sam2 $SAM2_REF"
Push-Location sam2
& "..\backend\.venv\Scripts\pip.exe" install -e . 2>&1 | Out-Null
Assert-LastExit "pip install -e sam2"
Pop-Location
Write-Host "  SAM2 installed" -ForegroundColor Green

# Install EdgeTAM (optional but recommended)
Write-Host "`n[4/6] Installing EdgeTAM (ref: $EDGETAM_REF)..." -ForegroundColor Yellow
if (-not (Test-Path "EdgeTAM")) {
    git clone https://github.com/facebookresearch/EdgeTAM.git --quiet
    Assert-LastExit "git clone EdgeTAM"
}
git -C EdgeTAM fetch --quiet --tags
git -C EdgeTAM checkout --quiet $EDGETAM_REF
Assert-LastExit "git checkout EdgeTAM $EDGETAM_REF"
Push-Location EdgeTAM
& "..\backend\.venv\Scripts\pip.exe" install -e . 2>&1 | Out-Null
Assert-LastExit "pip install -e EdgeTAM"
& "..\backend\.venv\Scripts\pip.exe" install timm 2>&1 | Out-Null
Assert-LastExit "pip install timm"
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
Assert-LastExit "npm install (frontend)"
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
