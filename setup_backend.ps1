param(
  [ValidateSet("cuda","cpu")] [string]$Torch = "cuda"
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $root "backend"
$venv = Join-Path $backend ".venv"
if (!(Test-Path $venv)) { python -m venv $venv }

$py = Join-Path $venv "Scripts\\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r (Join-Path $backend "requirements.txt")

if ($Torch -eq "cuda") {
  & $py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
} else {
  & $py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

& $py -m pip install opencv-python matplotlib numpy pillow tqdm pandas psutil pyyaml

# Install sam2 package if present
$sam2 = Join-Path $root "sam2"
$edge = Join-Path $root "EdgeTAM"
if (Test-Path $sam2) {
  & $py -m pip install -e $sam2
} elseif (Test-Path $edge) {
  & $py -m pip install -e $edge
} else {
  Write-Host "sam2/EdgeTAM not found. Clone sam2 or add EdgeTAM." -ForegroundColor Yellow
}