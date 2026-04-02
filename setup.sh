#!/bin/bash
# EnvisionObjectAnnotator Setup Script (Mac/Linux)
# Run: chmod +x setup.sh && ./setup.sh

set -e

echo ""
echo "========================================"
echo "  EnvisionObjectAnnotator Setup"
echo "========================================"
echo ""

# Check prerequisites
echo "[1/6] Checking prerequisites..."

# Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Install Python 3.10+"
    exit 1
fi
echo "  Found: $(python3 --version)"

# Node.js
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js not found. Install Node.js 18+"
    exit 1
fi
echo "  Found: Node.js $(node --version)"

# ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "WARNING: ffmpeg not found. Video processing may fail."
    echo "  Install: brew install ffmpeg (Mac) or apt install ffmpeg (Linux)"
else
    echo "  Found: ffmpeg"
fi

# Git
if ! command -v git &> /dev/null; then
    echo "ERROR: Git not found."
    exit 1
fi
echo "  Found: git"

# Create backend venv and install dependencies
echo ""
echo "[2/6] Setting up backend (Python venv + dependencies)..."
cd backend

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  Created virtual environment"
else
    echo "  Virtual environment already exists"
fi

PIP="./.venv/bin/pip"
PYTHON="./.venv/bin/python"

echo "  Installing backend requirements..."
$PIP install --upgrade pip -q
$PIP install -r requirements.txt -q
$PIP install numpy matplotlib tqdm opencv-python psutil -q

# Install PyTorch
echo "  Installing PyTorch..."
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS - use default (MPS support)
    $PIP install torch torchvision torchaudio -q
else
    # Linux - try CUDA first
    $PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q 2>/dev/null || \
    $PIP install torch torchvision torchaudio -q
fi

# Verify GPU availability
CUDA_CHECK=$($PYTHON -c "import torch; print(torch.cuda.is_available())" 2>&1)
if [[ "$CUDA_CHECK" == "True" ]]; then
    GPU_NAME=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1)
    echo "  GPU detected: $GPU_NAME"
elif [[ "$(uname)" == "Darwin" ]]; then
    MPS_CHECK=$($PYTHON -c "import torch; print(torch.backends.mps.is_available())" 2>&1)
    if [[ "$MPS_CHECK" == "True" ]]; then
        echo "  Apple GPU (MPS) detected"
    else
        echo "  WARNING: No GPU detected! Processing will be very slow."
    fi
else
    echo ""
    echo "  WARNING: No GPU detected! Processing will be very slow."
    echo "  To enable GPU acceleration:"
    echo "    1. Install NVIDIA drivers: https://www.nvidia.com/drivers"
    echo "    2. Install CUDA 12.1+: https://developer.nvidia.com/cuda-downloads"
    echo "    3. Re-run this setup script"
    echo ""
fi

cd ..
echo "  Backend setup complete"

# Install SAM2
echo ""
echo "[3/6] Installing SAM2..."
if [ ! -d "sam2" ]; then
    git clone https://github.com/facebookresearch/sam2.git -q
fi
cd sam2
../backend/.venv/bin/pip install -e . -q
cd ..
echo "  SAM2 installed"

# Install EdgeTAM
echo ""
echo "[4/6] Installing EdgeTAM..."
if [ ! -d "EdgeTAM" ]; then
    git clone https://github.com/facebookresearch/EdgeTAM.git -q
fi
cd EdgeTAM
../backend/.venv/bin/pip install -e . -q
../backend/.venv/bin/pip install timm -q
cd ..
echo "  EdgeTAM installed"

# Download checkpoints
echo ""
echo "[5/6] Downloading model checkpoints..."

mkdir -p checkpoints

SAM2_BASE="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
CHECKPOINTS=(
    "sam2.1_hiera_tiny.pt"
    "sam2.1_hiera_small.pt"
    "sam2.1_hiera_base_plus.pt"
    "sam2.1_hiera_large.pt"
)

for ckpt in "${CHECKPOINTS[@]}"; do
    if [ ! -f "checkpoints/$ckpt" ]; then
        echo "  Downloading $ckpt..."
        curl -sL "$SAM2_BASE/$ckpt" -o "checkpoints/$ckpt"
    else
        echo "  $ckpt already exists"
    fi
done

# EdgeTAM checkpoint
mkdir -p EdgeTAM/checkpoints
if [ ! -f "EdgeTAM/checkpoints/edgetam.pt" ]; then
    echo "  Downloading edgetam.pt..."
    curl -sL "https://huggingface.co/Arnav0400/EdgeTAM/resolve/main/edgetam.pt" -o "EdgeTAM/checkpoints/edgetam.pt"
else
    echo "  edgetam.pt already exists"
fi

echo "  Checkpoints downloaded"

# Install frontend
echo ""
echo "[6/6] Setting up frontend (Node.js)..."
cd frontend
npm install --silent 2>/dev/null
cd ..
echo "  Frontend setup complete"

# Done
echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "To run the application:"
echo "  ./run.sh"
echo ""
echo "Or manually:"
echo "  Terminal 1: cd backend && ./.venv/bin/uvicorn app.main:app --reload"
echo "  Terminal 2: cd frontend && npm run dev"
echo ""
echo "Then open: http://localhost:5173"
echo ""
