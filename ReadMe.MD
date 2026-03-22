<h1 align="center" style=font-size:200px>EnvisionObjectAnnotator:</h1>
<h2 align="center" style=font-size:200px>An Automatic Object-to-Object Overlap Detector with SAM2</h2>

<p align="center">
  <a href="https://envisionbox.org/embedded_EnvisionObjectAnnotator.html">
    <b>EnvisionBox Module — Full Documentation</b>
  </a>
</p>

---

<p align="center">
  <b>Authors</b><br>
  Levani Khaduri — <a href="mailto:Levani.Khaduri@student.hpi.uni-potsdam.de">Levani.Khaduri@student.hpi.uni-potsdam.de</a>  
  Davide Ahmar — <a href="mailto:ahmar.davide@gmail.com">ahmar.davide@gmail.com</a><br>
  Wim Pouw — <a href="mailto:wim.pouw@donders.ru.nl">wim.pouw@donders.ru.nl</a><br>
  Babajide Owoyele — <a href="mailto:babajide.owoyele@hpi.de">babajide.owoyele@hpi.de</a><br>
</p>

<p align="center">
  <img src="extra/Video_Process.gif">
</p>

This repository provides a user-friendly application built on [Meta AI’s SAM2](https://github.com/facebookresearch/sam2) model for **object tracking and overlap (“looking at”) detection** in videos.  

The tool was developed as part of the **EnvisionBOXBABY project**, with a focus on analyzing infant–adult interactions using videos recorded from an infant’s head-mounted camera. However, it can be used for **any scenario** where you want to annotate objects and detect when one *target* object overlaps with others.

---

##  Features

- **Interactive annotation**: select a reference frame, click to add positive/negative points, and name each object.
- **Target detection**: any object named with `”target”` (case-insensitive) is treated as the gaze/marker object.
- **Event detection**: logs “looking at” events whenever the target overlaps another object:
  - By pixel overlap above a threshold
  - Or by centroid inclusion
- **Model selection**: choose between SAM2.1 sizes or EdgeTAM (if installed).
- **Outputs**:
  - Annotated **video** with masks and status overlays
  - Frame-by-frame **CSV** with bounding boxes, centroids, overlap info
  - Time-aligned **ELAN (.eaf)** file for qualitative coding
- **Profiling**: timings, FPS, and GPU/RAM snapshots shown in Results.

---

## Quick Start

### Prerequisites
- **Python 3.10+** — [python.org](https://python.org)
- **Node.js 18+** — [nodejs.org](https://nodejs.org)
- **Git** — [git-scm.com](https://git-scm.com)
- **ffmpeg** — [ffmpeg.org](https://ffmpeg.org/download.html) (add to PATH)

### 1. Clone
```bash
git clone https://github.com/DavAhm/EnvisionObjectAnnotator.git
cd EnvisionObjectAnnotator
```

### 2. Setup (one command)
**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**Mac/Linux:**
```bash
chmod +x setup.sh && ./setup.sh
```

This installs everything: Python venv, SAM2, EdgeTAM, model checkpoints, and frontend.

### 3. Run
**Windows:**
```powershell
.\run.ps1
```

**Mac/Linux:**
```bash
./run.sh
```

Then open **http://localhost:5173**

---

## Manual Setup (advanced)

<details>
<summary>Click to expand manual installation steps</summary>

### Backend (Python)
```bash
cd backend
python3 -m venv .venv

# Windows: .\.venv\Scripts\pip
# Mac/Linux: ./.venv/bin/pip

pip install -r requirements.txt
pip install torch torchvision torchaudio
```

### SAM2
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e .
```

### EdgeTAM (optional)
```bash
git clone https://github.com/facebookresearch/EdgeTAM.git
cd EdgeTAM && pip install -e . && pip install timm
```

### Checkpoints
Download to `checkpoints/` folder:
- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

EdgeTAM checkpoint to `EdgeTAM/checkpoints/`:
- [edgetam.pt](https://huggingface.co/Arnav0400/EdgeTAM/resolve/main/edgetam.pt)

### Frontend
```bash
cd frontend && npm install
```

### Run manually
```bash
# Terminal 1 - Backend
cd backend
./.venv/bin/uvicorn app.main:app --reload

# Terminal 2 - Frontend
cd frontend
npm run dev
```

</details>

---

## Notes

- **GPU/CUDA**: The setup script tries CUDA-enabled PyTorch first, falls back to CPU.
- **Apple Silicon**: MPS acceleration is experimental; CPU fallback is stable.
- **Standalone GUI**: The original Tk GUI still exists in `Script/runEnvisionObjectAnnotator.py`.
- **Models**: SAM2 and EdgeTAM repos are cloned locally (not committed to git).

---

##  How It Works

1. **Load your video** → supports `.mp4`, `.mov`, `.avi`, etc.  
2. **Pick a reference frame** → usually frame `0`.  
3. **Annotate objects**:  
   - Left-click = positive point  
   - Right-click = negative point  
   - Press **C** to name the object (must contain `"target"` for gaze markers)  
   - Press **T** to test masks  
   - Press **Enter** when done  
4. **Set detection threshold** → default is 10% overlap.  
5. **Choose model** → select SAM2.1 or EdgeTAM on the Config page.  
6. **Process video** → masks are propagated, overlaps are detected, and outputs are generated.

---

##  What it outputs:

- **Annotated video**: shows objects with color-coded masks and on-screen event labels  
- **CSV file**: frame-by-frame details with bounding boxes, centroids, areas, and overlaps  
- **ELAN file**: time-aligned tiers with “Looking at: [object]” events for qualitative coding

<p align="center">
  <b>An example of the raw (left) and annotated (right) video output </b><br>
  <img src="extra/Annotated_Video.gif">
</p>

---

## Citation
If you use this tool in your research, please cite the EnvisionBox project.  
*(Full reference will be added when publication is available.)*


---

##  Related Resources
- [Meta AI SAM2](https://github.com/facebookresearch/sam2)  
- [EnvisionBox Project](https://www.envisionbox.org)  
