<h1 align="center" style=font-size:200px>EnvisionObjectAnnotator:</h1>
<h2 align="center" style=font-size:200px>An Automatic Object-to-Object Overlap Detector with SAM2</h2>

<p align="center">
  <a href="https://envisionbox.org/embedded_EnvisionObjectAnnotator.html">
    <b>📖 EnvisionBox Module — Full Documentation</b>
  </a>
</p>

---

<p align="center">
  <b>Authors</b><br>
  Davide Ahmar — <a href="mailto:ahmar.davide@gmail.com">ahmar.davide@gmail.com</a><br>
  Wim Pouw — <a href="mailto:wim.pouw@donders.ru.nl">wim.pouw@donders.ru.nl</a><br>
  Babajide Owoyele — <a href="mailto:babajide.owoyele@hpi.de">babajide.owoyele@hpi.de</a>
</p>

<p align="center">
  <img src="extra/Video_Process.gif">
</p>

This repository provides a user-friendly application built on [Meta AI’s SAM2](https://github.com/facebookresearch/sam2) model for **object tracking and overlap (“looking at”) detection** in videos.  

The tool was developed as part of the **EnvisionBOXBABY project**, with a focus on analyzing infant–adult interactions using videos recorded from an infant’s head-mounted camera. However, it can be used for **any scenario** where you want to annotate objects and detect when one *target* object overlaps with others.

---

##  Features

- 🖼️ **Interactive annotation**: select a reference frame, click to add positive/negative points, and name each object.
- 🎯 **Target detection**: any object named with `"target"` (case-insensitive) is treated as the gaze/marker object.
- 🔍 **Event detection**: logs “looking at” events whenever the target overlaps another object:
  - By pixel overlap above a threshold  
  - Or by centroid inclusion  
- ⚙️ **Model selection**: choose between SAM2.1 sizes or EdgeTAM (if installed).
- 📂 **Outputs**:
  - Annotated **video** with masks and status overlays
  - Frame-by-frame **CSV** with bounding boxes, centroids, overlap info
  - Time-aligned **ELAN (.eaf)** file for qualitative coding
- 📊 **Profiling**: timings, FPS, and GPU/RAM snapshots shown in Results.

---

##  Getting Started

### 1. Clone this repository
Click the green **Code** button (top right) → **Download ZIP** → extract it to a folder (e.g., `C:\EnvisionObjectAnnotator`).  
Or use git:
```bash
git clone https://github.com/DavAhm/EnvisionObjectAnnotator.git
cd EnvisionObjectAnnotator
```

### 2. Install prerequisites (all platforms)
- **Python 3.10+**
- **Node.js 18+** (for the React UI)
- **ffmpeg** (for frame extraction)

### 3. Install backend dependencies (FastAPI + SAM2)
```bash
cd backend
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install numpy matplotlib tqdm opencv-python torch torchvision torchaudio
```

Install SAM2 into the same venv:
```bash
cd ../sam2
../backend/.venv/bin/pip install -e .
```

Optional: enable EdgeTAM model selection by cloning EdgeTAM in the repo root and installing it into the same venv:
```bash
git clone https://github.com/facebookresearch/EdgeTAM.git
cd EdgeTAM
../backend/.venv/bin/pip install -e .
../backend/.venv/bin/pip install timm
```

Download checkpoints:
```bash
cd ../checkpoints
./download_ckpts.sh
```

### 4. Install frontend dependencies (React)
```bash
cd ../frontend
npm install
```

### 5. Run the app (web UI + API)
```bash
# backend API
cd ../backend
./.venv/bin/uvicorn app.main:app --reload

# frontend UI
cd ../frontend
npm run dev
```

### Notes
- The **original Tk GUI** still exists in `Script/runEnvisionObjectAnnotator.py`, but the primary flow now runs through the **React + FastAPI** app.
- The `sam2/` and `EdgeTAM/` repositories are cloned locally and **not committed**; follow the install steps above on each machine.
- On Apple Silicon, MPS acceleration is experimental; CPU fallback is stable.
- If you enable MPS, ensure `sam2/sam2_video_predictor.py` uses float32 for the image tensor on `_get_image_feature` (see local patch applied during setup).

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
