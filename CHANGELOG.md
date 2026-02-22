# Changelog — `feature/carbon-ui-v2-improvements`

## What's New

### UI Overhaul
- Migrated all pages to **IBM Carbon Design System** (consistent components, typography, spacing)
- Redesigned landing page with 4 feature cards and system diagnostics
- Added **hover tooltips** on all technical settings (model, batch size, stride, chunking, etc.)
- Fixed icon alignment across all pages
- Changed "AI" terminology to "Smart" throughout

### Smart Frame Suggestion
- Backend analyzes extracted frames and suggests optimal reference frames based on visual diversity, sharpness, and temporal spread
- Modal with checkbox selection — pick one or multiple frames to annotate
- Visual loading feedback while analyzing

### Multiframe Annotation
- Annotate objects across multiple reference frames (not just one)
- Earliest frame used as primary, rest as conditioning for better tracking

### Bidirectional Propagation
- New toggle on Config page: propagate masks both forward and backward from reference frame
- Works in both standard and **chunked processing modes** — backward chunks are seeded from the reference frame and propagated in reverse
- Post-processing pass runs overlap analysis on backward frames in chronological order for correct ELAN/CSV output
- Useful when reference frame is in the middle of the video

### Processing Improvements
- **Removed `CUDA_LAUNCH_BLOCKING=1`** — GPU was being severely underutilized due to forced synchronous execution
- **Resource profiler** — logs GPU/CPU/RAM usage every 2 seconds during processing, generates downloadable HTML chart on Results page
- **Fixed all-objects save bug** — Process button now saves ALL annotated objects across ALL frames before starting (previously only saved the last-selected object)
- Fixed ELAN export: file path now validated before storing (no more "File missing on disk" errors)
- Fixed ELAN/CSV output gaps when using chunked mode with bidirectional propagation
- Target object nudge: helper text and notification when no objects are marked as targets

### Video Preview Fix
- Config page video preview now works correctly (fixed blob URL memory leak)
- Jump-to-start/end buttons are now visible labeled buttons instead of tiny icons

### New Files
- `setup.ps1` / `setup.sh` — one-command setup (installs frontend + backend deps)
- `run.ps1` / `run.sh` — one-command launch (starts both servers)
- `demo.html` — standalone interactive demo with animated cursor walkthrough
- `backend/app/logger.py` — session-specific debug logging (console + file)
- `backend/app/resource_profiler.py` — time-series GPU/CPU/RAM profiler with HTML chart output
- `backend/tests/` — **100 pytest tests** covering schemas, API, frame analysis, multiframe, bidirectional propagation, ELAN export, annotation saving, and resource profiler

---

## How to Test

```bash
git fetch origin
git checkout feature/carbon-ui-v2-improvements

# Windows
.\setup.ps1
.\run.ps1

# Linux/Mac
chmod +x setup.sh run.sh
./setup.sh
./run.sh
```

Then open http://localhost:5173

### Run tests

```bash
cd backend
python -m pytest tests/ -v
```
