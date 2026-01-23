from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from uuid import uuid4
import subprocess
import json
import cv2
import math
import mimetypes
import numpy as np

from .schemas import (
    SessionCreate,
    Session,
    ConfigUpdate,
    AnnotationPayload,
    ProcessingStartRequest,
    ProcessingStatus,
    ResultsResponse,
    FrameExtractionRequest,
    FrameListResponse,
)
from .state import state
from .processing import start_background_job, test_mask_preview, list_available_models


def _parse_ffprobe_rate(value):
    value = (value or "").strip()
    if not value:
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            num_f = float(num)
            den_f = float(den)
        except ValueError:
            return None
        if den_f == 0:
            return None
        return num_f / den_f
    try:
        return float(value)
    except ValueError:
        return None


def _probe_video_fps(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        fps = _parse_ffprobe_rate(line)
        if fps and fps > 0 and math.isfinite(fps):
            return fps
    return None


class EdgeRiseSharpness:
    def __init__(self, max_samples=600, radius=4, edge_percentile=90):
        self.max_samples = max_samples
        self.radius = radius
        self.edge_percentile = edge_percentile

    @staticmethod
    def _bilinear_sample(img, x, y):
        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        x1 = x0 + 1
        y1 = y0 + 1
        h, w = img.shape
        if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
            return None
        dx = x - x0
        dy = y - y0
        v00 = img[y0, x0]
        v10 = img[y0, x1]
        v01 = img[y1, x0]
        v11 = img[y1, x1]
        return (
            v00 * (1.0 - dx) * (1.0 - dy)
            + v10 * dx * (1.0 - dy)
            + v01 * (1.0 - dx) * dy
            + v11 * dx * dy
        )

    def compute(self, img):
        if img is None or img.size == 0:
            return None
        img_f = img.astype(np.float32)
        gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        if mag.size == 0:
            return 0.0
        threshold = float(np.percentile(mag, self.edge_percentile))
        if not math.isfinite(threshold) or threshold <= 0:
            threshold = float(np.max(mag))
        ys, xs = np.where(mag >= max(threshold, 1e-6))
        if ys.size == 0:
            return 0.0
        points = np.column_stack((ys, xs))
        if points.shape[0] > self.max_samples:
            step = int(math.ceil(points.shape[0] / self.max_samples))
            points = points[::step]

        rise_distances = []
        radius = self.radius
        steps = radius * 2 + 1
        step_size = 2.0 * radius / max(1, steps - 1)
        h, w = img.shape
        for y, x in points:
            dx = float(gx[y, x])
            dy = float(gy[y, x])
            norm = math.hypot(dx, dy)
            if norm <= 1e-6:
                continue
            ux = dx / norm
            uy = dy / norm
            x0 = x - ux * radius
            x1 = x + ux * radius
            y0 = y - uy * radius
            y1 = y + uy * radius
            if (
                x0 < 0
                or x0 >= w - 1
                or x1 < 0
                or x1 >= w - 1
                or y0 < 0
                or y0 >= h - 1
                or y1 < 0
                or y1 >= h - 1
            ):
                continue
            profile = []
            for i in range(steps):
                t = -radius + i * step_size
                fx = x + ux * t
                fy = y + uy * t
                sample = self._bilinear_sample(img_f, fx, fy)
                if sample is None:
                    profile = []
                    break
                profile.append(sample)
            if not profile:
                continue
            if profile[0] > profile[-1]:
                profile = profile[::-1]
            min_v = min(profile)
            max_v = max(profile)
            span = max_v - min_v
            if span <= 1.0:
                continue
            low = min_v + 0.1 * span
            high = min_v + 0.9 * span
            idx_low = next((i for i, v in enumerate(profile) if v >= low), None)
            idx_high = next((i for i, v in enumerate(profile) if v >= high), None)
            if idx_low is None or idx_high is None or idx_high <= idx_low:
                continue
            rise_distances.append((idx_high - idx_low) * step_size)

        if not rise_distances:
            return 0.0
        avg_rise = float(np.mean(rise_distances))
        return 100.0 / (avg_rise + 1e-6)


EDGE_RISE_SHARPNESS = EdgeRiseSharpness()


def _auto_pick_reference_frames(
    frames_dir,
    frame_count,
    sample_stride,
    threshold,
    min_refs,
    max_refs,
):
    if frame_count <= 0:
        return []
    sample_stride = max(1, int(sample_stride))
    threshold = float(threshold)
    min_refs = max(1, int(min_refs))
    max_refs = max(min_refs, int(max_refs))

    metrics = []
    prev = None
    for frame_idx in range(0, frame_count, sample_stride):
        frame_path = frames_dir / f"{frame_idx:05d}.jpg"
        if not frame_path.exists():
            continue
        img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        if prev is None:
            prev = small
        diff = float(np.mean(np.abs(small.astype(np.float32) - prev.astype(np.float32))))
        prev = small
        sharpness = float(EDGE_RISE_SHARPNESS.compute(img))
        edges = cv2.Canny(img, 60, 120)
        edge_density = float(np.mean(edges) / 255.0)
        brightness = float(np.mean(img))
        metrics.append(
            {
                "idx": frame_idx,
                "diff": diff,
                "sharpness": sharpness,
                "edge": edge_density,
                "brightness": brightness,
            }
        )

    if not metrics:
        return []

    sharp_values = [item["sharpness"] for item in metrics]
    blur_threshold = float(np.percentile(sharp_values, 30)) if sharp_values else 0.0
    sharp_scale = float(np.percentile(sharp_values, 90)) if sharp_values else 1.0
    if not math.isfinite(sharp_scale) or sharp_scale <= 0:
        sharp_scale = 1.0

    def _score(item):
        diff_norm = min(1.0, item["diff"] / 255.0)
        edge_norm = min(1.0, item["edge"])
        sharp_norm = min(1.0, item["sharpness"] / sharp_scale)
        return diff_norm * 0.5 + edge_norm * 0.3 + sharp_norm * 0.2

    candidates = [
        item
        for item in metrics
        if item["diff"] >= threshold and item["sharpness"] >= blur_threshold and 15.0 <= item["brightness"] <= 240.0
    ]
    if not candidates:
        candidates = [
            item for item in metrics if item["sharpness"] >= blur_threshold and 15.0 <= item["brightness"] <= 240.0
        ]
    if not candidates:
        candidates = metrics[:]

    candidates.sort(key=_score, reverse=True)
    selected = []
    min_gap = max(1, sample_stride * 4)
    for item in candidates:
        idx = item["idx"]
        if len(selected) >= max_refs:
            break
        if any(abs(idx - picked) < min_gap for picked in selected):
            continue
        selected.append(idx)

    if len(selected) < min_refs:
        for item in candidates:
            idx = item["idx"]
            if len(selected) >= min_refs:
                break
            if idx in selected:
                continue
            selected.append(idx)

    return sorted(set(selected))


def _compute_frame_sharpness(frame_path):
    img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return float(EDGE_RISE_SHARPNESS.compute(img))

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
SESSIONS_DIR = BASE_DIR / "data" / "sessions"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="EnvisionObjectAnnotator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def get_models():
    return {"models": list_available_models()}


@app.post("/sessions", response_model=Session)
def create_session(payload: SessionCreate):
    session_id = str(uuid4())
    session = Session(id=session_id, name=payload.name)
    state.create_session(session)
    return session


@app.get("/sessions/{session_id}", response_model=Session)
def get_session(session_id: str):
    try:
        return state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/uploads/video", response_model=Session)
def upload_video(session_id: str, file: UploadFile = File(...)):
    try:
        session = state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    dest = session_dir / file.filename

    with dest.open("wb") as f:
        f.write(file.file.read())

    return state.update_session(session_id, video_path=str(dest))


@app.post("/frames/extract")
def extract_frames(payload: FrameExtractionRequest):
    try:
        session = state.get_session(payload.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.video_path:
        raise HTTPException(status_code=400, detail="Video not uploaded")

    frames_dir = SESSIONS_DIR / payload.session_id / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = SESSIONS_DIR / payload.session_id / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(frames_dir / "%05d.jpg")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        session.video_path,
        "-q:v",
        str(payload.quality),
        "-start_number",
        "0",
        output_pattern,
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr.strip() or "Frame extraction failed")

    thumb_pattern = str(thumbs_dir / "%05d.jpg")
    thumb_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        session.video_path,
        "-q:v",
        "5",
        "-vf",
        "scale='min(640,iw)':-1",
        "-start_number",
        "0",
        thumb_pattern,
    ]
    thumb_result = subprocess.run(thumb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if thumb_result.returncode != 0:
        print(f"[thumbs] generation failed: {thumb_result.stderr.strip()}")

    fps = _probe_video_fps(session.video_path)
    config = session.config or {}
    if fps:
        updated_config = {**config, "video_fps": float(fps)}
        session = state.update_session(payload.session_id, config=updated_config)
        config = session.config or updated_config

    auto_reference = bool(payload.auto_reference)
    auto_frames = []
    if auto_reference:
        config = session.config or {}
        frame_files = sorted([p.name for p in frames_dir.glob("*.jpg")])
        frame_count = len(frame_files)
        sample_stride = payload.auto_reference_stride
        if sample_stride is None or sample_stride <= 0:
            sample_stride = max(1, int(round(fps))) if fps else 30
        threshold = payload.auto_reference_threshold
        if threshold is None:
            threshold = 12.0
        min_refs = payload.auto_reference_min
        if min_refs is None:
            min_refs = 2
        max_refs = payload.auto_reference_max
        if max_refs is None:
            if fps and frame_count:
                minutes = max(1, int((frame_count / fps) // 60))
                max_refs = min(8, max(min_refs, minutes + 1))
            else:
                max_refs = min(8, max(min_refs, 4))

        auto_frames = _auto_pick_reference_frames(
            frames_dir=frames_dir,
            frame_count=frame_count,
            sample_stride=sample_stride,
            threshold=threshold,
            min_refs=min_refs,
            max_refs=max_refs,
        )
        if auto_frames:
            updated_config = {
                **config,
                "reference_frames": auto_frames,
                "reference_frame": auto_frames[0],
                "multi_reference": True,
            }
            state.update_session(payload.session_id, config=updated_config)

    return {
        "status": "ok",
        "frames_dir": str(frames_dir),
        "auto_reference_frames": auto_frames,
    }


@app.get("/frames/list/{session_id}", response_model=FrameListResponse)
def list_frames(session_id: str):
    try:
        state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    frames_dir = SESSIONS_DIR / session_id / "frames"
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="Frames not extracted")

    frame_files = sorted([p.name for p in frames_dir.glob("*.jpg")])
    frame_width = None
    frame_height = None
    if frame_files:
        sample_path = frames_dir / frame_files[0]
        img = cv2.imread(str(sample_path))
        if img is not None:
            frame_height, frame_width = img.shape[:2]
    thumbs_dir = SESSIONS_DIR / session_id / "thumbs"
    has_thumbs = thumbs_dir.exists() and any(thumbs_dir.glob("*.jpg"))
    return FrameListResponse(
        session_id=session_id,
        frame_count=len(frame_files),
        frame_files=frame_files,
        frame_width=frame_width,
        frame_height=frame_height,
        has_thumbnails=has_thumbs,
    )


@app.get("/frames/{session_id}/{frame_name}")
def get_frame(session_id: str, frame_name: str):
    try:
        state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    frame_path = SESSIONS_DIR / session_id / "frames" / frame_name
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")

    return FileResponse(frame_path)


@app.get("/frames/sharpness/{session_id}/{frame_index}")
def get_frame_sharpness(session_id: str, frame_index: int):
    try:
        state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    frames_dir = SESSIONS_DIR / session_id / "frames"
    frame_path = frames_dir / f"{frame_index:05d}.jpg"
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")

    sharpness = _compute_frame_sharpness(frame_path)
    if sharpness is None:
        raise HTTPException(status_code=500, detail="Failed to compute sharpness")

    return {"frame_index": frame_index, "sharpness": sharpness}


@app.get("/frames/thumbs/{session_id}/{frame_name}")
def get_frame_thumb(session_id: str, frame_name: str):
    try:
        state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    thumb_path = SESSIONS_DIR / session_id / "thumbs" / frame_name
    frame_path = SESSIONS_DIR / session_id / "frames" / frame_name
    if thumb_path.exists():
        return FileResponse(thumb_path)
    if frame_path.exists():
        return FileResponse(frame_path)
    raise HTTPException(status_code=404, detail="Frame not found")


@app.get("/annotation/frames/{session_id}/{frame_index}")
def get_frame_annotations(session_id: str, frame_index: int):
    try:
        state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    annotations_dir = SESSIONS_DIR / session_id / "annotations"
    frame_file = annotations_dir / f"frame_{frame_index:05d}.json"
    if not frame_file.exists():
        return {"frame_index": frame_index, "objects": {}}

    try:
        data = json.loads(frame_file.read_text())
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Corrupted annotation file")

    return data


@app.post("/config", response_model=Session)
def update_config(payload: ConfigUpdate):
    try:
        session = state.get_session(payload.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    config = payload.model_dump(exclude={"session_id"}, exclude_none=True)
    updated_config = {**session.config, **config}
    return state.update_session(payload.session_id, config=updated_config, output_dir=payload.output_dir)


@app.post("/annotation/points")
def add_annotation_points(payload: AnnotationPayload):
    try:
        session = state.get_session(payload.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    annotations_dir = SESSIONS_DIR / payload.session_id / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    frame_file = annotations_dir / f"frame_{payload.frame_index:05d}.json"

    existing = {"frame_index": payload.frame_index, "objects": {}}
    if frame_file.exists():
        try:
            existing = json.loads(frame_file.read_text())
        except json.JSONDecodeError:
            existing = {"frame_index": payload.frame_index, "objects": {}}

    existing["objects"][payload.object_name] = [
        {"x": p.x, "y": p.y, "label": p.label} for p in payload.points
    ]
    frame_file.write_text(json.dumps(existing, indent=2))
    config = session.config or {}
    reference_frames = config.get("reference_frames")
    if config.get("multi_reference"):
        frames = []
        if isinstance(reference_frames, list):
            for f in reference_frames:
                try:
                    frames.append(int(f))
                except (TypeError, ValueError):
                    continue
        if payload.frame_index not in frames:
            frames.append(payload.frame_index)
        frames = sorted(set(frames))
        updated_config = {
            **config,
            "reference_frames": frames,
            "reference_frame": frames[0] if frames else payload.frame_index,
        }
        state.update_session(payload.session_id, config=updated_config)
        reference_frames = frames
    elif config.get("reference_frame") != payload.frame_index:
        updated_config = {**config, "reference_frame": payload.frame_index}
        state.update_session(payload.session_id, config=updated_config)

    return {
        "status": "saved",
        "points": len(payload.points),
        "frame": payload.frame_index,
        "object": payload.object_name,
        "path": str(frame_file),
        "reference_frame": payload.frame_index,
        "reference_frames": reference_frames,
    }


@app.post("/annotation/test-mask")
def test_mask(payload: AnnotationPayload):
    try:
        state.get_session(payload.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    if not payload.points:
        raise HTTPException(status_code=400, detail="No points provided")

    try:
        preview_path = test_mask_preview(
            payload.session_id,
            payload.frame_index,
            payload.object_name,
            [p.model_dump() for p in payload.points],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "status": "ok",
        "message": "Mask preview generated",
        "preview_url": f"/previews/{payload.session_id}/{preview_path.name}",
    }


@app.get("/previews/{session_id}/{file_name}")
def get_preview(session_id: str, file_name: str):
    preview_path = SESSIONS_DIR / session_id / "previews" / file_name
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(preview_path)


@app.post("/processing/start", response_model=ProcessingStatus)
def start_processing(payload: ProcessingStartRequest):
    try:
        state.get_session(payload.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    active = state.get_thread(payload.session_id)
    if active and active.is_alive():
        raise HTTPException(status_code=400, detail="Processing already running")

    status = ProcessingStatus(
        session_id=payload.session_id,
        status="starting",
        progress=0.0,
        message="Starting processing",
    )
    state.set_processing(status)
    start_background_job(payload.session_id)
    return status


@app.get("/processing/status/{session_id}", response_model=ProcessingStatus)
def get_processing_status(session_id: str):
    try:
        return state.get_processing(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/processing/preview/{session_id}")
def get_processing_preview(session_id: str):
    preview_path = SESSIONS_DIR / session_id / "previews" / "latest.jpg"
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not available")
    return FileResponse(preview_path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/results/{session_id}", response_model=ResultsResponse)
def get_results(session_id: str):
    try:
        session = state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    outputs = {}
    if session.config:
        outputs = session.config.get("outputs", {}) or {}
    outputs_meta = session.config.get("outputs_meta", {}) if session.config else None
    profiling = session.config.get("profiling") if session.config else None

    return ResultsResponse(
        session_id=session_id,
        outputs={
            "annotated_video": outputs.get("annotated_video"),
            "csv": outputs.get("csv"),
            "elan": outputs.get("elan"),
        },
        profiling=profiling,
        outputs_meta=outputs_meta,
    )


@app.get("/results/download/{session_id}")
def download_result(session_id: str, kind: str):
    try:
        session = state.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    outputs = {}
    if session.config:
        outputs = session.config.get("outputs", {}) or {}

    path = outputs.get(kind)
    if not path:
        raise HTTPException(status_code=404, detail="Output not found")

    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File missing on disk")

    media_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    headers = {"Content-Disposition": f'attachment; filename="{file_path.name}"'}
    return FileResponse(file_path, media_type=media_type, headers=headers)
