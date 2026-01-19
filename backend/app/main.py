from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from uuid import uuid4
import subprocess
import json
import cv2
import math

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
    if fps:
        config = session.config or {}
        updated_config = {**config, "video_fps": float(fps)}
        state.update_session(payload.session_id, config=updated_config)

    return {"status": "ok", "frames_dir": str(frames_dir)}


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
    if config.get("reference_frame") != payload.frame_index:
        updated_config = {**config, "reference_frame": payload.frame_index}
        state.update_session(payload.session_id, config=updated_config)

    return {
        "status": "saved",
        "points": len(payload.points),
        "frame": payload.frame_index,
        "object": payload.object_name,
        "path": str(frame_file),
        "reference_frame": payload.frame_index,
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

    return FileResponse(file_path)
