import json
from pathlib import Path
import threading
import torch
import traceback
import os
import re
import cv2
import numpy as np

from .schemas import ProcessingStatus
from .state import state
from .pipeline import (
    setup_device_ultra_optimized,
    configure_torch_ultra_conservative,
    UltraOptimizedProcessor,
    get_video_fps,
)

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BASE_DIR.parent
SESSIONS_DIR = BASE_DIR / "data" / "sessions"


def _resolve_model_config():
    checkpoint_large = REPO_DIR / "checkpoints" / "sam2.1_hiera_large.pt"
    checkpoint_small = REPO_DIR / "checkpoints" / "sam2.1_hiera_small.pt"
    if checkpoint_large.exists():
        return "configs/sam2.1/sam2.1_hiera_l.yaml", checkpoint_large
    if checkpoint_small.exists():
        return "configs/sam2.1/sam2.1_hiera_s.yaml", checkpoint_small
    raise FileNotFoundError("SAM2 checkpoints not found")


def _build_predictor(use_mps=False):
    from sam2.build_sam import build_sam2_video_predictor

    configure_torch_ultra_conservative()
    torch.set_default_dtype(torch.float32)
    device = setup_device_ultra_optimized()
    if device.type == "mps" and not use_mps:
        device = torch.device("cpu")

    model_cfg, checkpoint = _resolve_model_config()
    predictor = build_sam2_video_predictor(model_cfg, str(checkpoint), device=device)
    if hasattr(predictor, "model"):
        try:
            predictor.model = predictor.model.to(device=device, dtype=torch.float32)
        except Exception:
            pass
    return predictor, device


def test_mask_preview(session_id, frame_index, object_name, points):
    session = state.get_session(session_id)
    frames_dir = SESSIONS_DIR / session_id / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError("Frames not found")

    use_mps = bool((session.config or {}).get("use_mps", False))
    predictor, _ = _build_predictor(use_mps=use_mps)

    inference_state = predictor.init_state(
        video_path=str(frames_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=True,
    )
    predictor.reset_state(inference_state)

    points_arr = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
    labels_arr = np.array([p["label"] for p in points], dtype=np.int32)
    if points_arr.size == 0:
        raise ValueError("No points provided")

    _, _, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_index,
        obj_id=1,
        points=points_arr,
        labels=labels_arr,
    )

    mask = (out_mask_logits[0] > 0.0).cpu().numpy()
    if len(mask.shape) == 3:
        mask = mask[0]

    frame_path = frames_dir / f"{frame_index:05d}.jpg"
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise FileNotFoundError("Frame not found")

    overlay = frame.copy()
    color = np.array([34, 197, 94], dtype=np.uint8)
    color_mask = np.zeros_like(overlay)
    for c in range(3):
        color_mask[:, :, c][mask] = color[c]
    cv2.addWeighted(overlay, 0.5, color_mask, 0.5, 0, overlay)

    preview_dir = SESSIONS_DIR / session_id / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", object_name).strip("_") or "object"
    preview_path = preview_dir / f"test_mask_{frame_index:05d}_{safe_name}.png"
    cv2.imwrite(str(preview_path), overlay)

    predictor.reset_state(inference_state)
    return preview_path


class HeadlessProcessor:
    def __init__(self, processor):
        self._processor = processor

    def process(self, points_dict, labels_dict, object_names):
        return self._processor.process_video_with_memory_management(
            points_dict,
            labels_dict,
            object_names,
            debug=False,
        )

    def save_video(self, results, output_path, fps):
        return self._processor.save_results_video_with_enhanced_annotations(
            results,
            output_path,
            fps=fps,
            show_original=True,
            alpha=0.5,
        )

    def save_elan(self, video_path, output_path, fps):
        return self._processor.create_elan_file(video_path, output_path, fps=fps, frame_offset=0)

    def save_csv(self, results, object_names, output_path):
        return self._processor.export_framewise_csv(results, object_names, output_path)


def _set_status(session_id, status, progress, message):
    state.set_processing(
        ProcessingStatus(
            session_id=session_id,
            status=status,
            progress=progress,
            message=message,
        )
    )


def _load_reference_annotations(session_id, reference_frame):
    annotations_dir = SESSIONS_DIR / session_id / "annotations"
    frame_file = annotations_dir / f"frame_{reference_frame:05d}.json"
    if not frame_file.exists():
        raise FileNotFoundError("No annotations found for reference frame")

    data = json.loads(frame_file.read_text())
    objects = data.get("objects", {})
    if not objects:
        raise ValueError("No objects annotated")

    points_dict = {}
    labels_dict = {}
    object_names = {}
    for idx, (name, points) in enumerate(objects.items(), start=1):
        if not points:
            continue
        coords = [[p["x"], p["y"]] for p in points]
        labels = [p["label"] for p in points]
        if len(coords) != len(labels) or not coords:
            continue
        object_names[idx] = name
        points_dict[idx] = coords
        labels_dict[idx] = labels

    if not points_dict:
        raise ValueError("No valid points found for reference frame")

    return points_dict, labels_dict, object_names


def run_processing(session_id):
    error_log = SESSIONS_DIR / session_id / "processing_error.log"
    try:
        session = state.get_session(session_id)
    except KeyError:
        _set_status(session_id, "error", 0.0, "Session not found")
        return

    frames_dir = SESSIONS_DIR / session_id / "frames"
    if not frames_dir.exists():
        _set_status(session_id, "error", 0.0, "Frames not found. Extract frames first.")
        return

    config = session.config or {}
    overlap_threshold = float(config.get("overlap_threshold", 0.1))
    batch_size = int(config.get("batch_size", 50))
    auto_fallback = bool(config.get("auto_fallback", True))
    export_video = bool(config.get("export_video", True))
    export_elan = bool(config.get("export_elan", True))
    export_csv = bool(config.get("export_csv", True))
    reference_frame = int(config.get("reference_frame", 0))
    use_mps = bool(config.get("use_mps", False))

    try:
        _set_status(session_id, "initializing", 0.05, "Loading annotations")
        points_dict, labels_dict, object_names = _load_reference_annotations(session_id, reference_frame)
    except Exception as exc:
        _set_status(session_id, "error", 0.05, f"Annotation error: {exc}")
        return

    try:
        _set_status(session_id, "initializing", 0.1, "Loading SAM2 model")
        os.environ["HYDRA_FULL_ERROR"] = "1"

        from sam2.build_sam import build_sam2_video_predictor
        import sam2.sam2_video_predictor  # noqa: F401

        configure_torch_ultra_conservative()
        torch.set_default_dtype(torch.float32)
        device = setup_device_ultra_optimized()
        if device.type == "mps" and not use_mps:
            device = torch.device("cpu")

        checkpoint_large = REPO_DIR / "checkpoints" / "sam2.1_hiera_large.pt"
        checkpoint_small = REPO_DIR / "checkpoints" / "sam2.1_hiera_small.pt"
        if checkpoint_large.exists():
            checkpoint = checkpoint_large
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        elif checkpoint_small.exists():
            checkpoint = checkpoint_small
            model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        else:
            _set_status(session_id, "error", 0.1, "SAM2 checkpoints not found")
            return

        predictor = build_sam2_video_predictor(model_cfg, str(checkpoint), device=device)
        if hasattr(predictor, "model"):
            try:
                predictor.model = predictor.model.to(device=device, dtype=torch.float32)
            except Exception:
                pass
        processor = UltraOptimizedProcessor(
            predictor,
            str(frames_dir),
            overlap_threshold=overlap_threshold,
            reference_frame=reference_frame,
            batch_size=batch_size,
            auto_fallback=auto_fallback,
        )
        wrapped = HeadlessProcessor(processor)
    except ImportError as exc:
        error_log.write_text(traceback.format_exc())
        _set_status(session_id, "error", 0.1, f"Missing dependency: {exc}")
        return
    except Exception as exc:
        error_log.write_text(traceback.format_exc())
        _set_status(session_id, "error", 0.1, f"Model init error: {exc}")
        return

    try:
        _set_status(session_id, "processing", 0.2, "Processing frames")
        results = wrapped.process(points_dict, labels_dict, object_names)
        if not results:
            _set_status(session_id, "error", 0.6, "Processing failed")
            return
    except Exception as exc:
        error_log.write_text(traceback.format_exc())
        _set_status(session_id, "error", 0.6, f"Processing error: {exc}")
        return

    try:
        _set_status(session_id, "saving", 0.85, "Writing outputs")
        output_dir = Path(session.output_dir) if session.output_dir else Path(session.video_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        video_stem = Path(session.video_path).stem
        fps, _ = get_video_fps(session.video_path)

        outputs = {}
        if export_video:
            video_path = output_dir / f"{video_stem}_ANNOTATED.mp4"
            wrapped.save_video(results, str(video_path), fps=fps)
            outputs["annotated_video"] = str(video_path)
        if export_elan:
            elan_path = output_dir / f"{video_stem}_ELAN_TIMELINE.eaf"
            wrapped.save_elan(session.video_path, str(elan_path), fps=fps)
            outputs["elan"] = str(elan_path)
        if export_csv:
            csv_path = output_dir / f"{video_stem}_FRAME_BY_FRAME.csv"
            wrapped.save_csv(results, object_names, str(csv_path))
            outputs["csv"] = str(csv_path)

        updated_config = {**session.config, "outputs": outputs}
        state.update_session(session_id, config=updated_config)
        _set_status(session_id, "completed", 1.0, "Processing complete")
    except Exception as exc:
        error_log.write_text(traceback.format_exc())
        _set_status(session_id, "error", 0.9, f"Output error: {exc}")
    finally:
        state.clear_thread(session_id)


def start_background_job(session_id):
    worker = threading.Thread(target=run_processing, args=(session_id,), daemon=True)
    state.set_thread(session_id, worker)
    worker.start()
    return worker
