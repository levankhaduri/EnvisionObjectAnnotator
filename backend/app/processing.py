import json
import shutil
from pathlib import Path
import threading
import torch
import traceback
import os
import re
import cv2
import numpy as np
import time

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

from .schemas import ProcessingStatus
from .state import state
from .pipeline import (
    setup_device_ultra_optimized,
    configure_torch_ultra_conservative,
    ultra_cleanup_memory,
    UltraOptimizedProcessor,
    get_video_fps,
    get_gpu_memory_info,
)
from .resource_profiler import ResourceProfiler
from .logger import get_logger, get_session_logger

log = get_logger("processing")

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BASE_DIR.parent
SESSIONS_DIR = BASE_DIR / "data" / "sessions"


MODEL_CATALOG = [
    {
        "key": "sam2.1_hiera_l",
        "label": "SAM2.1 Large (hiera_l)",
        "checkpoint": REPO_DIR / "checkpoints" / "sam2.1_hiera_large.pt",
        "config": REPO_DIR / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml",
    },
    {
        "key": "sam2.1_hiera_b+",
        "label": "SAM2.1 Base+ (hiera_b+)",
        "checkpoint": REPO_DIR / "checkpoints" / "sam2.1_hiera_base_plus.pt",
        "config": REPO_DIR / "configs" / "sam2.1" / "sam2.1_hiera_b+.yaml",
    },
    {
        "key": "sam2.1_hiera_s",
        "label": "SAM2.1 Small (hiera_s)",
        "checkpoint": REPO_DIR / "checkpoints" / "sam2.1_hiera_small.pt",
        "config": REPO_DIR / "configs" / "sam2.1" / "sam2.1_hiera_s.yaml",
    },
    {
        "key": "sam2.1_hiera_t",
        "label": "SAM2.1 Tiny (hiera_t)",
        "checkpoint": REPO_DIR / "checkpoints" / "sam2.1_hiera_tiny.pt",
        "config": REPO_DIR / "configs" / "sam2.1" / "sam2.1_hiera_t.yaml",
    },
    {
        "key": "edgetam",
        "label": "EdgeTAM (edgetam)",
        "checkpoint": REPO_DIR / "EdgeTAM" / "checkpoints" / "edgetam.pt",
        "config": REPO_DIR / "configs" / "edgetam.yaml",
    },
]

DEFAULT_MODEL_KEY = "sam2.1_hiera_b+"

_IMAGE_PREDICTOR_CACHE = {}
_VIDEO_PREDICTOR_CACHE = None  # (cache_key, predictor, device) or None


def list_available_models():
    models = []
    for item in MODEL_CATALOG:
        config_path = item.get("config")
        config_ok = config_path.exists() if isinstance(config_path, Path) else True
        models.append(
            {
                "key": item["key"],
                "label": item["label"],
                "available": item["checkpoint"].exists() and config_ok,
            }
        )
    return models


def _evict_video_predictor_cache():
    """Free the cached video predictor to reclaim GPU memory."""
    global _VIDEO_PREDICTOR_CACHE
    if _VIDEO_PREDICTOR_CACHE is not None:
        log.info("evicting cached video predictor")
        try:
            del _VIDEO_PREDICTOR_CACHE
        except Exception:
            pass
        _VIDEO_PREDICTOR_CACHE = None
    ultra_cleanup_memory()


def _evict_image_predictor_cache():
    """Free all cached image predictors to reclaim GPU memory."""
    global _IMAGE_PREDICTOR_CACHE
    if not _IMAGE_PREDICTOR_CACHE:
        return
    log.info("evicting %d cached image predictor(s)", len(_IMAGE_PREDICTOR_CACHE))
    for _, (predictor, _) in list(_IMAGE_PREDICTOR_CACHE.items()):
        try:
            if hasattr(predictor, "model"):
                del predictor.model
            del predictor
        except Exception:
            pass
    _IMAGE_PREDICTOR_CACHE.clear()
    ultra_cleanup_memory()


def _resolve_model_config(model_key=None):
    entry = _select_model_entry(model_key)
    return str(entry["config"]), entry["checkpoint"]


def _select_model_entry(model_key=None):
    if model_key and model_key != "auto":
        for item in MODEL_CATALOG:
            if item["key"] == model_key:
                if not item["checkpoint"].exists():
                    raise FileNotFoundError(f"Checkpoint missing for model {model_key}")
                config_path = item.get("config")
                if config_path and isinstance(config_path, Path) and not config_path.exists():
                    raise FileNotFoundError(f"Config missing for model {model_key}")
                return item
        raise ValueError(f"Unknown model key: {model_key}")

    # Try the recommended default model first
    for item in MODEL_CATALOG:
        if item["key"] == DEFAULT_MODEL_KEY:
            config_path = item.get("config")
            config_ok = config_path.exists() if isinstance(config_path, Path) else True
            if item["checkpoint"].exists() and config_ok:
                return item
            break

    # Fall back to first available model
    for item in MODEL_CATALOG:
        config_path = item.get("config")
        config_ok = config_path.exists() if isinstance(config_path, Path) else True
        if item["checkpoint"].exists() and config_ok:
            return item
    raise FileNotFoundError("SAM2 checkpoints not found")


def _build_predictor(use_mps=False, model_key=None, use_cache=True):
    global _VIDEO_PREDICTOR_CACHE
    cache_key = (model_key or "auto", bool(use_mps))

    if use_cache and _VIDEO_PREDICTOR_CACHE is not None:
        cached_key, predictor, device = _VIDEO_PREDICTOR_CACHE
        if cached_key == cache_key:
            log.debug("video predictor cache hit: %s", cache_key)
            return predictor, device
        # Different model requested — evict the old one
        _evict_video_predictor_cache()

    log.info("building video predictor: model_key=%s use_mps=%s", model_key, use_mps)
    from sam2.build_sam import build_sam2_video_predictor

    configure_torch_ultra_conservative()
    torch.set_default_dtype(torch.float32)
    device = setup_device_ultra_optimized()
    if device.type == "mps" and not use_mps:
        device = torch.device("cpu")

    entry = _select_model_entry(model_key=model_key)
    model_cfg, checkpoint = str(entry["config"]), entry["checkpoint"]
    predictor = build_sam2_video_predictor(model_cfg, str(checkpoint), device=device)
    if hasattr(predictor, "model"):
        try:
            predictor.model = predictor.model.to(device=device, dtype=torch.float32)
        except Exception:
            pass

    if use_cache:
        _VIDEO_PREDICTOR_CACHE = (cache_key, predictor, device)
        log.info("video predictor cached on %s", device)

    return predictor, device


def _build_image_predictor(use_mps=False, model_key=None):
    cache_key = (model_key or "auto", bool(use_mps))
    cached = _IMAGE_PREDICTOR_CACHE.get(cache_key)
    if cached:
        return cached

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    configure_torch_ultra_conservative()
    torch.set_default_dtype(torch.float32)
    device = setup_device_ultra_optimized()
    if device.type == "mps" and not use_mps:
        device = torch.device("cpu")

    entry = _select_model_entry(model_key=model_key)
    model_cfg, checkpoint = str(entry["config"]), entry["checkpoint"]
    sam_model = build_sam2(model_cfg, str(checkpoint), device=device)
    try:
        sam_model = sam_model.to(device=device, dtype=torch.float32)
    except Exception:
        pass

    predictor = SAM2ImagePredictor(sam_model)
    _IMAGE_PREDICTOR_CACHE[cache_key] = (predictor, device)
    return predictor, device


def _get_ram_info():
    if psutil is None:
        return None
    mem = psutil.virtual_memory()
    return {
        "used_gb": mem.used / 1024**3,
        "available_gb": mem.available / 1024**3,
        "total_gb": mem.total / 1024**3,
    }


def _clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def _auto_tune_settings(frame_count, predictor, config, log_cb=None, fps=None):
    target = float(config.get("tuning_target", 0.75) or 0.75)
    target = _clamp(target, 0.5, 0.9)
    reserve_gb = float(config.get("tuning_reserve_gb", 8.0) or 0.0)
    reserve_gb = max(0.0, reserve_gb)
    max_cache_override = config.get("max_cache_frames")
    max_cache_cap = config.get("max_cache_cap")
    preview_stride_override = config.get("preview_stride")
    chunk_size = config.get("chunk_size")
    chunk_seconds = config.get("chunk_seconds")
    chunk_overlap = config.get("chunk_overlap", 1)
    compress_masks = config.get("compress_masks")

    cpu_pct = None
    if psutil is not None:
        try:
            cpu_pct = psutil.cpu_percent(interval=0.1)
        except Exception:
            cpu_pct = None

    ram_info = _get_ram_info()
    gpu_info = get_gpu_memory_info()

    image_size = getattr(predictor, "image_size", 1024)
    bytes_per_frame = int(image_size) * int(image_size) * 3 * 4
    estimated_video_gb = (bytes_per_frame * frame_count) / 1024**3 if frame_count else 0.0

    max_cache_frames = None
    if max_cache_override is not None:
        try:
            max_cache_frames = int(max_cache_override)
        except (TypeError, ValueError):
            max_cache_frames = None
    elif ram_info is not None and bytes_per_frame > 0:
        available_gb = max(0.0, ram_info["available_gb"] - reserve_gb)
        target_bytes = available_gb * 1024**3 * target
        max_cache_frames = int(target_bytes // bytes_per_frame)

    if max_cache_cap is None and frame_count >= 5000:
        max_cache_cap = 2048
    if max_cache_cap is not None:
        try:
            max_cache_cap = int(max_cache_cap)
        except (TypeError, ValueError):
            max_cache_cap = None
    if max_cache_cap is not None:
        if max_cache_frames is None:
            max_cache_frames = max_cache_cap
        else:
            max_cache_frames = min(max_cache_frames, max_cache_cap)

    if max_cache_frames is not None:
        if max_cache_frames >= frame_count:
            max_cache_frames = None
        else:
            max_cache_frames = max(2, max_cache_frames)

    if chunk_seconds is not None:
        try:
            chunk_seconds = float(chunk_seconds)
        except (TypeError, ValueError):
            chunk_seconds = None
    if chunk_seconds and fps and fps > 0:
        chunk_size = max(1, int(round(fps * chunk_seconds)))
    if chunk_size is None and frame_count >= 5000:
        chunk_size = 1000
    if chunk_size is not None:
        try:
            chunk_size = int(chunk_size)
        except (TypeError, ValueError):
            chunk_size = None
    if chunk_overlap is None:
        chunk_overlap = 1
    try:
        chunk_overlap = max(1, int(chunk_overlap))
    except (TypeError, ValueError):
        chunk_overlap = 1

    if compress_masks is None:
        compress_masks = frame_count >= 2000

    if preview_stride_override is not None:
        try:
            preview_stride = max(1, int(preview_stride_override))
        except (TypeError, ValueError):
            preview_stride = 1
    else:
        # Reduced from 600 to 200 to lower overhead - updates every ~8-9 frames for typical videos
        target_previews = 200
        preview_stride = max(1, int(frame_count / target_previews)) if frame_count else 1

    gpu_memory_fraction = None
    if torch.cuda.is_available():
        gpu_memory_fraction = _clamp(target, 0.4, 0.95)

    if log_cb:
        log_cb(
            "auto_tune: target=%.2f reserve_gb=%.1f cpu_pct=%s ram_avail_gb=%s gpu_free_gb=%s "
            "bytes_per_frame=%s est_video_gb=%.2f max_cache_frames=%s max_cache_cap=%s "
            "chunk_size=%s chunk_seconds=%s chunk_overlap=%s compress_masks=%s preview_stride=%s "
            "gpu_mem_fraction=%s"
            % (
                target,
                reserve_gb,
                cpu_pct if cpu_pct is not None else "n/a",
                ram_info["available_gb"] if ram_info else "n/a",
                gpu_info["free_gb"] if gpu_info else "n/a",
                bytes_per_frame,
                estimated_video_gb,
                max_cache_frames if max_cache_frames is not None else "full",
                max_cache_cap if max_cache_cap is not None else "n/a",
                chunk_size if chunk_size is not None else "off",
                chunk_seconds if chunk_seconds is not None else "n/a",
                chunk_overlap,
                compress_masks,
                preview_stride,
                gpu_memory_fraction if gpu_memory_fraction is not None else "n/a",
            )
        )

    return {
        "target": target,
        "reserve_gb": reserve_gb,
        "cpu_pct": cpu_pct,
        "ram_info": ram_info,
        "gpu_info": gpu_info,
        "bytes_per_frame": bytes_per_frame,
        "estimated_video_gb": estimated_video_gb,
        "max_cache_frames": max_cache_frames,
        "max_cache_cap": max_cache_cap,
        "chunk_size": chunk_size,
        "chunk_seconds": chunk_seconds,
        "chunk_overlap": chunk_overlap,
        "compress_masks": compress_masks,
        "preview_stride": preview_stride,
        "gpu_memory_fraction": gpu_memory_fraction,
    }


def test_mask_preview(session_id, frame_index, object_name, points, bbox=None):
    """Generate a mask preview using video predictor on a single frame (fast)."""
    import tempfile
    import shutil

    slog = get_session_logger(session_id)
    slog.info("test_mask_preview: frame=%d obj=%s pts=%d bbox=%s", frame_index, object_name, len(points), bbox is not None)
    session = state.get_session(session_id)
    frames_dir = SESSIONS_DIR / session_id / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError("Frames not found")

    use_mps = bool((session.config or {}).get("use_mps", False))
    model_key = (session.config or {}).get("model_key") or "auto"

    # Build coordinate arrays
    coords_list = [[float(p["x"]), float(p["y"])] for p in points]
    labels_list = [int(p["label"]) for p in points]

    if len(coords_list) == 0:
        raise ValueError("No points provided")

    frame_path = frames_dir / f"{frame_index:05d}.jpg"
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise FileNotFoundError("Frame not found")

    # Create a temp directory with just the one frame for fast loading
    temp_dir = tempfile.mkdtemp(prefix="sam2_test_")
    try:
        # Copy single frame to temp dir as 00000.jpg (video predictor expects this format)
        temp_frame_path = Path(temp_dir) / "00000.jpg"
        enhance_target = bool((session.config or {}).get("enhance_target", False))
        if enhance_target:
            enhanced = UltraOptimizedProcessor._enhance_red_channel(frame)
            cv2.imwrite(str(temp_frame_path), enhanced)
        else:
            shutil.copy2(frame_path, temp_frame_path)

        predictor, device = _build_predictor(use_mps=use_mps, model_key=model_key)

        with torch.inference_mode():
            # Initialize state with temp directory (only 1 frame)
            inference_state = predictor.init_state(
                video_path=temp_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=False,
            )

            # Add points for the object (frame_idx=0 since it's the only frame)
            points_arr = np.array(coords_list, dtype=np.float32)
            labels_arr = np.array(labels_list, dtype=np.int32)

            sam_kwargs = dict(
                inference_state=inference_state,
                frame_idx=0,  # Always 0 since temp dir has only 1 frame
                obj_id=1,
                points=points_arr,
                labels=labels_arr,
            )
            if bbox:
                sam_kwargs["box"] = np.array(
                    [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]], dtype=np.float32
                )
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(**sam_kwargs)

            # Get the mask from the logits
            mask_logits = out_mask_logits[0]  # First object
            mask = (mask_logits > 0.0).squeeze().cpu().numpy()

            # Clean up predictor state
            predictor.reset_state(inference_state)
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Resize mask if needed (video predictor may output different size)
    if mask.shape != (frame.shape[0], frame.shape[1]):
        mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0])) > 0

    mask = mask.astype(bool)

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

    return preview_path


class HeadlessProcessor:
    def __init__(self, processor):
        self._processor = processor

    def process(self, points_dict, labels_dict, object_names, multiframe_data=None, bboxes_dict=None, progress_callback=None):
        return self._processor.process_video_with_memory_management(
            points_dict,
            labels_dict,
            object_names,
            debug=True,
            multiframe_data=multiframe_data,
            bboxes_dict=bboxes_dict,
            progress_callback=progress_callback,
        )

    def get_partial_results(self):
        return self._processor.get_partial_results()

    def save_video(self, results, output_path, fps, frame_limit=None, progress_callback=None):
        return self._processor.save_results_video_with_enhanced_annotations(
            results,
            output_path,
            fps=fps,
            show_original=True,
            alpha=0.5,
            frame_limit=frame_limit,
            progress_callback=progress_callback,
        )

    def save_elan(self, video_path, output_path, fps):
        return self._processor.create_elan_file(video_path, output_path, fps=fps, frame_offset=0)

    def save_csv(self, results, object_names, output_path, progress_callback=None):
        return self._processor.export_framewise_csv(
            results, object_names, output_path, progress_callback=progress_callback
        )

    def save_masks_json(self, results, object_names, output_path, progress_callback=None):
        return self._processor.export_masks_json(
            results, object_names, output_path, progress_callback=progress_callback
        )

    def cleanup_mask_store(self):
        return self._processor.cleanup_mask_store()


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

    bboxes_data = data.get("bboxes", {})

    points_dict = {}
    labels_dict = {}
    object_names = {}
    bboxes_dict = {}
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
        if name in bboxes_data:
            bb = bboxes_data[name]
            bboxes_dict[idx] = np.array(
                [bb["x1"], bb["y1"], bb["x2"], bb["y2"]], dtype=np.float32
            )

    if not points_dict:
        raise ValueError("No valid points found for reference frame")

    return points_dict, labels_dict, object_names, bboxes_dict


def _load_multiframe_annotations(session_id):
    """Load annotations from all annotated frames.

    Scans all frame_*.json files and builds a consistent object ID mapping
    so the same object name gets the same ID across all frames.

    Returns:
        Tuple of (multiframe_data, object_name_to_id) where multiframe_data
        is {frame_idx: (points_dict, labels_dict, object_names)}.
    """
    annotations_dir = SESSIONS_DIR / session_id / "annotations"
    if not annotations_dir.exists():
        raise FileNotFoundError("No annotations directory found")

    frame_annotations = {}
    frame_bboxes = {}
    global_object_names: set[str] = set()

    for frame_file in sorted(annotations_dir.glob("frame_*.json")):
        try:
            frame_idx = int(frame_file.stem.split("_")[1])
            data = json.loads(frame_file.read_text())
            objects = data.get("objects", {})
            if objects:
                frame_annotations[frame_idx] = objects
                frame_bboxes[frame_idx] = data.get("bboxes", {})
                global_object_names.update(objects.keys())
        except (ValueError, json.JSONDecodeError):
            continue

    if not frame_annotations:
        raise FileNotFoundError("No valid annotations found in any frame")

    # Consistent object IDs across frames
    object_name_to_id = {
        name: idx for idx, name in enumerate(sorted(global_object_names), start=1)
    }

    multiframe_data = {}
    all_bboxes_dict = {}
    for frame_idx, objects in frame_annotations.items():
        points_dict = {}
        labels_dict = {}
        object_names = {}
        bboxes_for_frame = frame_bboxes.get(frame_idx, {})

        for obj_name, points in objects.items():
            if not points:
                continue
            coords = [[p["x"], p["y"]] for p in points]
            labels = [p["label"] for p in points]
            if len(coords) != len(labels) or not coords:
                continue

            obj_id = object_name_to_id[obj_name]
            object_names[obj_id] = obj_name
            points_dict[obj_id] = coords
            labels_dict[obj_id] = labels
            if obj_name in bboxes_for_frame:
                bb = bboxes_for_frame[obj_name]
                all_bboxes_dict[obj_id] = np.array(
                    [bb["x1"], bb["y1"], bb["x2"], bb["y2"]], dtype=np.float32
                )

        if points_dict:
            multiframe_data[frame_idx] = (points_dict, labels_dict, object_names)

    if not multiframe_data:
        raise ValueError("No valid points found in any annotated frame")

    return multiframe_data, object_name_to_id, all_bboxes_dict


def run_processing(session_id):
    slog = get_session_logger(session_id)
    error_log = SESSIONS_DIR / session_id / "processing_error.log"
    debug_log = SESSIONS_DIR / session_id / "processing_debug.log"
    started_at = time.perf_counter()

    def _append_log(line):
        try:
            debug_log.parent.mkdir(parents=True, exist_ok=True)
            with debug_log.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:
            pass

    def _log_debug(message):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        _append_log(f"[{stamp}] {message}")

    def _log_trace():
        trace = traceback.format_exc().rstrip()
        if not trace:
            return
        for line in trace.splitlines():
            _append_log(f"    {line}")

    slog.info("run_processing: starting")
    _log_debug(f"run start: session_id={session_id}")
    try:
        session = state.get_session(session_id)
    except KeyError:
        slog.error("run_processing: session not found")
        _set_status(session_id, "error", 0.0, "Session not found")
        _log_debug("session not found")
        return

    frames_dir = SESSIONS_DIR / session_id / "frames"
    if not frames_dir.exists():
        _set_status(session_id, "error", 0.0, "Frames not found. Extract frames first.")
        _log_debug(f"frames not found: {frames_dir}")
        return

    preview_dir = SESSIONS_DIR / session_id / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / "latest.jpg"

    # Start resource profiler (samples GPU/CPU/RAM every 2 seconds)
    profile_output_dir = (
        Path(session.output_dir) if session.output_dir
        else Path(session.video_path).parent if session.video_path
        else SESSIONS_DIR / session_id
    )
    profiler = ResourceProfiler(
        output_dir=profile_output_dir,
        interval_seconds=2.0,
        session_id=session_id,
    )
    profiler.start()
    _log_debug("resource profiler started")

    def _save_preview(frame):
        try:
            cv2.imwrite(str(preview_path), frame)
        except Exception:
            pass

    config = session.config or {}
    overlap_threshold = float(config.get("overlap_threshold", 0.1))
    batch_size = int(config.get("batch_size", 50))
    auto_fallback = bool(config.get("auto_fallback", True))
    auto_tune = bool(config.get("auto_tune", True))
    auto_tune_info = None
    fps = None
    # Prefer effective_fps (computed from actual extracted frames / duration)
    if "effective_fps" in config:
        try:
            fps = float(config.get("effective_fps"))
        except (TypeError, ValueError):
            fps = None
    if fps is None and "video_fps" in config:
        try:
            fps = float(config.get("video_fps"))
        except (TypeError, ValueError):
            fps = None
    if fps is None and session.video_path:
        try:
            fps, _ = get_video_fps(session.video_path)
        except Exception:
            fps = None
    export_video = bool(config.get("export_video", True))
    export_elan = bool(config.get("export_elan", True))
    export_csv = bool(config.get("export_csv", True))
    reference_frame = int(config.get("reference_frame", 0))
    frame_stride = config.get("frame_stride")
    frame_interpolation = config.get("frame_interpolation")
    roi_enabled = bool(config.get("roi_enabled", False))
    enhance_target = bool(config.get("enhance_target", False))
    try:
        roi_margin = float(config.get("roi_margin", 0.15))
    except (TypeError, ValueError):
        roi_margin = 0.15
    try:
        roi_min_size = int(config.get("roi_min_size", 256))
    except (TypeError, ValueError):
        roi_min_size = 256
    try:
        roi_max_coverage = float(config.get("roi_max_coverage", 0.95))
    except (TypeError, ValueError):
        roi_max_coverage = 0.95
    process_start_frame = config.get("process_start_frame")
    process_end_frame = config.get("process_end_frame")
    try:
        process_start_frame = int(process_start_frame)
    except (TypeError, ValueError):
        process_start_frame = None
    try:
        process_end_frame = int(process_end_frame)
    except (TypeError, ValueError):
        process_end_frame = None
    use_mps = bool(config.get("use_mps", False))
    enable_bidirectional = bool(config.get("enable_bidirectional", False))
    model_key = config.get("model_key") or "auto"
    resolved_model_key = model_key
    model_label = model_key
    frame_count = len(list(frames_dir.glob("*.jpg")))
    gpu_start = get_gpu_memory_info()
    ram_start = _get_ram_info()
    _log_debug(
        "config: model_key=%s reference_frame=%s batch_size=%s frames=%s"
        % (model_key, reference_frame, batch_size, frame_count)
    )

    multiframe_data = None
    is_multiframe = False

    try:
        _set_status(session_id, "initializing", 0.05, "Loading annotations")
        mf_data, object_name_to_id, bboxes_dict = _load_multiframe_annotations(session_id)

        # Build global object names registry
        object_names = {obj_id: obj_name for obj_name, obj_id in object_name_to_id.items()}

        if len(mf_data) == 1:
            # Single frame — use traditional approach
            frame_idx = list(mf_data.keys())[0]
            points_dict, labels_dict, _ = mf_data[frame_idx]
            reference_frame = frame_idx
            is_multiframe = False
            _log_debug(
                f"single reference frame: frame={reference_frame} objects={len(object_names)} "
                f"points={sum(len(v) for v in points_dict.values())}"
            )
        else:
            # Multiple frames — use earliest as primary, rest as conditioning
            sorted_frames = sorted(mf_data.keys())
            reference_frame = sorted_frames[0]
            points_dict, labels_dict, _ = mf_data[reference_frame]
            multiframe_data = mf_data
            is_multiframe = True
            _log_debug(
                f"multiframe mode: frames={sorted_frames} total_objects={len(object_names)} "
                f"primary_ref={reference_frame}"
            )
    except Exception as exc:
        _set_status(session_id, "error", 0.05, f"Annotation error: {exc}")
        _log_debug(f"annotation error: {exc}")
        _log_trace()
        return
    _log_debug(
        "annotations loaded: objects=%s points=%s"
        % (len(object_names), sum(len(v) for v in points_dict.values()))
    )
    annotations_done = time.perf_counter()

    try:
        _set_status(session_id, "initializing", 0.1, "Loading SAM2 model")
        os.environ["HYDRA_FULL_ERROR"] = "1"

        import sam2.sam2_video_predictor  # noqa: F401

        # Evict cached test-mask predictor to free GPU memory for full processing
        _evict_video_predictor_cache()

        try:
            model_entry = _select_model_entry(model_key=model_key)
            resolved_model_key = model_entry["key"]
            model_label = model_entry["label"]
        except Exception as exc:
            _set_status(session_id, "error", 0.1, f"Model selection error: {exc}")
            return

        predictor, device = _build_predictor(use_mps=use_mps, model_key=model_entry["key"], use_cache=False)
        _log_debug(
            "model ready: key=%s label=%s device=%s checkpoint=%s"
            % (resolved_model_key, model_label, device, model_entry["checkpoint"])
        )

        if auto_tune:
            auto_tune_info = _auto_tune_settings(
                frame_count=frame_count,
                predictor=predictor,
                config=config,
                log_cb=_log_debug,
                fps=fps,
            )
        preview_stride = auto_tune_info["preview_stride"] if auto_tune_info else 1
        max_cache_frames = auto_tune_info["max_cache_frames"] if auto_tune_info else None
        gpu_memory_fraction = auto_tune_info["gpu_memory_fraction"] if auto_tune_info else None
        chunk_size = auto_tune_info["chunk_size"] if auto_tune_info else config.get("chunk_size")
        chunk_overlap = auto_tune_info["chunk_overlap"] if auto_tune_info else config.get("chunk_overlap", 1)
        compress_masks = auto_tune_info["compress_masks"] if auto_tune_info else config.get("compress_masks")

        processor = UltraOptimizedProcessor(
            predictor,
            str(frames_dir),
            overlap_threshold=overlap_threshold,
            reference_frame=reference_frame,
            batch_size=batch_size,
            auto_fallback=auto_fallback,
            preview_callback=_save_preview,
            log_callback=_log_debug,
            preview_stride=preview_stride,
            preview_max_dim=720,
            max_cache_frames=max_cache_frames,
            gpu_memory_fraction=gpu_memory_fraction,
            frame_stride=frame_stride,
            frame_interpolation=frame_interpolation,
            roi_enabled=roi_enabled,
            roi_margin=roi_margin,
            roi_min_size=roi_min_size,
            roi_max_coverage=roi_max_coverage,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            compress_masks=compress_masks,
            mask_store_dir=str(SESSIONS_DIR / session_id / "mask_cache"),
            process_start_frame=process_start_frame,
            process_end_frame=process_end_frame,
            enable_bidirectional=enable_bidirectional,
            enhance_target=enhance_target,
        )
        wrapped = HeadlessProcessor(processor)
    except ImportError as exc:
        error_log.write_text(traceback.format_exc())
        _set_status(session_id, "error", 0.1, f"Missing dependency: {exc}")
        _log_debug(f"missing dependency: {exc}")
        _log_trace()
        return
    except Exception as exc:
        error_log.write_text(traceback.format_exc())
        _set_status(session_id, "error", 0.1, f"Model init error: {exc}")
        _log_debug(f"model init error: {exc}")
        _log_trace()
        return
    model_done = time.perf_counter()

    processing_done = None

    def _write_primary_outputs(results_to_save, suffix="", frame_limit=None, fps_override=None):
        output_dir = Path(session.output_dir) if session.output_dir else Path(session.video_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        video_stem = Path(session.video_path).stem
        fps = fps_override
        if fps is None:
            fps, _ = get_video_fps(session.video_path)

        outputs = {}
        if export_video:
            video_path = output_dir / f"{video_stem}{suffix}_ANNOTATED.mp4"

            def _save_progress(frame_idx, max_frame):
                total = max(1, int(max_frame) + 1)
                current = min(int(frame_idx) + 1, total)
                progress = 0.85 + 0.15 * (current / float(total))
                _set_status(
                    session_id,
                    "saving",
                    min(progress, 0.99),
                    f"Writing outputs: {current}/{total}",
                )

            wrapped.save_video(
                results_to_save,
                str(video_path),
                fps=fps,
                frame_limit=frame_limit,
                progress_callback=_save_progress,
            )
            outputs["annotated_video"] = str(video_path)
        if export_elan:
            elan_path = output_dir / f"{video_stem}{suffix}_ELAN_TIMELINE.eaf"
            wrapped.save_elan(session.video_path, str(elan_path), fps=fps)
            if elan_path.exists():
                outputs["elan"] = str(elan_path)
            else:
                _log_debug("ELAN file not created (no targets registered?)")

        return outputs, output_dir, video_stem

    def _update_outputs_meta(csv_status=None, csv_progress=None, csv_error=None, outputs_override=None):
        try:
            session_state = state.get_session(session_id)
        except KeyError:
            return
        config_state = session_state.config or {}
        outputs_state = outputs_override or config_state.get("outputs", {}) or {}
        outputs_meta = config_state.get("outputs_meta", {}) or {}
        if csv_status is not None:
            outputs_meta["csv_status"] = csv_status
        if csv_progress is not None:
            outputs_meta["csv_progress"] = csv_progress
        if csv_error is not None:
            outputs_meta["csv_error"] = csv_error
        updated_config = {**config_state, "outputs": outputs_state, "outputs_meta": outputs_meta}
        state.update_session(session_id, config=updated_config)

    def _export_masks(results_to_save, outputs, output_dir, video_stem, suffix=""):
        """Export segmentation masks as JSON before mask store cleanup."""
        try:
            masks_path = output_dir / f"{video_stem}{suffix}_MASKS.json"
            wrapped.save_masks_json(results_to_save, object_names, str(masks_path))
            outputs["masks_json"] = str(masks_path)
        except Exception as exc:
            _log_debug(f"masks json export error: {exc}")
            _log_trace()

    def _start_csv_export(results_to_save, outputs, output_dir, video_stem, suffix=""):
        if not export_csv:
            _update_outputs_meta(csv_status="disabled", csv_progress=0.0, outputs_override=outputs)
            _export_masks(results_to_save, outputs, output_dir, video_stem, suffix)
            try:
                wrapped.cleanup_mask_store()
            except Exception:
                pass
            return

        _update_outputs_meta(csv_status="pending", csv_progress=0.0, outputs_override=outputs)

        def _csv_worker():
            _update_outputs_meta(csv_status="running", csv_progress=0.0)
            try:
                csv_path = output_dir / f"{video_stem}{suffix}_FRAME_BY_FRAME.csv"

                def _csv_progress(done, total):
                    total = max(1, int(total))
                    current = min(int(done), total)
                    progress = current / float(total)
                    _update_outputs_meta(csv_status="running", csv_progress=progress)

                wrapped.save_csv(
                    results_to_save,
                    object_names,
                    str(csv_path),
                    progress_callback=_csv_progress,
                )
                _export_masks(results_to_save, outputs, output_dir, video_stem, suffix)
                outputs_done = {**outputs, "csv": str(csv_path)}
                _update_outputs_meta(
                    csv_status="completed",
                    csv_progress=1.0,
                    outputs_override=outputs_done,
                )
                try:
                    wrapped.cleanup_mask_store()
                except Exception:
                    pass
            except Exception as exc:
                _log_debug(f"csv export error: {exc}")
                _log_trace()
                _update_outputs_meta(csv_status="error", csv_error=str(exc))

        worker = threading.Thread(target=_csv_worker, daemon=True)
        worker.start()

    try:
        _set_status(session_id, "processing", 0.2, "Processing frames")

        def _frame_progress(current, total):
            pct = 0.2 + 0.65 * (current / max(total, 1))
            _set_status(session_id, "processing", min(pct, 0.85), f"Processing frames: {current}/{total}")

        mf_data = multiframe_data if is_multiframe else None
        results = wrapped.process(points_dict, labels_dict, object_names, multiframe_data=mf_data, bboxes_dict=bboxes_dict, progress_callback=_frame_progress)
        if not results:
            _set_status(session_id, "error", 0.6, "Processing failed")
            _log_debug("processing failed: no results")
            return
    except Exception as exc:
        slog.error("run_processing: failed — %s", exc, exc_info=True)
        error_log.write_text(traceback.format_exc())
        _log_debug(f"processing error: {exc}")
        _log_trace()

        partial_results = {}
        try:
            partial_results = wrapped.get_partial_results() or {}
        except Exception:
            partial_results = {}

        if partial_results:
            _set_status(session_id, "saving", 0.85, "Writing partial outputs")
            try:
                frame_limit = max(partial_results.keys()) if partial_results else None
                outputs, output_dir, video_stem = _write_primary_outputs(
                    partial_results,
                    suffix="_PARTIAL",
                    frame_limit=frame_limit,
                    fps_override=fps,
                )
                failed_at = time.perf_counter()
                processing_seconds = max(failed_at - model_done, 1e-6)
                profiling = {
                    "model_key": resolved_model_key,
                    "model_label": model_label,
                    "device": str(device),
                    "frames_total": frame_count,
                    "frames_processed": len(partial_results),
                    "objects_total": len(object_names),
                    "processing_fps": len(partial_results) / processing_seconds,
                    "auto_tune": auto_tune_info,
                    "partial": True,
                    "error": str(exc),
                    "timings_s": {
                        "load_annotations": annotations_done - started_at,
                        "model_init": model_done - annotations_done,
                        "processing": failed_at - model_done,
                        "total": failed_at - started_at,
                    },
                    "gpu_mem_start": gpu_start,
                    "gpu_mem_end": get_gpu_memory_info(),
                    "ram_start": ram_start,
                    "ram_end": _get_ram_info(),
                }
                updated_config = {
                    **session.config,
                    "outputs": outputs,
                    "profiling": profiling,
                }
                state.update_session(session_id, config=updated_config)
                _start_csv_export(
                    partial_results,
                    outputs,
                    output_dir,
                    video_stem,
                    suffix="_PARTIAL",
                )
                _set_status(session_id, "error", 0.9, "Processing failed; partial outputs saved")
            except Exception as save_exc:
                error_log.write_text(traceback.format_exc())
                _set_status(session_id, "error", 0.9, f"Output error: {save_exc}")
                _log_debug(f"output error: {save_exc}")
                _log_trace()
        else:
            _set_status(session_id, "error", 0.6, f"Processing error: {exc}")
        return
    processing_done = time.perf_counter()

    try:
        _set_status(session_id, "saving", 0.85, "Writing outputs")
        outputs, output_dir, video_stem = _write_primary_outputs(results, fps_override=fps)

        finished_at = time.perf_counter()
        processing_seconds = max(processing_done - model_done, 1e-6)
        profiling = {
            "model_key": resolved_model_key,
            "model_label": model_label,
            "device": str(device),
            "frames_total": frame_count,
            "objects_total": len(object_names),
            "processing_fps": frame_count / processing_seconds,
            "auto_tune": auto_tune_info,
            "timings_s": {
                "load_annotations": annotations_done - started_at,
                "model_init": model_done - annotations_done,
                "processing": processing_done - model_done,
                "saving": finished_at - processing_done,
                "total": finished_at - started_at,
            },
            "gpu_mem_start": gpu_start,
            "gpu_mem_end": get_gpu_memory_info(),
            "ram_start": ram_start,
            "ram_end": _get_ram_info(),
        }
        # Stop profiler early so we can include the path in outputs
        try:
            profile_html = profiler.stop()
            outputs["resource_profile"] = str(profile_html)
            profiling["resource_profile"] = str(profile_html)
        except Exception:
            pass
        updated_config = {**session.config, "outputs": outputs, "profiling": profiling}
        state.update_session(session_id, config=updated_config)
        _start_csv_export(results, outputs, output_dir, video_stem)
        message = "Processing complete"
        if export_csv:
            message = "Video ready. CSV exporting..."
        _set_status(session_id, "completed", 1.0, message)
        slog.info("run_processing: completed — %d frames", frame_count)
        _log_debug("processing complete")

        try:
            removed_count = 0
            if frames_dir.exists():
                frame_files = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
                for f in frame_files:
                    f.unlink()
                removed_count = len(frame_files)
            session_dir = frames_dir.parent
            for d in session_dir.iterdir():
                if d.is_dir() and d.name not in ("frames", "thumbs", "mask_cache"):
                    shutil.rmtree(d, ignore_errors=True)
            if removed_count:
                slog.info("run_processing: cleaned up %d frames", removed_count)
                _log_debug(f"frame cleanup: removed {removed_count} files + temp dirs")
        except Exception as cleanup_exc:
            slog.warning("run_processing: frame cleanup failed — %s", cleanup_exc)
    except Exception as exc:
        slog.error("run_processing: output error — %s", exc, exc_info=True)
        error_log.write_text(traceback.format_exc())
        _set_status(session_id, "error", 0.9, f"Output error: {exc}")
        _log_debug(f"output error: {exc}")
        _log_trace()
    finally:
        # Stop resource profiler and save results
        try:
            profile_html = profiler.stop()
            _log_debug(f"resource profile saved: {profile_html}")
        except Exception as prof_exc:
            _log_debug(f"profiler stop error: {prof_exc}")
        # Free GPU memory so the next session starts clean
        try:
            _evict_video_predictor_cache()
            _evict_image_predictor_cache()
            slog.info("run_processing: GPU cleanup completed")
            _log_debug("GPU cleanup completed")
        except Exception as gpu_exc:
            slog.warning("run_processing: GPU cleanup failed — %s", gpu_exc)
        state.clear_thread(session_id)


def start_background_job(session_id):
    worker = threading.Thread(target=run_processing, args=(session_id,), daemon=True)
    state.set_thread(session_id, worker)
    worker.start()
    return worker
