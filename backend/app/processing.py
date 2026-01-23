import json
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
    UltraOptimizedProcessor,
    ImprovedTargetOverlapTracker,
    get_video_fps,
    get_gpu_memory_info,
)

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

_IMAGE_PREDICTOR_CACHE = {}


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

    for item in MODEL_CATALOG:
        config_path = item.get("config")
        config_ok = config_path.exists() if isinstance(config_path, Path) else True
        if item["checkpoint"].exists() and config_ok:
            return item
    raise FileNotFoundError("SAM2 checkpoints not found")


def _build_predictor(use_mps=False, model_key=None):
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
        target_previews = 600
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


def test_mask_preview(session_id, frame_index, object_name, points):
    session = state.get_session(session_id)
    frames_dir = SESSIONS_DIR / session_id / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError("Frames not found")

    use_mps = bool((session.config or {}).get("use_mps", False))
    model_key = (session.config or {}).get("model_key") or "auto"
    predictor, _ = _build_image_predictor(use_mps=use_mps, model_key=model_key)

    points_arr = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
    labels_arr = np.array([p["label"] for p in points], dtype=np.int32)
    if points_arr.size == 0:
        raise ValueError("No points provided")

    frame_path = frames_dir / f"{frame_index:05d}.jpg"
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise FileNotFoundError("Frame not found")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb_frame)
    masks, _, _ = predictor.predict(
        point_coords=points_arr,
        point_labels=labels_arr,
        multimask_output=False,
        return_logits=True,
        normalize_coords=True,
    )

    mask = masks[0] > 0.0

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

    def process(
        self,
        points_dict,
        labels_dict,
        object_names,
        end_frame=None,
        seed_masks=None,
        roi_points_dict=None,
        finalize_events=True,
    ):
        return self._processor.process_video_with_memory_management(
            points_dict,
            labels_dict,
            object_names,
            debug=True,
            end_frame=end_frame,
            seed_masks=seed_masks,
            roi_points_dict=roi_points_dict,
            finalize_events=finalize_events,
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


def _set_status(session_id, status, progress, message):
    state.set_processing(
        ProcessingStatus(
            session_id=session_id,
            status=status,
            progress=progress,
            message=message,
        )
    )


def _read_frame_annotations(session_id, frame_index):
    annotations_dir = SESSIONS_DIR / session_id / "annotations"
    frame_file = annotations_dir / f"frame_{frame_index:05d}.json"
    if not frame_file.exists():
        return {}
    data = json.loads(frame_file.read_text())
    return data.get("objects", {}) or {}


def _build_points_from_objects(objects, name_to_id, object_names, roi_points_dict=None):
    points_dict = {}
    labels_dict = {}
    for name, points in objects.items():
        if not points:
            continue
        coords = [[p["x"], p["y"]] for p in points]
        labels = [p["label"] for p in points]
        if len(coords) != len(labels) or not coords:
            continue
        obj_id = name_to_id.get(name)
        if obj_id is None:
            obj_id = len(name_to_id) + 1
            name_to_id[name] = obj_id
            object_names[obj_id] = name
        points_dict[obj_id] = coords
        labels_dict[obj_id] = labels
        if roi_points_dict is not None:
            roi_points_dict.setdefault(obj_id, []).extend(coords)
    return points_dict, labels_dict


def _load_reference_annotations(session_id, reference_frame):
    objects = _read_frame_annotations(session_id, reference_frame)
    if not objects:
        raise FileNotFoundError("No annotations found for reference frame")

    name_to_id = {}
    object_names = {}
    roi_points_dict = {}
    points_dict, labels_dict = _build_points_from_objects(
        objects, name_to_id, object_names, roi_points_dict=roi_points_dict
    )

    if not points_dict:
        raise ValueError("No valid points found for reference frame")

    return points_dict, labels_dict, object_names, roi_points_dict


def _load_reference_annotations_multi(session_id, reference_frames):
    name_to_id = {}
    object_names = {}
    roi_points_dict = {}
    entries = []

    for frame_index in reference_frames:
        objects = _read_frame_annotations(session_id, frame_index)
        if not objects:
            continue
        points_dict, labels_dict = _build_points_from_objects(
            objects, name_to_id, object_names, roi_points_dict=roi_points_dict
        )
        if points_dict:
            entries.append(
                {
                    "frame": frame_index,
                    "points": points_dict,
                    "labels": labels_dict,
                }
            )

    if not entries:
        raise ValueError("No valid points found for reference frames")

    entries = sorted(entries, key=lambda item: item["frame"])
    return entries, object_names, roi_points_dict


def run_processing(session_id):
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

    _log_debug(f"run start: session_id={session_id}")
    try:
        session = state.get_session(session_id)
    except KeyError:
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
    if "video_fps" in config:
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
    multi_reference = bool(config.get("multi_reference", False))
    reference_frames = config.get("reference_frames") if multi_reference else None
    if reference_frames is not None and not isinstance(reference_frames, list):
        reference_frames = None
    frame_stride = config.get("frame_stride")
    frame_interpolation = config.get("frame_interpolation")
    roi_enabled = bool(config.get("roi_enabled", False))
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
    use_mps = bool(config.get("use_mps", False))
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

    reference_entries = None
    points_dict = {}
    labels_dict = {}
    object_names = {}
    roi_points_dict = {}
    try:
        _set_status(session_id, "initializing", 0.05, "Loading annotations")
        if multi_reference and reference_frames:
            sanitized = []
            for f in reference_frames:
                try:
                    sanitized.append(int(f))
                except (TypeError, ValueError):
                    continue
            reference_frames = sorted(set(sanitized))
        if multi_reference and reference_frames:
            reference_entries, object_names, roi_points_dict = _load_reference_annotations_multi(
                session_id, reference_frames
            )
        else:
            points_dict, labels_dict, object_names, roi_points_dict = _load_reference_annotations(
                session_id, reference_frame
            )
            reference_entries = [
                {
                    "frame": reference_frame,
                    "points": points_dict,
                    "labels": labels_dict,
                }
            ]
    except Exception as exc:
        _set_status(session_id, "error", 0.05, f"Annotation error: {exc}")
        _log_debug(f"annotation error: {exc}")
        _log_trace()
        return
    total_points = sum(len(v) for v in points_dict.values()) if points_dict else 0
    if reference_entries:
        total_points = sum(
            sum(len(v) for v in entry["points"].values()) for entry in reference_entries
        )
    _log_debug(
        "annotations loaded: objects=%s points=%s reference_frames=%s"
        % (len(object_names), total_points, [e["frame"] for e in reference_entries or []])
    )
    annotations_done = time.perf_counter()

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

        try:
            model_entry = _select_model_entry(model_key=model_key)
            model_cfg, checkpoint = str(model_entry["config"]), model_entry["checkpoint"]
            resolved_model_key = model_entry["key"]
            model_label = model_entry["label"]
        except Exception as exc:
            _set_status(session_id, "error", 0.1, f"Model selection error: {exc}")
            return

        predictor = build_sam2_video_predictor(model_cfg, str(checkpoint), device=device)
        if hasattr(predictor, "model"):
            try:
                predictor.model = predictor.model.to(device=device, dtype=torch.float32)
            except Exception:
                pass
        _log_debug(
            "model ready: key=%s label=%s device=%s checkpoint=%s"
            % (resolved_model_key, model_label, device, checkpoint)
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

        shared_overlap_tracker = ImprovedTargetOverlapTracker(overlap_threshold)
        shared_frame_analyses = {}

        def _make_processor(reference_idx):
            processor = UltraOptimizedProcessor(
                predictor,
                str(frames_dir),
                overlap_threshold=overlap_threshold,
                reference_frame=reference_idx,
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
            )
            processor.overlap_tracker = shared_overlap_tracker
            return processor

        processor = _make_processor(reference_frame)
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
            outputs["elan"] = str(elan_path)

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

    def _start_csv_export(results_to_save, outputs, output_dir, video_stem, suffix=""):
        if not export_csv:
            _update_outputs_meta(csv_status="disabled", csv_progress=0.0, outputs_override=outputs)
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
                outputs_done = {**outputs, "csv": str(csv_path)}
                _update_outputs_meta(
                    csv_status="completed",
                    csv_progress=1.0,
                    outputs_override=outputs_done,
                )
            except Exception as exc:
                _log_debug(f"csv export error: {exc}")
                _log_trace()
                _update_outputs_meta(csv_status="error", csv_error=str(exc))

        worker = threading.Thread(target=_csv_worker, daemon=True)
        worker.start()

    try:
        _set_status(session_id, "processing", 0.2, "Processing frames")
        results = {}
        partial_results = {}
        total_segments = len(reference_entries or [])
        for idx, entry in enumerate(reference_entries or []):
            segment_start = entry["frame"]
            segment_end = None
            if reference_entries and idx + 1 < len(reference_entries):
                segment_end = max(0, reference_entries[idx + 1]["frame"])

            if total_segments > 1:
                _set_status(
                    session_id,
                    "processing",
                    0.2,
                    f"Processing segment {idx + 1}/{total_segments} (frame {segment_start})",
                )

            segment_processor = _make_processor(segment_start)
            segment_wrapped = HeadlessProcessor(segment_processor)
            seed_masks = None
            if idx > 0 and segment_start in results:
                seed_masks = {}
                for obj_id, mask in results.get(segment_start, {}).items():
                    mask = segment_processor._decompress_mask(mask)
                    if mask is None:
                        continue
                    seed_masks[obj_id] = mask

            segment_results = segment_wrapped.process(
                entry["points"],
                entry["labels"],
                object_names,
                end_frame=segment_end,
                seed_masks=seed_masks,
                roi_points_dict=roi_points_dict,
                finalize_events=(idx + 1 == total_segments),
            )
            if segment_results:
                results.update(segment_results)
                partial_results = results
                segment_analyses = getattr(segment_processor, "frame_analyses", None) or {}
                if segment_analyses:
                    shared_frame_analyses.update(segment_analyses)
            else:
                raise RuntimeError("Processing failed: no results in segment")

        if not results:
            _set_status(session_id, "error", 0.6, "Processing failed")
            _log_debug("processing failed: no results")
            return
    except Exception as exc:
        error_log.write_text(traceback.format_exc())
        _log_debug(f"processing error: {exc}")
        _log_trace()

        if not partial_results:
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
        writer_reference = reference_entries[0]["frame"] if reference_entries else reference_frame
        writer_processor = _make_processor(writer_reference)
        writer_processor.overlap_tracker = shared_overlap_tracker
        writer_processor.frame_analyses = shared_frame_analyses
        try:
            writer_processor._prepare_frame_source(roi_points_dict or points_dict)
        except Exception:
            pass
        wrapped = HeadlessProcessor(writer_processor)
        _set_status(session_id, "saving", 0.85, "Writing outputs")
        outputs, output_dir, video_stem = _write_primary_outputs(results, fps_override=fps)

        finished_at = time.perf_counter()
        processing_seconds = max(processing_done - model_done, 1e-6)
        frames_processed = len(results)
        profiling = {
            "model_key": resolved_model_key,
            "model_label": model_label,
            "device": str(device),
            "frames_total": frame_count,
            "objects_total": len(object_names),
            "processing_fps": frames_processed / processing_seconds,
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
        updated_config = {**session.config, "outputs": outputs, "profiling": profiling}
        state.update_session(session_id, config=updated_config)
        _start_csv_export(results, outputs, output_dir, video_stem)
        message = "Processing complete"
        if export_csv:
            message = "Video ready. CSV exporting..."
        _set_status(session_id, "completed", 1.0, message)
        _log_debug("processing complete")
    except Exception as exc:
        error_log.write_text(traceback.format_exc())
        _set_status(session_id, "error", 0.9, f"Output error: {exc}")
        _log_debug(f"output error: {exc}")
        _log_trace()
    finally:
        state.clear_thread(session_id)


def start_background_job(session_id):
    worker = threading.Thread(target=run_processing, args=(session_id,), daemon=True)
    state.set_thread(session_id, worker)
    worker.start()
    return worker
