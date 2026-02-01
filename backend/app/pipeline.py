import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import traceback

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


os.environ.setdefault("SAM2_OFFLOAD_VIDEO_TO_CPU", "true")
os.environ.setdefault("SAM2_OFFLOAD_STATE_TO_CPU", "true")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

RAM_PRESSURE_THRESHOLD = 80


def _format_tensor_info(value):
    if value is None:
        return "none"
    if torch.is_tensor(value):
        try:
            return (
                f"shape={list(value.shape)} dtype={value.dtype} device={value.device} "
                f"contig={value.is_contiguous()} stride={list(value.stride())} "
                f"storage_offset={value.storage_offset()}"
            )
        except Exception as exc:
            return f"tensor<unavailable> error={exc}"
    if isinstance(value, np.ndarray):
        return f"ndarray shape={list(value.shape)} dtype={value.dtype}"
    return f"{type(value)}"


def _format_points_info(points):
    if points is None:
        return "none"
    try:
        arr = np.asarray(points)
        if arr.size == 0:
            return "empty"
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        return f"shape={list(arr.shape)} min={mins.tolist()} max={maxs.tolist()}"
    except Exception as exc:
        return f"unavailable error={exc}"


def get_gpu_memory_info():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": total - reserved,
            "utilization_pct": (reserved / total) * 100,
        }
    return None


def get_system_memory_info():
    if psutil is None:
        return None
    try:
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / 1024**3,
            "available_gb": mem.available / 1024**3,
            "used_gb": mem.used / 1024**3,
            "percent_used": mem.percent,
        }
    except Exception:
        return None


def check_memory_pressure(threshold_pct=None):
    if threshold_pct is None:
        threshold_pct = RAM_PRESSURE_THRESHOLD
    mem_info = get_system_memory_info()
    if mem_info and mem_info["percent_used"] >= threshold_pct:
        return True, mem_info
    return False, mem_info


def ultra_cleanup_memory():
    """Memory cleanup."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def configure_torch_ultra_conservative():
    """Configure PyTorch for conservative memory usage."""
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.70)
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except Exception:
            pass
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        ultra_cleanup_memory()
        print(f"GPU Memory after setup: {get_gpu_memory_info()}")


def setup_device_ultra_optimized():
    """Setup computation device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_info = get_gpu_memory_info()
        print(
            f"Initial GPU Memory: {gpu_info['allocated_gb']:.1f}GB allocated, {gpu_info['free_gb']:.1f}GB free"
        )
        if gpu_info["total_gb"] < 8:
            print("WARNING: Low GPU memory detected. Using conservative settings.")
            torch.cuda.set_per_process_memory_fraction(0.60)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Support for MPS devices is preliminary.")
        torch.set_default_dtype(torch.float32)
    else:
        device = torch.device("cpu")
        print("Using CPU - this will be slow but stable")

    print(f"Using device: {device}")
    return device


class DiskBackedMaskStore:
    """Store masks on disk with a small in-memory cache."""

    def __init__(self, base_dir, max_in_memory=50):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index = {}
        self._in_memory = {}
        self._max_in_memory = max_in_memory

    def store(self, frame_idx, frame_results):
        frame_dir = self.base_dir / f"frame_{int(frame_idx):06d}"
        frame_dir.mkdir(exist_ok=True)
        obj_ids = []
        for obj_id, mask in frame_results.items():
            if mask is None:
                continue
            mask_path = frame_dir / f"obj_{int(obj_id)}.npz"
            np.savez_compressed(mask_path, mask=mask.astype(np.uint8))
            obj_ids.append(int(obj_id))
        self._index[int(frame_idx)] = obj_ids
        self._in_memory[int(frame_idx)] = frame_results
        if len(self._in_memory) > self._max_in_memory:
            oldest = min(self._in_memory.keys())
            del self._in_memory[oldest]
        return obj_ids

    def load(self, frame_idx):
        frame_idx = int(frame_idx)
        if frame_idx in self._in_memory:
            return self._in_memory[frame_idx]
        obj_ids = self._index.get(frame_idx)
        if not obj_ids:
            return {}
        frame_dir = self.base_dir / f"frame_{frame_idx:06d}"
        results = {}
        for obj_id in obj_ids:
            mask_path = frame_dir / f"obj_{obj_id}.npz"
            if not mask_path.exists():
                continue
            data = np.load(mask_path)
            results[obj_id] = data["mask"].astype(bool)
        return results

    def has_frame(self, frame_idx):
        return int(frame_idx) in self._index

    def frame_indices(self):
        return sorted(self._index.keys())

    def cleanup(self):
        try:
            shutil.rmtree(self.base_dir)
        except Exception:
            pass

def get_video_fps(video_path):
    """Get video FPS using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


class EnhancedOverlapDetector:
    """Enhanced overlap detector that properly handles inclusion and complex overlaps."""

    def __init__(self, overlap_threshold=0.1):
        self.overlap_threshold = overlap_threshold
        self.inclusion_threshold = 0.1

    def calculate_detailed_overlap(self, mask1, mask2):
        """Enhanced overlap detection with both pixel overlap and spatial containment."""
        if mask1.shape != mask2.shape:
            print(f"Shape mismatch: {mask1.shape} vs {mask2.shape}")
            return None

        if len(mask1.shape) > 2:
            mask1 = mask1.squeeze()
        if len(mask2.shape) > 2:
            mask2 = mask2.squeeze()

        if mask1.dtype in [np.float32, np.float64]:
            mask1_bool = mask1 > 0.0001
        else:
            mask1_bool = mask1 > 0

        if mask2.dtype in [np.float32, np.float64]:
            mask2_bool = mask2 > 0.0001
        else:
            mask2_bool = mask2 > 0

        area1 = np.sum(mask1_bool)
        area2 = np.sum(mask2_bool)

        if area1 == 0 or area2 == 0:
            return None

        intersection = mask1_bool & mask2_bool
        intersection_area = np.sum(intersection)

        overlap_pct_1 = intersection_area / area1 if area1 > 0 else 0
        overlap_pct_2 = intersection_area / area2 if area2 > 0 else 0
        max_overlap = max(overlap_pct_1, overlap_pct_2)

        spatial_relationship = False
        containment_type = None

        try:
            contours1, _ = cv2.findContours(mask1_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(mask2_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours1 and contours2:
                contour1 = max(contours1, key=cv2.contourArea)
                contour2 = max(contours2, key=cv2.contourArea)

                M1 = cv2.moments(contour1)
                M2 = cv2.moments(contour2)

                if M1["m00"] != 0 and M2["m00"] != 0:
                    cx1, cy1 = int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"])
                    cx2, cy2 = int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"])

                    mask1_contour = np.zeros_like(mask1_bool, dtype=np.uint8)
                    mask2_contour = np.zeros_like(mask2_bool, dtype=np.uint8)
                    cv2.fillPoly(mask1_contour, [contour1], 1)
                    cv2.fillPoly(mask2_contour, [contour2], 1)

                    object2_in_contour1 = np.any(mask2_bool & mask1_contour)
                    object1_in_contour2 = np.any(mask1_bool & mask2_contour)

                    if object2_in_contour1:
                        spatial_relationship = True
                        containment_type = "object2_partial_inside_object1"
                    elif object1_in_contour2:
                        spatial_relationship = True
                        containment_type = "object1_partial_inside_object2"
                    else:
                        inside_1 = cv2.pointPolygonTest(contour1, (cx2, cy2), False) >= 0
                        inside_2 = cv2.pointPolygonTest(contour2, (cx1, cy1), False) >= 0

                        if inside_1:
                            spatial_relationship = True
                            containment_type = "object2_centroid_inside_object1"
                        elif inside_2:
                            spatial_relationship = True
                            containment_type = "object1_centroid_inside_object2"
        except Exception as exc:
            print(f"    Error in spatial containment detection: {exc}")
            spatial_relationship = False

        has_meaningful_pixel_overlap = intersection_area > 0 and max_overlap >= self.overlap_threshold
        has_spatial_relationship = spatial_relationship

        if not has_meaningful_pixel_overlap and not has_spatial_relationship:
            return None

        meets_basic_threshold = has_meaningful_pixel_overlap or has_spatial_relationship
        meets_continuation_threshold = has_meaningful_pixel_overlap or has_spatial_relationship

        if has_meaningful_pixel_overlap and has_spatial_relationship:
            relationship_type = "overlap_and_containment"
        elif has_meaningful_pixel_overlap:
            relationship_type = "pixel_overlap"
        elif has_spatial_relationship:
            relationship_type = "spatial_containment"
        else:
            relationship_type = "none"

        return {
            "intersection_area": intersection_area,
            "overlap_pct_1": overlap_pct_1,
            "overlap_pct_2": overlap_pct_2,
            "min_overlap_pct": min(overlap_pct_1, overlap_pct_2),
            "max_overlap_pct": max_overlap,
            "spatial_relationship": spatial_relationship,
            "containment_type": containment_type,
            "has_meaningful_pixel_overlap": has_meaningful_pixel_overlap,
            "has_spatial_relationship": has_spatial_relationship,
            "relationship_type": relationship_type,
            "meets_threshold": meets_basic_threshold,
            "meets_continuation_threshold": meets_continuation_threshold,
        }


class ImprovedTargetOverlapTracker:
    """Improved overlap tracker with better inclusion detection and annotations."""

    def __init__(self, overlap_threshold=0.1):
        self.overlap_threshold = overlap_threshold
        self.overlap_events = {}
        self.target_objects = {}
        self.detector = EnhancedOverlapDetector(overlap_threshold)

    def register_target(self, obj_id, obj_name):
        """Register target objects."""
        if "target" in obj_name.lower():
            self.target_objects[obj_id] = obj_name
            self.overlap_events[obj_id] = []
            print(f"Target registered: {obj_name} (ID: {obj_id})")
            return True
        return False

    def get_overlap_summary(self):
        """Get overlap summary."""
        summary = {}
        for target_id, events in self.overlap_events.items():
            target_name = self.target_objects[target_id]
            summary[target_name] = {
                "total_events": len(events),
                "events": events,
                "total_overlap_frames": sum(event["duration_frames"] for event in events),
            }
        return summary

    def finalize_tracking(self, last_frame_idx):
        """Finalize events without extending duration."""
        for target_id, events in self.overlap_events.items():
            if events and events[-1].get("end_frame") is None:
                last_event = events[-1]
                if "duration_frames" in last_event and last_event["duration_frames"] > 0:
                    last_event["end_frame"] = last_event["start_frame"] + last_event["duration_frames"] - 1
                else:
                    last_event["end_frame"] = last_event["start_frame"]
                    last_event["duration_frames"] = 1

    def analyze_frame_overlaps(self, frame_results, object_names):
        """Enhanced frame analysis with continuation validation."""
        frame_analysis = {
            "target_overlaps": {},
            "object_relationships": {},
            "looking_at_events": [],
        }

        try:
            for target_id in self.target_objects:
                if target_id not in frame_results:
                    continue

                target_mask = frame_results[target_id]
                if len(target_mask.shape) > 2:
                    target_mask = target_mask.squeeze()

                target_name = self.target_objects[target_id]
                looking_at_objects = []

                has_ongoing_event = (
                    self.overlap_events[target_id] and not self.overlap_events[target_id][-1].get("end_frame")
                )

                for obj_id, mask in frame_results.items():
                    if obj_id == target_id:
                        continue

                    if len(mask.shape) > 2:
                        mask = mask.squeeze()

                    obj_name = object_names.get(obj_id, f"Object_{obj_id}")

                    try:
                        overlap_info = self.detector.calculate_detailed_overlap(target_mask, mask)
                        if overlap_info:
                            if has_ongoing_event:
                                if overlap_info.get("meets_continuation_threshold", False):
                                    looking_at_objects.append(
                                        {
                                            "object_id": obj_id,
                                            "object_name": obj_name,
                                            "event_type": "looking_at",
                                            "relationship_desc": f"OVERLAPS {obj_name} (continuing)",
                                        }
                                    )
                            else:
                                if overlap_info.get("meets_threshold", False):
                                    looking_at_objects.append(
                                        {
                                            "object_id": obj_id,
                                            "object_name": obj_name,
                                            "event_type": "looking_at",
                                            "relationship_desc": f"LOOKING AT {obj_name}",
                                        }
                                    )

                            if looking_at_objects and looking_at_objects[-1]["object_id"] == obj_id:
                                frame_analysis["looking_at_events"].append(
                                    {
                                        "target_id": target_id,
                                        "target_name": target_name,
                                        "object_id": obj_id,
                                        "object_name": obj_name,
                                    }
                                )
                    except Exception as exc:
                        print(f"      Error checking {obj_name}: {exc}")
                        continue

                if looking_at_objects:
                    frame_analysis["target_overlaps"][target_id] = looking_at_objects
        except Exception as exc:
            print(f"  Error in analyze_frame_overlaps: {exc}")
        return frame_analysis

    def track_frame_overlaps_batch(self, frame_idx, frame_results, object_names):
        """Track 'looking at' events with accurate offset detection."""
        try:
            frame_analysis = self.analyze_frame_overlaps(frame_results, object_names)

            if not hasattr(self, "frame_analyses"):
                self.frame_analyses = {}
            self.frame_analyses[frame_idx] = frame_analysis

            for target_id in self.target_objects:
                current_overlaps = []
                if target_id in frame_analysis.get("target_overlaps", {}):
                    looking_at_objects = frame_analysis["target_overlaps"][target_id]
                    current_overlaps = [obj["object_name"] for obj in looking_at_objects]

                self._update_overlap_event(target_id, frame_idx, current_overlaps)

            return frame_analysis
        except Exception as exc:
            print(f"  Error in track_frame_overlaps_batch for frame {frame_idx}: {exc}")
            return {
                "target_overlaps": {},
                "object_relationships": {},
                "looking_at_events": [],
            }

    def _update_overlap_event(self, target_id, frame_idx, overlapping_names):
        events = self.overlap_events[target_id]
        current_overlap_set = set(overlapping_names)

        if events and events[-1].get("end_frame") is None:
            last_event = events[-1]
            last_overlap_set = set(last_event["overlapping_objects"])

            if current_overlap_set == last_overlap_set and current_overlap_set:
                last_event["duration_frames"] = frame_idx - last_event["start_frame"] + 1
            else:
                last_event["end_frame"] = frame_idx - 1
                last_event["duration_frames"] = last_event["end_frame"] - last_event["start_frame"] + 1

                if current_overlap_set:
                    new_event = {
                        "start_frame": frame_idx,
                        "end_frame": None,
                        "duration_frames": 1,
                        "overlapping_objects": list(overlapping_names),
                        "event_type": "looking_at",
                    }
                    events.append(new_event)
        else:
            if current_overlap_set:
                new_event = {
                    "start_frame": frame_idx,
                    "end_frame": None,
                    "duration_frames": 1,
                    "overlapping_objects": list(overlapping_names),
                    "event_type": "looking_at",
                }
                events.append(new_event)

    def has_targets(self):
        return bool(self.target_objects)


class UltraOptimizedProcessor:
    """Ultra memory-optimized processor with improved overlap detection."""

    def __init__(
        self,
        predictor,
        video_dir,
        overlap_threshold=0.1,
        reference_frame=0,
        batch_size=50,
        auto_fallback=True,
        preview_callback=None,
        log_callback=None,
        preview_stride=15,
        preview_max_dim=720,
        max_cache_frames=None,
        gpu_memory_fraction=None,
        frame_stride=None,
        frame_interpolation=None,
        roi_enabled=False,
        roi_margin=0.15,
        roi_min_size=256,
        roi_max_coverage=0.95,
        chunk_size=None,
        chunk_overlap=1,
        compress_masks=None,
        process_start_frame=None,
        process_end_frame=None,
        mask_store_dir=None,
        disk_store_max_in_memory=50,
        ram_pressure_threshold=RAM_PRESSURE_THRESHOLD,
        disk_store_enabled=True,
        enable_bidirectional=True,
    ):
        self.predictor = predictor
        self.full_video_dir = video_dir
        self.video_dir = video_dir
        self.overlap_threshold = overlap_threshold
        self.reference_frame = reference_frame
        self.reference_frame_full = reference_frame
        self.batch_size = batch_size
        self.auto_fallback = auto_fallback
        self.preview_callback = preview_callback
        self.log_callback = log_callback
        self.preview_stride = max(1, int(preview_stride)) if preview_stride else None
        self.preview_max_dim = int(preview_max_dim)
        self.max_cache_frames = max_cache_frames
        self.gpu_memory_fraction = gpu_memory_fraction
        self.frame_stride = frame_stride
        self.frame_interpolation = frame_interpolation
        self.roi_enabled = roi_enabled
        self.roi_margin = roi_margin
        self.roi_min_size = roi_min_size
        self.roi_max_coverage = roi_max_coverage
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.compress_masks = compress_masks
        self.process_start_frame_full = process_start_frame
        self.process_end_frame_full = process_end_frame
        self.process_start_frame = None
        self.process_end_frame = None
        self.mask_store_dir = mask_store_dir
        self.disk_store_max_in_memory = disk_store_max_in_memory
        self.ram_pressure_threshold = ram_pressure_threshold
        self.disk_store_enabled = disk_store_enabled
        self.enable_bidirectional = enable_bidirectional
        self._mask_store = None
        self._disk_store_active = False
        self.frame_index_map = None
        self.roi_info = None
        self._prepared = False

        self.overlap_tracker = ImprovedTargetOverlapTracker(overlap_threshold)
        self.partial_results = {}

        self.full_frame_names = sorted(
            [
                p for p in os.listdir(self.full_video_dir)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
            ],
            key=lambda p: int(os.path.splitext(p)[0]),
        )

        if not self.full_frame_names:
            raise ValueError("No frames found in the specified directory!")
        self.frame_names = list(self.full_frame_names)

        try:
            _cmap = plt.get_cmap("tab10")
            self._preview_colors = [tuple(int(c * 255) for c in _cmap(i)[:3][::-1]) for i in range(10)]
        except Exception:
            self._preview_colors = [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
                (128, 128, 0),
                (128, 0, 128),
                (0, 128, 128),
                (200, 200, 200),
            ]
        self.offload_video_to_cpu = os.environ.get("SAM2_OFFLOAD_VIDEO_TO_CPU", "true") == "true"
        self.offload_state_to_cpu = os.environ.get("SAM2_OFFLOAD_STATE_TO_CPU", "true") == "true"

    def _compress_mask(self, mask):
        if not self.compress_masks:
            return mask
        if mask is None:
            return None
        if isinstance(mask, dict) and mask.get("_packed"):
            return mask
        m = mask[0] if hasattr(mask, "shape") and len(mask.shape) == 3 else mask
        m = np.asarray(m).astype(np.uint8)
        flat = m.reshape(-1)
        packed = np.packbits(flat)
        return {
            "_packed": True,
            "shape": tuple(m.shape),
            "size": int(m.size),
            "data": packed,
        }

    def _decompress_mask(self, mask):
        if not (isinstance(mask, dict) and mask.get("_packed")):
            return mask
        shape = tuple(mask.get("shape") or ())
        size = int(mask.get("size") or 0)
        data = mask.get("data")
        if data is None or not shape or size <= 0:
            return None
        flat = np.unpackbits(data)[:size]
        return flat.reshape(shape).astype(bool)

    def _expand_roi_mask(self, mask):
        if mask is None or not self.roi_info:
            return mask
        x0 = int(self.roi_info["x0"])
        y0 = int(self.roi_info["y0"])
        x1 = int(self.roi_info["x1"])
        y1 = int(self.roi_info["y1"])
        full_w = int(self.roi_info["full_width"])
        full_h = int(self.roi_info["full_height"])
        roi_w = x1 - x0 + 1
        roi_h = y1 - y0 + 1

        if roi_w <= 0 or roi_h <= 0 or full_w <= 0 or full_h <= 0:
            return mask

        if mask.shape != (roi_h, roi_w) and mask.shape[0] > 0 and mask.shape[1] > 0:
            try:
                mask = cv2.resize(mask.astype(np.float32), (roi_w, roi_h), interpolation=cv2.INTER_LINEAR) > 0.5
            except Exception:
                return mask

        full_mask = np.zeros((full_h, full_w), dtype=bool)
        full_mask[y0 : y1 + 1, x0 : x1 + 1] = mask
        return full_mask

    def _log(self, message):
        cb = getattr(self, "log_callback", None)
        if cb is None:
            return
        try:
            cb(message)
        except Exception:
            pass

    def cleanup_mask_store(self):
        if self._mask_store is None:
            return
        try:
            self._mask_store.cleanup()
        except Exception:
            pass
        self._mask_store = None
        self._disk_store_active = False

    def _normalize_frame_results(self, frame_results):
        normalized = {}
        if not frame_results:
            return normalized
        for obj_id, mask in frame_results.items():
            if mask is None:
                continue
            m = self._decompress_mask(mask)
            if m is None:
                continue
            if hasattr(m, "shape") and len(m.shape) == 3:
                m = m[0]
            normalized[obj_id] = m.astype(bool)
        return normalized

    def _ensure_disk_store(self):
        if self._mask_store is not None or not self.disk_store_enabled:
            return self._mask_store
        base_dir = self.mask_store_dir
        if not base_dir:
            base_dir = os.path.join(os.path.dirname(self.full_video_dir), "mask_cache")
        self._mask_store = DiskBackedMaskStore(base_dir, max_in_memory=self.disk_store_max_in_memory)
        self._disk_store_active = True
        return self._mask_store

    def _enable_disk_store(self, results, reason=None):
        store = self._ensure_disk_store()
        if store is None:
            return False
        if reason:
            self._log(f"disk store enabled: {reason}")
        # Offload any in-memory results to disk
        for frame_idx, frame_results in list(results.items()):
            if not frame_results:
                continue
            if isinstance(frame_results, dict) and frame_results.get("_disk_only"):
                continue
            normalized = self._normalize_frame_results(frame_results)
            if normalized:
                store.store(frame_idx, normalized)
                results[frame_idx] = {"_disk_only": True}
        ultra_cleanup_memory()
        return True

    def _store_frame_results(self, results, frame_idx, frame_results):
        if self._mask_store is not None:
            self._mask_store.store(frame_idx, frame_results)
            results[frame_idx] = {"_disk_only": True}
            return
        if self.compress_masks:
            results[frame_idx] = {
                obj_id: self._compress_mask(mask) for obj_id, mask in frame_results.items()
            }
        else:
            results[frame_idx] = frame_results

    def _get_frame_results(self, results, frame_idx):
        frame_results = results.get(frame_idx)
        if self._mask_store is not None:
            if not frame_results or (isinstance(frame_results, dict) and frame_results.get("_disk_only")):
                return self._mask_store.load(frame_idx)
        return frame_results or {}

    def get_partial_results(self):
        return self.partial_results or {}

    def _list_frame_files(self, video_dir):
        return sorted(
            [
                p
                for p in os.listdir(video_dir)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
            ],
            key=lambda p: int(os.path.splitext(p)[0]),
        )

    def _map_frame_idx(self, local_idx):
        if not self.frame_index_map:
            return local_idx
        if 0 <= local_idx < len(self.frame_index_map):
            return self.frame_index_map[local_idx]
        return local_idx

    def _map_full_range_to_local(self, full_start, full_end):
        if not self.frame_index_map:
            return full_start, full_end
        local_start = None
        local_end = None
        for idx, full_idx in enumerate(self.frame_index_map):
            if local_start is None and full_idx >= full_start:
                local_start = idx
            if full_idx <= full_end:
                local_end = idx
        return local_start, local_end

    def _compute_roi_box(self, points_dict, frame_width, frame_height):
        if not self.roi_enabled:
            return None
        all_points = []
        for points in points_dict.values():
            for pt in points:
                if pt and len(pt) >= 2:
                    all_points.append(pt)
        if not all_points:
            return None

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        min_x = max(0, int(min(xs)))
        max_x = min(frame_width - 1, int(max(xs)))
        min_y = max(0, int(min(ys)))
        max_y = min(frame_height - 1, int(max(ys)))
        if max_x <= min_x or max_y <= min_y:
            return None

        box_w = max(1, max_x - min_x + 1)
        box_h = max(1, max_y - min_y + 1)
        pad_x = int(box_w * float(self.roi_margin))
        pad_y = int(box_h * float(self.roi_margin))

        x0 = max(0, min_x - pad_x)
        y0 = max(0, min_y - pad_y)
        x1 = min(frame_width - 1, max_x + pad_x)
        y1 = min(frame_height - 1, max_y + pad_y)

        roi_w = x1 - x0 + 1
        roi_h = y1 - y0 + 1
        min_size = int(self.roi_min_size) if self.roi_min_size else 0
        if min_size > 0:
            if roi_w < min_size:
                extra = min_size - roi_w
                x0 = max(0, x0 - extra // 2)
                x1 = min(frame_width - 1, x1 + (extra - extra // 2))
            if roi_h < min_size:
                extra = min_size - roi_h
                y0 = max(0, y0 - extra // 2)
                y1 = min(frame_height - 1, y1 + (extra - extra // 2))

        roi_w = x1 - x0 + 1
        roi_h = y1 - y0 + 1
        if roi_w <= 0 or roi_h <= 0:
            return None

        coverage = (roi_w * roi_h) / float(frame_width * frame_height)
        if coverage >= float(self.roi_max_coverage):
            return None

        return (x0, y0, x1, y1)

    def _ensure_stride_dir(self, indices):
        stride = int(self.frame_stride)
        stride_dir = f"{self.full_video_dir}_stride_{stride}_ref_{self.reference_frame_full}"
        os.makedirs(stride_dir, exist_ok=True)
        meta_path = os.path.join(stride_dir, "_stride_meta.json")
        meta = {
            "stride": stride,
            "reference_frame": self.reference_frame_full,
            "frame_count": len(self.full_frame_names),
        }
        if os.path.exists(meta_path):
            try:
                existing = json.loads(open(meta_path, "r", encoding="utf-8").read())
            except Exception:
                existing = None
            if existing == meta and len(self._list_frame_files(stride_dir)) == len(indices):
                return stride_dir

        for idx, src_idx in enumerate(indices):
            src_name = self.full_frame_names[src_idx]
            src_path = os.path.join(self.full_video_dir, src_name)
            dst_name = f"{idx:05d}{os.path.splitext(src_name)[-1]}"
            dst_path = os.path.join(stride_dir, dst_name)
            if os.path.exists(dst_path):
                continue
            try:
                os.link(src_path, dst_path)
            except Exception:
                shutil.copy2(src_path, dst_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return stride_dir

    def _ensure_roi_dir(self, source_dir, source_names, roi_box):
        x0, y0, x1, y1 = roi_box
        roi_dir = f"{source_dir}_roi_{x0}_{y0}_{x1}_{y1}"
        os.makedirs(roi_dir, exist_ok=True)
        meta_path = os.path.join(roi_dir, "_roi_meta.json")
        meta = {"roi_box": [x0, y0, x1, y1], "frame_count": len(source_names)}
        if os.path.exists(meta_path):
            try:
                existing = json.loads(open(meta_path, "r", encoding="utf-8").read())
            except Exception:
                existing = None
            if existing == meta and len(self._list_frame_files(roi_dir)) == len(source_names):
                return roi_dir

        for name in source_names:
            src_path = os.path.join(source_dir, name)
            dst_path = os.path.join(roi_dir, name)
            if os.path.exists(dst_path):
                continue
            frame = cv2.imread(src_path)
            if frame is None:
                continue
            crop = frame[y0 : y1 + 1, x0 : x1 + 1]
            cv2.imwrite(dst_path, crop)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return roi_dir

    def _prepare_frame_source(self, points_dict):
        if self._prepared:
            return
        self._prepared = True

        total_frames = len(self.full_frame_names)
        self.frame_index_map = list(range(total_frames))
        self.video_dir = self.full_video_dir
        self.frame_names = list(self.full_frame_names)
        self.reference_frame = max(0, min(self.reference_frame_full, total_frames - 1))

        stride = None
        if self.frame_stride is not None:
            try:
                stride = int(self.frame_stride)
            except (TypeError, ValueError):
                stride = None
        if stride is not None and stride > 1:
            indices = list(range(self.reference_frame, total_frames, stride))
            if not indices:
                indices = [self.reference_frame]
            self.frame_index_map = indices
            self.reference_frame = 0
            self.video_dir = self._ensure_stride_dir(indices)
            self.frame_names = self._list_frame_files(self.video_dir)

        if self.roi_enabled:
            first_frame = cv2.imread(os.path.join(self.full_video_dir, self.full_frame_names[0]))
            if first_frame is not None:
                full_h, full_w = first_frame.shape[:2]
                roi_box = self._compute_roi_box(points_dict, full_w, full_h)
                if roi_box:
                    self.roi_info = {
                        "x0": roi_box[0],
                        "y0": roi_box[1],
                        "x1": roi_box[2],
                        "y1": roi_box[3],
                        "full_width": full_w,
                        "full_height": full_h,
                    }
                    self.video_dir = self._ensure_roi_dir(self.video_dir, self.frame_names, roi_box)
                    self.frame_names = self._list_frame_files(self.video_dir)
                else:
                    self.roi_info = None

        stride_val = None
        if self.frame_stride is not None:
            try:
                stride_val = int(self.frame_stride)
            except (TypeError, ValueError):
                stride_val = None
        if stride_val and stride_val > 1:
            self._log(
                "frame stride enabled: stride=%s processed_frames=%s total_frames=%s"
                % (stride_val, len(self.frame_names), len(self.full_frame_names))
            )
        if self.roi_info:
            self._log(
                "roi enabled: box=(%s,%s)-(%s,%s) full_size=%sx%s"
                % (
                    self.roi_info["x0"],
                    self.roi_info["y0"],
                    self.roi_info["x1"],
                    self.roi_info["y1"],
                    self.roi_info["full_width"],
                    self.roi_info["full_height"],
                )
            )

        full_start = self.process_start_frame_full
        full_end = self.process_end_frame_full
        if full_start is not None or full_end is not None:
            try:
                full_start = int(full_start) if full_start is not None else 0
            except (TypeError, ValueError):
                full_start = 0
            try:
                full_end = int(full_end) if full_end is not None else total_frames - 1
            except (TypeError, ValueError):
                full_end = total_frames - 1
            full_start = max(0, min(full_start, total_frames - 1))
            full_end = max(0, min(full_end, total_frames - 1))
            if full_end < full_start:
                full_end = full_start
            if self.reference_frame_full < full_start:
                full_start = self.reference_frame_full
            if self.reference_frame_full > full_end:
                full_end = self.reference_frame_full
            self.process_start_frame_full = full_start
            self.process_end_frame_full = full_end
            local_start, local_end = self._map_full_range_to_local(full_start, full_end)
            if local_start is not None and local_end is not None and local_end >= local_start:
                self.process_start_frame = local_start
                self.process_end_frame = local_end
                self._log(
                    "frame range enabled: full=%s-%s local=%s-%s"
                    % (full_start, full_end, local_start, local_end)
                )

    def _fill_missing_frames(self, results):
        if not self.frame_index_map:
            return results
        stride = None
        if self.frame_stride is not None:
            try:
                stride = int(self.frame_stride)
            except (TypeError, ValueError):
                stride = None
        if stride is None or stride <= 1:
            return results

        full_last = len(self.full_frame_names) - 1
        range_start = 0
        range_end = full_last
        if self.process_start_frame_full is not None or self.process_end_frame_full is not None:
            if self.process_start_frame_full is not None:
                range_start = max(0, int(self.process_start_frame_full))
            if self.process_end_frame_full is not None:
                range_end = min(full_last, int(self.process_end_frame_full))
            if range_end < range_start:
                range_end = range_start

        processed = sorted([idx for idx in results.keys() if range_start <= idx <= range_end])
        if len(processed) < 2:
            return results

        interpolation = (self.frame_interpolation or "nearest").lower()
        for prev_idx, next_idx in zip(processed, processed[1:]):
            if next_idx <= prev_idx + 1:
                continue
            gap = next_idx - prev_idx
            prev_masks = self._get_frame_results(results, prev_idx)
            next_masks = self._get_frame_results(results, next_idx)
            for missing in range(prev_idx + 1, next_idx):
                if interpolation == "linear" and next_masks:
                    alpha = (missing - prev_idx) / float(gap)
                    interp_masks = {}
                    obj_ids = set(prev_masks.keys()) | set(next_masks.keys())
                    for obj_id in obj_ids:
                        pm = self._decompress_mask(prev_masks.get(obj_id))
                        nm = self._decompress_mask(next_masks.get(obj_id))
                        if pm is None and nm is None:
                            continue
                        if pm is None:
                            mask = nm
                        elif nm is None:
                            mask = pm
                        else:
                            if pm.shape != nm.shape:
                                try:
                                    nm = cv2.resize(
                                        nm.astype(np.float32),
                                        (pm.shape[1], pm.shape[0]),
                                        interpolation=cv2.INTER_LINEAR,
                                    )
                                    nm = nm > 0.5
                                except Exception:
                                    nm = pm
                            blend = (1.0 - alpha) * pm.astype(np.float32) + alpha * nm.astype(np.float32)
                            mask = blend > 0.5
                        interp_masks[obj_id] = mask
                    self._store_frame_results(results, missing, interp_masks)
                else:
                    self._store_frame_results(results, missing, prev_masks)

        last_idx = processed[-1]
        if last_idx < range_end:
            prev_masks = self._get_frame_results(results, last_idx)
            for missing in range(last_idx + 1, range_end + 1):
                self._store_frame_results(results, missing, prev_masks)
        return results

    def process_video_with_memory_management(self, points_dict, labels_dict, object_names, debug=True, multiframe_data=None):
        """Process video with ultra memory management and improved overlap detection."""
        last_exc = None
        self.partial_results = {}
        self._prepare_frame_source(points_dict)
        try:
            if debug:
                self._log(
                    "processing start: frames=%s objects=%s batch_size=%s offload_video=%s offload_state=%s"
                    % (
                        len(self.frame_names),
                        len(object_names),
                        self.batch_size,
                        self.offload_video_to_cpu,
                        self.offload_state_to_cpu,
                    )
                )
            configure_torch_ultra_conservative()
            if self.gpu_memory_fraction is not None and torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
            for attempt in range(3):
                try:
                    if debug:
                        self._log(
                            "processing attempt %s: batch_size=%s device=%s"
                            % (attempt + 1, self.batch_size, getattr(self.predictor, "device", "unknown"))
                        )
                    if attempt == 0:
                        return self._process_standard_optimized(points_dict, labels_dict, object_names, debug, multiframe_data)
                    if attempt == 1:
                        self.batch_size = self.batch_size // 2
                        return self._process_standard_optimized(points_dict, labels_dict, object_names, debug, multiframe_data)
                    return self._process_cpu_fallback(points_dict, labels_dict, object_names, debug)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        ultra_cleanup_memory()
                        if torch.cuda.is_available():
                            current_fraction = torch.cuda.get_per_process_memory_fraction() * 0.8
                            torch.cuda.set_per_process_memory_fraction(max(0.3, current_fraction))
                        if attempt == 2:
                            raise exc
                    else:
                        if debug:
                            self._log("processing attempt failed: %s" % exc)
                            self._log(traceback.format_exc())
                        raise exc
        except Exception as exc:
            last_exc = exc
            if debug:
                self._log("all processing attempts failed: %s" % exc)
                self._log(traceback.format_exc())
            print(f"All processing attempts failed: {exc}")
            raise
        finally:
            ultra_cleanup_memory()

    def _process_standard_optimized(self, points_dict, labels_dict, object_names, debug, multiframe_data=None):
        self._prepare_frame_source(points_dict)
        image_size = getattr(self.predictor, "image_size", 1024)
        bytes_per_frame = image_size * image_size * 3 * 4
        estimated_bytes = bytes_per_frame * len(self.frame_names)
        max_preload_bytes = 2 * 1024**3
        max_cache_frames = self.max_cache_frames
        if max_cache_frames is not None and max_cache_frames <= 0:
            max_cache_frames = 1
        if max_cache_frames is not None and max_cache_frames >= len(self.frame_names):
            max_cache_frames = None
        if max_cache_frames is None:
            max_cache_frames = 8 if estimated_bytes > max_preload_bytes else None
        async_loading_frames = max_cache_frames is None
        inference_state = self.predictor.init_state(
            video_path=self.video_dir,
            offload_video_to_cpu=self.offload_video_to_cpu,
            offload_state_to_cpu=self.offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
            max_cache_frames=max_cache_frames,
        )

        self.predictor.reset_state(inference_state)
        ultra_cleanup_memory()
        if debug:
            self._log(
                "init_state: num_frames=%s video_size=%sx%s image_size=%s async_loading_frames=%s max_cache_frames=%s estimated_video_gb=%.2f"
                % (
                    inference_state.get("num_frames"),
                    inference_state.get("video_width"),
                    inference_state.get("video_height"),
                    image_size,
                    async_loading_frames,
                    max_cache_frames,
                    estimated_bytes / (1024**3),
                )
            )

        self.object_names = object_names
        targets_found = False
        for obj_id, obj_name in object_names.items():
            if self.overlap_tracker.register_target(obj_id, obj_name):
                targets_found = True

        results = {}
        self.partial_results = results
        frame_analyses = {}
        frame_count = 0
        overlap_count = 0
        last_memory_check = 0

        use_disk_mode = False
        mem_info = get_system_memory_info()
        if self.disk_store_enabled:
            if len(self.frame_names) >= 2000:
                use_disk_mode = True
                self._enable_disk_store(results, reason="long video")
            elif mem_info and mem_info.get("percent_used", 0) >= 70:
                use_disk_mode = True
                self._enable_disk_store(
                    results,
                    reason=f"ram usage {mem_info.get('percent_used', 0):.0f}%",
                )

        total_frames = len(self.frame_names)
        start_frame = max(0, min(self.reference_frame, total_frames - 1))
        range_start = self.process_start_frame if self.process_start_frame is not None else 0
        range_end = self.process_end_frame if self.process_end_frame is not None else total_frames - 1
        range_start = max(0, min(int(range_start), total_frames - 1))
        range_end = max(range_start, min(int(range_end), total_frames - 1))
        total_to_process = max(0, range_end - range_start + 1)
        if self.process_start_frame is not None or self.process_end_frame is not None:
            self._log(
                "processing range: start=%s end=%s (local frames)"
                % (range_start, range_end)
            )
        chunk_size = None
        if self.chunk_size is not None:
            try:
                chunk_size = int(self.chunk_size)
            except (TypeError, ValueError):
                chunk_size = None
        if chunk_size is not None and chunk_size <= 0:
            chunk_size = None
        if chunk_size is not None and chunk_size >= total_to_process:
            chunk_size = None
        chunk_overlap = self.chunk_overlap if self.chunk_overlap is not None else 1
        try:
            chunk_overlap = max(1, int(chunk_overlap))
        except (TypeError, ValueError):
            chunk_overlap = 1

        def _process_frames(
            inference_state,
            start_frame_idx,
            max_frame_num_to_track,
            skip_until,
        ):
            nonlocal frame_count, last_memory_check, overlap_count
            last_masks = {}
            last_processed = None
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
            ):
                if out_frame_idx < skip_until:
                    continue
                try:
                    global_frame_idx = self._map_frame_idx(out_frame_idx)
                    if frame_count - last_memory_check >= 50:
                        gpu_info = get_gpu_memory_info()
                        if gpu_info and gpu_info["utilization_pct"] > 90:
                            ultra_cleanup_memory()
                        is_pressured, mem_info = check_memory_pressure(self.ram_pressure_threshold)
                        if is_pressured and not self._disk_store_active:
                            use_disk_mode = True
                            pct = mem_info.get("percent_used", 0) if mem_info else 0
                            self._enable_disk_store(results, reason=f"ram pressure {pct:.0f}%")
                        last_memory_check = frame_count

                    frame_results = {}
                    for i, out_obj_id in enumerate(out_obj_ids):
                        mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                        if len(mask.shape) == 3:
                            mask = mask[0]
                        frame_results[out_obj_id] = mask.copy()
                        last_masks[out_obj_id] = mask.copy()
                        del mask

                    if self._disk_store_active:
                        self._store_frame_results(results, global_frame_idx, frame_results)
                    else:
                        self._store_frame_results(results, global_frame_idx, frame_results)

                    if targets_found:
                        frame_analysis = self.overlap_tracker.track_frame_overlaps_batch(
                            global_frame_idx, frame_results, object_names
                        )
                        self._maybe_emit_preview(out_frame_idx, frame_results, frame_analysis)
                        frame_analyses[global_frame_idx] = frame_analysis
                        if frame_analysis.get("target_overlaps"):
                            overlap_count += 1
                    else:
                        self._maybe_emit_preview(out_frame_idx, frame_results, None)

                    frame_count += 1
                    pbar.update(1)

                    if frame_count % 25 == 0:
                        ultra_cleanup_memory()

                    last_processed = out_frame_idx
                    del out_mask_logits, frame_results
                except Exception as exc:
                    if debug:
                        self._log(
                            "frame error: frame_idx=%s obj_ids=%s mask_logits=%s error=%s"
                            % (
                                out_frame_idx,
                                list(out_obj_ids) if out_obj_ids is not None else None,
                                _format_tensor_info(out_mask_logits),
                                exc,
                            )
                        )
                        self._log(traceback.format_exc())
                    print(f"  Error processing frame {out_frame_idx}: {exc}")
                    pbar.update(1)
                    ultra_cleanup_memory()
                    continue
            return last_processed, last_masks

        with tqdm(total=total_to_process, desc="Processing frames") as pbar:
            if chunk_size is None:
                # Add prompts from primary reference frame
                for obj_id in points_dict:
                    try:
                        points = np.array(points_dict[obj_id], dtype=np.float32)
                        labels = np.array(labels_dict[obj_id], dtype=np.int32)
                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=start_frame,
                            obj_id=obj_id,
                            points=points,
                            labels=labels,
                        )
                        del out_mask_logits, points, labels
                        ultra_cleanup_memory()
                    except Exception as exc:
                        if debug:
                            self._log(
                                "add prompts failed: obj_id=%s points=%s labels=%s error=%s"
                                % (
                                    obj_id,
                                    _format_points_info(points_dict.get(obj_id)),
                                    _format_tensor_info(labels_dict.get(obj_id)),
                                    exc,
                                )
                            )
                            self._log(traceback.format_exc())
                        print(f"  Error adding prompts for object {obj_id}: {exc}")
                        continue

                # Multiframe: Add conditioning frames from other annotated frames
                if multiframe_data and len(multiframe_data) > 1:
                    if debug:
                        self._log(f"multiframe: adding {len(multiframe_data)-1} additional conditioning frames")

                    for frame_idx, (frame_points, frame_labels, frame_obj_names) in multiframe_data.items():
                        if frame_idx == start_frame:
                            continue  # Skip primary reference (already added)

                        for obj_id in frame_points:
                            try:
                                points = np.array(frame_points[obj_id], dtype=np.float32)
                                labels = np.array(frame_labels[obj_id], dtype=np.int32)
                                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                                    inference_state=inference_state,
                                    frame_idx=frame_idx,
                                    obj_id=obj_id,
                                    points=points,
                                    labels=labels,
                                )
                                del out_mask_logits, points, labels
                                ultra_cleanup_memory()
                                if debug:
                                    self._log(f"multiframe: added conditioning frame {frame_idx} for obj_id={obj_id}")
                            except Exception as exc:
                                if debug:
                                    self._log(f"multiframe: failed to add frame {frame_idx} obj_id={obj_id} error={exc}")
                                continue
                # Forward propagation
                max_track_forward = range_end - start_frame
                max_track_forward = max(0, max_track_forward)
                last_processed, _ = _process_frames(
                    inference_state,
                    start_frame,
                    max_track_forward,
                    range_start,
                )
                if last_processed is None:
                    ultra_cleanup_memory()

                # Backward propagation (if enabled and reference frame > range_start)
                if self.enable_bidirectional and start_frame > range_start:
                    if debug:
                        self._log(f"bidirectional: starting backward propagation from frame {start_frame}")

                    # Reset state for backward propagation
                    self.predictor.reset_state(inference_state)
                    ultra_cleanup_memory()

                    # Re-add prompts
                    for obj_id in points_dict:
                        try:
                            points = np.array(points_dict[obj_id], dtype=np.float32)
                            labels = np.array(labels_dict[obj_id], dtype=np.int32)
                            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=start_frame,
                                obj_id=obj_id,
                                points=points,
                                labels=labels,
                            )
                            del out_mask_logits, points, labels
                            ultra_cleanup_memory()
                        except Exception as exc:
                            if debug:
                                self._log(f"backward add prompts failed: obj_id={obj_id} error={exc}")
                            continue

                    # Propagate backward
                    max_track_backward = start_frame - range_start
                    max_track_backward = max(0, max_track_backward)

                    # Backward propagation loop
                    last_masks_backward = {}
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=start_frame,
                        max_frame_num_to_track=max_track_backward,
                        reverse=True,
                    ):
                        if out_frame_idx >= start_frame or out_frame_idx < range_start:
                            continue

                        try:
                            global_frame_idx = self._map_frame_idx(out_frame_idx)

                            if frame_count - last_memory_check >= 50:
                                gpu_info = get_gpu_memory_info()
                                if gpu_info and gpu_info["utilization_pct"] > 90:
                                    ultra_cleanup_memory()
                                is_pressured, mem_info = check_memory_pressure(self.ram_pressure_threshold)
                                if is_pressured and not self._disk_store_active:
                                    use_disk_mode = True
                                    pct = mem_info.get("percent_used", 0) if mem_info else 0
                                    self._enable_disk_store(results, reason=f"ram pressure {pct:.0f}%")
                                last_memory_check = frame_count

                            frame_results = {}
                            for i, out_obj_id in enumerate(out_obj_ids):
                                mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                                if len(mask.shape) == 3:
                                    mask = mask[0]
                                frame_results[out_obj_id] = mask.copy()
                                last_masks_backward[out_obj_id] = mask.copy()
                                del mask

                            if self._disk_store_active:
                                self._store_frame_results(results, global_frame_idx, frame_results)
                            else:
                                self._store_frame_results(results, global_frame_idx, frame_results)

                            if targets_found:
                                frame_analysis = self.overlap_tracker.track_frame_overlaps_batch(
                                    global_frame_idx, frame_results, object_names
                                )
                                self._maybe_emit_preview(out_frame_idx, frame_results, frame_analysis)
                                frame_analyses[global_frame_idx] = frame_analysis
                                if frame_analysis.get("target_overlaps"):
                                    overlap_count += 1
                            else:
                                self._maybe_emit_preview(out_frame_idx, frame_results, None)

                            frame_count += 1
                            pbar.update(1)

                            if frame_count % 25 == 0:
                                ultra_cleanup_memory()

                            del out_mask_logits, frame_results
                        except Exception as exc:
                            if debug:
                                self._log(f"backward frame error: frame_idx={out_frame_idx} error={exc}")
                                self._log(traceback.format_exc())
                            print(f"  Error processing backward frame {out_frame_idx}: {exc}")
                            pbar.update(1)
                            ultra_cleanup_memory()
                            continue

                    ultra_cleanup_memory()
            else:
                if debug:
                    self._log("chunking enabled: chunk_size=%s chunk_overlap=%s" % (chunk_size, chunk_overlap))
                next_start = range_start
                seed_masks = None
                chunk_index = 0
                while next_start <= range_end:
                    if chunk_index > 0:
                        inference_state = self.predictor.init_state(
                            video_path=self.video_dir,
                            offload_video_to_cpu=self.offload_video_to_cpu,
                            offload_state_to_cpu=self.offload_state_to_cpu,
                            async_loading_frames=async_loading_frames,
                            max_cache_frames=max_cache_frames,
                        )
                        self.predictor.reset_state(inference_state)
                        ultra_cleanup_memory()
                    seed_frame = start_frame if chunk_index == 0 else max(start_frame, next_start - chunk_overlap)
                    if chunk_index == 0:
                        # Add prompts from primary reference frame
                        for obj_id in points_dict:
                            try:
                                points = np.array(points_dict[obj_id], dtype=np.float32)
                                labels = np.array(labels_dict[obj_id], dtype=np.int32)
                                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                                    inference_state=inference_state,
                                    frame_idx=seed_frame,
                                    obj_id=obj_id,
                                    points=points,
                                    labels=labels,
                                )
                                del out_mask_logits, points, labels
                                ultra_cleanup_memory()
                            except Exception as exc:
                                if debug:
                                    self._log(
                                        "add prompts failed: obj_id=%s points=%s labels=%s error=%s"
                                        % (
                                            obj_id,
                                            _format_points_info(points_dict.get(obj_id)),
                                            _format_tensor_info(labels_dict.get(obj_id)),
                                            exc,
                                        )
                                    )
                                    self._log(traceback.format_exc())
                                print(f"  Error adding prompts for object {obj_id}: {exc}")
                                continue

                        # Multiframe: Add conditioning frames (chunked mode)
                        if multiframe_data and len(multiframe_data) > 1:
                            if debug:
                                self._log(f"multiframe (chunked): adding {len(multiframe_data)-1} conditioning frames")
                            for frame_idx, (frame_points, frame_labels, frame_obj_names) in multiframe_data.items():
                                if frame_idx == start_frame:
                                    continue
                                for obj_id in frame_points:
                                    try:
                                        points = np.array(frame_points[obj_id], dtype=np.float32)
                                        labels = np.array(frame_labels[obj_id], dtype=np.int32)
                                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                                            inference_state=inference_state,
                                            frame_idx=frame_idx,
                                            obj_id=obj_id,
                                            points=points,
                                            labels=labels,
                                        )
                                        del out_mask_logits, points, labels
                                        ultra_cleanup_memory()
                                    except Exception as exc:
                                        if debug:
                                            self._log(f"multiframe (chunked): failed frame {frame_idx} obj_id={obj_id} error={exc}")
                                        continue
                    else:
                        for obj_id, mask in (seed_masks or {}).items():
                            if mask is None:
                                continue
                            try:
                                self.predictor.add_new_mask(
                                    inference_state=inference_state,
                                    frame_idx=seed_frame,
                                    obj_id=obj_id,
                                    mask=mask,
                                )
                            except Exception as exc:
                                if debug:
                                    self._log("seed mask failed: obj_id=%s error=%s" % (obj_id, exc))
                                    self._log(traceback.format_exc())
                                continue

                    remaining = range_end - next_start + 1
                    desired_new = min(chunk_size, remaining)
                    if chunk_index == 0:
                        max_track = max(0, desired_new - 1)
                    else:
                        max_track = max(0, desired_new + chunk_overlap - 1)
                    skip_needed = max(0, next_start - seed_frame)
                    max_track = max(max_track, skip_needed)
                    max_track = min(max_track, max(0, range_end - seed_frame))

                    last_processed, seed_masks = _process_frames(
                        inference_state,
                        seed_frame,
                        max_track,
                        next_start,
                    )
                    self.predictor.reset_state(inference_state)
                    ultra_cleanup_memory()
                    if last_processed is None:
                        break
                    next_start = last_processed + 1
                    chunk_index += 1

                # Backward propagation after chunking (chunked backward if needed)
                if self.enable_bidirectional and start_frame > range_start:
                    backward_range = start_frame - range_start
                    if debug:
                        self._log(f"bidirectional (chunked): starting backward, range={backward_range} frames")

                    # Use chunking if backward range is large
                    if chunk_size and backward_range > chunk_size:
                        # CHUNKED backward propagation
                        if debug:
                            self._log(f"backward: using CHUNKED approach, chunks={chunk_size}")

                        current_end = start_frame
                        back_chunk_idx = 0
                        seed_masks_back = None

                        while current_end > range_start:
                            chunk_start = max(range_start, current_end - chunk_size)

                            # Re-init for each chunk
                            inference_state = self.predictor.init_state(
                                video_path=self.video_dir,
                                offload_video_to_cpu=self.offload_video_to_cpu,
                                offload_state_to_cpu=self.offload_state_to_cpu,
                                async_loading_frames=async_loading_frames,
                                max_cache_frames=max_cache_frames,
                            )
                            self.predictor.reset_state(inference_state)

                            # First chunk: use prompts; later chunks: use seed masks
                            if back_chunk_idx == 0:
                                for obj_id in points_dict:
                                    try:
                                        points = np.array(points_dict[obj_id], dtype=np.float32)
                                        labels = np.array(labels_dict[obj_id], dtype=np.int32)
                                        self.predictor.add_new_points_or_box(
                                            inference_state, current_end, obj_id, points, labels
                                        )
                                        del points, labels
                                    except Exception:
                                        continue
                                # Multiframe conditioning
                                if multiframe_data and len(multiframe_data) > 1:
                                    for fr_idx, (fr_pts, fr_lbl, _) in multiframe_data.items():
                                        if fr_idx == start_frame:
                                            continue
                                        for o_id in fr_pts:
                                            try:
                                                pts = np.array(fr_pts[o_id], dtype=np.float32)
                                                lbl = np.array(fr_lbl[o_id], dtype=np.int32)
                                                self.predictor.add_new_points_or_box(
                                                    inference_state, fr_idx, o_id, pts, lbl
                                                )
                                                del pts, lbl
                                            except Exception:
                                                continue
                            else:
                                for obj_id, mask in (seed_masks_back or {}).items():
                                    if mask is not None:
                                        try:
                                            self.predictor.add_new_mask(inference_state, current_end, obj_id, mask)
                                        except Exception:
                                            continue

                            # Propagate this chunk backward
                            chunk_seeds = {}
                            for o_fi, o_oi, o_ml in self.predictor.propagate_in_video(
                                inference_state, start_frame_idx=current_end,
                                max_frame_num_to_track=current_end - chunk_start, reverse=True
                            ):
                                if o_fi >= current_end or o_fi < chunk_start:
                                    continue
                                try:
                                    g_fi = self._map_frame_idx(o_fi)
                                    fr_res = {}
                                    for i, o_id in enumerate(o_oi):
                                        msk = (o_ml[i] > 0.0).cpu().numpy()
                                        if len(msk.shape) == 3:
                                            msk = msk[0]
                                        fr_res[o_id] = msk.copy()
                                        if o_fi == chunk_start:
                                            chunk_seeds[o_id] = msk.copy()
                                        del msk
                                    self._store_frame_results(results, g_fi, fr_res)
                                    if targets_found:
                                        fa = self.overlap_tracker.track_frame_overlaps_batch(g_fi, fr_res, object_names)
                                        frame_analyses[g_fi] = fa
                                    pbar.update(1)
                                    del o_ml, fr_res
                                except Exception:
                                    pbar.update(1)
                                    continue

                            seed_masks_back = chunk_seeds
                            current_end = chunk_start
                            back_chunk_idx += 1
                            ultra_cleanup_memory()

                        if debug:
                            self._log(f"backward: completed {back_chunk_idx} backward chunks")
                    else:
                        # SINGLE backward pass (small range)
                        if debug:
                            self._log(f"backward: using SINGLE pass, range={backward_range}")

                        inference_state = self.predictor.init_state(
                            video_path=self.video_dir,
                            offload_video_to_cpu=self.offload_video_to_cpu,
                            offload_state_to_cpu=self.offload_state_to_cpu,
                            async_loading_frames=async_loading_frames,
                            max_cache_frames=max_cache_frames,
                        )
                        self.predictor.reset_state(inference_state)

                        # Add prompts
                        for obj_id in points_dict:
                            try:
                                points = np.array(points_dict[obj_id], dtype=np.float32)
                                labels = np.array(labels_dict[obj_id], dtype=np.int32)
                                self.predictor.add_new_points_or_box(inference_state, start_frame, obj_id, points, labels)
                                del points, labels
                            except Exception:
                                continue
                        # Multiframe conditioning
                        if multiframe_data and len(multiframe_data) > 1:
                            for fr_idx, (fr_pts, fr_lbl, _) in multiframe_data.items():
                                if fr_idx == start_frame:
                                    continue
                                for o_id in fr_pts:
                                    try:
                                        pts = np.array(fr_pts[o_id], dtype=np.float32)
                                        lbl = np.array(fr_lbl[o_id], dtype=np.int32)
                                        self.predictor.add_new_points_or_box(inference_state, fr_idx, o_id, pts, lbl)
                                        del pts, lbl
                                    except Exception:
                                        continue

                        # Single backward propagation
                        for o_fi, o_oi, o_ml in self.predictor.propagate_in_video(
                            inference_state, start_frame_idx=start_frame,
                            max_frame_num_to_track=backward_range, reverse=True
                        ):
                            if o_fi >= start_frame or o_fi < range_start:
                                continue
                            try:
                                g_fi = self._map_frame_idx(o_fi)
                                fr_res = {}
                                for i, o_id in enumerate(o_oi):
                                    msk = (o_ml[i] > 0.0).cpu().numpy()
                                    if len(msk.shape) == 3:
                                        msk = msk[0]
                                    fr_res[o_id] = msk.copy()
                                    del msk
                                self._store_frame_results(results, g_fi, fr_res)
                                if targets_found:
                                    fa = self.overlap_tracker.track_frame_overlaps_batch(g_fi, fr_res, object_names)
                                    frame_analyses[g_fi] = fa
                                pbar.update(1)
                                del o_ml, fr_res
                            except Exception:
                                pbar.update(1)
                                continue
                        ultra_cleanup_memory()

        if targets_found:
            last_frame = max(results.keys()) if results else 0
            self.overlap_tracker.finalize_tracking(last_frame)

        results = self._fill_missing_frames(results)
        self.frame_analyses = frame_analyses
        if chunk_size is None:
            self.predictor.reset_state(inference_state)
            ultra_cleanup_memory()
        return results

    def _process_cpu_fallback(self, points_dict, labels_dict, object_names, debug):
        print("Emergency CPU fallback - this will be slow but stable")
        if debug:
            self._log("cpu fallback: moving model to cpu")
        if hasattr(self.predictor.model, "to"):
            self.predictor.model = self.predictor.model.to("cpu")
        ultra_cleanup_memory()
        return None

    def _maybe_emit_preview(self, frame_idx, frame_results, frame_analysis):
        cb = getattr(self, "preview_callback", None)
        stride = getattr(self, "preview_stride", None)
        if cb is None or stride is None:
            return
        if frame_idx % max(1, int(stride)) != 0:
            return

        try:
            frame_path = os.path.join(self.video_dir, self.frame_names[frame_idx])
            frame = cv2.imread(frame_path)
            if frame is None:
                return
            H, W = frame.shape[:2]
            overlay = frame.copy()

            colors = getattr(self, "_preview_colors", [(0, 255, 0)])
            for i, (obj_id, mask) in enumerate(frame_results.items()):
                if mask is None:
                    continue
                m = mask[0] if hasattr(mask, "shape") and len(mask.shape) == 3 else mask
                if m is None:
                    continue

                if getattr(m, "shape", None) != (H, W):
                    try:
                        m = cv2.resize(m.astype(float), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
                    except Exception:
                        continue
                else:
                    m = m.astype(float) > 0.5

                color = colors[i % len(colors)]
                idx = m.astype(bool)
                if np.any(idx):
                    overlay[idx] = (0.6 * np.array(color, dtype=np.float32) + 0.4 * overlay[idx]).astype(np.uint8)

            text = f"Frame {frame_idx}"
            try:
                if frame_analysis and frame_analysis.get("target_overlaps"):
                    text += " | overlaps"
            except Exception:
                pass
            cv2.putText(overlay, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            max_side = int(getattr(self, "preview_max_dim", 720))
            if max(H, W) > max_side and max_side > 0:
                scale = max_side / max(H, W)
                overlay = cv2.resize(overlay, (int(W * scale), int(H * scale)))

            cb(overlay)
        except Exception as exc:
            print(f"[preview] skipped: {exc}")

    def save_results_video_with_enhanced_annotations(
        self,
        results,
        output_path,
        fps=30,
        show_original=True,
        alpha=0.5,
        frame_limit=None,
        progress_callback=None,
    ):
        """Save results video with enhanced visual feedback for looking-at events."""
        if not results:
            print("No results to save!")
            return

        frame_dir = self.full_video_dir if self.full_video_dir else self.video_dir
        frame_names = self.full_frame_names if self.full_frame_names else self.frame_names
        first_frame = cv2.imread(os.path.join(frame_dir, frame_names[0]))
        height, width = first_frame.shape[:2]

        out_width = width * 2 if show_original else width
        out = None
        for codec in ("mp4v", "H264", "avc1"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (int(out_width), int(height)))
            if out.isOpened():
                print(f"VideoWriter initialized with codec {codec}")
                break
        if out is None or not out.isOpened():
            raise RuntimeError("Failed to initialize video writer (check codec support).")

        cmap = plt.get_cmap("tab10")
        overlap_frame_count = 0
        last_target_states = {}

        max_frame = len(frame_names) - 1
        if frame_limit is not None:
            try:
                frame_limit = int(frame_limit)
            except (TypeError, ValueError):
                frame_limit = None
        if frame_limit is not None:
            max_frame = min(max_frame, max(0, frame_limit))

        for frame_idx in tqdm(range(max_frame + 1), desc="Saving frames"):
            frame = cv2.imread(os.path.join(frame_dir, frame_names[frame_idx]))
            if frame is None:
                continue

            overlay = frame.copy()
            frame_analysis = None
            if hasattr(self, "frame_analyses") and frame_idx in self.frame_analyses:
                frame_analysis = self.frame_analyses[frame_idx]

            has_looking_at_events = (
                frame_analysis
                and frame_analysis.get("target_overlaps")
                and any(frame_analysis["target_overlaps"].values())
            )
            if has_looking_at_events:
                overlap_frame_count += 1

            looking_at_info = {}
            if frame_analysis:
                frame_results = self._get_frame_results(results, frame_idx)
                for obj_id in frame_results:
                    looking_at_info[obj_id] = {
                        "is_target": False,
                        "looking_at": [],
                        "looked_at_by": [],
                        "is_being_looked_at": False,
                    }

                for target_id, looking_at_objects in frame_analysis.get("target_overlaps", {}).items():
                    if target_id in looking_at_info:
                        looking_at_info[target_id]["is_target"] = True
                        looking_at_info[target_id]["looking_at"] = [
                            obj["object_name"] for obj in looking_at_objects
                        ]

                    for obj_info in looking_at_objects:
                        obj_id = obj_info["object_id"]
                        if obj_id in looking_at_info:
                            target_name = self.overlap_tracker.target_objects.get(target_id, f"Target_{target_id}")
                            looking_at_info[obj_id]["looked_at_by"].append(target_name)
                            looking_at_info[obj_id]["is_being_looked_at"] = True

            frame_results = self._get_frame_results(results, frame_idx)
            if frame_results:
                for obj_id, mask in frame_results.items():
                    mask = self._decompress_mask(mask)
                    mask = self._expand_roi_mask(mask)
                    if mask is None:
                        continue
                    if len(mask.shape) == 3:
                        mask = mask[0]

                    if mask.shape != (height, width) and mask.shape[0] > 0 and mask.shape[1] > 0:
                        try:
                            mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR) > 0.5
                        except cv2.error:
                            continue

                    if mask.shape == (height, width):
                        obj_info = looking_at_info.get(obj_id, {})
                        is_target = obj_info.get("is_target", False)
                        is_being_looked_at = obj_info.get("is_being_looked_at", False)
                        looking_at = obj_info.get("looking_at", [])
                        looked_at_by = obj_info.get("looked_at_by", [])

                        base_color = np.array(cmap(obj_id % 10)[:3]) * 255
                        if is_target and looking_at:
                            color = np.minimum(base_color + [100, 100, 0], 255)
                            border_color = (0, 255, 255)
                            border_thickness = 8
                        elif is_being_looked_at:
                            color = np.minimum(base_color + [120, 0, 0], 255)
                            border_color = (0, 0, 255)
                            border_thickness = 8
                        else:
                            color = base_color
                            border_color = (255, 255, 255)
                            border_thickness = 2

                        # Blend only inside the mask to avoid darkening the whole frame
                        for c in range(3):
                            channel = overlay[:, :, c]
                            channel[mask] = (
                                channel[mask] * (1 - alpha) + color[c] * alpha
                            ).astype(np.uint8)

                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay, contours, -1, border_color, border_thickness)

                        obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
                        if is_target and looking_at:
                            if len(looking_at) == 1:
                                status_text = f"{obj_name} -> OVERLAPS {looking_at[0]}"
                            else:
                                status_text = f"{obj_name} -> OVERLAPS {len(looking_at)} OBJECTS"
                        elif is_being_looked_at:
                            continue
                        else:
                            status_text = obj_name
                        if is_target and looking_at:
                            pass
                        elif is_being_looked_at:
                            pass
                        # place label near mask centroid (fallback to top-left stack)
                        ys, xs = np.where(mask)
                        if len(xs) > 0:
                            cx, cy = int(xs.mean()), int(ys.mean())
                        else:
                            cx, cy = 10, 30 + (obj_id * 25) % max(25, height - 30)
                        font_scale = 0.6 if len(status_text) > 30 else 0.7
                        (tw, th), baseline = cv2.getTextSize(
                            status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
                        )
                        x = max(0, min(cx - tw // 2, width - tw - 2))
                        y = max(th + 4, min(cy, height - 4))
                        if is_target and looking_at:
                            bg_color = (0, 100, 200)
                            text_color = (0, 255, 255)
                            padding = 10
                        elif is_being_looked_at:
                            bg_color = (0, 0, 200)
                            text_color = (255, 255, 255)
                            padding = 10
                        else:
                            bg_color = (0, 0, 0)
                            text_color = (255, 255, 255)
                            padding = 5
                        cv2.rectangle(
                            overlay,
                            (x - padding, y - th - padding),
                            (x + tw + padding, y + baseline + padding),
                            bg_color,
                            -1,
                        )
                        cv2.putText(
                            overlay,
                            status_text,
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            text_color,
                            2,
                        )

            status_messages = []
            event_messages = []
            if frame_analysis:
                target_overlaps = frame_analysis.get("target_overlaps", {}) or {}
                if target_overlaps and any(target_overlaps.values()):
                    target_events = []
                    for target_id, looking_at_objects in target_overlaps.items():
                        target_name = self.overlap_tracker.target_objects.get(
                            target_id, f"Target_{target_id}"
                        )
                        object_names = [obj["object_name"] for obj in looking_at_objects]
                        if len(object_names) == 1:
                            target_events.append(f"{target_name} -> {object_names[0]}")
                        else:
                            target_events.append(f"{target_name} -> {len(object_names)} objects")
                    if target_events:
                        status_messages.append(f"LOOKING AT DETECTED: {'; '.join(target_events)}")

                for target_id, target_name in self.overlap_tracker.target_objects.items():
                    current = set(
                        obj["object_name"] for obj in (target_overlaps.get(target_id) or [])
                    )
                    last = last_target_states.get(target_id, set())
                    if current and current == last:
                        event_messages.append(
                            f"EVENT CONTINUES: {target_name} -> {', '.join(sorted(current))}"
                        )
                    elif current and not last:
                        event_messages.append(
                            f"EVENT STARTED: {target_name} -> {', '.join(sorted(current))}"
                        )
                    elif not current and last:
                        event_messages.append(
                            f"EVENT ENDED: {target_name} -> {', '.join(sorted(last))}"
                        )
                    elif current and last and current != last:
                        event_messages.append(
                            f"EVENT ENDED: {target_name} -> {', '.join(sorted(last))}"
                        )
                        event_messages.append(
                            f"EVENT STARTED: {target_name} -> {', '.join(sorted(current))}"
                        )
                    last_target_states[target_id] = current

            info_y = 30
            for message in status_messages:
                text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(
                    overlay,
                    (5, info_y - 25),
                    (text_size[0] + 15, info_y + 10),
                    (0, 0, 180),
                    -1,
                )
                cv2.putText(
                    overlay,
                    message,
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                info_y += 35

            for message in event_messages:
                text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(
                    overlay,
                    (5, info_y - 22),
                    (text_size[0] + 15, info_y + 8),
                    (0, 100, 200),
                    -1,
                )
                cv2.putText(
                    overlay,
                    message,
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                info_y += 28

            if show_original:
                output_frame = np.concatenate([frame, overlay], axis=1)
            else:
                output_frame = overlay

            out.write(output_frame)
            if progress_callback and (frame_idx % 50 == 0 or frame_idx == max_frame):
                try:
                    progress_callback(frame_idx, max_frame)
                except Exception:
                    pass

        out.release()

    def create_elan_file(self, video_path, output_path, fps, frame_offset=0):
        """Create ELAN file with corrected timing alignment."""
        if not self.overlap_tracker.has_targets():
            print("No targets found - skipping ELAN export")
            return

        try:
            cap = cv2.VideoCapture(video_path)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if abs(fps - actual_fps) > 1.0:
                fps = actual_fps
        except Exception:
            pass

        summary = self.overlap_tracker.get_overlap_summary()

        header = f'''<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="SAM2_Looking_At_Events" DATE="{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}" FORMAT="3.0" VERSION="3.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
        <MEDIA_DESCRIPTOR MEDIA_URL="file://{os.path.abspath(video_path)}"
            MIME_TYPE="video/mp4" RELATIVE_MEDIA_URL="{os.path.basename(video_path)}"/>
        <PROPERTY NAME="lastUsedAnnotationId">0</PROPERTY>
    </HEADER>
    <TIME_ORDER>
'''

        time_slots = []
        time_slot_id = 1
        time_slot_refs = {}

        all_time_points = set()
        for target_name, target_data in summary.items():
            for event in target_data["events"]:
                start_frame_corrected = event["start_frame"] + frame_offset
                end_frame_corrected = event["end_frame"] + frame_offset

                start_time = start_frame_corrected / fps
                end_time = end_frame_corrected / fps

                all_time_points.add(start_time)
                all_time_points.add(end_time)

        for time_point in sorted(all_time_points):
            time_ms = int(time_point * 1000)
            time_slots.append(f'        <TIME_SLOT TIME_SLOT_ID="ts{time_slot_id}" TIME_VALUE="{time_ms}"/>')
            time_slot_refs[time_ms] = f"ts{time_slot_id}"
            time_slot_id += 1

        header += "\n".join(time_slots) + "\n    </TIME_ORDER>\n"

        tier_content = ""
        annotation_id = 1

        for target_name, target_data in summary.items():
            tier_id = target_name.upper().replace(" ", "_").replace("-", "_")
            tier_content += f'    <TIER DEFAULT_LOCALE="en" LINGUISTIC_TYPE_REF="default" TIER_ID="{tier_id}_LOOKING_AT">\n'

            for event in target_data["events"]:
                start_frame_corrected = event["start_frame"] + frame_offset
                end_frame_corrected = event["end_frame"] + frame_offset

                start_time = start_frame_corrected / fps
                end_time = end_frame_corrected / fps
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)

                start_slot = time_slot_refs[start_ms]
                end_slot = time_slot_refs[end_ms]

                overlapping_objects_str = ", ".join(event["overlapping_objects"])

                if len(event["overlapping_objects"]) == 1:
                    annotation_value = f"Looking at: {overlapping_objects_str}"
                else:
                    annotation_value = f"Looking at {len(event['overlapping_objects'])} objects: {overlapping_objects_str}"

                annotation = f'''        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="a{annotation_id}" TIME_SLOT_REF1="{start_slot}" TIME_SLOT_REF2="{end_slot}">
                <ANNOTATION_VALUE>{annotation_value}</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>'''

                tier_content += annotation + "\n"
                annotation_id += 1

            tier_content += "    </TIER>\n"

        footer = '''    <LINGUISTIC_TYPE GRAPHIC_REFERENCES="false" LINGUISTIC_TYPE_ID="default" TIME_ALIGNABLE="true"/>
    <LOCALE LANGUAGE_CODE="en"/>
    <CONSTRAINT DESCRIPTION="Time subdivision of parent annotation's time interval, no time gaps allowed within this interval" STEREOTYPE="Time_Subdivision"/>
    <CONSTRAINT DESCRIPTION="Symbolic subdivision of a parent annotation. Annotations cannot be time-aligned" STEREOTYPE="Symbolic_Subdivision"/>
    <CONSTRAINT DESCRIPTION="1-1 association with a parent annotation" STEREOTYPE="Symbolic_Association"/>
    <CONSTRAINT DESCRIPTION="Time alignable annotations within the parent annotation's time interval, gaps are allowed" STEREOTYPE="Included_In"/>
</ANNOTATION_DOCUMENT>'''

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header + tier_content + footer)

    def export_framewise_csv(self, results, object_names, csv_path, progress_callback=None):
        import csv

        if not results:
            print("No results to export to CSV.")
            return

        analyses = getattr(self, "frame_analyses", {}) or {}
        target_ids = set(
            getattr(self, "overlap_tracker", None).target_objects.keys()
            if getattr(self, "overlap_tracker", None)
            else []
        )

        fieldnames = [
            "frame_idx",
            "obj_id",
            "obj_name",
            "is_target",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "centroid_x",
            "centroid_y",
            "area_px",
            "overlapped_by_targets",
            "target_looking_at",
            "looking_at_count",
        ]

        roi_info = getattr(self, "roi_info", None)
        roi_x0 = int(roi_info["x0"]) if roi_info else 0
        roi_y0 = int(roi_info["y0"]) if roi_info else 0
        roi_w = int(roi_info["x1"] - roi_info["x0"] + 1) if roi_info else None
        roi_h = int(roi_info["y1"] - roi_info["y0"] + 1) if roi_info else None

        def _mask_stats(m):
            try:
                if roi_info and m is not None and roi_w and roi_h and m.shape != (roi_h, roi_w):
                    try:
                        m = cv2.resize(m.astype(np.float32), (roi_w, roi_h), interpolation=cv2.INTER_LINEAR) > 0.5
                    except Exception:
                        pass
                m = m.astype(np.uint8)
                ys, xs = np.where(m > 0)
                if ys.size == 0:
                    return (None,) * 7
                x0, y0 = int(xs.min()), int(ys.min())
                x1, y1 = int(xs.max()), int(ys.max())
                w, h = (x1 - x0 + 1), (y1 - y0 + 1)
                cx = float(xs.mean())
                cy = float(ys.mean())
                area = int(ys.size)
                if roi_info:
                    x0 += roi_x0
                    y0 += roi_y0
                    cx += roi_x0
                    cy += roi_y0
                return x0, y0, w, h, cx, cy, area
            except Exception:
                return (None,) * 7

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            total_frames = len(results)
            processed_frames = 0
            for frame_idx in sorted(results.keys()):
                frame_results = self._get_frame_results(results, frame_idx)
                frame_analysis = analyses.get(frame_idx, {}) or {}
                target_overlaps = frame_analysis.get("target_overlaps", {}) or {}

                overlapped_by = {}
                for t_id, objs in target_overlaps.items():
                    t_name = object_names.get(t_id, f"Object_{t_id}")
                    for entry in objs or []:
                        oid = entry.get("object_id")
                        if oid is None:
                            continue
                        overlapped_by.setdefault(oid, []).append(t_name)

                for obj_id, mask in frame_results.items():
                    obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                    mask = self._decompress_mask(mask)
                    if mask is None:
                        continue
                    if hasattr(mask, "shape") and len(mask.shape) == 3:
                        mask = mask.squeeze()

                    bx, by, bw, bh, cx, cy, area = _mask_stats(mask)

                    is_target = obj_id in target_ids

                    looking_at = []
                    if is_target and obj_id in target_overlaps:
                        looking_at = [
                            e.get("object_name", f"Object_{e.get('object_id')}")
                            for e in (target_overlaps.get(obj_id) or [])
                        ]

                    row = dict(
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        obj_name=obj_name,
                        is_target=bool(is_target),
                        bbox_x=bx,
                        bbox_y=by,
                        bbox_w=bw,
                        bbox_h=bh,
                        centroid_x=cx,
                        centroid_y=cy,
                        area_px=area,
                        overlapped_by_targets=";".join(overlapped_by.get(obj_id, [])),
                        target_looking_at=";".join(looking_at) if is_target else "",
                        looking_at_count=(len(looking_at) if is_target else 0),
                    )
                    writer.writerow(row)
                processed_frames += 1
                if progress_callback and (
                    processed_frames % 50 == 0 or processed_frames == total_frames
                ):
                    try:
                        progress_callback(processed_frames, total_frames)
                    except Exception:
                        pass

        print(f"CSV exported: {csv_path}")
