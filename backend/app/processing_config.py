"""Typed configuration for video processing jobs.

Replaces the ad-hoc ``config.get()`` / ``try/except`` pattern in
``run_processing`` with a validated dataclass.
"""

from __future__ import annotations

import dataclasses
from typing import Optional


def _safe_float(value: object, default: float) -> float:
    """Convert *value* to float, returning *default* on failure."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int) -> int:
    """Convert *value* to int, returning *default* on failure."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_optional_int(value: object) -> Optional[int]:
    """Convert *value* to int or return ``None``."""
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _safe_optional_float(value: object) -> Optional[float]:
    """Convert *value* to float or return ``None``."""
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


@dataclasses.dataclass
class ProcessingConfig:
    """Validated processing parameters parsed from a raw session config dict.

    Use :meth:`from_dict` to create an instance from the raw ``session.config``
    dictionary.  Every field has a sensible default so partial configs are fine.
    """

    overlap_threshold: float = 0.1
    batch_size: int = 50
    auto_fallback: bool = True
    auto_tune: bool = True
    fps: Optional[float] = None
    export_video: bool = True
    export_elan: bool = True
    export_csv: bool = True
    reference_frame: int = 0
    frame_stride: Optional[int] = None
    frame_interpolation: Optional[str] = None
    roi_enabled: bool = False
    roi_margin: float = 0.15
    roi_min_size: int = 256
    roi_max_coverage: float = 0.95
    process_start_frame: Optional[int] = None
    process_end_frame: Optional[int] = None
    use_mps: bool = False
    enable_bidirectional: bool = True
    model_key: str = "auto"

    @classmethod
    def from_dict(
        cls,
        config: dict,
        video_path: Optional[str] = None,
        fps_resolver: object = None,
    ) -> ProcessingConfig:
        """Build a ``ProcessingConfig`` from a raw config dict.

        Args:
            config: The ``session.config`` dictionary.
            video_path: Path to the video file (used for FPS detection fallback).
            fps_resolver: Optional callable ``(video_path) -> (fps, ...)`` used
                when ``video_fps`` is not in *config*.
        """
        fps: Optional[float] = _safe_optional_float(config.get("video_fps"))
        if fps is None and video_path and fps_resolver is not None:
            try:
                fps, _ = fps_resolver(video_path)  # type: ignore[misc]
            except Exception:
                fps = None

        return cls(
            overlap_threshold=_safe_float(config.get("overlap_threshold", 0.1), 0.1),
            batch_size=_safe_int(config.get("batch_size", 50), 50),
            auto_fallback=bool(config.get("auto_fallback", True)),
            auto_tune=bool(config.get("auto_tune", True)),
            fps=fps,
            export_video=bool(config.get("export_video", True)),
            export_elan=bool(config.get("export_elan", True)),
            export_csv=bool(config.get("export_csv", True)),
            reference_frame=_safe_int(config.get("reference_frame", 0), 0),
            frame_stride=_safe_optional_int(config.get("frame_stride")),
            frame_interpolation=config.get("frame_interpolation"),
            roi_enabled=bool(config.get("roi_enabled", False)),
            roi_margin=_safe_float(config.get("roi_margin", 0.15), 0.15),
            roi_min_size=_safe_int(config.get("roi_min_size", 256), 256),
            roi_max_coverage=_safe_float(config.get("roi_max_coverage", 0.95), 0.95),
            process_start_frame=_safe_optional_int(config.get("process_start_frame")),
            process_end_frame=_safe_optional_int(config.get("process_end_frame")),
            use_mps=bool(config.get("use_mps", False)),
            enable_bidirectional=bool(config.get("enable_bidirectional", False)),
            model_key=config.get("model_key") or "auto",
        )
