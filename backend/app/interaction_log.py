"""Session interaction logging for research reproducibility."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SESSIONS_DIR = Path(__file__).parent.parent / "sessions"


def _get_log_path(session_id: str) -> Path:
    """Get the interaction log file path for a session."""
    return SESSIONS_DIR / session_id / "interaction_log.json"


def _load_log(session_id: str) -> list[dict[str, Any]]:
    """Load existing log entries for a session."""
    log_path = _get_log_path(session_id)
    if log_path.exists():
        try:
            return json.loads(log_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_log(session_id: str, entries: list[dict[str, Any]]) -> None:
    """Save log entries for a session."""
    log_path = _get_log_path(session_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(entries, indent=2, default=str), encoding="utf-8")


def log_event(
    session_id: str,
    event_type: str,
    data: dict[str, Any] | None = None,
) -> None:
    """
    Log an interaction event for a session.

    Args:
        session_id: The session ID
        event_type: Type of event (e.g., "session_created", "point_added")
        data: Additional event data
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        "data": data or {},
    }
    entries = _load_log(session_id)
    entries.append(entry)
    _save_log(session_id, entries)


def get_session_log(session_id: str) -> list[dict[str, Any]]:
    """Get all interaction log entries for a session."""
    return _load_log(session_id)


# Convenience functions for common events
def log_session_created(session_id: str, name: str | None = None) -> None:
    log_event(session_id, "session_created", {"name": name})


def log_video_uploaded(session_id: str, filename: str, size_bytes: int) -> None:
    log_event(session_id, "video_uploaded", {"filename": filename, "size_bytes": size_bytes})


def log_frames_extracted(session_id: str, frame_count: int, quality: int) -> None:
    log_event(session_id, "frames_extracted", {"frame_count": frame_count, "quality": quality})


def log_config_updated(session_id: str, config: dict[str, Any]) -> None:
    log_event(session_id, "config_updated", {"config": config})


def log_object_created(session_id: str, object_name: str, frame_index: int) -> None:
    log_event(session_id, "object_created", {"object_name": object_name, "frame_index": frame_index})


def log_points_saved(
    session_id: str,
    frame_index: int,
    object_name: str,
    points: list[dict[str, Any]],
) -> None:
    log_event(
        session_id,
        "points_saved",
        {
            "frame_index": frame_index,
            "object_name": object_name,
            "point_count": len(points),
            "points": points,
        },
    )


def log_test_mask(session_id: str, frame_index: int, object_name: str, point_count: int) -> None:
    log_event(
        session_id,
        "test_mask",
        {"frame_index": frame_index, "object_name": object_name, "point_count": point_count},
    )


def log_processing_started(session_id: str, object_count: int) -> None:
    log_event(session_id, "processing_started", {"object_count": object_count})


def log_processing_completed(session_id: str, duration_seconds: float, frame_count: int) -> None:
    log_event(
        session_id,
        "processing_completed",
        {"duration_seconds": duration_seconds, "frame_count": frame_count},
    )


def log_processing_failed(session_id: str, error: str) -> None:
    log_event(session_id, "processing_failed", {"error": error})


def log_export(session_id: str, export_type: str, filename: str) -> None:
    log_event(session_id, "export", {"type": export_type, "filename": filename})
