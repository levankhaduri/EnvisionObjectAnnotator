"""Tests for multiframe annotation loading."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from app.processing import _load_multiframe_annotations


# We need SESSIONS_DIR to point to our tmp fixtures
@pytest.fixture
def sessions_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def session_with_single_frame(sessions_dir: Path) -> str:
    """Create a session with annotations on a single frame."""
    session_id = "test-single"
    ann_dir = sessions_dir / session_id / "annotations"
    ann_dir.mkdir(parents=True)
    ann_data = {
        "objects": {
            "hand": [
                {"x": 0.5, "y": 0.5, "label": 1},
                {"x": 0.6, "y": 0.6, "label": 1},
            ],
            "cup": [
                {"x": 0.2, "y": 0.3, "label": 1},
            ],
        }
    }
    (ann_dir / "frame_00010.json").write_text(json.dumps(ann_data))
    return session_id


@pytest.fixture
def session_with_multi_frame(sessions_dir: Path) -> str:
    """Create a session with annotations on multiple frames."""
    session_id = "test-multi"
    ann_dir = sessions_dir / session_id / "annotations"
    ann_dir.mkdir(parents=True)

    frame_0 = {
        "objects": {
            "hand": [
                {"x": 0.5, "y": 0.5, "label": 1},
            ],
            "cup": [
                {"x": 0.2, "y": 0.3, "label": 1},
            ],
        }
    }
    frame_50 = {
        "objects": {
            "hand": [
                {"x": 0.55, "y": 0.45, "label": 1},
            ],
            "ball": [
                {"x": 0.7, "y": 0.7, "label": 1},
            ],
        }
    }
    (ann_dir / "frame_00000.json").write_text(json.dumps(frame_0))
    (ann_dir / "frame_00050.json").write_text(json.dumps(frame_50))
    return session_id


@pytest.fixture
def session_empty(sessions_dir: Path) -> str:
    """Create a session with an empty annotations directory."""
    session_id = "test-empty"
    ann_dir = sessions_dir / session_id / "annotations"
    ann_dir.mkdir(parents=True)
    return session_id


@pytest.fixture
def session_no_dir(sessions_dir: Path) -> str:
    """Create a session with no annotations directory at all."""
    session_id = "test-nodir"
    (sessions_dir / session_id).mkdir(parents=True)
    return session_id


class TestLoadMultiframeAnnotations:
    def test_single_frame(self, sessions_dir: Path, session_with_single_frame: str):
        with patch("app.processing.SESSIONS_DIR", sessions_dir):
            mf_data, name_to_id = _load_multiframe_annotations(session_with_single_frame)

        assert len(mf_data) == 1
        assert 10 in mf_data
        points_dict, labels_dict, obj_names = mf_data[10]
        assert len(points_dict) == 2  # hand + cup
        assert len(name_to_id) == 2

    def test_multi_frame_consistent_ids(self, sessions_dir: Path, session_with_multi_frame: str):
        with patch("app.processing.SESSIONS_DIR", sessions_dir):
            mf_data, name_to_id = _load_multiframe_annotations(session_with_multi_frame)

        assert len(mf_data) == 2
        assert 0 in mf_data
        assert 50 in mf_data

        # "hand" appears in both frames — must have same ID
        hand_id = name_to_id["hand"]
        assert hand_id == name_to_id["hand"]

        # Check frame 0 has hand + cup
        pd0, _, on0 = mf_data[0]
        assert hand_id in pd0

        # Check frame 50 has hand + ball
        pd50, _, on50 = mf_data[50]
        assert hand_id in pd50

        # Total unique objects across all frames: hand, cup, ball
        assert len(name_to_id) == 3

    def test_object_ids_are_sequential(self, sessions_dir: Path, session_with_multi_frame: str):
        with patch("app.processing.SESSIONS_DIR", sessions_dir):
            _, name_to_id = _load_multiframe_annotations(session_with_multi_frame)

        ids = sorted(name_to_id.values())
        assert ids == [1, 2, 3]

    def test_empty_dir_raises(self, sessions_dir: Path, session_empty: str):
        with patch("app.processing.SESSIONS_DIR", sessions_dir):
            with pytest.raises(FileNotFoundError, match="No valid annotations"):
                _load_multiframe_annotations(session_empty)

    def test_no_dir_raises(self, sessions_dir: Path, session_no_dir: str):
        with patch("app.processing.SESSIONS_DIR", sessions_dir):
            with pytest.raises(FileNotFoundError, match="No annotations directory"):
                _load_multiframe_annotations(session_no_dir)

    def test_ignores_malformed_json(self, sessions_dir: Path):
        session_id = "test-malformed"
        ann_dir = sessions_dir / session_id / "annotations"
        ann_dir.mkdir(parents=True)
        # Write one valid and one malformed
        valid = {"objects": {"hand": [{"x": 0.5, "y": 0.5, "label": 1}]}}
        (ann_dir / "frame_00000.json").write_text(json.dumps(valid))
        (ann_dir / "frame_00001.json").write_text("NOT JSON{{{")

        with patch("app.processing.SESSIONS_DIR", sessions_dir):
            mf_data, _ = _load_multiframe_annotations(session_id)

        assert len(mf_data) == 1
        assert 0 in mf_data

    def test_ignores_empty_objects(self, sessions_dir: Path):
        session_id = "test-empty-obj"
        ann_dir = sessions_dir / session_id / "annotations"
        ann_dir.mkdir(parents=True)
        # Frame with empty points list
        data = {"objects": {"hand": []}}
        (ann_dir / "frame_00000.json").write_text(json.dumps(data))
        # Frame with valid points
        valid = {"objects": {"cup": [{"x": 0.1, "y": 0.2, "label": 1}]}}
        (ann_dir / "frame_00001.json").write_text(json.dumps(valid))

        with patch("app.processing.SESSIONS_DIR", sessions_dir):
            mf_data, _ = _load_multiframe_annotations(session_id)

        # Only frame 1 should be loaded (frame 0 had empty points)
        assert len(mf_data) == 1
        assert 1 in mf_data
