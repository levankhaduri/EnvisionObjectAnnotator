"""Tests for multi-object annotation saving via the /annotation/points API.

Verifies that saving multiple objects across multiple frames produces the
correct annotation JSON files, and that _load_multiframe_annotations() can
read them back with consistent object IDs.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import state
from app.schemas import Session
from app.processing import _load_multiframe_annotations


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def session_with_frames(tmp_path: Path):
    """Create a session with a frames directory (needed for annotation endpoint)."""
    session_id = "test-annotation-save"
    # Create the session directory structure
    session_dir = tmp_path / session_id
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True)
    ann_dir = session_dir / "annotations"
    ann_dir.mkdir(parents=True)

    # Register session
    state.create_session(Session(id=session_id, status="created"))

    yield session_id, tmp_path

    state.sessions.pop(session_id, None)
    state.processing.pop(session_id, None)


class TestAnnotationSaveEndpoint:
    """Test the /annotation/points endpoint saves all objects correctly."""

    def test_save_single_object(self, client: TestClient, session_with_frames):
        session_id, tmp_path = session_with_frames
        with patch("app.main.SESSIONS_DIR", tmp_path):
            resp = client.post("/annotation/points", json={
                "session_id": session_id,
                "frame_index": 0,
                "object_name": "ball",
                "points": [{"x": 0.5, "y": 0.5, "label": 1}],
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "saved"
        assert data["points"] == 1
        assert data["object"] == "ball"

        # Verify file on disk
        frame_file = tmp_path / session_id / "annotations" / "frame_00000.json"
        assert frame_file.exists()
        content = json.loads(frame_file.read_text())
        assert "ball" in content["objects"]
        assert len(content["objects"]["ball"]) == 1

    def test_save_multiple_objects_same_frame(self, client: TestClient, session_with_frames):
        """Multiple objects saved to same frame should accumulate in one JSON file."""
        session_id, tmp_path = session_with_frames
        with patch("app.main.SESSIONS_DIR", tmp_path):
            # Save first object
            client.post("/annotation/points", json={
                "session_id": session_id,
                "frame_index": 10,
                "object_name": "target",
                "points": [{"x": 0.3, "y": 0.3, "label": 1}],
            })
            # Save second object
            client.post("/annotation/points", json={
                "session_id": session_id,
                "frame_index": 10,
                "object_name": "ball",
                "points": [{"x": 0.7, "y": 0.7, "label": 1}],
            })
            # Save third object
            client.post("/annotation/points", json={
                "session_id": session_id,
                "frame_index": 10,
                "object_name": "cup",
                "points": [
                    {"x": 0.1, "y": 0.1, "label": 1},
                    {"x": 0.2, "y": 0.2, "label": 1},
                ],
            })

        frame_file = tmp_path / session_id / "annotations" / "frame_00010.json"
        content = json.loads(frame_file.read_text())

        assert len(content["objects"]) == 3
        assert "target" in content["objects"]
        assert "ball" in content["objects"]
        assert "cup" in content["objects"]
        assert len(content["objects"]["cup"]) == 2

    def test_save_objects_across_multiple_frames(self, client: TestClient, session_with_frames):
        """Objects saved to different frames should create separate JSON files."""
        session_id, tmp_path = session_with_frames
        with patch("app.main.SESSIONS_DIR", tmp_path):
            client.post("/annotation/points", json={
                "session_id": session_id,
                "frame_index": 0,
                "object_name": "target",
                "points": [{"x": 0.5, "y": 0.5, "label": 1}],
            })
            client.post("/annotation/points", json={
                "session_id": session_id,
                "frame_index": 50,
                "object_name": "target",
                "points": [{"x": 0.6, "y": 0.4, "label": 1}],
            })
            client.post("/annotation/points", json={
                "session_id": session_id,
                "frame_index": 50,
                "object_name": "ball",
                "points": [{"x": 0.2, "y": 0.8, "label": 1}],
            })

        frame_0 = tmp_path / session_id / "annotations" / "frame_00000.json"
        frame_50 = tmp_path / session_id / "annotations" / "frame_00050.json"
        assert frame_0.exists()
        assert frame_50.exists()

        data_0 = json.loads(frame_0.read_text())
        data_50 = json.loads(frame_50.read_text())
        assert len(data_0["objects"]) == 1  # Only target on frame 0
        assert len(data_50["objects"]) == 2  # target + ball on frame 50

    def test_overwrite_object_same_name(self, client: TestClient, session_with_frames):
        """Saving the same object name again should overwrite its points."""
        session_id, tmp_path = session_with_frames
        with patch("app.main.SESSIONS_DIR", tmp_path):
            # First save
            client.post("/annotation/points", json={
                "session_id": session_id,
                "frame_index": 5,
                "object_name": "ball",
                "points": [{"x": 0.1, "y": 0.1, "label": 1}],
            })
            # Overwrite with new points
            client.post("/annotation/points", json={
                "session_id": session_id,
                "frame_index": 5,
                "object_name": "ball",
                "points": [
                    {"x": 0.9, "y": 0.9, "label": 1},
                    {"x": 0.8, "y": 0.8, "label": 0},
                ],
            })

        frame_file = tmp_path / session_id / "annotations" / "frame_00005.json"
        content = json.loads(frame_file.read_text())
        assert len(content["objects"]["ball"]) == 2  # Updated to 2 points
        assert content["objects"]["ball"][0]["x"] == 0.9

    def test_rename_object(self, client: TestClient, session_with_frames):
        """Renaming an object should remove the old name and add the new one."""
        session_id, tmp_path = session_with_frames
        with patch("app.main.SESSIONS_DIR", tmp_path):
            # Save as old name
            client.post("/annotation/points", json={
                "session_id": session_id,
                "frame_index": 0,
                "object_name": "hand",
                "points": [{"x": 0.5, "y": 0.5, "label": 1}],
            })
            # Rename to new name
            client.post("/annotation/points", json={
                "session_id": session_id,
                "frame_index": 0,
                "object_name": "target_hand",
                "previous_object_name": "hand",
                "points": [{"x": 0.5, "y": 0.5, "label": 1}],
            })

        frame_file = tmp_path / session_id / "annotations" / "frame_00000.json"
        content = json.loads(frame_file.read_text())
        assert "hand" not in content["objects"]
        assert "target_hand" in content["objects"]

    def test_session_not_found(self, client: TestClient):
        resp = client.post("/annotation/points", json={
            "session_id": "nonexistent",
            "frame_index": 0,
            "object_name": "ball",
            "points": [{"x": 0.5, "y": 0.5, "label": 1}],
        })
        assert resp.status_code == 404


class TestMultiObjectRoundTrip:
    """Test saving multiple objects then loading with _load_multiframe_annotations."""

    def test_roundtrip_single_frame_multiple_objects(self, client: TestClient, session_with_frames):
        session_id, tmp_path = session_with_frames
        with patch("app.main.SESSIONS_DIR", tmp_path):
            for obj_name in ["target", "ball", "box", "corn"]:
                client.post("/annotation/points", json={
                    "session_id": session_id,
                    "frame_index": 3330,
                    "object_name": obj_name,
                    "points": [{"x": 0.5, "y": 0.5, "label": 1}],
                })

        # Now load using multiframe loader
        with patch("app.processing.SESSIONS_DIR", tmp_path):
            mf_data, name_to_id = _load_multiframe_annotations(session_id)

        assert len(mf_data) == 1
        assert 3330 in mf_data
        points_dict, labels_dict, obj_names = mf_data[3330]
        assert len(points_dict) == 4  # All 4 objects loaded
        assert len(name_to_id) == 4
        assert "target" in name_to_id
        assert "ball" in name_to_id
        assert "box" in name_to_id
        assert "corn" in name_to_id

    def test_roundtrip_multi_frame_multi_object(self, client: TestClient, session_with_frames):
        """Save objects across frames, then verify multiframe loader gets them all."""
        session_id, tmp_path = session_with_frames
        with patch("app.main.SESSIONS_DIR", tmp_path):
            # Frame 0: target + ball
            client.post("/annotation/points", json={
                "session_id": session_id,
                "frame_index": 0,
                "object_name": "target",
                "points": [{"x": 0.3, "y": 0.3, "label": 1}],
            })
            client.post("/annotation/points", json={
                "session_id": session_id,
                "frame_index": 0,
                "object_name": "ball",
                "points": [{"x": 0.7, "y": 0.7, "label": 1}],
            })
            # Frame 50: target + cup (new object)
            client.post("/annotation/points", json={
                "session_id": session_id,
                "frame_index": 50,
                "object_name": "target",
                "points": [{"x": 0.4, "y": 0.4, "label": 1}],
            })
            client.post("/annotation/points", json={
                "session_id": session_id,
                "frame_index": 50,
                "object_name": "cup",
                "points": [{"x": 0.1, "y": 0.9, "label": 1}],
            })

        with patch("app.processing.SESSIONS_DIR", tmp_path):
            mf_data, name_to_id = _load_multiframe_annotations(session_id)

        assert len(mf_data) == 2
        assert 0 in mf_data
        assert 50 in mf_data

        # 3 unique objects: target, ball, cup
        assert len(name_to_id) == 3

        # Same object name = same ID across frames
        target_id = name_to_id["target"]
        pd0, _, _ = mf_data[0]
        pd50, _, _ = mf_data[50]
        assert target_id in pd0
        assert target_id in pd50
