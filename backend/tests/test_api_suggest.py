"""Tests for the /frames/suggest/{session_id} API endpoint."""

import cv2
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.state import state
from app.schemas import Session


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_session(tmp_path: Path):
    """Create a mock session with extracted frames."""
    session_id = "test-suggest-session"
    frames_dir = tmp_path / session_id / "frames"
    frames_dir.mkdir(parents=True)

    # Create 10 synthetic frames
    rng = np.random.default_rng(42)
    for i in range(10):
        img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), img)

    # Register session using proper API
    state.create_session(Session(id=session_id, status="created"))

    yield session_id, tmp_path

    # Cleanup
    state.sessions.pop(session_id, None)
    state.processing.pop(session_id, None)


class TestSuggestFramesEndpoint:
    def test_success(self, client: TestClient, mock_session):
        session_id, tmp_path = mock_session
        with patch("app.main.SESSIONS_DIR", tmp_path):
            resp = client.get(f"/frames/suggest/{session_id}?use_dinov2=false")

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert len(data["suggested_frames"]) > 0
        assert data["total_analyzed"] == 10
        assert data["method_used"] == "basic"

    def test_frame_structure(self, client: TestClient, mock_session):
        session_id, tmp_path = mock_session
        with patch("app.main.SESSIONS_DIR", tmp_path):
            resp = client.get(f"/frames/suggest/{session_id}?top_k=3&use_dinov2=false")

        data = resp.json()
        assert len(data["suggested_frames"]) <= 3
        for frame in data["suggested_frames"]:
            assert "frame_index" in frame
            assert "score" in frame
            assert "sharpness" in frame
            assert "brightness" in frame
            assert "method" in frame

    def test_session_not_found(self, client: TestClient):
        resp = client.get("/frames/suggest/nonexistent-id")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_frames_not_extracted(self, client: TestClient, tmp_path: Path):
        session_id = "test-no-frames"
        state.create_session(Session(id=session_id, status="created"))
        try:
            with patch("app.main.SESSIONS_DIR", tmp_path):
                # Session dir exists but no frames subdir
                (tmp_path / session_id).mkdir(parents=True, exist_ok=True)
                resp = client.get(f"/frames/suggest/{session_id}")

            assert resp.status_code == 404
            assert "not extracted" in resp.json()["detail"].lower()
        finally:
            state.sessions.pop(session_id, None)
            state.processing.pop(session_id, None)

    def test_top_k_parameter(self, client: TestClient, mock_session):
        session_id, tmp_path = mock_session
        with patch("app.main.SESSIONS_DIR", tmp_path):
            resp = client.get(f"/frames/suggest/{session_id}?top_k=2&use_dinov2=false")

        data = resp.json()
        assert len(data["suggested_frames"]) == 2

    def test_scores_are_valid_floats(self, client: TestClient, mock_session):
        session_id, tmp_path = mock_session
        with patch("app.main.SESSIONS_DIR", tmp_path):
            resp = client.get(f"/frames/suggest/{session_id}?use_dinov2=false")

        for frame in resp.json()["suggested_frames"]:
            assert isinstance(frame["score"], float)
            assert isinstance(frame["sharpness"], float)
            assert isinstance(frame["brightness"], float)
            assert 0.0 <= frame["score"] <= 1.0
