"""Tests for new Pydantic schemas (frame suggestion + bidirectional config)."""

import pytest
from app.schemas import (
    ConfigUpdate,
    SuggestedFrame,
    FrameSuggestionResponse,
)


class TestSuggestedFrame:
    def test_valid_basic(self):
        frame = SuggestedFrame(
            frame_index=10,
            score=0.85,
            sharpness=45.2,
            brightness=0.7,
            method="basic",
        )
        assert frame.frame_index == 10
        assert frame.score == pytest.approx(0.85)
        assert frame.method == "basic"

    def test_valid_dinov2(self):
        frame = SuggestedFrame(
            frame_index=0,
            score=0.92,
            sharpness=120.5,
            brightness=0.55,
            method="dinov2",
        )
        assert frame.method == "dinov2"

    def test_missing_required_field(self):
        with pytest.raises(Exception):  # ValidationError
            SuggestedFrame(
                frame_index=10,
                score=0.5,
                # missing sharpness, brightness, method
            )


class TestFrameSuggestionResponse:
    def test_valid_response(self):
        resp = FrameSuggestionResponse(
            session_id="abc-123",
            suggested_frames=[
                SuggestedFrame(
                    frame_index=5,
                    score=0.8,
                    sharpness=50.0,
                    brightness=0.6,
                    method="basic",
                )
            ],
            total_analyzed=50,
            method_used="basic",
        )
        assert resp.session_id == "abc-123"
        assert len(resp.suggested_frames) == 1
        assert resp.total_analyzed == 50

    def test_empty_suggestions(self):
        resp = FrameSuggestionResponse(
            session_id="abc",
            suggested_frames=[],
            total_analyzed=0,
            method_used="none",
        )
        assert len(resp.suggested_frames) == 0


class TestConfigUpdateBidirectional:
    def test_default_false(self):
        config = ConfigUpdate(session_id="test")
        assert config.enable_bidirectional is False

    def test_set_true(self):
        config = ConfigUpdate(session_id="test", enable_bidirectional=True)
        assert config.enable_bidirectional is True

    def test_serialization_roundtrip(self):
        config = ConfigUpdate(session_id="test", enable_bidirectional=True)
        data = config.model_dump()
        assert data["enable_bidirectional"] is True
        restored = ConfigUpdate(**data)
        assert restored.enable_bidirectional is True
