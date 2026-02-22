"""Tests for ELAN export with target objects.

Verifies that:
- ELAN export is skipped when no targets are registered
- ELAN export produces valid XML when targets are present
- Overlap events between target and other objects generate ELAN annotations
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from xml.etree import ElementTree

import cv2
import numpy as np
import pytest

from app.pipeline import (
    ImprovedTargetOverlapTracker,
    UltraOptimizedProcessor,
)


@pytest.fixture
def frames_dir(tmp_path: Path) -> Path:
    """Create a directory of numbered JPEG frames."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True)
    rng = np.random.default_rng(42)
    for i in range(10):
        img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), img)
    return frames_dir


@pytest.fixture
def video_file(tmp_path: Path) -> str:
    """Create a small test video file for ELAN export."""
    video_path = str(tmp_path / "test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 30.0, (64, 64))
    for _ in range(10):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


class TestElanExportSkipped:
    """Test that ELAN export is skipped when there are no targets."""

    def test_no_targets_skips_elan(self, frames_dir, video_file, tmp_path):
        """When no target objects are registered, create_elan_file should not create a file."""
        predictor = MagicMock()
        predictor.image_size = 64

        processor = UltraOptimizedProcessor(
            predictor=predictor,
            video_dir=str(frames_dir),
            overlap_threshold=0.1,
            reference_frame=0,
            batch_size=50,
            enable_bidirectional=False,
        )

        # Register non-target objects only
        processor.overlap_tracker.register_target(1, "ball")
        processor.overlap_tracker.register_target(2, "cup")

        assert processor.overlap_tracker.has_targets() is False

        elan_path = str(tmp_path / "output.eaf")
        processor.create_elan_file(video_file, elan_path, fps=30.0)

        # File should NOT be created
        assert not Path(elan_path).exists()

    def test_target_present_creates_elan(self, frames_dir, video_file, tmp_path):
        """When target objects are registered, ELAN file should be created."""
        predictor = MagicMock()
        predictor.image_size = 64

        processor = UltraOptimizedProcessor(
            predictor=predictor,
            video_dir=str(frames_dir),
            overlap_threshold=0.1,
            reference_frame=0,
            batch_size=50,
            enable_bidirectional=False,
        )

        # Register a target object
        processor.overlap_tracker.register_target(1, "target")
        processor.overlap_tracker.register_target(2, "ball")

        assert processor.overlap_tracker.has_targets() is True

        # Add a fake overlap event
        processor.overlap_tracker.overlap_events[1] = [
            {
                "start_frame": 0,
                "end_frame": 5,
                "duration_frames": 6,
                "overlapping_objects": ["ball"],
                "event_type": "looking_at",
            }
        ]

        elan_path = str(tmp_path / "output.eaf")
        processor.create_elan_file(video_file, elan_path, fps=30.0)

        assert Path(elan_path).exists()

    def test_elan_xml_structure(self, frames_dir, video_file, tmp_path):
        """Verify the ELAN XML structure is valid."""
        predictor = MagicMock()
        predictor.image_size = 64

        processor = UltraOptimizedProcessor(
            predictor=predictor,
            video_dir=str(frames_dir),
            overlap_threshold=0.1,
            reference_frame=0,
            batch_size=50,
            enable_bidirectional=False,
        )

        processor.overlap_tracker.register_target(1, "target_hand")
        processor.overlap_tracker.overlap_events[1] = [
            {
                "start_frame": 2,
                "end_frame": 7,
                "duration_frames": 6,
                "overlapping_objects": ["ball"],
                "event_type": "looking_at",
            },
            {
                "start_frame": 15,
                "end_frame": 20,
                "duration_frames": 6,
                "overlapping_objects": ["cup", "box"],
                "event_type": "looking_at",
            },
        ]

        elan_path = str(tmp_path / "test.eaf")
        processor.create_elan_file(video_file, elan_path, fps=30.0)

        assert Path(elan_path).exists()

        # Parse and validate XML
        tree = ElementTree.parse(elan_path)
        root = tree.getroot()
        assert root.tag == "ANNOTATION_DOCUMENT"

        # Check TIME_ORDER has time slots
        time_order = root.find("TIME_ORDER")
        assert time_order is not None
        time_slots = time_order.findall("TIME_SLOT")
        assert len(time_slots) >= 4  # At least 2 events x 2 slots each

        # Check TIER exists
        tiers = root.findall("TIER")
        assert len(tiers) >= 1
        tier_id = tiers[0].get("TIER_ID")
        assert "TARGET_HAND" in tier_id
        assert "LOOKING_AT" in tier_id

        # Check annotations
        annotations = tiers[0].findall(".//ANNOTATION/ALIGNABLE_ANNOTATION")
        assert len(annotations) == 2

        # First event should mention "ball"
        first_value = annotations[0].find("ANNOTATION_VALUE").text
        assert "ball" in first_value.lower()

        # Second event should mention "2 objects"
        second_value = annotations[1].find("ANNOTATION_VALUE").text
        assert "2 objects" in second_value.lower()

    def test_elan_timing_correct(self, frames_dir, video_file, tmp_path):
        """Verify ELAN time values are correctly calculated from FPS."""
        predictor = MagicMock()
        predictor.image_size = 64

        processor = UltraOptimizedProcessor(
            predictor=predictor,
            video_dir=str(frames_dir),
            overlap_threshold=0.1,
            reference_frame=0,
            batch_size=50,
            enable_bidirectional=False,
        )

        processor.overlap_tracker.register_target(1, "target")
        # Event from frame 30 to frame 60 at 30fps = 1.0s to 2.0s
        processor.overlap_tracker.overlap_events[1] = [
            {
                "start_frame": 30,
                "end_frame": 60,
                "duration_frames": 31,
                "overlapping_objects": ["ball"],
                "event_type": "looking_at",
            }
        ]

        elan_path = str(tmp_path / "timing.eaf")
        processor.create_elan_file(video_file, elan_path, fps=30.0)

        tree = ElementTree.parse(elan_path)
        root = tree.getroot()
        time_order = root.find("TIME_ORDER")
        time_slots = time_order.findall("TIME_SLOT")

        # Extract time values
        time_values = sorted(
            int(ts.get("TIME_VALUE")) for ts in time_slots
        )
        # frame 30 / 30fps = 1000ms, frame 60 / 30fps = 2000ms
        assert 1000 in time_values
        assert 2000 in time_values


class TestElanWithTargetNaming:
    """Test that 'target' name matching works case-insensitively."""

    def test_target_case_insensitive(self):
        tracker = ImprovedTargetOverlapTracker()
        assert tracker.register_target(1, "Target") is True
        assert tracker.register_target(2, "TARGET") is True
        assert tracker.register_target(3, "my_target") is True
        assert tracker.register_target(4, "hand") is False
        assert len(tracker.target_objects) == 3

    def test_target_in_compound_name(self):
        tracker = ImprovedTargetOverlapTracker()
        assert tracker.register_target(1, "target_hand") is True
        assert tracker.register_target(2, "left_target_finger") is True
        assert tracker.register_target(3, "ball") is False
