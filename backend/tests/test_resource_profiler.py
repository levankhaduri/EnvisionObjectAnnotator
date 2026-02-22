"""Tests for the resource profiler module.

Verifies that the ResourceProfiler correctly:
- Starts and stops without errors
- Takes samples with expected fields
- Saves CSV, JSON, and HTML outputs
"""

import json
import time
from pathlib import Path

import pytest

from app.resource_profiler import ResourceProfiler


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "profile_output"
    d.mkdir()
    return d


class TestResourceProfilerBasic:
    """Test basic profiler lifecycle."""

    def test_start_stop_no_errors(self, output_dir):
        profiler = ResourceProfiler(
            output_dir=output_dir,
            interval_seconds=0.1,
            session_id="test-session",
        )
        profiler.start()
        time.sleep(0.3)  # Let a few samples be taken
        html_path = profiler.stop()
        assert html_path.exists()
        assert html_path.suffix == ".html"

    def test_samples_collected(self, output_dir):
        profiler = ResourceProfiler(
            output_dir=output_dir,
            interval_seconds=0.05,
            session_id="test-session",
        )
        profiler.start()
        time.sleep(0.25)
        profiler.stop()

        # Should have collected at least 2 samples
        assert len(profiler.samples) >= 2

    def test_sample_has_elapsed_field(self, output_dir):
        profiler = ResourceProfiler(
            output_dir=output_dir,
            interval_seconds=0.05,
            session_id="test-session",
        )
        profiler.start()
        time.sleep(0.15)
        profiler.stop()

        for sample in profiler.samples:
            assert "elapsed_s" in sample
            assert isinstance(sample["elapsed_s"], float)
            assert sample["elapsed_s"] >= 0

    def test_sample_has_cpu_fields(self, output_dir):
        """If psutil is available, CPU fields should be present."""
        try:
            import psutil  # noqa: F401
            has_psutil = True
        except ImportError:
            has_psutil = False

        profiler = ResourceProfiler(
            output_dir=output_dir,
            interval_seconds=0.05,
            session_id="test-session",
        )
        profiler.start()
        time.sleep(0.15)
        profiler.stop()

        if has_psutil and profiler.samples:
            sample = profiler.samples[-1]  # Last sample most likely to have data
            assert "cpu_percent" in sample
            assert "ram_used_gb" in sample
            assert "ram_total_gb" in sample
            assert "ram_percent" in sample

    def test_stop_idempotent(self, output_dir):
        """Calling stop() twice should not raise."""
        profiler = ResourceProfiler(
            output_dir=output_dir,
            interval_seconds=0.05,
            session_id="test-session",
        )
        profiler.start()
        time.sleep(0.1)
        profiler.stop()
        profiler.stop()  # Second call should not raise


class TestResourceProfilerOutputs:
    """Test output files are created correctly."""

    def test_csv_created(self, output_dir):
        profiler = ResourceProfiler(
            output_dir=output_dir,
            interval_seconds=0.05,
            session_id="test-csv",
        )
        profiler.start()
        time.sleep(0.15)
        profiler.stop()

        csv_path = output_dir / "resource_profile.csv"
        assert csv_path.exists()

        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) >= 2  # Header + at least 1 data row
        header = lines[0]
        assert "elapsed_s" in header

    def test_json_created(self, output_dir):
        profiler = ResourceProfiler(
            output_dir=output_dir,
            interval_seconds=0.05,
            session_id="test-json",
        )
        profiler.start()
        time.sleep(0.15)
        profiler.stop()

        json_path = output_dir / "resource_profile.json"
        assert json_path.exists()

        data = json.loads(json_path.read_text())
        assert data["session_id"] == "test-json"
        assert data["sample_count"] >= 1
        assert data["interval_seconds"] == 0.05
        assert isinstance(data["samples"], list)
        assert len(data["samples"]) >= 1

    def test_html_created_and_valid(self, output_dir):
        profiler = ResourceProfiler(
            output_dir=output_dir,
            interval_seconds=0.05,
            session_id="test-html",
        )
        profiler.start()
        time.sleep(0.15)
        profiler.stop()

        html_path = output_dir / "resource_profile.html"
        assert html_path.exists()

        html_content = html_path.read_text()
        assert "<!DOCTYPE html>" in html_content
        assert "Processing Resource Profile" in html_content
        assert "test-html" in html_content  # Session ID in title
        assert "drawChart" in html_content  # Chart JS function

    def test_html_contains_stats(self, output_dir):
        profiler = ResourceProfiler(
            output_dir=output_dir,
            interval_seconds=0.05,
            session_id="test-stats",
        )
        profiler.start()
        time.sleep(0.2)
        profiler.stop()

        html_content = (output_dir / "resource_profile.html").read_text()
        # Should have stats for Duration, Avg CPU, Peak RAM
        assert "Duration" in html_content
        assert "Avg CPU" in html_content
        assert "Peak RAM" in html_content

    def test_output_dir_created_if_missing(self, tmp_path):
        """Profiler should create the output directory if it doesn't exist."""
        nested_dir = tmp_path / "deep" / "nested" / "dir"
        profiler = ResourceProfiler(
            output_dir=nested_dir,
            interval_seconds=0.05,
            session_id="test-mkdir",
        )
        profiler.start()
        time.sleep(0.1)
        html_path = profiler.stop()

        assert nested_dir.exists()
        assert html_path.exists()


class TestResourceProfilerEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_samples(self, output_dir):
        """Profiler stopped immediately should not crash."""
        profiler = ResourceProfiler(
            output_dir=output_dir,
            interval_seconds=10.0,  # Long interval so no samples taken
            session_id="test-zero",
        )
        profiler.start()
        # Stop immediately — might get 0 or 1 sample
        html_path = profiler.stop()
        assert html_path.exists()

    def test_take_sample_returns_dict(self, output_dir):
        """_take_sample() should return a dict with at least elapsed_s."""
        profiler = ResourceProfiler(
            output_dir=output_dir,
            interval_seconds=1.0,
            session_id="test-sample",
        )
        profiler._start_time = time.perf_counter()
        sample = profiler._take_sample()

        assert isinstance(sample, dict)
        assert "elapsed_s" in sample

    def test_session_id_in_outputs(self, output_dir):
        profiler = ResourceProfiler(
            output_dir=output_dir,
            interval_seconds=0.05,
            session_id="my-unique-session-123",
        )
        profiler.start()
        time.sleep(0.1)
        profiler.stop()

        json_data = json.loads((output_dir / "resource_profile.json").read_text())
        assert json_data["session_id"] == "my-unique-session-123"

        html_content = (output_dir / "resource_profile.html").read_text()
        assert "my-unique-session-123" in html_content
