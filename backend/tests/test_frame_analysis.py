"""Tests for the frame analysis module (frame suggestion feature)."""

import json
import numpy as np
import cv2
import pytest
from pathlib import Path

from app.frame_analysis import (
    calculate_sharpness,
    calculate_brightness_score,
    calculate_edge_density,
    calculate_color_variance,
    calculate_basic_score,
    sample_frames_evenly,
    select_diverse_frames,
    suggest_optimal_frames,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def solid_gray_image() -> np.ndarray:
    """A uniform mid-gray 100x100 BGR image (low sharpness, mid brightness)."""
    return np.full((100, 100, 3), 127, dtype=np.uint8)


@pytest.fixture
def noisy_image() -> np.ndarray:
    """A random noise 100x100 BGR image (high sharpness, high edge density)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)


@pytest.fixture
def dark_image() -> np.ndarray:
    """A near-black 100x100 BGR image."""
    return np.full((100, 100, 3), 10, dtype=np.uint8)


@pytest.fixture
def bright_image() -> np.ndarray:
    """A near-white 100x100 BGR image."""
    return np.full((100, 100, 3), 245, dtype=np.uint8)


@pytest.fixture
def frames_dir(tmp_path: Path) -> Path:
    """Create a temp directory with 10 synthetic JPEG frames."""
    rng = np.random.default_rng(42)
    for i in range(10):
        img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"{i:05d}.jpg"), img)
    return tmp_path


@pytest.fixture
def mixed_frames_dir(tmp_path: Path) -> Path:
    """Create a directory with a mix of sharp and blurry frames."""
    for i in range(20):
        if i % 3 == 0:
            # Sharp: high-frequency noise
            rng = np.random.default_rng(i)
            img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        else:
            # Blurry: uniform gray with slight variation
            img = np.full((64, 64, 3), 120 + i, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"{i:05d}.jpg"), img)
    return tmp_path


# ---------------------------------------------------------------------------
# calculate_sharpness
# ---------------------------------------------------------------------------

class TestCalculateSharpness:
    def test_solid_image_low_sharpness(self, solid_gray_image: np.ndarray):
        score = calculate_sharpness(solid_gray_image)
        assert score == pytest.approx(0.0, abs=1.0)

    def test_noisy_image_high_sharpness(self, noisy_image: np.ndarray):
        score = calculate_sharpness(noisy_image)
        assert score > 100.0  # Random noise has very high Laplacian variance

    def test_returns_float(self, solid_gray_image: np.ndarray):
        assert isinstance(calculate_sharpness(solid_gray_image), float)


# ---------------------------------------------------------------------------
# calculate_brightness_score
# ---------------------------------------------------------------------------

class TestCalculateBrightnessScore:
    def test_mid_gray_optimal(self, solid_gray_image: np.ndarray):
        score = calculate_brightness_score(solid_gray_image)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_dark_penalized(self, dark_image: np.ndarray):
        score = calculate_brightness_score(dark_image)
        assert score < 0.15  # 10/127 = very low

    def test_bright_penalized(self, bright_image: np.ndarray):
        score = calculate_brightness_score(bright_image)
        assert score < 0.15

    def test_score_range(self, noisy_image: np.ndarray):
        score = calculate_brightness_score(noisy_image)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# calculate_edge_density
# ---------------------------------------------------------------------------

class TestCalculateEdgeDensity:
    def test_solid_no_edges(self, solid_gray_image: np.ndarray):
        score = calculate_edge_density(solid_gray_image)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_noisy_has_edges(self, noisy_image: np.ndarray):
        score = calculate_edge_density(noisy_image)
        assert score > 0.1

    def test_score_range(self, noisy_image: np.ndarray):
        score = calculate_edge_density(noisy_image)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# calculate_color_variance
# ---------------------------------------------------------------------------

class TestCalculateColorVariance:
    def test_solid_no_variance(self, solid_gray_image: np.ndarray):
        score = calculate_color_variance(solid_gray_image)
        assert score == pytest.approx(0.0, abs=1.0)

    def test_noisy_has_variance(self, noisy_image: np.ndarray):
        score = calculate_color_variance(noisy_image)
        assert score > 10.0


# ---------------------------------------------------------------------------
# calculate_basic_score
# ---------------------------------------------------------------------------

class TestCalculateBasicScore:
    def test_returns_all_keys(self, frames_dir: Path):
        result = calculate_basic_score(frames_dir / "00000.jpg")
        expected_keys = {"sharpness", "brightness", "edge_density", "color_variance", "combined"}
        assert set(result.keys()) == expected_keys

    def test_combined_in_range(self, frames_dir: Path):
        result = calculate_basic_score(frames_dir / "00000.jpg")
        assert 0.0 <= result["combined"] <= 1.0

    def test_missing_file_returns_zeros(self, tmp_path: Path):
        result = calculate_basic_score(tmp_path / "nonexistent.jpg")
        assert result["combined"] == 0.0
        assert result["sharpness"] == 0.0

    def test_all_values_are_floats(self, frames_dir: Path):
        result = calculate_basic_score(frames_dir / "00000.jpg")
        for v in result.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# sample_frames_evenly
# ---------------------------------------------------------------------------

class TestSampleFramesEvenly:
    def test_fewer_than_max(self, frames_dir: Path):
        sampled = sample_frames_evenly(frames_dir, max_samples=50)
        assert len(sampled) == 10  # Only 10 frames exist

    def test_more_than_max(self, mixed_frames_dir: Path):
        sampled = sample_frames_evenly(mixed_frames_dir, max_samples=5)
        assert len(sampled) == 5

    def test_returns_tuples(self, frames_dir: Path):
        sampled = sample_frames_evenly(frames_dir, max_samples=50)
        for idx, path in sampled:
            assert isinstance(idx, int)
            assert isinstance(path, Path)
            assert path.exists()

    def test_empty_dir(self, tmp_path: Path):
        sampled = sample_frames_evenly(tmp_path, max_samples=10)
        assert len(sampled) == 0

    def test_even_spacing(self, mixed_frames_dir: Path):
        sampled = sample_frames_evenly(mixed_frames_dir, max_samples=4)
        indices = [idx for idx, _ in sampled]
        # With 20 frames and 4 samples, step=5, indices should be [0, 5, 10, 15]
        assert indices == [0, 5, 10, 15]


# ---------------------------------------------------------------------------
# select_diverse_frames
# ---------------------------------------------------------------------------

class TestSelectDiverseFrames:
    def test_empty_input(self):
        assert select_diverse_frames([]) == []

    def test_selects_top_k(self):
        embeddings = [np.random.randn(384) for _ in range(10)]
        # Make them very different so diversity filter doesn't kick in
        for i, emb in enumerate(embeddings):
            embeddings[i] = np.zeros(384)
            embeddings[i][i * 38 : (i + 1) * 38] = 1.0

        scored = [(i, 1.0 - i * 0.05, embeddings[i]) for i in range(10)]
        result = select_diverse_frames(scored, top_k=5, min_distance=0.1)
        assert len(result) <= 5

    def test_filters_similar(self):
        # All identical embeddings => only first one selected
        emb = np.ones(384)
        scored = [(i, 1.0 - i * 0.01, emb.copy()) for i in range(5)]
        result = select_diverse_frames(scored, top_k=5, min_distance=0.3)
        assert len(result) == 1  # All others too similar

    def test_returns_index_score_tuples(self):
        emb = np.random.randn(384)
        scored = [(0, 0.9, emb)]
        result = select_diverse_frames(scored, top_k=1)
        assert len(result) == 1
        idx, score = result[0]
        assert idx == 0
        assert score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# suggest_optimal_frames (integration)
# ---------------------------------------------------------------------------

class TestSuggestOptimalFrames:
    def test_returns_list(self, frames_dir: Path):
        result = suggest_optimal_frames(frames_dir, top_k=3, use_dinov2=False)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_result_structure(self, frames_dir: Path):
        result = suggest_optimal_frames(frames_dir, top_k=3, use_dinov2=False)
        for item in result:
            assert "frame_index" in item
            assert "score" in item
            assert "sharpness" in item
            assert "brightness" in item
            assert "method" in item
            assert item["method"] == "basic"

    def test_scores_sorted_descending(self, frames_dir: Path):
        result = suggest_optimal_frames(frames_dir, top_k=5, use_dinov2=False)
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_dir(self, tmp_path: Path):
        result = suggest_optimal_frames(tmp_path, top_k=3, use_dinov2=False)
        assert result == []

    def test_top_k_limits_output(self, mixed_frames_dir: Path):
        result = suggest_optimal_frames(mixed_frames_dir, top_k=2, use_dinov2=False)
        assert len(result) == 2

    def test_frame_indices_valid(self, frames_dir: Path):
        result = suggest_optimal_frames(frames_dir, top_k=5, use_dinov2=False)
        for item in result:
            assert 0 <= item["frame_index"] < 10

    def test_dinov2_fallback_on_cpu(self, frames_dir: Path):
        """DINOv2 should gracefully fall back to basic on CPU-only machines."""
        result = suggest_optimal_frames(frames_dir, top_k=3, use_dinov2=True)
        assert isinstance(result, list)
        assert len(result) > 0
        # On CPU, should fall back to basic
        # (or if GPU available, dinov2 is fine too)
        for item in result:
            assert item["method"] in ("basic", "dinov2")
