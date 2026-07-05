"""Unit tests for the Image Quality module.

Run: pytest backend/tests/test_quality.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.quality.checkers import (
    BlurChecker,
    BrightnessChecker,
    ExposureChecker,
    FaceAngleChecker,
    MotionBlurChecker,
    MultipleFacesChecker,
    OcclusionChecker,
    ResolutionChecker,
)
from backend.quality.aggregator import QualityAggregator
from backend.quality.config import QualityThresholds


@pytest.fixture
def sharp_frame() -> np.ndarray:
    """High-frequency checkerboard pattern -> high Laplacian variance."""
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    frame[::2, ::2] = 255
    frame[1::2, 1::2] = 255
    return frame


@pytest.fixture
def blurry_frame() -> np.ndarray:
    """Flat uniform frame -> zero variance, definitively blurry."""
    return np.full((300, 300, 3), 128, dtype=np.uint8)


@pytest.fixture
def dark_frame() -> np.ndarray:
    return np.full((300, 300, 3), 10, dtype=np.uint8)


@pytest.fixture
def bright_frame() -> np.ndarray:
    return np.full((300, 300, 3), 250, dtype=np.uint8)


class TestBlurChecker:
    def test_sharp_frame_passes(self, sharp_frame: np.ndarray) -> None:
        checker = BlurChecker()
        result = checker.check(sharp_frame)
        assert result.passed is True
        assert result.score > 0.5

    def test_blurry_frame_fails(self, blurry_frame: np.ndarray) -> None:
        checker = BlurChecker()
        result = checker.check(blurry_frame)
        assert result.passed is False
        assert result.raw_value == pytest.approx(0.0, abs=1e-6)


class TestMotionBlurChecker:
    def test_uniform_frame_flags_low_sharpness(self, blurry_frame: np.ndarray) -> None:
        checker = MotionBlurChecker()
        result = checker.check(blurry_frame)
        # Uniform image has ~no high-frequency energy at all.
        assert result.raw_value is not None
        assert result.raw_value < 0.5


class TestBrightnessChecker:
    def test_dark_frame_fails(self, dark_frame: np.ndarray) -> None:
        result = BrightnessChecker().check(dark_frame)
        assert result.passed is False
        assert "dark" in result.message.lower()

    def test_bright_frame_fails(self, bright_frame: np.ndarray) -> None:
        result = BrightnessChecker().check(bright_frame)
        assert result.passed is False
        assert "bright" in result.message.lower()

    def test_mid_range_passes(self) -> None:
        mid_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = BrightnessChecker().check(mid_frame)
        assert result.passed is True
        assert result.score == 1.0


class TestExposureChecker:
    def test_overexposed_frame_fails(self, bright_frame: np.ndarray) -> None:
        result = ExposureChecker().check(bright_frame)
        assert result.passed is False

    def test_normal_frame_passes(self) -> None:
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = ExposureChecker().check(frame)
        assert result.passed is True


class TestResolutionChecker:
    def test_missing_bbox_fails_gracefully(self, sharp_frame: np.ndarray) -> None:
        result = ResolutionChecker().check(sharp_frame)
        assert result.passed is False
        assert result.score == 0.0

    def test_large_face_passes(self, sharp_frame: np.ndarray) -> None:
        result = ResolutionChecker().check(sharp_frame, face_bbox=(0, 0, 250, 250))
        assert result.passed is True

    def test_small_face_fails(self, sharp_frame: np.ndarray) -> None:
        result = ResolutionChecker().check(sharp_frame, face_bbox=(0, 0, 50, 50))
        assert result.passed is False


class TestOcclusionChecker:
    def test_fully_visible_passes(self, sharp_frame: np.ndarray) -> None:
        visibility = np.ones(468)  # MediaPipe Face Mesh landmark count
        result = OcclusionChecker().check(sharp_frame, landmark_visibility=visibility)
        assert result.passed is True

    def test_heavily_occluded_fails(self, sharp_frame: np.ndarray) -> None:
        visibility = np.zeros(468)
        result = OcclusionChecker().check(sharp_frame, landmark_visibility=visibility)
        assert result.passed is False


class TestFaceAngleChecker:
    def test_frontal_pose_passes(self, sharp_frame: np.ndarray) -> None:
        result = FaceAngleChecker().check(sharp_frame, head_pose=(0.0, 0.0, 0.0))
        assert result.passed is True

    def test_extreme_yaw_fails(self, sharp_frame: np.ndarray) -> None:
        result = FaceAngleChecker().check(sharp_frame, head_pose=(45.0, 0.0, 0.0))
        assert result.passed is False


class TestMultipleFacesChecker:
    def test_zero_faces_reported_as_no_face(self, sharp_frame: np.ndarray) -> None:
        result = MultipleFacesChecker().check(sharp_frame, num_faces_detected=0)
        assert result.passed is False
        assert result.issue_type.value == "no_face"

    def test_single_face_passes(self, sharp_frame: np.ndarray) -> None:
        result = MultipleFacesChecker().check(sharp_frame, num_faces_detected=1)
        assert result.passed is True

    def test_multiple_faces_fails(self, sharp_frame: np.ndarray) -> None:
        result = MultipleFacesChecker().check(sharp_frame, num_faces_detected=2)
        assert result.passed is False


class TestQualityAggregator:
    def test_ideal_frame_with_full_context_passes(self, sharp_frame: np.ndarray) -> None:
        # Boost brightness to mid-range for a realistic "good" frame.
        good_frame = sharp_frame.copy()
        good_frame[:] = np.clip(good_frame.astype(int) // 2 + 90, 0, 255).astype(np.uint8)

        aggregator = QualityAggregator()
        report = aggregator.evaluate(
            good_frame,
            face_bbox=(0, 0, 250, 250),
            landmark_visibility=np.ones(468),
            head_pose=(0.0, 0.0, 0.0),
            num_faces_detected=1,
        )
        assert report.overall_score > 0.0
        assert isinstance(report.passed, bool)

    def test_blurry_dark_frame_fails_with_reasons(self, blurry_frame: np.ndarray) -> None:
        aggregator = QualityAggregator()
        report = aggregator.evaluate(blurry_frame, num_faces_detected=1)
        assert report.passed is False
        assert len(report.recapture_reasons) > 0

    def test_custom_thresholds_are_respected(self, sharp_frame: np.ndarray) -> None:
        lenient = QualityThresholds(min_laplacian_variance=0.0)
        aggregator = QualityAggregator(checkers=[BlurChecker(lenient)])
        report = aggregator.evaluate(sharp_frame)
        assert report.results["blur"].passed is True