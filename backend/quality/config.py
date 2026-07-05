"""Centralized thresholds for the Image Quality module.

Design rationale
-----------------
Magic numbers never live inside checker logic. All tunable thresholds sit
here so recalibration (e.g. after collecting real user data) never touches
business logic, and non-engineers (e.g. a QA lead) can adjust behavior by
editing one file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityThresholds:
    """Immutable threshold values used by all checkers.

    All thresholds were chosen as reasonable defaults for a
    webcam-distance frontal face capture (~40-80cm). They should be
    recalibrated against real capture data before production launch.
    """

    # Blur (Laplacian variance) — below this = too blurry
    min_laplacian_variance: float = 100.0

    # Motion blur (FFT high-frequency energy ratio)
    min_motion_sharpness_ratio: float = 0.15

    # Brightness (mean pixel intensity, 0-255)
    min_brightness: float = 60.0
    max_brightness: float = 200.0

    # Exposure (percentage of clipped pixels, 0-1)
    max_overexposed_ratio: float = 0.05
    max_underexposed_ratio: float = 0.05

    # Resolution (minimum face bounding box side, px)
    min_face_size_px: int = 200

    # Occlusion (fraction of expected landmarks that must be visible)
    min_landmark_visibility_ratio: float = 0.90

    # Face angle (degrees, absolute deviation from frontal)
    max_yaw_degrees: float = 20.0
    max_pitch_degrees: float = 20.0
    max_roll_degrees: float = 15.0

    # Multiple faces
    max_allowed_faces: int = 1

    # Overall aggregation
    min_overall_score_to_pass: float = 0.75


DEFAULT_THRESHOLDS = QualityThresholds()