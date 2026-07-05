"""Interfaces (ports) for the Image Quality module.

Design rationale
-----------------
Every concrete quality check (blur, brightness, occlusion, ...) implements
`QualityChecker`. This means:
  - New checks can be added without touching existing code (Open/Closed).
  - Any check can be swapped for a different implementation (e.g. replace
    Laplacian blur detector with a learned model) without breaking the
    aggregator or the pipeline (Liskov substitution).
  - The aggregator depends only on this abstraction, never on concrete
    classes (Dependency Inversion).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np


class QualityIssueType(str, Enum):
    """Enumerates every quality dimension the system can flag."""

    BLUR = "blur"
    MOTION_BLUR = "motion_blur"
    BRIGHTNESS = "brightness"
    EXPOSURE = "exposure"
    RESOLUTION = "resolution"
    OCCLUSION = "occlusion"
    FACE_ANGLE = "face_angle"
    MULTIPLE_FACES = "multiple_faces"
    NO_FACE = "no_face"
    FILTER_DETECTED = "filter_detected"


@dataclass(frozen=True)
class QualityCheckResult:
    """Result of a single quality check.

    Attributes:
        issue_type: Which dimension this result belongs to.
        passed: Whether the frame passes this specific check.
        score: Normalized score in [0.0, 1.0], 1.0 = perfect.
        message: Human-readable explanation (used in recapture prompts).
        raw_value: Optional raw measurement (e.g. Laplacian variance),
            kept for debugging/logging/analytics.
    """

    issue_type: QualityIssueType
    passed: bool
    score: float
    message: str
    raw_value: float | None = None


class QualityChecker(ABC):
    """Base contract for any single-responsibility quality check.

    Each implementation must be stateless and side-effect free so checks
    can run in parallel and be unit-tested in isolation.
    """

    @property
    @abstractmethod
    def issue_type(self) -> QualityIssueType:
        """The QualityIssueType this checker is responsible for."""
        raise NotImplementedError

    @abstractmethod
    def check(self, frame_bgr: np.ndarray, **context) -> QualityCheckResult:
        """Run the check on a single BGR frame.

        Args:
            frame_bgr: Image as a numpy array in OpenCV BGR format,
                shape (H, W, 3), dtype uint8.
            **context: Optional shared context computed upstream (e.g.
                face landmarks, bounding boxes) so expensive detectors
                (like face mesh) run once and get reused across checkers.

        Returns:
            QualityCheckResult for this dimension.
        """
        raise NotImplementedError