"""Aggregates all QualityChecker results into one verdict.

Design rationale
-----------------
The aggregator is the only class that knows about "all checks together".
Individual checkers know nothing about each other (loose coupling). To
add a new check: write a QualityChecker, add it to the list passed into
QualityAggregator — no other code changes. This is the Composite pattern
applied to independent validators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .checkers import (
    BlurChecker,
    BrightnessChecker,
    ExposureChecker,
    FaceAngleChecker,
    MotionBlurChecker,
    MultipleFacesChecker,
    OcclusionChecker,
    ResolutionChecker,
)
from .config import DEFAULT_THRESHOLDS, QualityThresholds
from .interfaces import QualityChecker, QualityCheckResult


@dataclass
class FrameQualityReport:
    """Full quality verdict for a single frame.

    Attributes:
        passed: True only if every individual check passed.
        overall_score: Mean of all individual check scores, in [0, 1].
        results: Per-dimension results, keyed by issue_type value.
        recapture_reasons: Human-readable messages for failed checks
            only — this is what gets shown to the user.
    """

    passed: bool
    overall_score: float
    results: dict[str, QualityCheckResult] = field(default_factory=dict)
    recapture_reasons: list[str] = field(default_factory=list)


class QualityAggregator:
    """Runs a configurable list of QualityChecker instances and combines results."""

    def __init__(
        self,
        checkers: list[QualityChecker] | None = None,
        thresholds: QualityThresholds = DEFAULT_THRESHOLDS,
    ) -> None:
        """Args:
        checkers: Custom checker list. Defaults to the standard set
            covering every quality dimension. Pass a subset/superset to
            customize behavior without touching this class's internals.
        thresholds: Shared threshold config used to build default checkers.
        """
        self._thresholds = thresholds
        self._checkers: list[QualityChecker] = checkers or self._build_default_checkers()

    def _build_default_checkers(self) -> list[QualityChecker]:
        t = self._thresholds
        return [
            MultipleFacesChecker(t),
            BlurChecker(t),
            MotionBlurChecker(t),
            BrightnessChecker(t),
            ExposureChecker(t),
            ResolutionChecker(t),
            OcclusionChecker(t),
            FaceAngleChecker(t),
        ]

    def evaluate(self, frame_bgr: np.ndarray, **context) -> FrameQualityReport:
        """Run every checker against a frame and produce a combined verdict.

        Args:
            frame_bgr: Frame to evaluate, OpenCV BGR uint8 array.
            **context: Shared upstream data (face_bbox, landmark_visibility,
                head_pose, num_faces_detected) reused across checkers to
                avoid redundant face-detection/landmark computation.

        Returns:
            FrameQualityReport summarizing pass/fail + per-dimension detail.
        """
        results: dict[str, QualityCheckResult] = {}
        recapture_reasons: list[str] = []

        for checker in self._checkers:
            result = checker.check(frame_bgr, **context)
            results[result.issue_type.value] = result
            if not result.passed:
                recapture_reasons.append(result.message)

        overall_score = (
            float(np.mean([r.score for r in results.values()])) if results else 0.0
        )
        passed = (
            all(r.passed for r in results.values())
            and overall_score >= self._thresholds.min_overall_score_to_pass
        )

        return FrameQualityReport(
            passed=passed,
            overall_score=round(overall_score, 3),
            results=results,
            recapture_reasons=recapture_reasons,
        )