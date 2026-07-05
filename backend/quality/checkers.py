"""Concrete implementations of QualityChecker.

Design rationale
-----------------
Classic CV (Laplacian variance, histogram stats, FFT) is used for blur,
brightness, exposure. These need zero model weights, run in <5ms on CPU,
and are proven techniques for this exact problem — no reason to pay a
model's latency/complexity cost here.

Face-geometry checks (occlusion, angle, multi-face) reuse MediaPipe
landmarks/detections passed via `context` (computed once upstream in the
preprocessing stage) instead of re-running detection per checker. This
avoids the classic mistake of running a face detector 5 times per frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import DEFAULT_THRESHOLDS, QualityThresholds
from .interfaces import QualityChecker, QualityCheckResult, QualityIssueType


class BlurChecker(QualityChecker):
    """Detects blur via variance of the Laplacian.

    A sharp image has high-frequency edges everywhere -> high variance
    after the Laplacian (2nd derivative) filter. A blurry image loses
    those edges -> low variance. This is a well-established, cheap
    proxy for sharpness (Pech-Pacheco et al., 2000).
    """

    @property
    def issue_type(self) -> QualityIssueType:
        return QualityIssueType.BLUR

    def __init__(self, thresholds: QualityThresholds = DEFAULT_THRESHOLDS) -> None:
        self._thresholds = thresholds

    def check(self, frame_bgr: np.ndarray, **context) -> QualityCheckResult:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        threshold = self._thresholds.min_laplacian_variance
        passed = variance >= threshold
        score = min(variance / threshold, 1.0) if threshold > 0 else 1.0

        return QualityCheckResult(
            issue_type=self.issue_type,
            passed=passed,
            score=round(score, 3),
            message=(
                "Image sharp."
                if passed
                else "Image too blurry. Hold camera steady and refocus."
            ),
            raw_value=variance,
        )


class MotionBlurChecker(QualityChecker):
    """Detects directional motion blur via FFT high-frequency energy.

    Ordinary out-of-focus blur is somewhat isotropic; motion blur smears
    energy along one direction, which shows up as a reduced ratio of
    high-frequency to total energy in the 2D frequency spectrum. This
    catches "moved during capture" cases that Laplacian variance alone
    can miss (a fast pan can still leave locally sharp-looking edges).
    """

    @property
    def issue_type(self) -> QualityIssueType:
        return QualityIssueType.MOTION_BLUR

    def __init__(self, thresholds: QualityThresholds = DEFAULT_THRESHOLDS) -> None:
        self._thresholds = thresholds

    def check(self, frame_bgr: np.ndarray, **context) -> QualityCheckResult:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        f_transform = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = np.abs(f_transform)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 8  # low-frequency disk radius

        y, x = np.ogrid[:h, :w]
        low_freq_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2

        total_energy = float(magnitude.sum()) + 1e-8
        high_freq_energy = float(magnitude[~low_freq_mask].sum())
        ratio = high_freq_energy / total_energy

        threshold = self._thresholds.min_motion_sharpness_ratio
        passed = ratio >= threshold
        score = min(ratio / threshold, 1.0) if threshold > 0 else 1.0

        return QualityCheckResult(
            issue_type=self.issue_type,
            passed=passed,
            score=round(score, 3),
            message=(
                "No motion blur detected."
                if passed
                else "Motion blur detected. Stay still during capture."
            ),
            raw_value=ratio,
        )


class BrightnessChecker(QualityChecker):
    """Flags frames that are too dark or too bright via mean luminance."""

    @property
    def issue_type(self) -> QualityIssueType:
        return QualityIssueType.BRIGHTNESS

    def __init__(self, thresholds: QualityThresholds = DEFAULT_THRESHOLDS) -> None:
        self._thresholds = thresholds

    def check(self, frame_bgr: np.ndarray, **context) -> QualityCheckResult:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(gray.mean())
        t = self._thresholds
        passed = t.min_brightness <= mean_brightness <= t.max_brightness

        if mean_brightness < t.min_brightness:
            message = "Too dark. Move to a brighter area."
            score = mean_brightness / t.min_brightness
        elif mean_brightness > t.max_brightness:
            message = "Too bright. Reduce direct light/glare."
            score = t.max_brightness / mean_brightness
        else:
            message = "Brightness acceptable."
            score = 1.0

        return QualityCheckResult(
            issue_type=self.issue_type,
            passed=passed,
            score=round(max(0.0, min(score, 1.0)), 3),
            message=message,
            raw_value=mean_brightness,
        )


class ExposureChecker(QualityChecker):
    """Flags clipped highlights/shadows via histogram tail analysis.

    Distinct from BrightnessChecker: an image can have acceptable mean
    brightness while still having blown-out highlights or crushed
    shadows in specific regions (e.g. window backlight). We measure the
    fraction of pixels sitting at the extreme ends of the histogram.
    """

    @property
    def issue_type(self) -> QualityIssueType:
        return QualityIssueType.EXPOSURE

    def __init__(self, thresholds: QualityThresholds = DEFAULT_THRESHOLDS) -> None:
        self._thresholds = thresholds

    def check(self, frame_bgr: np.ndarray, **context) -> QualityCheckResult:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        total_pixels = gray.size

        overexposed_ratio = float(np.sum(gray >= 250)) / total_pixels
        underexposed_ratio = float(np.sum(gray <= 5)) / total_pixels

        t = self._thresholds
        passed = (
            overexposed_ratio <= t.max_overexposed_ratio
            and underexposed_ratio <= t.max_underexposed_ratio
        )

        if overexposed_ratio > t.max_overexposed_ratio:
            message = "Overexposed regions detected. Avoid direct light/flash."
        elif underexposed_ratio > t.max_underexposed_ratio:
            message = "Underexposed regions detected. Improve lighting."
        else:
            message = "Exposure acceptable."

        worst_ratio = max(overexposed_ratio, underexposed_ratio)
        score = 1.0 - min(worst_ratio / max(t.max_overexposed_ratio, 1e-6), 1.0)

        return QualityCheckResult(
            issue_type=self.issue_type,
            passed=passed,
            score=round(max(0.0, score), 3),
            message=message,
            raw_value=worst_ratio,
        )


class ResolutionChecker(QualityChecker):
    """Ensures the detected face is large enough for downstream analysis.

    Requires `context["face_bbox"]` = (x, y, w, h) supplied by the
    upstream face-detection step. Pore/wrinkle/texture analysis needs
    enough pixels per cm^2 of skin; a tiny face crop makes every
    downstream model unreliable regardless of its own quality.
    """

    @property
    def issue_type(self) -> QualityIssueType:
        return QualityIssueType.RESOLUTION

    def __init__(self, thresholds: QualityThresholds = DEFAULT_THRESHOLDS) -> None:
        self._thresholds = thresholds

    def check(self, frame_bgr: np.ndarray, **context) -> QualityCheckResult:
        face_bbox = context.get("face_bbox")
        if face_bbox is None:
            return QualityCheckResult(
                issue_type=self.issue_type,
                passed=False,
                score=0.0,
                message="No face bounding box available for resolution check.",
                raw_value=None,
            )

        _, _, w, h = face_bbox
        min_side = min(w, h)
        threshold = self._thresholds.min_face_size_px
        passed = min_side >= threshold
        score = min(min_side / threshold, 1.0) if threshold > 0 else 1.0

        return QualityCheckResult(
            issue_type=self.issue_type,
            passed=passed,
            score=round(score, 3),
            message=(
                "Face resolution sufficient."
                if passed
                else "Face too small in frame. Move closer to camera."
            ),
            raw_value=float(min_side),
        )


class OcclusionChecker(QualityChecker):
    """Flags occluded faces via missing/low-confidence facial landmarks.

    Requires `context["landmark_visibility"]`: a 1D array of per-landmark
    visibility scores (0-1) from MediaPipe Face Mesh. Hair, masks,
    hands, or glasses glare reduce visibility on specific landmark
    clusters; a global visibility ratio is a cheap, model-free way to
    catch this without a dedicated occlusion classifier.
    """

    @property
    def issue_type(self) -> QualityIssueType:
        return QualityIssueType.OCCLUSION

    def __init__(self, thresholds: QualityThresholds = DEFAULT_THRESHOLDS) -> None:
        self._thresholds = thresholds

    def check(self, frame_bgr: np.ndarray, **context) -> QualityCheckResult:
        visibility = context.get("landmark_visibility")
        if visibility is None or len(visibility) == 0:
            return QualityCheckResult(
                issue_type=self.issue_type,
                passed=False,
                score=0.0,
                message="No landmarks available for occlusion check.",
                raw_value=None,
            )

        visible_ratio = float(np.mean(np.asarray(visibility) > 0.5))
        threshold = self._thresholds.min_landmark_visibility_ratio
        passed = visible_ratio >= threshold

        return QualityCheckResult(
            issue_type=self.issue_type,
            passed=passed,
            score=round(visible_ratio, 3),
            message=(
                "Face fully visible."
                if passed
                else "Face partially occluded. Remove mask/hair/glasses/hand."
            ),
            raw_value=visible_ratio,
        )


class FaceAngleChecker(QualityChecker):
    """Flags non-frontal poses via yaw/pitch/roll from head-pose estimation.

    Requires `context["head_pose"]` = (yaw, pitch, roll) in degrees,
    computed upstream via solvePnP against MediaPipe landmarks. Skin
    analysis (symmetry, region segmentation) assumes a near-frontal
    face; large angles distort region boundaries (e.g. cheek area
    foreshortened) and bias every downstream measurement.
    """

    @property
    def issue_type(self) -> QualityIssueType:
        return QualityIssueType.FACE_ANGLE

    def __init__(self, thresholds: QualityThresholds = DEFAULT_THRESHOLDS) -> None:
        self._thresholds = thresholds

    def check(self, frame_bgr: np.ndarray, **context) -> QualityCheckResult:
        head_pose = context.get("head_pose")
        if head_pose is None:
            return QualityCheckResult(
                issue_type=self.issue_type,
                passed=False,
                score=0.0,
                message="No head pose available for angle check.",
                raw_value=None,
            )

        yaw, pitch, roll = head_pose
        t = self._thresholds
        passed = (
            abs(yaw) <= t.max_yaw_degrees
            and abs(pitch) <= t.max_pitch_degrees
            and abs(roll) <= t.max_roll_degrees
        )

        worst_ratio = max(
            abs(yaw) / t.max_yaw_degrees,
            abs(pitch) / t.max_pitch_degrees,
            abs(roll) / t.max_roll_degrees,
        )
        score = max(0.0, 1.0 - min(worst_ratio, 1.0))

        return QualityCheckResult(
            issue_type=self.issue_type,
            passed=passed,
            score=round(score, 3),
            message=(
                "Face angle frontal enough."
                if passed
                else "Face not frontal. Look straight at the camera."
            ),
            raw_value=float(worst_ratio),
        )


class MultipleFacesChecker(QualityChecker):
    """Flags frames containing more than one detected face.

    Requires `context["num_faces_detected"]`. Multi-person frames break
    every downstream assumption (which face to align/segment/score).
    """

    @property
    def issue_type(self) -> QualityIssueType:
        return QualityIssueType.MULTIPLE_FACES

    def __init__(self, thresholds: QualityThresholds = DEFAULT_THRESHOLDS) -> None:
        self._thresholds = thresholds

    def check(self, frame_bgr: np.ndarray, **context) -> QualityCheckResult:
        num_faces = context.get("num_faces_detected")

        if num_faces is None:
            return QualityCheckResult(
                issue_type=self.issue_type,
                passed=False,
                score=0.0,
                message="Face count unavailable.",
                raw_value=None,
            )

        if num_faces == 0:
            return QualityCheckResult(
                issue_type=QualityIssueType.NO_FACE,
                passed=False,
                score=0.0,
                message="No face detected. Center your face in frame.",
                raw_value=0.0,
            )

        passed = num_faces <= self._thresholds.max_allowed_faces
        return QualityCheckResult(
            issue_type=self.issue_type,
            passed=passed,
            score=1.0 if passed else 0.0,
            message=(
                "Single face detected."
                if passed
                else f"{num_faces} faces detected. Only one person allowed in frame."
            ),
            raw_value=float(num_faces),
        )