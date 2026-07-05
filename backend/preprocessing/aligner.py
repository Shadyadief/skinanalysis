"""Face alignment: rotate + crop so the face is upright and centered.

Design rationale
-----------------
Alignment uses the classic eye-line method: compute the angle between
the two outer eye corners relative to horizontal, rotate the whole
frame by the negative of that angle, then crop to a padded square
around the (rotated) face bounding box. This is the same approach used
by dlib's face alignment utilities and is standard in face-recognition
pipelines — cheap (one affine warp), robust to moderate roll, and
requires no extra model beyond landmarks already extracted.

Alignment corrects roll (head tilt). It intentionally does not attempt
to correct yaw/pitch (turning/nodding) — Stage 1's FaceAngleChecker
already rejects frames with excessive yaw/pitch before they reach this
module, so this module only needs to handle in-plane rotation.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import (
    DEFAULT_CONFIG,
    LEFT_EYE_OUTER_CORNER_IDX,
    RIGHT_EYE_OUTER_CORNER_IDX,
    PreprocessingConfig,
)
from .interfaces import AlignedFace, FaceAligner, FaceLandmarks


class EyeLineFaceAligner(FaceAligner):
    """Aligns a face by rotating around the eye-line and cropping square."""

    def __init__(self, config: PreprocessingConfig = DEFAULT_CONFIG) -> None:
        self._config = config

    def align(self, frame_bgr: np.ndarray, landmarks: FaceLandmarks) -> AlignedFace:
        """See FaceAligner.align."""
        left_eye = landmarks.points_px[LEFT_EYE_OUTER_CORNER_IDX]
        right_eye = landmarks.points_px[RIGHT_EYE_OUTER_CORNER_IDX]

        dx = float(right_eye[0] - left_eye[0])
        dy = float(right_eye[1] - left_eye[1])
        angle_degrees = float(np.degrees(np.arctan2(dy, dx)))

        h, w = frame_bgr.shape[:2]
        eye_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)

        rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle_degrees, scale=1.0)
        rotated = cv2.warpAffine(
            frame_bgr, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR
        )

        # Re-project all landmarks through the same rotation matrix so
        # downstream segmentation lines up with the rotated pixels.
        ones = np.ones((landmarks.points_px.shape[0], 1), dtype=np.float32)
        homogeneous_points = np.hstack([landmarks.points_px, ones])
        rotated_points = (rotation_matrix @ homogeneous_points.T).T

        x_min, y_min = rotated_points.min(axis=0)
        x_max, y_max = rotated_points.max(axis=0)
        box_w, box_h = x_max - x_min, y_max - y_min

        pad = self._config.crop_padding_ratio
        x_min -= box_w * pad
        y_min -= box_h * pad
        box_w *= 1 + 2 * pad
        box_h *= 1 + 2 * pad

        # Square crop centered on the padded box, clamped to frame bounds.
        side = max(box_w, box_h)
        cx, cy = x_min + box_w / 2.0, y_min + box_h / 2.0
        crop_x_min = int(max(cx - side / 2.0, 0))
        crop_y_min = int(max(cy - side / 2.0, 0))
        crop_x_max = int(min(cx + side / 2.0, w))
        crop_y_max = int(min(cy + side / 2.0, h))

        cropped = rotated[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        target_size = self._config.aligned_output_size
        if cropped.size == 0:
            # Degenerate crop (landmarks at frame edge) — fall back to
            # the full rotated frame rather than producing an empty image.
            cropped = rotated
            crop_x_min, crop_y_min = 0, 0
            scale_x = scale_y = 1.0
        else:
            scale_x = target_size / cropped.shape[1]
            scale_y = target_size / cropped.shape[0]

        resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        # Re-project landmarks into the crop+resize coordinate space.
        final_points_px = rotated_points.copy()
        final_points_px[:, 0] = (final_points_px[:, 0] - crop_x_min) * scale_x
        final_points_px[:, 1] = (final_points_px[:, 1] - crop_y_min) * scale_y

        final_x_min, final_y_min = final_points_px.min(axis=0)
        final_x_max, final_y_max = final_points_px.max(axis=0)
        final_bbox = (
            int(max(final_x_min, 0)),
            int(max(final_y_min, 0)),
            int(min(final_x_max, target_size) - max(final_x_min, 0)),
            int(min(final_y_max, target_size) - max(final_y_min, 0)),
        )

        aligned_landmarks = FaceLandmarks(
            points_px=final_points_px.astype(np.float32),
            points_norm=landmarks.points_norm,  # depth/normalized unaffected by 2D warp
            bbox_px=final_bbox,
        )

        return AlignedFace(
            image=resized,
            landmarks=aligned_landmarks,
            rotation_degrees=angle_degrees,
        )