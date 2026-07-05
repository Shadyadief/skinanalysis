"""MediaPipe-based implementation of LandmarkExtractor.

Uses the modern MediaPipe Tasks API (`FaceLandmarker`), not the legacy
`mp.solutions.face_mesh` (deprecated by Google, and unavailable in some
recent MediaPipe distributions). Requires the FaceLandmarker model
bundle (~3MB) downloaded separately — see PreprocessingConfig docstring
for the URL. This class raises a clear error at construction time if
the model file is missing, rather than failing confusingly later.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .config import DEFAULT_CONFIG, PreprocessingConfig
from .interfaces import FaceLandmarks, LandmarkExtractor


class MediaPipeLandmarkExtractor(LandmarkExtractor):
    """Extracts 468 facial landmarks using MediaPipe FaceLandmarker."""

    def __init__(self, config: PreprocessingConfig = DEFAULT_CONFIG) -> None:
        """Args:
            config: Preprocessing settings, including model path and
                confidence thresholds.

        Raises:
            FileNotFoundError: if config.model_asset_path does not exist.
                Fail fast and loud — a silently-None landmarker would
                make every downstream extract() call fail mysteriously.
        """
        if not os.path.exists(config.model_asset_path):
            raise FileNotFoundError(
                f"FaceLandmarker model not found at '{config.model_asset_path}'. "
                "Download it from https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/latest/face_landmarker.task "
                "and place it at that path (or update config.model_asset_path)."
            )

        # Imported lazily so the rest of the package can be imported/tested
        # without mediapipe's tasks submodule being importable in every env.
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base_options = mp_python.BaseOptions(model_asset_path=config.model_asset_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=config.min_face_detection_confidence,
            min_face_presence_confidence=config.min_face_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._vision = vision

    def extract(self, frame_bgr: np.ndarray) -> FaceLandmarks | None:
        """See LandmarkExtractor.extract."""
        import mediapipe as mp

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        raw_landmarks = result.face_landmarks[0]  # num_faces=1

        points_norm = np.array(
            [[lm.x, lm.y, lm.z] for lm in raw_landmarks], dtype=np.float32
        )
        points_px = np.stack(
            [points_norm[:, 0] * w, points_norm[:, 1] * h], axis=1
        ).astype(np.float32)

        x_min, y_min = points_px.min(axis=0)
        x_max, y_max = points_px.max(axis=0)
        bbox_px = (
            int(max(x_min, 0)),
            int(max(y_min, 0)),
            int(min(x_max, w) - max(x_min, 0)),
            int(min(y_max, h) - max(y_min, 0)),
        )

        return FaceLandmarks(
            points_px=points_px,
            points_norm=points_norm,
            bbox_px=bbox_px,
        )

    def close(self) -> None:
        """Release the underlying MediaPipe landmarker resources."""
        self._landmarker.close()

    def __enter__(self) -> "MediaPipeLandmarkExtractor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()