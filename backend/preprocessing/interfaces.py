"""Interfaces (ports) for the Preprocessing module.

Design rationale
-----------------
Three independent responsibilities, three independent interfaces:
  - LandmarkExtractor: pixels -> 468 (x, y, z) facial landmark points.
  - FaceAligner: raw frame + landmarks -> rotation-corrected crop.
  - RegionSegmenter: aligned frame + landmarks -> named region masks/crops.

Each is swappable independently. Example: replace MediaPipe with a
custom-trained landmark model later without touching alignment or
segmentation code, as long as the new extractor still returns
FaceLandmarks with the same 468-point MediaPipe index convention (or a
documented alternative + updated region index maps in config.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceLandmarks:
    """468 facial landmark points in normalized and pixel coordinates.

    Attributes:
        points_px: shape (468, 2) array of (x, y) in pixel coordinates,
            relative to the frame the landmarks were extracted from.
        points_norm: shape (468, 3) array of (x, y, z) normalized to
            [0, 1] (z is relative depth), as returned by MediaPipe.
        bbox_px: (x, y, w, h) tight bounding box around all landmarks,
            in pixel coordinates.
    """

    points_px: np.ndarray
    points_norm: np.ndarray
    bbox_px: tuple[int, int, int, int]


@dataclass(frozen=True)
class AlignedFace:
    """Result of face alignment.

    Attributes:
        image: Rotation-corrected (and optionally cropped/resized) BGR
            image, dtype uint8.
        landmarks: Landmarks re-expressed in the aligned image's
            coordinate space (so downstream segmentation lines up).
        rotation_degrees: The rotation applied, for logging/debugging.
    """

    image: np.ndarray
    landmarks: FaceLandmarks
    rotation_degrees: float


@dataclass(frozen=True)
class RegionMask:
    """A single named facial region.

    Attributes:
        name: Region identifier, e.g. "forehead", "left_cheek".
        mask: Binary mask (same H, W as the aligned image), dtype uint8,
            255 = inside region, 0 = outside.
        bbox_px: Tight (x, y, w, h) bounding box of the mask, useful for
            cropping before feeding a region into a downstream model.
    """

    name: str
    mask: np.ndarray
    bbox_px: tuple[int, int, int, int]


class LandmarkExtractor(ABC):
    """Extracts facial landmarks from a single frame."""

    @abstractmethod
    def extract(self, frame_bgr: np.ndarray) -> FaceLandmarks | None:
        """Detect a face and return its landmarks.

        Args:
            frame_bgr: OpenCV BGR uint8 image.

        Returns:
            FaceLandmarks if exactly one face is confidently detected,
            otherwise None (caller should treat this as "no usable
            face" — Stage 1's quality gate should normally prevent this
            case from reaching here, but this module must not assume
            that and must not raise on a missing face).
        """
        raise NotImplementedError


class FaceAligner(ABC):
    """Rotates/crops a frame so the face is upright and centered."""

    @abstractmethod
    def align(self, frame_bgr: np.ndarray, landmarks: FaceLandmarks) -> AlignedFace:
        """Produce a rotation-corrected version of the frame.

        Args:
            frame_bgr: Original OpenCV BGR uint8 image.
            landmarks: Landmarks from LandmarkExtractor, in the
                original frame's coordinate space.

        Returns:
            AlignedFace with corrected image + re-projected landmarks.
        """
        raise NotImplementedError


class RegionSegmenter(ABC):
    """Splits an aligned face into named anatomical regions."""

    @abstractmethod
    def segment(self, aligned: AlignedFace) -> list[RegionMask]:
        """Produce per-region masks from an aligned face.

        Args:
            aligned: Output of FaceAligner.align().

        Returns:
            List of RegionMask, one per known region (forehead, cheeks,
            nose, chin, jaw, eyes, under-eye, ...).
        """
        raise NotImplementedError