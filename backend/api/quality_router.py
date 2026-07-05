"""FastAPI router for the Image Quality module.

Note on context
-----------------
Checks that need face geometry (resolution, occlusion, face_angle,
multiple_faces) require `face_bbox`, `landmark_visibility`, `head_pose`,
`num_faces_detected` — all produced by the Preprocessing stage (Stage 2,
not yet built/approved). Until that stage exists, this endpoint runs
honestly: those checkers execute and correctly report "unavailable"
(see OcclusionChecker/FaceAngleChecker/etc. None-context handling) rather
than being stubbed out or faked. Once preprocessing lands, this router
requires zero changes — it already forwards **context.
"""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from ..quality import QualityAggregator
from .schemas import FrameQualityReportDTO, QualityCheckResultDTO

router = APIRouter(prefix="/quality", tags=["quality"])

_aggregator = QualityAggregator()


def _decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into an OpenCV BGR array.

    Raises:
        HTTPException(400): if bytes are not a valid image.
    """
    buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file.")
    return frame


@router.post("/check", response_model=FrameQualityReportDTO)
async def check_frame_quality(image: UploadFile = File(...)) -> FrameQualityReportDTO:
    """Run all quality checks against a single uploaded frame.

    Args:
        image: Uploaded image file (jpeg/png), typically one frame
            extracted client-side from the 3-4s capture window.

    Returns:
        FrameQualityReportDTO with pass/fail, overall score, and
        per-dimension breakdown + recapture reasons if failed.
    """
    raw_bytes = await image.read()
    frame_bgr = _decode_image(raw_bytes)

    report = _aggregator.evaluate(frame_bgr)

    return FrameQualityReportDTO(
        passed=report.passed,
        overall_score=report.overall_score,
        results={
            key: QualityCheckResultDTO(
                issue_type=result.issue_type.value,
                passed=result.passed,
                score=result.score,
                message=result.message,
                raw_value=result.raw_value,
            )
            for key, result in report.results.items()
        },
        recapture_reasons=report.recapture_reasons,
    )