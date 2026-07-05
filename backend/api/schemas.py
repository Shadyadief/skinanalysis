"""Pydantic DTOs for the quality-check API surface.

Kept separate from domain dataclasses (interfaces.py, aggregator.py) so
the API contract can evolve independently of internal logic — a change
to FrameQualityReport's internals never breaks API consumers unless this
mapping is also updated (Dependency Inversion at the API boundary).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QualityCheckResultDTO(BaseModel):
    issue_type: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    message: str
    raw_value: float | None = None


class FrameQualityReportDTO(BaseModel):
    passed: bool
    overall_score: float = Field(ge=0.0, le=1.0)
    results: dict[str, QualityCheckResultDTO]
    recapture_reasons: list[str]