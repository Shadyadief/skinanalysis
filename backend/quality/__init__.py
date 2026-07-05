"""Image Quality module.

Public API:
    QualityAggregator: runs all checks, returns FrameQualityReport.
    QualityThresholds: tunable config.
    QualityIssueType: enum of quality dimensions.
"""

from .aggregator import FrameQualityReport, QualityAggregator
from .config import DEFAULT_THRESHOLDS, QualityThresholds
from .interfaces import QualityCheckResult, QualityIssueType

__all__ = [
    "QualityAggregator",
    "FrameQualityReport",
    "QualityThresholds",
    "DEFAULT_THRESHOLDS",
    "QualityCheckResult",
    "QualityIssueType",
]