"""
Causal Research Engine - Core Data Models
Version: v5.0.5

Defines the core structures used by the causal research layer.

The module deliberately separates:
- predictive evidence
- causal evidence
- confounders
- leakage risks
- falsification evidence
- event/intervention evidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class HypothesisStatus(str, Enum):
    """Lifecycle state of a causal hypothesis."""

    PROPOSED = "proposed"
    TESTING = "testing"
    SUPPORTED = "supported"
    WEAK = "weak"
    FALSIFIED = "falsified"
    INCONCLUSIVE = "inconclusive"


class EvidenceType(str, Enum):
    """Supported categories of causal evidence."""

    OBSERVATIONAL = "observational"
    EVENT_STUDY = "event_study"
    INTERVENTION = "intervention"
    PLACEBO = "placebo"
    FALSIFICATION = "falsification"
    NEGATIVE_CONTROL = "negative_control"


class TestOutcome(str, Enum):
    """Generic outcome for causal diagnostic tests."""

    __test__ = False

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    INCONCLUSIVE = "inconclusive"


@dataclass
class CausalHypothesis:
    """
    Structured representation of a causal hypothesis.

    Example:
        Higher funding rates -> future BTC downside risk.
    """

    hypothesis_id: str
    cause: str
    effect: str
    description: str = ""

    expected_direction: Optional[str] = None
    mechanism: Optional[str] = None

    confounders: List[str] = field(default_factory=list)
    controls: List[str] = field(default_factory=list)

    status: HypothesisStatus = HypothesisStatus.PROPOSED

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfounderCheck:
    """Result of testing a potential confounding variable."""

    variable: str
    outcome: TestOutcome

    association_with_cause: Optional[float] = None
    association_with_effect: Optional[float] = None

    severity: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        self.severity = max(
            0.0,
            min(1.0, float(self.severity)),
        )


@dataclass
class LeakageCheck:
    """Result of a temporal or informational leakage diagnostic."""

    check_name: str
    outcome: TestOutcome

    leakage_detected: bool = False
    severity: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        self.severity = max(
            0.0,
            min(1.0, float(self.severity)),
        )


@dataclass
class CausalTestResult:
    """
    Result produced by placebo, falsification or negative-control testing.
    """

    test_name: str
    evidence_type: EvidenceType
    outcome: TestOutcome

    statistic: Optional[float] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None

    confidence: float = 0.0
    notes: str = ""

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = max(
            0.0,
            min(1.0, float(self.confidence)),
        )

        if self.p_value is not None:
            self.p_value = max(
                0.0,
                min(1.0, float(self.p_value)),
            )


@dataclass
class EventStudyResult:
    """Result of event-study style causal research."""

    event_name: str

    pre_event_mean: Optional[float] = None
    post_event_mean: Optional[float] = None

    abnormal_effect: Optional[float] = None
    cumulative_abnormal_effect: Optional[float] = None

    p_value: Optional[float] = None
    confidence: float = 0.0

    notes: str = ""

    def __post_init__(self) -> None:
        self.confidence = max(
            0.0,
            min(1.0, float(self.confidence)),
        )

        if self.p_value is not None:
            self.p_value = max(
                0.0,
                min(1.0, float(self.p_value)),
            )


@dataclass
class CausalEvidenceScore:
    """
    Final causal-evidence assessment.

    Predictive quality and causal quality are intentionally stored
    independently.
    """

    predictive_score: float
    causal_score: float

    confounder_score: float = 1.0
    leakage_score: float = 1.0
    falsification_score: float = 1.0
    event_evidence_score: float = 0.0

    confidence: float = 0.0

    explanation: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        bounded_fields = (
            "predictive_score",
            "causal_score",
            "confounder_score",
            "leakage_score",
            "falsification_score",
            "event_evidence_score",
            "confidence",
        )

        for field_name in bounded_fields:
            value = float(getattr(self, field_name))

            setattr(
                self,
                field_name,
                max(0.0, min(1.0, value)),
            )


@dataclass
class CausalResearchReport:
    """Complete result of one causal research investigation."""

    hypothesis: CausalHypothesis

    confounder_checks: List[ConfounderCheck] = field(
        default_factory=list
    )

    leakage_checks: List[LeakageCheck] = field(
        default_factory=list
    )

    causal_tests: List[CausalTestResult] = field(
        default_factory=list
    )

    event_studies: List[EventStudyResult] = field(
        default_factory=list
    )

    evidence_score: Optional[CausalEvidenceScore] = None

    warnings: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )