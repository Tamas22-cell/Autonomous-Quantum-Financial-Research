"""
v5.0.9 - Autonomous Research Loop
Core domain models.

The models in this module represent one autonomous research cycle:

research task
    -> hypothesis
    -> strategy
    -> validation
    -> reject / promote
    -> shadow allocation
    -> observed outcome
    -> learning feedback

The module intentionally contains no execution logic.
It defines stable, auditable data contracts for the rest of v5.0.9.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def generate_id(prefix: str) -> str:
    """Generate a readable unique identifier."""
    return f"{prefix}_{uuid4().hex}"


class ResearchPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ResearchTrigger(str, Enum):
    MANUAL = "MANUAL"
    UNCERTAINTY = "UNCERTAINTY"
    REGIME_CHANGE = "REGIME_CHANGE"
    PERFORMANCE_GAP = "PERFORMANCE_GAP"
    ALPHA_DECAY = "ALPHA_DECAY"
    AGENT_SIGNAL = "AGENT_SIGNAL"
    RISK_EVENT = "RISK_EVENT"


class ResearchStatus(str, Enum):
    CREATED = "CREATED"
    PRIORITIZED = "PRIORITIZED"
    HYPOTHESIS_GENERATED = "HYPOTHESIS_GENERATED"
    STRATEGY_GENERATED = "STRATEGY_GENERATED"
    VALIDATING = "VALIDATING"
    REJECTED = "REJECTED"
    PROMOTED = "PROMOTED"
    SHADOW = "SHADOW"
    OBSERVING = "OBSERVING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ValidationDecision(str, Enum):
    REJECT = "REJECT"
    PROMOTE = "PROMOTE"
    REVIEW = "REVIEW"


class RiskDecision(str, Enum):
    APPROVE = "APPROVE"
    REDUCE = "REDUCE"
    VETO = "VETO"
    SHUTDOWN = "SHUTDOWN"


@dataclass
class ResearchTask:
    """
    A research problem selected by the Research Orchestrator.
    """

    title: str
    description: str
    trigger: ResearchTrigger

    asset: Optional[str] = None
    regime: Optional[str] = None
    horizon: Optional[str] = None

    priority: ResearchPriority = ResearchPriority.MEDIUM
    status: ResearchStatus = ResearchStatus.CREATED

    uncertainty_score: float = 0.0
    performance_gap_score: float = 0.0

    task_id: str = field(
        default_factory=lambda: generate_id("research")
    )

    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_status(self, status: ResearchStatus) -> None:
        self.status = status
        self.updated_at = utc_now()


@dataclass
class Hypothesis:
    """
    Testable market hypothesis produced from a research task.
    """

    task_id: str
    statement: str

    asset: Optional[str] = None
    regime: Optional[str] = None
    horizon: Optional[str] = None

    confidence: float = 0.0

    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    hypothesis_id: str = field(
        default_factory=lambda: generate_id("hypothesis")
    )

    created_at: datetime = field(default_factory=utc_now)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyCandidate:
    """
    Strategy generated from a validated research hypothesis.
    """

    hypothesis_id: str
    name: str

    entry_rule: str
    exit_rule: str

    asset: Optional[str] = None
    horizon: Optional[str] = None

    position_size: float = 0.0
    stop_condition: Optional[str] = None

    strategy_id: str = field(
        default_factory=lambda: generate_id("strategy")
    )

    created_at: datetime = field(default_factory=utc_now)

    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """
    Result returned by the institutional validation layer.
    """

    strategy_id: str
    decision: ValidationDecision

    score: float = 0.0

    out_of_sample_score: Optional[float] = None
    robustness_score: Optional[float] = None
    causal_score: Optional[float] = None

    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None

    reasons: List[str] = field(default_factory=list)

    validation_id: str = field(
        default_factory=lambda: generate_id("validation")
    )

    created_at: datetime = field(default_factory=utc_now)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAssessment:
    """
    Independent Risk Governor decision.

    Risk remains a hard gate and can veto, reduce or shut down
    a candidate independently of research confidence.
    """

    strategy_id: str
    decision: RiskDecision

    risk_score: float = 0.0

    approved_position_size: Optional[float] = None

    reasons: List[str] = field(default_factory=list)

    assessment_id: str = field(
        default_factory=lambda: generate_id("risk")
    )

    created_at: datetime = field(default_factory=utc_now)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShadowAllocation:
    """
    Virtual allocation used before any higher-trust state.

    This represents research/shadow capital only.
    It does not represent automatic real-money execution.
    """

    strategy_id: str
    asset: str

    weight: float
    notional_value: float

    entry_price: Optional[float] = None

    allocation_id: str = field(
        default_factory=lambda: generate_id("shadow")
    )

    created_at: datetime = field(default_factory=utc_now)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchOutcome:
    """
    Observed result of a promoted/shadow research candidate.
    """

    strategy_id: str

    pnl: float = 0.0
    return_pct: float = 0.0

    benchmark_return_pct: Optional[float] = None
    alpha: Optional[float] = None

    hit: Optional[bool] = None

    observation_start: datetime = field(default_factory=utc_now)
    observation_end: Optional[datetime] = None

    outcome_id: str = field(
        default_factory=lambda: generate_id("outcome")
    )

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningFeedback:
    """
    Feedback produced from observed outcomes.

    This can later feed:
    - Meta-Learning
    - Alpha Lifecycle
    - Agent Tournament
    - new Research Tasks
    """

    strategy_id: str
    outcome_id: str

    performance_score: float

    lessons: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)

    create_new_research_task: bool = False

    feedback_id: str = field(
        default_factory=lambda: generate_id("feedback")
    )

    created_at: datetime = field(default_factory=utc_now)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchCycle:
    """
    Complete auditable state of one autonomous research cycle.
    """

    task: ResearchTask

    hypothesis: Optional[Hypothesis] = None
    strategy: Optional[StrategyCandidate] = None

    validation: Optional[ValidationResult] = None
    risk_assessment: Optional[RiskAssessment] = None

    shadow_allocation: Optional[ShadowAllocation] = None
    outcome: Optional[ResearchOutcome] = None
    feedback: Optional[LearningFeedback] = None

    cycle_id: str = field(
        default_factory=lambda: generate_id("cycle")
    )

    started_at: datetime = field(default_factory=utc_now)
    completed_at: Optional[datetime] = None

    audit_log: List[Dict[str, Any]] = field(default_factory=list)

    def add_audit_event(
        self,
        event: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an immutable-style event record to the cycle audit trail.
        """

        self.audit_log.append(
            {
                "timestamp": utc_now(),
                "event": event,
                "details": details or {},
            }
        )

    def complete(self) -> None:
        """
        Mark the research cycle as completed.
        """

        self.task.update_status(ResearchStatus.COMPLETED)
        self.completed_at = utc_now()

        self.add_audit_event(
            "RESEARCH_CYCLE_COMPLETED",
            {
                "cycle_id": self.cycle_id,
                "task_id": self.task.task_id,
            },
        )