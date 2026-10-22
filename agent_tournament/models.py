"""
Agent Tournament System - Core Data Models
==========================================

v5.0.6 of the Multi-Agent Financial Research Platform.

Defines the domain models used by the champion/challenger tournament:

- agent competitors
- market snapshots
- tournament tasks
- blind submissions
- evaluation metrics
- tournament results
- promotion / demotion decisions
- rollback and audit records
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


class CompetitorRole(str, Enum):
    CHAMPION = "champion"
    CHALLENGER = "challenger"


class TournamentStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DecisionType(str, Enum):
    KEEP_CHAMPION = "keep_champion"
    PROMOTE_CHALLENGER = "promote_challenger"
    DEMOTE_COMPETITOR = "demote_competitor"
    NO_DECISION = "no_decision"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class AgentCompetitor:
    """
    A versioned competitor participating in the tournament.

    A competitor can represent an agent configuration, model,
    prompt version or research policy.
    """

    competitor_id: str
    name: str
    version: str
    role: CompetitorRole

    model_name: Optional[str] = None
    prompt_version: Optional[str] = None
    policy_version: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class MarketSnapshot:
    """
    Immutable market state supplied identically to every competitor.

    This is the foundation of blind evaluation: all competitors must
    receive the same market information for a tournament task.
    """

    snapshot_id: str
    timestamp: datetime

    asset: str
    regime: str
    horizon: str

    features: Dict[str, Any]

    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TournamentTask:
    """
    A research or prediction task evaluated by the tournament.
    """

    task_id: str
    snapshot_id: str

    asset: str
    regime: str
    horizon: str

    instruction: str

    expected_output_schema: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(default_factory=utc_now)


@dataclass
class BlindSubmission:
    """
    Competitor output stored using an anonymous evaluation identifier.

    The evaluator should operate on blind_id instead of competitor_id.
    """

    submission_id: str
    tournament_id: str
    task_id: str

    blind_id: str
    competitor_id: str

    prediction: Any

    confidence: Optional[float] = None

    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    submitted_at: datetime = field(default_factory=utc_now)


@dataclass
class EvaluationMetrics:
    """
    Metrics used to compare tournament competitors.

    The roadmap explicitly extends evaluation beyond prediction quality
    to calibration, consistency, latency and cost.
    """

    prediction_score: float = 0.0
    calibration_score: float = 0.0
    consistency_score: float = 0.0

    latency_ms: float = 0.0
    cost_usd: float = 0.0

    composite_score: float = 0.0

    sample_count: int = 0

    extra_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CompetitorResult:
    """
    Aggregated result for one competitor.
    """

    competitor_id: str

    asset: str
    regime: str
    horizon: str

    metrics: EvaluationMetrics

    rank: Optional[int] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TournamentResult:
    """
    Complete result of one champion/challenger tournament.
    """

    tournament_id: str

    champion_id: str
    challenger_ids: List[str]

    results: List[CompetitorResult]

    winner_id: Optional[str] = None

    status: TournamentStatus = TournamentStatus.CREATED

    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromotionDecision:
    """
    Promotion/demotion proposal generated from tournament evidence.
    """

    decision_id: str
    tournament_id: str

    decision_type: DecisionType

    current_champion_id: str
    proposed_champion_id: Optional[str]

    reason: str

    champion_score: Optional[float] = None
    challenger_score: Optional[float] = None

    approved: bool = False

    created_at: datetime = field(default_factory=utc_now)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditRecord:
    """
    Immutable-style audit event for tournament governance.
    """

    audit_id: str

    event_type: str
    entity_type: str
    entity_id: str

    description: str

    previous_state: Optional[Dict[str, Any]] = None
    new_state: Optional[Dict[str, Any]] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(default_factory=utc_now)


@dataclass
class RollbackRecord:
    """
    Records a reversible champion change.
    """

    rollback_id: str
    decision_id: str

    previous_champion_id: str
    replacement_champion_id: str

    reason: str

    executed: bool = False

    created_at: datetime = field(default_factory=utc_now)
    executed_at: Optional[datetime] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


def generate_id(prefix: str) -> str:
    """
    Generate a readable unique identifier.

    Example:
        tournament_1d7a4c92...
    """

    return f"{prefix}_{uuid4().hex}"


__all__ = [
    "AgentCompetitor",
    "AuditRecord",
    "BlindSubmission",
    "CompetitorResult",
    "CompetitorRole",
    "DecisionType",
    "EvaluationMetrics",
    "MarketSnapshot",
    "PromotionDecision",
    "RollbackRecord",
    "TournamentResult",
    "TournamentStatus",
    "TournamentTask",
    "generate_id",
    "utc_now",
]