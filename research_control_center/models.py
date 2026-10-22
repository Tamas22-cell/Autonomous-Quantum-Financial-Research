from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class ExperimentStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


class AlphaStatus(str, Enum):
    DISCOVERY = "DISCOVERY"
    VALIDATING = "VALIDATING"
    ACTIVE = "ACTIVE"
    DECAYING = "DECAYING"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


class RobustnessStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


class AllocationMode(str, Enum):
    RESEARCH = "RESEARCH"
    SHADOW = "SHADOW"


class ExperimentType(str, Enum):
    AGENT = "AGENT"
    MODEL = "MODEL"
    STRATEGY = "STRATEGY"
    PORTFOLIO = "PORTFOLIO"
    QUANTUM = "QUANTUM"
    CLASSICAL = "CLASSICAL"
    HYBRID = "HYBRID"


@dataclass
class ExperimentRecord:
    name: str
    experiment_type: ExperimentType

    asset: Optional[str] = None
    regime: Optional[str] = None
    horizon: Optional[str] = None

    status: ExperimentStatus = ExperimentStatus.CREATED

    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)

    random_seed: Optional[int] = None
    code_version: Optional[str] = None
    data_version: Optional[str] = None

    experiment_id: str = field(
        default_factory=lambda: generate_id("experiment")
    )

    created_at: datetime = field(default_factory=utc_now)
    completed_at: Optional[datetime] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LeaderboardEntry:
    participant_name: str

    participant_type: str

    asset: str
    regime: str
    horizon: str

    score: float

    prediction_quality: Optional[float] = None
    calibration_score: Optional[float] = None
    consistency_score: Optional[float] = None

    latency_ms: Optional[float] = None
    cost: Optional[float] = None

    observations: int = 0

    rank: Optional[int] = None

    entry_id: str = field(
        default_factory=lambda: generate_id("leaderboard")
    )

    created_at: datetime = field(default_factory=utc_now)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlphaRecord:
    name: str

    asset: str
    regime: Optional[str]
    horizon: str

    status: AlphaStatus = AlphaStatus.DISCOVERY

    robustness_status: RobustnessStatus = RobustnessStatus.UNKNOWN

    expected_return: Optional[float] = None
    realized_return: Optional[float] = None

    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None

    max_drawdown: Optional[float] = None

    decay_score: float = 0.0
    robustness_score: float = 0.0

    validation_score: Optional[float] = None
    out_of_sample_score: Optional[float] = None

    source_strategy_id: Optional[str] = None
    source_hypothesis_id: Optional[str] = None
    source_experiment_id: Optional[str] = None

    alpha_id: str = field(
        default_factory=lambda: generate_id("alpha")
    )

    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestDiagnostic:
    strategy_id: str

    in_sample_return: Optional[float] = None
    out_of_sample_return: Optional[float] = None

    in_sample_sharpe: Optional[float] = None
    out_of_sample_sharpe: Optional[float] = None

    in_sample_drawdown: Optional[float] = None
    out_of_sample_drawdown: Optional[float] = None

    robustness_score: Optional[float] = None

    p_value: Optional[float] = None
    confidence_interval_low: Optional[float] = None
    confidence_interval_high: Optional[float] = None

    observations: int = 0

    diagnostic_id: str = field(
        default_factory=lambda: generate_id("diagnostic")
    )

    created_at: datetime = field(default_factory=utc_now)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShadowFundSnapshot:
    total_capital: float

    allocated_capital: float = 0.0
    cash: float = 0.0

    pnl: float = 0.0
    return_pct: float = 0.0

    gross_exposure: float = 0.0
    net_exposure: float = 0.0

    max_drawdown: float = 0.0

    positions: Dict[str, float] = field(default_factory=dict)

    asset_exposure: Dict[str, float] = field(default_factory=dict)
    sector_exposure: Dict[str, float] = field(default_factory=dict)
    factor_exposure: Dict[str, float] = field(default_factory=dict)

    attribution: Dict[str, float] = field(default_factory=dict)

    mode: AllocationMode = AllocationMode.SHADOW

    snapshot_id: str = field(
        default_factory=lambda: generate_id("shadow_fund")
    )

    created_at: datetime = field(default_factory=utc_now)

    metadata: Dict[str, Any] = field(
        default_factory=lambda: {
            "research_only": True,
            "real_money_execution": False,
        }
    )


@dataclass
class QuantumClassicalExperiment:
    problem_name: str

    classical_method: str
    quantum_method: str

    asset_universe: List[str]

    classical_score: Optional[float] = None
    quantum_score: Optional[float] = None

    classical_runtime_seconds: Optional[float] = None
    quantum_runtime_seconds: Optional[float] = None

    classical_objective: Optional[float] = None
    quantum_objective: Optional[float] = None

    approximation_ratio: Optional[float] = None

    qubits: Optional[int] = None
    circuit_depth: Optional[int] = None

    repetitions: int = 1

    experiment_id: str = field(
        default_factory=lambda: generate_id("quantum_compare")
    )

    created_at: datetime = field(default_factory=utc_now)

    metadata: Dict[str, Any] = field(
        default_factory=lambda: {
            "research_only": True,
        }
    )


@dataclass
class DecisionLineageEvent:
    stage: str
    entity_type: str
    entity_id: str

    parent_entity_id: Optional[str] = None

    summary: Optional[str] = None

    evidence: Dict[str, Any] = field(default_factory=dict)

    event_id: str = field(
        default_factory=lambda: generate_id("lineage")
    )

    created_at: datetime = field(default_factory=utc_now)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionLineage:
    root_id: str

    events: List[DecisionLineageEvent] = field(default_factory=list)

    lineage_id: str = field(
        default_factory=lambda: generate_id("decision_lineage")
    )

    created_at: datetime = field(default_factory=utc_now)

    metadata: Dict[str, Any] = field(
        default_factory=lambda: {
            "auditable": True,
            "research_only": True,
            "real_money_execution": False,
        }
    )

    def add_event(
        self,
        event: DecisionLineageEvent,
    ) -> None:
        if not isinstance(event, DecisionLineageEvent):
            raise TypeError(
                "event must be a DecisionLineageEvent"
            )

        self.events.append(event)


@dataclass
class ResearchControlCenterSnapshot:
    experiments: List[ExperimentRecord] = field(default_factory=list)

    leaderboard: List[LeaderboardEntry] = field(default_factory=list)

    alpha_library: List[AlphaRecord] = field(default_factory=list)

    backtest_diagnostics: List[BacktestDiagnostic] = field(
        default_factory=list
    )

    shadow_fund: Optional[ShadowFundSnapshot] = None

    quantum_classical_experiments: List[
        QuantumClassicalExperiment
    ] = field(default_factory=list)

    decision_lineages: List[DecisionLineage] = field(
        default_factory=list
    )

    snapshot_id: str = field(
        default_factory=lambda: generate_id("control_center")
    )

    created_at: datetime = field(default_factory=utc_now)

    metadata: Dict[str, Any] = field(
        default_factory=lambda: {
            "version": "v5.0.10",
            "research_only": True,
            "real_money_execution": False,
        }
    )