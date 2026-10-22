from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Dict, List, Optional


# ============================================================
# Version
# ============================================================

VERSION = "v6.0.3"


# ============================================================
# Helpers
# ============================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_hash(value: Any) -> str:
    return sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


# ============================================================
# Exceptions
# ============================================================


class AlphaDiscoveryError(Exception):
    """Base exception for Alpha Discovery Laboratory."""


class AlphaValidationError(AlphaDiscoveryError):
    """Raised when alpha-discovery input is invalid."""


# ============================================================
# Enums
# ============================================================


class FeatureType(str, Enum):
    PRICE = "price"
    RETURN = "return"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    LIQUIDITY = "liquidity"
    ORDER_FLOW = "order_flow"
    DERIVATIVES = "derivatives"
    ONCHAIN = "onchain"
    MACRO = "macro"
    SENTIMENT = "sentiment"
    NEWS = "news"
    REGIME = "regime"
    CROSS_ASSET = "cross_asset"
    CUSTOM = "custom"


class FeatureTransform(str, Enum):
    RAW = "raw"
    LAG = "lag"
    DIFFERENCE = "difference"
    PCT_CHANGE = "pct_change"
    LOG_RETURN = "log_return"
    ROLLING_MEAN = "rolling_mean"
    ROLLING_STD = "rolling_std"
    ZSCORE = "zscore"
    RANK = "rank"
    NORMALIZE = "normalize"
    CLIP = "clip"
    CUSTOM = "custom"


class CombinationOperator(str, Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    RATIO = "ratio"
    SPREAD = "spread"
    WEIGHTED_SUM = "weighted_sum"
    INTERACTION = "interaction"
    CUSTOM = "custom"


class StrategyDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    LONG_SHORT = "long_short"
    MARKET_NEUTRAL = "market_neutral"


class CandidateStatus(str, Enum):
    GENERATED = "generated"
    BACKTESTING = "backtesting"
    CAUSAL_VALIDATION = "causal_validation"
    OOS_VALIDATION = "oos_validation"
    ROBUSTNESS_VALIDATION = "robustness_validation"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ValidationStage(str, Enum):
    BACKTEST = "backtest"
    CAUSAL = "causal"
    OUT_OF_SAMPLE = "out_of_sample"
    ROBUSTNESS = "robustness"


class ValidationStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    REJECTED = "rejected"
    ERROR = "error"


class RejectionReason(str, Enum):
    INSUFFICIENT_DATA = "insufficient_data"
    INVALID_FEATURE = "invalid_feature"
    LOW_SIGNAL_STRENGTH = "low_signal_strength"
    NEGATIVE_EXPECTANCY = "negative_expectancy"
    EXCESSIVE_DRAWDOWN = "excessive_drawdown"
    LOW_SHARPE = "low_sharpe"
    LOW_STABILITY = "low_stability"
    OVERFIT = "overfit"
    OOS_FAILURE = "oos_failure"
    CAUSAL_FAILURE = "causal_failure"
    ROBUSTNESS_FAILURE = "robustness_failure"
    HIGH_TURNOVER = "high_turnover"
    HIGH_EXECUTION_COST = "high_execution_cost"
    REGIME_DEPENDENT = "regime_dependent"
    DUPLICATE_ALPHA = "duplicate_alpha"
    DATA_LEAKAGE = "data_leakage"
    LOOKAHEAD_BIAS = "lookahead_bias"
    OTHER = "other"


# ============================================================
# Market Feature
# ============================================================


@dataclass(frozen=True)
class MarketFeature:
    name: str
    feature_type: FeatureType
    source: str

    transform: FeatureTransform = FeatureTransform.RAW

    parameters: Dict[str, Any] = field(
        default_factory=dict
    )

    description: str = ""

    dataset_id: Optional[str] = None
    dataset_fingerprint: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "name": self.name,
                "feature_type": self.feature_type.value,
                "source": self.source,
                "transform": self.transform.value,
                "parameters": self.parameters,
                "dataset_id": self.dataset_id,
                "dataset_fingerprint": (
                    self.dataset_fingerprint
                ),
            }
        )

    @property
    def feature_id(self) -> str:
        return f"feature_{self.fingerprint[:16]}"


# ============================================================
# Feature Combination
# ============================================================


@dataclass(frozen=True)
class FeatureCombination:
    name: str

    feature_ids: List[str]

    operator: CombinationOperator

    weights: List[float] = field(
        default_factory=list
    )

    parameters: Dict[str, Any] = field(
        default_factory=dict
    )

    description: str = ""

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        if len(self.feature_ids) < 2:
            raise AlphaValidationError(
                "FeatureCombination requires "
                "at least two feature IDs."
            )

        if (
            self.weights
            and len(self.weights)
            != len(self.feature_ids)
        ):
            raise AlphaValidationError(
                "weights length must match "
                "feature_ids length."
            )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "name": self.name,
                "feature_ids": self.feature_ids,
                "operator": self.operator.value,
                "weights": self.weights,
                "parameters": self.parameters,
            }
        )

    @property
    def combination_id(self) -> str:
        return (
            f"combination_"
            f"{self.fingerprint[:16]}"
        )


# ============================================================
# Strategy Candidate
# ============================================================


@dataclass
class StrategyCandidate:
    name: str

    feature_ids: List[str]

    combination_ids: List[str] = field(
        default_factory=list
    )

    direction: StrategyDirection = (
        StrategyDirection.LONG_SHORT
    )

    entry_rule: Dict[str, Any] = field(
        default_factory=dict
    )

    exit_rule: Dict[str, Any] = field(
        default_factory=dict
    )

    position_rule: Dict[str, Any] = field(
        default_factory=dict
    )

    parameters: Dict[str, Any] = field(
        default_factory=dict
    )

    status: CandidateStatus = (
        CandidateStatus.GENERATED
    )

    hypothesis_id: Optional[str] = None

    parent_candidate_ids: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    rejection_reason: Optional[
        RejectionReason
    ] = None

    rejection_details: Optional[str] = None

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "name": self.name,
                "feature_ids": self.feature_ids,
                "combination_ids": (
                    self.combination_ids
                ),
                "direction": self.direction.value,
                "entry_rule": self.entry_rule,
                "exit_rule": self.exit_rule,
                "position_rule": (
                    self.position_rule
                ),
                "parameters": self.parameters,
                "hypothesis_id": (
                    self.hypothesis_id
                ),
                "parent_candidate_ids": (
                    self.parent_candidate_ids
                ),
            }
        )

    @property
    def candidate_id(self) -> str:
        return (
            f"alpha_"
            f"{self.fingerprint[:16]}"
        )

    @property
    def accepted(self) -> bool:
        return (
            self.status
            == CandidateStatus.ACCEPTED
        )

    @property
    def rejected(self) -> bool:
        return (
            self.status
            == CandidateStatus.REJECTED
        )

    def set_status(
        self,
        status: CandidateStatus,
    ) -> None:
        self.status = status
        self.updated_at = utc_now()

    def reject(
        self,
        reason: RejectionReason,
        details: Optional[str] = None,
    ) -> None:
        self.status = CandidateStatus.REJECTED
        self.rejection_reason = reason
        self.rejection_details = details
        self.updated_at = utc_now()

    def accept(self) -> None:
        self.status = CandidateStatus.ACCEPTED
        self.rejection_reason = None
        self.rejection_details = None
        self.updated_at = utc_now()


# ============================================================
# Performance Metrics
# ============================================================


@dataclass(frozen=True)
class PerformanceMetrics:
    total_return: float = 0.0
    annualized_return: float = 0.0

    volatility: float = 0.0

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    max_drawdown: float = 0.0

    win_rate: float = 0.0
    profit_factor: float = 0.0

    turnover: float = 0.0

    execution_cost: float = 0.0

    trade_count: int = 0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "total_return": self.total_return,
                "annualized_return": (
                    self.annualized_return
                ),
                "volatility": self.volatility,
                "sharpe_ratio": (
                    self.sharpe_ratio
                ),
                "sortino_ratio": (
                    self.sortino_ratio
                ),
                "calmar_ratio": (
                    self.calmar_ratio
                ),
                "max_drawdown": (
                    self.max_drawdown
                ),
                "win_rate": self.win_rate,
                "profit_factor": (
                    self.profit_factor
                ),
                "turnover": self.turnover,
                "execution_cost": (
                    self.execution_cost
                ),
                "trade_count": self.trade_count,
            }
        )


# ============================================================
# Validation Result
# ============================================================


@dataclass
class ValidationResult:
    candidate_id: str
    stage: ValidationStage

    status: ValidationStatus

    metrics: Optional[
        PerformanceMetrics
    ] = None

    score: Optional[float] = None

    passed_checks: List[str] = field(
        default_factory=list
    )

    failed_checks: List[str] = field(
        default_factory=list
    )

    rejection_reason: Optional[
        RejectionReason
    ] = None

    details: Dict[str, Any] = field(
        default_factory=dict
    )

    started_at: datetime = field(
        default_factory=utc_now
    )

    completed_at: Optional[datetime] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def successful(self) -> bool:
        return (
            self.status
            == ValidationStatus.PASSED
        )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "candidate_id": (
                    self.candidate_id
                ),
                "stage": self.stage.value,
                "status": self.status.value,
                "metrics": (
                    self.metrics.fingerprint
                    if self.metrics
                    else None
                ),
                "score": self.score,
                "passed_checks": (
                    self.passed_checks
                ),
                "failed_checks": (
                    self.failed_checks
                ),
                "rejection_reason": (
                    self.rejection_reason.value
                    if self.rejection_reason
                    else None
                ),
                "details": self.details,
            }
        )


# ============================================================
# Alpha Lineage
# ============================================================


@dataclass(frozen=True)
class AlphaLineageRecord:
    candidate_id: str

    feature_ids: List[str]

    combination_ids: List[str]

    dataset_ids: List[str] = field(
        default_factory=list
    )

    dataset_fingerprints: List[str] = field(
        default_factory=list
    )

    model_version_ids: List[str] = field(
        default_factory=list
    )

    prompt_version_ids: List[str] = field(
        default_factory=list
    )

    policy_version_ids: List[str] = field(
        default_factory=list
    )

    validation_fingerprints: List[str] = field(
        default_factory=list
    )

    experiment_id: Optional[str] = None
    run_id: Optional[str] = None

    parent_candidate_ids: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "candidate_id": (
                    self.candidate_id
                ),
                "feature_ids": (
                    self.feature_ids
                ),
                "combination_ids": (
                    self.combination_ids
                ),
                "dataset_ids": (
                    self.dataset_ids
                ),
                "dataset_fingerprints": (
                    self.dataset_fingerprints
                ),
                "model_version_ids": (
                    self.model_version_ids
                ),
                "prompt_version_ids": (
                    self.prompt_version_ids
                ),
                "policy_version_ids": (
                    self.policy_version_ids
                ),
                "validation_fingerprints": (
                    self.validation_fingerprints
                ),
                "experiment_id": (
                    self.experiment_id
                ),
                "run_id": self.run_id,
                "parent_candidate_ids": (
                    self.parent_candidate_ids
                ),
            }
        )


# ============================================================
# Discovery Experiment
# ============================================================


@dataclass
class AlphaDiscoveryExperiment:
    name: str

    dataset_ids: List[str]

    feature_ids: List[str] = field(
        default_factory=list
    )

    combination_ids: List[str] = field(
        default_factory=list
    )

    candidate_ids: List[str] = field(
        default_factory=list
    )

    accepted_candidate_ids: List[str] = field(
        default_factory=list
    )

    rejected_candidate_ids: List[str] = field(
        default_factory=list
    )

    configuration: Dict[str, Any] = field(
        default_factory=dict
    )

    seed: Optional[int] = None

    run_id: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    completed_at: Optional[datetime] = None

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "name": self.name,
                "dataset_ids": (
                    self.dataset_ids
                ),
                "feature_ids": (
                    self.feature_ids
                ),
                "combination_ids": (
                    self.combination_ids
                ),
                "candidate_ids": (
                    self.candidate_ids
                ),
                "configuration": (
                    self.configuration
                ),
                "seed": self.seed,
                "run_id": self.run_id,
            }
        )

    @property
    def experiment_id(self) -> str:
        return (
            f"alpha_experiment_"
            f"{self.fingerprint[:16]}"
        )


# ============================================================
# Laboratory Summary
# ============================================================


@dataclass(frozen=True)
class AlphaDiscoverySummary:
    experiment_id: str

    feature_count: int
    combination_count: int
    candidate_count: int

    accepted_count: int
    rejected_count: int

    validation_count: int

    research_only: bool = True
    real_money_execution: bool = False

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# Public exports
# ============================================================


__all__ = [
    "VERSION",
    "utc_now",
    "canonical_json",
    "stable_hash",
    "AlphaDiscoveryError",
    "AlphaValidationError",
    "FeatureType",
    "FeatureTransform",
    "CombinationOperator",
    "StrategyDirection",
    "CandidateStatus",
    "ValidationStage",
    "ValidationStatus",
    "RejectionReason",
    "MarketFeature",
    "FeatureCombination",
    "StrategyCandidate",
    "PerformanceMetrics",
    "ValidationResult",
    "AlphaLineageRecord",
    "AlphaDiscoveryExperiment",
    "AlphaDiscoverySummary",
]