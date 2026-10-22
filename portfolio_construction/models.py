"""
Core data models for the v5.0.7 Portfolio Construction Engine.

The models in this module provide the common data structures used by:

- Mean-Variance Optimization
- Minimum-Variance Optimization
- Risk Parity
- CVaR Optimization
- Robust Portfolio Optimization
- Scenario-Aware Allocation
- Drawdown-Aware De-Risking
- Portfolio Constraints
- Portfolio Risk Analytics
- Performance and Risk Attribution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


def utc_now() -> datetime:
    """
    Return a timezone-aware UTC timestamp.
    """

    return datetime.now(timezone.utc)


def generate_id(prefix: str) -> str:
    """
    Generate a unique identifier.
    """

    if not prefix:
        raise ValueError("prefix cannot be empty.")

    return f"{prefix}_{uuid4().hex}"


# ==============================================================
# Enums
# ==============================================================


class OptimizationMethod(str, Enum):
    """
    Supported portfolio construction methods.
    """

    MEAN_VARIANCE = "mean_variance"
    MINIMUM_VARIANCE = "minimum_variance"
    RISK_PARITY = "risk_parity"
    CVAR = "cvar"
    ROBUST = "robust"
    SCENARIO_AWARE = "scenario_aware"


class ConstraintType(str, Enum):
    """
    Supported portfolio constraint categories.
    """

    MIN_WEIGHT = "min_weight"
    MAX_WEIGHT = "max_weight"
    POSITION = "position"
    SECTOR = "sector"
    ASSET_CLASS = "asset_class"
    FACTOR = "factor"
    LIQUIDITY = "liquidity"
    TURNOVER = "turnover"
    LEVERAGE = "leverage"
    CASH = "cash"


class OptimizationStatus(str, Enum):
    """
    Portfolio optimization execution status.
    """

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INFEASIBLE = "infeasible"


class RiskMeasure(str, Enum):
    """
    Supported portfolio risk measures.
    """

    VOLATILITY = "volatility"
    VAR = "var"
    CVAR = "cvar"
    MAX_DRAWDOWN = "max_drawdown"
    DOWNSIDE_DEVIATION = "downside_deviation"
    TRACKING_ERROR = "tracking_error"


class AttributionSource(str, Enum):
    """
    Sources used for portfolio attribution.
    """

    ALPHA = "alpha"
    AGENT = "agent"
    ASSET = "asset"
    SECTOR = "sector"
    ASSET_CLASS = "asset_class"
    FACTOR = "factor"
    REGIME = "regime"
    RISK = "risk"


# ==============================================================
# Asset definitions
# ==============================================================


@dataclass(frozen=True)
class AssetDefinition:
    """
    Metadata describing one investable asset.
    """

    symbol: str
    asset_class: str

    sector: Optional[str] = None
    currency: str = "USD"

    liquidity_score: Optional[float] = None

    factor_exposures: Dict[str, float] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError(
                "Asset symbol cannot be empty."
            )

        if not self.asset_class:
            raise ValueError(
                "asset_class cannot be empty."
            )

        if (
            self.liquidity_score is not None
            and not 0.0
            <= self.liquidity_score
            <= 1.0
        ):
            raise ValueError(
                "liquidity_score must be between 0 and 1."
            )


# ==============================================================
# Expected return estimates
# ==============================================================


@dataclass(frozen=True)
class ExpectedReturnEstimate:
    """
    Expected return estimate for one asset.

    uncertainty can later be used by robust optimization.
    """

    symbol: str
    expected_return: float

    uncertainty: float = 0.0

    confidence: Optional[float] = None

    source: Optional[str] = None

    agent_id: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError(
                "symbol cannot be empty."
            )

        if self.uncertainty < 0.0:
            raise ValueError(
                "uncertainty cannot be negative."
            )

        if (
            self.confidence is not None
            and not 0.0
            <= self.confidence
            <= 1.0
        ):
            raise ValueError(
                "confidence must be between 0 and 1."
            )


# ==============================================================
# Covariance input
# ==============================================================


@dataclass(frozen=True)
class CovarianceMatrix:
    """
    Covariance matrix representation.

    Matrix ordering must correspond exactly to symbols.
    """

    symbols: List[str]

    matrix: List[List[float]]

    annualized: bool = True

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError(
                "symbols cannot be empty."
            )

        if len(set(self.symbols)) != len(
            self.symbols
        ):
            raise ValueError(
                "symbols must be unique."
            )

        size = len(self.symbols)

        if len(self.matrix) != size:
            raise ValueError(
                "Covariance matrix row count must "
                "match symbol count."
            )

        for row in self.matrix:
            if len(row) != size:
                raise ValueError(
                    "Covariance matrix must be square."
                )


# ==============================================================
# Portfolio allocation
# ==============================================================


@dataclass
class PortfolioAllocation:
    """
    Portfolio allocation represented as symbol -> weight.

    Example:

        {
            "SPY": 0.40,
            "BTC": 0.20,
            "NVDA": 0.15,
            "GLD": 0.15,
            "CASH": 0.10
        }
    """

    weights: Dict[str, float]

    cash_weight: float = 0.0

    gross_exposure: Optional[float] = None

    net_exposure: Optional[float] = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def invested_weight(self) -> float:
        """
        Total weight excluding explicit cash.
        """

        return sum(
            self.weights.values()
        )

    @property
    def total_weight(self) -> float:
        """
        Total portfolio weight including cash.
        """

        return (
            self.invested_weight
            + self.cash_weight
        )

    def validate(
        self,
        tolerance: float = 1e-6,
    ) -> None:
        """
        Validate basic portfolio weight consistency.
        """

        if tolerance < 0:
            raise ValueError(
                "tolerance cannot be negative."
            )

        if not self.weights:
            raise ValueError(
                "Portfolio weights cannot be empty."
            )

        if self.cash_weight < 0.0:
            raise ValueError(
                "cash_weight cannot be negative."
            )

        for symbol, weight in self.weights.items():
            if not symbol:
                raise ValueError(
                    "Portfolio symbol cannot be empty."
                )

            if weight < 0.0:
                raise ValueError(
                    "Long-only portfolio weights "
                    "cannot be negative."
                )

        if abs(
            self.total_weight - 1.0
        ) > tolerance:
            raise ValueError(
                "Portfolio weights plus cash must "
                "sum to 1.0."
            )


# ==============================================================
# Constraints
# ==============================================================


@dataclass(frozen=True)
class PortfolioConstraint:
    """
    Generic portfolio construction constraint.

    target can represent a symbol, sector, asset class,
    factor or other constrained dimension.
    """

    constraint_type: ConstraintType

    target: str

    minimum: Optional[float] = None

    maximum: Optional[float] = None

    enabled: bool = True

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError(
                "Constraint target cannot be empty."
            )

        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError(
                "Constraint minimum cannot exceed maximum."
            )


@dataclass(frozen=True)
class TurnoverConstraint:
    """
    Maximum allowed portfolio turnover.
    """

    maximum_turnover: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.maximum_turnover <= 2.0:
            raise ValueError(
                "maximum_turnover must be between 0 and 2."
            )


@dataclass(frozen=True)
class LeverageConstraint:
    """
    Maximum gross portfolio exposure.
    """

    maximum_gross_exposure: float = 1.0

    def __post_init__(self) -> None:
        if self.maximum_gross_exposure <= 0.0:
            raise ValueError(
                "maximum_gross_exposure must be positive."
            )


# ==============================================================
# Scenario definitions
# ==============================================================


@dataclass(frozen=True)
class MarketScenario:
    """
    Portfolio stress or allocation scenario.

    asset_returns contains assumed return shocks.
    """

    scenario_id: str
    name: str

    probability: Optional[float] = None

    regime: Optional[str] = None

    asset_returns: Dict[str, float] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError(
                "scenario_id cannot be empty."
            )

        if not self.name:
            raise ValueError(
                "Scenario name cannot be empty."
            )

        if (
            self.probability is not None
            and not 0.0
            <= self.probability
            <= 1.0
        ):
            raise ValueError(
                "Scenario probability must be "
                "between 0 and 1."
            )


# ==============================================================
# Drawdown control
# ==============================================================


@dataclass(frozen=True)
class DrawdownPolicy:
    """
    Drawdown-aware portfolio de-risking policy.

    Example:

        trigger_drawdown = 0.10
        target_risk_multiplier = 0.70

    means that after a 10% drawdown the portfolio risk
    budget may be reduced to 70% of normal.
    """

    trigger_drawdown: float

    target_risk_multiplier: float

    minimum_risk_multiplier: float = 0.25

    recovery_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.trigger_drawdown <= 1.0:
            raise ValueError(
                "trigger_drawdown must be between 0 and 1."
            )

        if not 0.0 <= self.target_risk_multiplier <= 1.0:
            raise ValueError(
                "target_risk_multiplier must be "
                "between 0 and 1."
            )

        if not 0.0 <= self.minimum_risk_multiplier <= 1.0:
            raise ValueError(
                "minimum_risk_multiplier must be "
                "between 0 and 1."
            )

        if (
            self.minimum_risk_multiplier
            > self.target_risk_multiplier
        ):
            raise ValueError(
                "minimum_risk_multiplier cannot exceed "
                "target_risk_multiplier."
            )

        if (
            self.recovery_threshold is not None
            and not 0.0
            <= self.recovery_threshold
            <= 1.0
        ):
            raise ValueError(
                "recovery_threshold must be between 0 and 1."
            )


# ==============================================================
# Risk metrics
# ==============================================================


@dataclass
class PortfolioRiskMetrics:
    """
    Portfolio-level risk measurements.
    """

    expected_return: Optional[float] = None

    volatility: Optional[float] = None

    sharpe_ratio: Optional[float] = None

    var: Optional[float] = None

    cvar: Optional[float] = None

    max_drawdown: Optional[float] = None

    downside_deviation: Optional[float] = None

    tracking_error: Optional[float] = None

    turnover: Optional[float] = None

    gross_exposure: Optional[float] = None

    net_exposure: Optional[float] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ==============================================================
# Attribution
# ==============================================================


@dataclass(frozen=True)
class AttributionRecord:
    """
    Contribution of one source to portfolio performance or risk.
    """

    source_type: AttributionSource

    source_id: str

    return_contribution: float = 0.0

    risk_contribution: float = 0.0

    weight_contribution: float = 0.0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError(
                "source_id cannot be empty."
            )


# ==============================================================
# Optimization request
# ==============================================================


@dataclass
class OptimizationRequest:
    """
    Complete request passed to a portfolio optimizer.
    """

    request_id: str

    method: OptimizationMethod

    assets: List[AssetDefinition]

    expected_returns: List[
        ExpectedReturnEstimate
    ]

    covariance: CovarianceMatrix

    constraints: List[
        PortfolioConstraint
    ] = field(
        default_factory=list
    )

    scenarios: List[
        MarketScenario
    ] = field(
        default_factory=list
    )

    current_allocation: Optional[
        PortfolioAllocation
    ] = None

    turnover_constraint: Optional[
        TurnoverConstraint
    ] = None

    leverage_constraint: Optional[
        LeverageConstraint
    ] = None

    drawdown_policy: Optional[
        DrawdownPolicy
    ] = None

    risk_free_rate: float = 0.0

    target_return: Optional[float] = None

    target_volatility: Optional[float] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError(
                "request_id cannot be empty."
            )

        if not self.assets:
            raise ValueError(
                "Optimization request requires assets."
            )

        asset_symbols = [
            asset.symbol
            for asset in self.assets
        ]

        if len(set(asset_symbols)) != len(
            asset_symbols
        ):
            raise ValueError(
                "Asset symbols must be unique."
            )

        covariance_symbols = (
            self.covariance.symbols
        )

        if asset_symbols != covariance_symbols:
            raise ValueError(
                "Asset ordering must exactly match "
                "covariance symbol ordering."
            )

        expected_return_symbols = {
            estimate.symbol
            for estimate
            in self.expected_returns
        }

        missing_returns = (
            set(asset_symbols)
            - expected_return_symbols
        )

        if missing_returns:
            raise ValueError(
                "Missing expected return estimates for: "
                + ", ".join(
                    sorted(missing_returns)
                )
            )


# ==============================================================
# Optimization result
# ==============================================================


@dataclass
class OptimizationResult:
    """
    Standard output from every portfolio optimizer.
    """

    result_id: str

    request_id: str

    method: OptimizationMethod

    status: OptimizationStatus

    allocation: Optional[
        PortfolioAllocation
    ] = None

    risk_metrics: Optional[
        PortfolioRiskMetrics
    ] = None

    attribution: List[
        AttributionRecord
    ] = field(
        default_factory=list
    )

    objective_value: Optional[float] = None

    solver_name: Optional[str] = None

    iterations: Optional[int] = None

    message: Optional[str] = None

    diagnostics: Dict[str, Any] = field(
        default_factory=dict
    )

    started_at: Optional[
        datetime
    ] = None

    completed_at: Optional[
        datetime
    ] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


__all__ = [
    "AssetDefinition",
    "AttributionRecord",
    "AttributionSource",
    "ConstraintType",
    "CovarianceMatrix",
    "DrawdownPolicy",
    "ExpectedReturnEstimate",
    "LeverageConstraint",
    "MarketScenario",
    "OptimizationMethod",
    "OptimizationRequest",
    "OptimizationResult",
    "OptimizationStatus",
    "PortfolioAllocation",
    "PortfolioConstraint",
    "PortfolioRiskMetrics",
    "RiskMeasure",
    "TurnoverConstraint",
    "generate_id",
    "utc_now",
]