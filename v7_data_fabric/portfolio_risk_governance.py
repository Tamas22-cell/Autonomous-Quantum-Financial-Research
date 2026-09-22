from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


class PortfolioRiskGovernanceError(ValueError):
    """Portfolio intelligence and risk governance error."""


class SignalAction(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    AVOID = "AVOID"


class RiskSeverity(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RecommendationStatus(str, Enum):
    PROPOSED = "PROPOSED"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"
    APPROVED_FOR_RESEARCH = "APPROVED_FOR_RESEARCH"
    REJECTED = "REJECTED"


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PortfolioRiskGovernanceError(
            f"{name} must be a non-empty string."
        )


def _require_probability(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise PortfolioRiskGovernanceError(
            f"{name} must be between 0 and 1."
        )


def _require_non_negative(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0.0
    ):
        raise PortfolioRiskGovernanceError(
            f"{name} must be a finite non-negative number."
        )


@dataclass(frozen=True)
class PortfolioResearchSignal:
    signal_id: str
    asset_id: str
    action: SignalAction
    confidence: float
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    rationale: str = ""
    evidence_fingerprint: str = ""
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _require_text(self.signal_id, "signal_id")
        _require_text(self.asset_id, "asset_id")

        if not isinstance(self.action, SignalAction):
            raise PortfolioRiskGovernanceError(
                "action must be a SignalAction."
            )

        _require_probability(
            self.confidence,
            "confidence",
        )

        if (
            isinstance(self.expected_return, bool)
            or not isinstance(
                self.expected_return,
                (int, float),
            )
            or not isfinite(self.expected_return)
        ):
            raise PortfolioRiskGovernanceError(
                "expected_return must be finite."
            )

        _require_non_negative(
            self.expected_volatility,
            "expected_volatility",
        )

        if not isinstance(self.rationale, str):
            raise PortfolioRiskGovernanceError(
                "rationale must be a string."
            )

        if not isinstance(
            self.evidence_fingerprint,
            str,
        ):
            raise PortfolioRiskGovernanceError(
                "evidence_fingerprint must be a string."
            )

        if not isinstance(self.metadata, dict):
            raise PortfolioRiskGovernanceError(
                "metadata must be a dictionary."
            )


@dataclass(frozen=True)
class RiskLimits:
    max_position_weight: float = 0.25
    max_asset_concentration: float = 0.40
    max_portfolio_drawdown: float = 0.20
    minimum_cash_weight: float = 0.05
    require_human_approval: bool = True

    def __post_init__(self) -> None:
        _require_probability(
            self.max_position_weight,
            "max_position_weight",
        )
        _require_probability(
            self.max_asset_concentration,
            "max_asset_concentration",
        )
        _require_probability(
            self.max_portfolio_drawdown,
            "max_portfolio_drawdown",
        )
        _require_probability(
            self.minimum_cash_weight,
            "minimum_cash_weight",
        )

        if (
            self.max_position_weight
            > self.max_asset_concentration
        ):
            raise PortfolioRiskGovernanceError(
                "max_position_weight cannot exceed "
                "max_asset_concentration."
            )

        if not isinstance(
            self.require_human_approval,
            bool,
        ):
            raise PortfolioRiskGovernanceError(
                "require_human_approval must be boolean."
            )


@dataclass(frozen=True)
class RiskAssessment:
    severity: RiskSeverity
    violations: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    drawdown_breach: bool = False
    concentration_breach: bool = False
    cash_breach: bool = False

    @property
    def is_clear(self) -> bool:
        return (
            self.severity == RiskSeverity.NONE
            and not self.violations
        )


@dataclass(frozen=True)
class PortfolioRecommendation:
    recommendation_id: str
    weights: Dict[str, float]
    cash_weight: float
    signals: Tuple[str, ...]
    rationale: str
    risk_assessment: RiskAssessment
    status: RecommendationStatus = (
        RecommendationStatus.PROPOSED
    )
    research_only: bool = True
    real_money_execution: bool = False
    human_approval_required: bool = True

    def __post_init__(self) -> None:
        _require_text(
            self.recommendation_id,
            "recommendation_id",
        )

        if not isinstance(self.weights, dict):
            raise PortfolioRiskGovernanceError(
                "weights must be a dictionary."
            )

        total = 0.0

        for asset_id, weight in self.weights.items():
            _require_text(asset_id, "asset_id")
            _require_probability(weight, "weight")
            total += weight

        _require_probability(
            self.cash_weight,
            "cash_weight",
        )
        total += self.cash_weight

        if abs(total - 1.0) > 1e-6:
            raise PortfolioRiskGovernanceError(
                "Portfolio weights must sum to 1."
            )

        if not isinstance(self.signals, tuple):
            raise PortfolioRiskGovernanceError(
                "signals must be a tuple."
            )

        if not isinstance(self.rationale, str):
            raise PortfolioRiskGovernanceError(
                "rationale must be a string."
            )

        if not isinstance(
            self.risk_assessment,
            RiskAssessment,
        ):
            raise PortfolioRiskGovernanceError(
                "risk_assessment must be a RiskAssessment."
            )

        if not isinstance(
            self.status,
            RecommendationStatus,
        ):
            raise PortfolioRiskGovernanceError(
                "status must be a RecommendationStatus."
            )

        if self.research_only is not True:
            raise PortfolioRiskGovernanceError(
                "Portfolio recommendations must be "
                "research-only."
            )

        if self.real_money_execution is not False:
            raise PortfolioRiskGovernanceError(
                "Real-money execution is not supported."
            )

        if not isinstance(
            self.human_approval_required,
            bool,
        ):
            raise PortfolioRiskGovernanceError(
                "human_approval_required must be boolean."
            )


class PortfolioRiskGovernance:
    """
    v7.0.8 portfolio intelligence and risk governance.

    The component translates validated research signals into
    research-only portfolio recommendations. It never places
    orders, allocates real capital or removes human approval.
    """

    VERSION = "v7.0.8"
    RESEARCH_ONLY = True
    REAL_MONEY_EXECUTION = False
    AUTOMATIC_CAPITAL_ALLOCATION = False

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
    ) -> None:
        self.limits = limits or RiskLimits()
        self._signals: Dict[
            str,
            PortfolioResearchSignal,
        ] = {}
        self._recommendations: Dict[
            str,
            PortfolioRecommendation,
        ] = {}

    @property
    def signal_count(self) -> int:
        return len(self._signals)

    @property
    def recommendation_count(self) -> int:
        return len(self._recommendations)

    def register_signal(
        self,
        signal: PortfolioResearchSignal,
    ) -> PortfolioResearchSignal:
        if not isinstance(
            signal,
            PortfolioResearchSignal,
        ):
            raise PortfolioRiskGovernanceError(
                "signal must be a PortfolioResearchSignal."
            )

        if signal.signal_id in self._signals:
            raise PortfolioRiskGovernanceError(
                "Duplicate signal ID: "
                f"{signal.signal_id}"
            )

        self._signals[signal.signal_id] = signal
        return signal

    def translate_signals(
        self,
        signals: Iterable[PortfolioResearchSignal],
    ) -> Tuple[PortfolioResearchSignal, ...]:
        if isinstance(signals, (str, bytes)):
            raise PortfolioRiskGovernanceError(
                "signals must be an iterable of signals."
            )

        try:
            values = tuple(signals)
        except TypeError as exc:
            raise PortfolioRiskGovernanceError(
                "signals must be iterable."
            ) from exc

        for signal in values:
            self.register_signal(signal)

        return values

    def build_recommendation(
        self,
        recommendation_id: str,
        signals: Iterable[PortfolioResearchSignal],
        *,
        cash_weight: Optional[float] = None,
        current_drawdown: float = 0.0,
    ) -> PortfolioRecommendation:
        _require_text(
            recommendation_id,
            "recommendation_id",
        )

        values = tuple(signals)

        if not values:
            raise PortfolioRiskGovernanceError(
                "At least one signal is required."
            )

        for signal in values:
            if not isinstance(
                signal,
                PortfolioResearchSignal,
            ):
                raise PortfolioRiskGovernanceError(
                    "signals must contain "
                    "PortfolioResearchSignal values."
                )

        if cash_weight is None:
            cash_weight = self.limits.minimum_cash_weight

        _require_probability(
            cash_weight,
            "cash_weight",
        )

        _require_probability(
            current_drawdown,
            "current_drawdown",
        )

        if current_drawdown > 0.0:
            drawdown_fraction = current_drawdown
        else:
            drawdown_fraction = 0.0

        weights: Dict[str, float] = {}

        positive_signals = [
            signal
            for signal in values
            if signal.action == SignalAction.BUY
        ]

        if positive_signals:
            raw_scores = {
                signal.asset_id: max(
                    signal.confidence,
                    0.0,
                )
                for signal in positive_signals
            }

            score_total = sum(raw_scores.values())

            if score_total <= 0.0:
                raise PortfolioRiskGovernanceError(
                    "Positive signals must have "
                    "non-zero confidence."
                )

            investable_weight = 1.0 - cash_weight

            for asset_id, score in raw_scores.items():
                proposed = (
                    investable_weight
                    * score
                    / score_total
                )
                weights[asset_id] = min(
                    proposed,
                    self.limits.max_position_weight,
                )

            allocated = sum(weights.values())
            remainder = investable_weight - allocated

            if remainder > 1e-9:
                weights["UNALLOCATED_RESEARCH"] = (
                    remainder
                )
        else:
            weights["UNALLOCATED_RESEARCH"] = (
                1.0 - cash_weight
            )

        assessment = self.assess_risk(
            weights=weights,
            cash_weight=cash_weight,
            current_drawdown=drawdown_fraction,
        )

        status = (
            RecommendationStatus.REQUIRES_APPROVAL
            if self.limits.require_human_approval
            or not assessment.is_clear
            else RecommendationStatus
            .APPROVED_FOR_RESEARCH
        )

        recommendation = PortfolioRecommendation(
            recommendation_id=recommendation_id,
            weights=weights,
            cash_weight=cash_weight,
            signals=tuple(
                signal.signal_id
                for signal in values
            ),
            rationale=(
                "Research-only translation of validated "
                "signals. Human approval remains required "
                "before any portfolio action."
            ),
            risk_assessment=assessment,
            status=status,
            research_only=True,
            real_money_execution=False,
            human_approval_required=(
                self.limits.require_human_approval
                or not assessment.is_clear
            ),
        )

        if recommendation_id in self._recommendations:
            raise PortfolioRiskGovernanceError(
                "Duplicate recommendation ID: "
                f"{recommendation_id}"
            )

        self._recommendations[
            recommendation_id
        ] = recommendation

        return recommendation

    def assess_risk(
        self,
        *,
        weights: Mapping[str, float],
        cash_weight: float,
        current_drawdown: float = 0.0,
    ) -> RiskAssessment:
        if not isinstance(weights, Mapping):
            raise PortfolioRiskGovernanceError(
                "weights must be a mapping."
            )

        _require_probability(
            cash_weight,
            "cash_weight",
        )
        _require_probability(
            current_drawdown,
            "current_drawdown",
        )

        violations: List[str] = []
        warnings: List[str] = []

        concentration_breach = False
        drawdown_breach = False
        cash_breach = False

        for asset_id, weight in weights.items():
            _require_text(asset_id, "asset_id")
            _require_probability(weight, "weight")

            if weight > self.limits.max_position_weight:
                concentration_breach = True
                violations.append(
                    f"Position exceeds limit: {asset_id}"
                )

            if weight > self.limits.max_asset_concentration:
                concentration_breach = True
                violations.append(
                    f"Concentration exceeds limit: {asset_id}"
                )

        if current_drawdown > self.limits.max_portfolio_drawdown:
            drawdown_breach = True
            violations.append(
                "Portfolio drawdown exceeds configured limit."
            )
        elif current_drawdown > (
            self.limits.max_portfolio_drawdown * 0.75
        ):
            warnings.append(
                "Portfolio drawdown is approaching its limit."
            )

        if cash_weight < self.limits.minimum_cash_weight:
            cash_breach = True
            violations.append(
                "Cash weight is below the configured minimum."
            )

        if violations:
            severity = RiskSeverity.HIGH
        elif warnings:
            severity = RiskSeverity.MODERATE
        else:
            severity = RiskSeverity.NONE

        return RiskAssessment(
            severity=severity,
            violations=tuple(violations),
            warnings=tuple(warnings),
            drawdown_breach=drawdown_breach,
            concentration_breach=concentration_breach,
            cash_breach=cash_breach,
        )

    def approve_for_research(
        self,
        recommendation_id: str,
        *,
        approved: bool = False,
    ) -> PortfolioRecommendation:
        if not isinstance(approved, bool):
            raise PortfolioRiskGovernanceError(
                "approved must be boolean."
            )

        try:
            recommendation = self._recommendations[
                recommendation_id
            ]
        except (KeyError, TypeError) as exc:
            raise PortfolioRiskGovernanceError(
                "Unknown recommendation ID: "
                f"{recommendation_id}"
            ) from exc

        if not approved:
            return recommendation

        if not recommendation.risk_assessment.is_clear:
            raise PortfolioRiskGovernanceError(
                "Risk violations must be resolved before "
                "research approval."
            )

        updated = PortfolioRecommendation(
            recommendation_id=(
                recommendation.recommendation_id
            ),
            weights=dict(recommendation.weights),
            cash_weight=recommendation.cash_weight,
            signals=recommendation.signals,
            rationale=recommendation.rationale,
            risk_assessment=(
                recommendation.risk_assessment
            ),
            status=(
                RecommendationStatus
                .APPROVED_FOR_RESEARCH
            ),
            research_only=True,
            real_money_execution=False,
            human_approval_required=True,
        )

        self._recommendations[
            recommendation_id
        ] = updated

        return updated

    def get_recommendation(
        self,
        recommendation_id: str,
    ) -> PortfolioRecommendation:
        try:
            return self._recommendations[
                recommendation_id
            ]
        except (KeyError, TypeError) as exc:
            raise PortfolioRiskGovernanceError(
                "Unknown recommendation ID: "
                f"{recommendation_id}"
            ) from exc

    def list_recommendations(
        self,
    ) -> Tuple[PortfolioRecommendation, ...]:
        return tuple(
            self._recommendations.values()
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "signal_count": self.signal_count,
            "recommendation_count": (
                self.recommendation_count
            ),
            "research_only": self.RESEARCH_ONLY,
            "real_money_execution": (
                self.REAL_MONEY_EXECUTION
            ),
            "automatic_capital_allocation": (
                self.AUTOMATIC_CAPITAL_ALLOCATION
            ),
            "human_approval_required": (
                self.limits.require_human_approval
            ),
            "max_position_weight": (
                self.limits.max_position_weight
            ),
            "max_portfolio_drawdown": (
                self.limits.max_portfolio_drawdown
            ),
        }