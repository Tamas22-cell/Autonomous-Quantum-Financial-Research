"""
v5.0.9 - Autonomous Research Loop
Independent Risk Governor Adapter.

Connects the existing RiskAgent to the Autonomous Research Loop.

Existing RiskAgent contract:

{
    "agent": "risk",
    "signal": "bullish" | "neutral" | "bearish",
    "score": float,
    "confidence": float,
    "summary": str,
    "risks": list,
    "evidence": dict
}

This adapter translates that output into:

- APPROVE
- REDUCE
- VETO
- SHUTDOWN

Important:
PROMOTE from the validation layer is NOT sufficient for shadow
allocation. The candidate must also pass this independent risk gate.

This module:
- does not execute trades
- does not allocate real capital
- does not modify RiskAgent
- does not bypass validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from autonomous_research_loop.models import (
    RiskAssessment,
    RiskDecision,
    StrategyCandidate,
    ValidationDecision,
    ValidationResult,
)


@dataclass(frozen=True)
class RiskGovernorConfig:
    """
    Thresholds for translating RiskAgent output into
    autonomous research risk decisions.

    Existing RiskAgent semantics:
        higher score = safer / more constructive
        lower score = higher risk
    """

    approve_score_threshold: float = 0.55
    reduce_score_threshold: float = 0.40
    veto_score_threshold: float = 0.20

    minimum_confidence: float = 0.50

    reduce_position_multiplier: float = 0.50

    severe_drawdown_threshold: float = 0.30
    extreme_volatility_threshold: float = 0.75
    extreme_correlation_threshold: float = 0.90

    def __post_init__(self) -> None:
        bounded_values = (
            self.approve_score_threshold,
            self.reduce_score_threshold,
            self.veto_score_threshold,
            self.minimum_confidence,
            self.reduce_position_multiplier,
            self.severe_drawdown_threshold,
            self.extreme_volatility_threshold,
            self.extreme_correlation_threshold,
        )

        if any(
            value < 0.0 or value > 1.0
            for value in bounded_values
        ):
            raise ValueError(
                "Risk Governor configuration values must "
                "be between 0.0 and 1.0."
            )

        if not (
            self.veto_score_threshold
            < self.reduce_score_threshold
            < self.approve_score_threshold
        ):
            raise ValueError(
                "Risk thresholds must satisfy: "
                "veto < reduce < approve."
            )


class RiskGovernorAdapter:
    """
    Independent risk gate for promoted research candidates.

    The adapter can operate in two ways:

    1. evaluate(candidate, validation, risk_output)
       Uses an already available RiskAgent result.

    2. run_risk_agent(candidate, validation, risk_agent)
       Calls the existing RiskAgent.analyze() method.

    This keeps the v5.0.9 research layer decoupled from the
    production RiskAgent implementation.
    """

    def __init__(
        self,
        config: Optional[RiskGovernorConfig] = None,
    ) -> None:
        self.config = config or RiskGovernorConfig()

    @staticmethod
    def _clamp(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                numeric,
            ),
        )

    @staticmethod
    def _safe_list(value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, (list, tuple)):
            return [
                str(item)
                for item in value
                if item is not None
            ]

        return [str(value)]

    @staticmethod
    def _safe_mapping(value: Any) -> Dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)

        return {}

    @staticmethod
    def _normalize_signal(value: Any) -> str:
        signal = str(
            value or "neutral"
        ).strip().lower()

        if signal not in {
            "bullish",
            "neutral",
            "bearish",
        }:
            return "neutral"

        return signal

    def detect_extreme_risk(
        self,
        evidence: Mapping[str, Any],
    ) -> List[str]:
        """
        Detect extreme conditions directly from RiskAgent evidence.

        The current RiskAgent evidence contains:
        - avg_volatility
        - avg_correlation
        - max_drawdown
        """

        extreme_conditions: List[str] = []

        try:
            volatility = abs(
                float(
                    evidence.get(
                        "avg_volatility",
                        0.0,
                    )
                )
            )
        except (TypeError, ValueError):
            volatility = 0.0

        try:
            correlation = abs(
                float(
                    evidence.get(
                        "avg_correlation",
                        0.0,
                    )
                )
            )
        except (TypeError, ValueError):
            correlation = 0.0

        try:
            drawdown = abs(
                float(
                    evidence.get(
                        "max_drawdown",
                        0.0,
                    )
                )
            )
        except (TypeError, ValueError):
            drawdown = 0.0

        if (
            drawdown
            >= self.config.severe_drawdown_threshold
        ):
            extreme_conditions.append(
                "Severe drawdown threshold breached."
            )

        if (
            volatility
            >= self.config.extreme_volatility_threshold
        ):
            extreme_conditions.append(
                "Extreme volatility threshold breached."
            )

        if (
            correlation
            >= self.config.extreme_correlation_threshold
        ):
            extreme_conditions.append(
                "Extreme cross-asset correlation threshold breached."
            )

        return extreme_conditions

    def determine_decision(
        self,
        *,
        score: float,
        confidence: float,
        signal: str,
        extreme_conditions: List[str],
    ) -> RiskDecision:
        """
        Translate risk evidence into a RiskDecision.

        Precedence:
            SHUTDOWN
            VETO
            REDUCE
            APPROVE

        SHUTDOWN is reserved for extreme measured risk conditions.
        """

        if extreme_conditions:
            return RiskDecision.SHUTDOWN

        if (
            score
            <= self.config.veto_score_threshold
        ):
            return RiskDecision.VETO

        if signal == "bearish":
            return RiskDecision.VETO

        if (
            confidence
            < self.config.minimum_confidence
        ):
            return RiskDecision.REDUCE

        if (
            score
            < self.config.reduce_score_threshold
        ):
            return RiskDecision.VETO

        if (
            score
            < self.config.approve_score_threshold
        ):
            return RiskDecision.REDUCE

        return RiskDecision.APPROVE

    def calculate_risk_score(
        self,
        risk_agent_score: float,
    ) -> float:
        """
        Convert RiskAgent safety score into explicit risk severity.

        Existing RiskAgent:
            1.0 = constructive / low risk
            0.0 = severe risk

        RiskAssessment:
            0.0 = low risk
            1.0 = high risk
        """

        safety_score = self._clamp(
            risk_agent_score
        )

        return round(
            1.0 - safety_score,
            6,
        )

    def calculate_approved_position_size(
        self,
        candidate: StrategyCandidate,
        decision: RiskDecision,
    ) -> float:
        """
        Determine maximum research/shadow position size.

        This remains research-only sizing.
        """

        requested_size = self._clamp(
            candidate.position_size
        )

        if decision == RiskDecision.APPROVE:
            return requested_size

        if decision == RiskDecision.REDUCE:
            return round(
                requested_size
                * self.config.reduce_position_multiplier,
                6,
            )

        return 0.0

    def build_reasons(
        self,
        *,
        decision: RiskDecision,
        score: float,
        confidence: float,
        signal: str,
        risks: List[str],
        extreme_conditions: List[str],
    ) -> List[str]:
        """
        Build auditable risk-decision reasons.
        """

        reasons: List[str] = [
            f"RiskAgent safety score: {score:.3f}.",
            f"RiskAgent confidence: {confidence:.3f}.",
            f"RiskAgent signal: {signal}.",
            f"Risk Governor decision: {decision.value}.",
        ]

        for risk in risks:
            reasons.append(
                f"RiskAgent finding: {risk}"
            )

        reasons.extend(
            extreme_conditions
        )

        if decision == RiskDecision.APPROVE:
            reasons.append(
                "Candidate may proceed to shadow allocation only."
            )

        elif decision == RiskDecision.REDUCE:
            reasons.append(
                "Candidate may proceed only with reduced "
                "shadow position size."
            )

        elif decision == RiskDecision.VETO:
            reasons.append(
                "Candidate is blocked from shadow allocation."
            )

        else:
            reasons.append(
                "Research exposure is shut down because an "
                "extreme risk condition was detected."
            )

        return reasons

    def evaluate(
        self,
        candidate: StrategyCandidate,
        validation: ValidationResult,
        risk_output: Mapping[str, Any],
    ) -> RiskAssessment:
        """
        Evaluate one promoted StrategyCandidate using existing
        RiskAgent output.
        """

        if not isinstance(
            candidate,
            StrategyCandidate,
        ):
            raise TypeError(
                "RiskGovernorAdapter requires a StrategyCandidate."
            )

        if not isinstance(
            validation,
            ValidationResult,
        ):
            raise TypeError(
                "validation must be a ValidationResult."
            )

        if (
            validation.strategy_id
            != candidate.strategy_id
        ):
            raise ValueError(
                "ValidationResult does not belong to "
                "the supplied StrategyCandidate."
            )

        if (
            validation.decision
            != ValidationDecision.PROMOTE
        ):
            raise ValueError(
                "Only PROMOTE candidates may enter "
                "the Risk Governor."
            )

        if not isinstance(
            risk_output,
            Mapping,
        ):
            raise TypeError(
                "risk_output must be a mapping."
            )

        score = self._clamp(
            risk_output.get(
                "score",
                0.0,
            )
        )

        confidence = self._clamp(
            risk_output.get(
                "confidence",
                0.0,
            )
        )

        signal = self._normalize_signal(
            risk_output.get(
                "signal",
                "neutral",
            )
        )

        risks = self._safe_list(
            risk_output.get(
                "risks"
            )
        )

        evidence = self._safe_mapping(
            risk_output.get(
                "evidence"
            )
        )

        extreme_conditions = (
            self.detect_extreme_risk(
                evidence
            )
        )

        decision = self.determine_decision(
            score=score,
            confidence=confidence,
            signal=signal,
            extreme_conditions=extreme_conditions,
        )

        risk_score = self.calculate_risk_score(
            score
        )

        approved_position_size = (
            self.calculate_approved_position_size(
                candidate,
                decision,
            )
        )

        reasons = self.build_reasons(
            decision=decision,
            score=score,
            confidence=confidence,
            signal=signal,
            risks=risks,
            extreme_conditions=extreme_conditions,
        )

        return RiskAssessment(
            strategy_id=candidate.strategy_id,
            decision=decision,
            risk_score=risk_score,
            approved_position_size=approved_position_size,
            reasons=reasons,
            metadata={
                "governor":
                    "v5.0.9-risk-governor-adapter",
                "source_agent":
                    risk_output.get(
                        "agent",
                        "risk",
                    ),
                "risk_agent_score":
                    score,
                "risk_agent_confidence":
                    confidence,
                "risk_agent_signal":
                    signal,
                "risk_agent_summary":
                    risk_output.get(
                        "summary"
                    ),
                "risk_agent_evidence":
                    evidence,
                "extreme_conditions":
                    list(
                        extreme_conditions
                    ),
                "validation_score":
                    validation.score,
                "validation_decision":
                    validation.decision.value,
                "real_money_authorized":
                    False,
                "shadow_only":
                    decision
                    in {
                        RiskDecision.APPROVE,
                        RiskDecision.REDUCE,
                    },
            },
        )

    def run_risk_agent(
        self,
        candidate: StrategyCandidate,
        validation: ValidationResult,
        risk_agent: Any,
    ) -> RiskAssessment:
        """
        Call the existing RiskAgent.analyze() method and translate
        its result into a v5.0.9 RiskAssessment.
        """

        analyze_method = getattr(
            risk_agent,
            "analyze",
            None,
        )

        if not callable(
            analyze_method
        ):
            raise TypeError(
                "RiskAgent must expose a callable analyze() method."
            )

        risk_output = analyze_method()

        if not isinstance(
            risk_output,
            Mapping,
        ):
            raise TypeError(
                "RiskAgent.analyze() must return a mapping."
            )

        return self.evaluate(
            candidate,
            validation,
            risk_output,
        )