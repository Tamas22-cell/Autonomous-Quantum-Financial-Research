from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable


class UncertaintyType(str, Enum):
    CONFIDENCE_GAP = "CONFIDENCE_GAP"
    REGIME_CHANGE = "REGIME_CHANGE"
    PERFORMANCE_GAP = "PERFORMANCE_GAP"
    EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class UncertaintySignal:
    signal_type: UncertaintyType
    severity: Severity
    score: float
    metric: str
    observed_value: float
    threshold: float
    description: str


@dataclass(frozen=True)
class UncertaintyReport:
    signals: tuple[UncertaintySignal, ...]

    @property
    def has_uncertainty(self) -> bool:
        return bool(self.signals)

    @property
    def highest_severity(self) -> Severity | None:
        if not self.signals:
            return None

        order = {
            Severity.LOW: 0,
            Severity.MEDIUM: 1,
            Severity.HIGH: 2,
            Severity.CRITICAL: 3,
        }

        return max(
            self.signals,
            key=lambda signal: order[signal.severity],
        ).severity

    @property
    def signal_count(self) -> int:
        return len(self.signals)


class UncertaintyDetector:
    """
    Detects research uncertainty that may justify a new experiment.

    The detector does not execute experiments.
    It only produces structured signals for the experiment designer.
    """

    def __init__(
        self,
        *,
        confidence_gap_threshold: float = 0.20,
        regime_change_threshold: float = 0.30,
        performance_gap_threshold: float = 0.10,
        evidence_conflict_threshold: float = 0.40,
    ) -> None:
        self._validate_threshold(
            confidence_gap_threshold,
            "confidence_gap_threshold",
        )
        self._validate_threshold(
            regime_change_threshold,
            "regime_change_threshold",
        )
        self._validate_threshold(
            performance_gap_threshold,
            "performance_gap_threshold",
        )
        self._validate_threshold(
            evidence_conflict_threshold,
            "evidence_conflict_threshold",
        )

        self.confidence_gap_threshold = confidence_gap_threshold
        self.regime_change_threshold = regime_change_threshold
        self.performance_gap_threshold = performance_gap_threshold
        self.evidence_conflict_threshold = evidence_conflict_threshold

    @staticmethod
    def _validate_threshold(
        value: float,
        name: str,
    ) -> None:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number.")

        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number.")

        if not isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number.")

        if not 0.0 < float(value) <= 1.0:
            raise ValueError(
                f"{name} must be greater than 0 and at most 1."
            )

    @staticmethod
    def _validate_metric(
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number.")

        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number.")

        value = float(value)

        if not isfinite(value):
            raise ValueError(f"{name} must be a finite number.")

        return value

    @staticmethod
    def _severity(
        ratio: float,
    ) -> Severity:
        if ratio >= 2.0:
            return Severity.CRITICAL
        if ratio >= 1.5:
            return Severity.HIGH
        if ratio >= 1.0:
            return Severity.MEDIUM
        return Severity.LOW

    def detect_confidence_gap(
        self,
        *,
        confidence: float,
        target_confidence: float,
    ) -> UncertaintySignal | None:
        confidence = self._validate_metric(
            confidence,
            "confidence",
        )
        target_confidence = self._validate_metric(
            target_confidence,
            "target_confidence",
        )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")

        if not 0.0 <= target_confidence <= 1.0:
            raise ValueError(
                "target_confidence must be between 0 and 1."
            )

        gap = max(0.0, target_confidence - confidence)

        if gap < self.confidence_gap_threshold:
            return None

        ratio = gap / self.confidence_gap_threshold

        return UncertaintySignal(
            signal_type=UncertaintyType.CONFIDENCE_GAP,
            severity=self._severity(ratio),
            score=gap,
            metric="confidence_gap",
            observed_value=confidence,
            threshold=target_confidence,
            description=(
                "Model confidence is materially below the "
                "target confidence."
            ),
        )

    def detect_regime_change(
        self,
        *,
        regime_distance: float,
    ) -> UncertaintySignal | None:
        regime_distance = self._validate_metric(
            regime_distance,
            "regime_distance",
        )

        if regime_distance < 0.0:
            raise ValueError(
                "regime_distance cannot be negative."
            )

        if regime_distance < self.regime_change_threshold:
            return None

        ratio = (
            regime_distance
            / self.regime_change_threshold
        )

        return UncertaintySignal(
            signal_type=UncertaintyType.REGIME_CHANGE,
            severity=self._severity(ratio),
            score=regime_distance,
            metric="regime_distance",
            observed_value=regime_distance,
            threshold=self.regime_change_threshold,
            description=(
                "The observed market regime differs materially "
                "from the reference regime."
            ),
        )

    def detect_performance_gap(
        self,
        *,
        observed_performance: float,
        expected_performance: float,
    ) -> UncertaintySignal | None:
        observed_performance = self._validate_metric(
            observed_performance,
            "observed_performance",
        )
        expected_performance = self._validate_metric(
            expected_performance,
            "expected_performance",
        )

        gap = abs(
            expected_performance - observed_performance
        )

        if gap < self.performance_gap_threshold:
            return None

        ratio = gap / self.performance_gap_threshold

        return UncertaintySignal(
            signal_type=UncertaintyType.PERFORMANCE_GAP,
            severity=self._severity(ratio),
            score=gap,
            metric="performance_gap",
            observed_value=observed_performance,
            threshold=expected_performance,
            description=(
                "Observed performance materially differs "
                "from expected performance."
            ),
        )

    def detect_evidence_conflict(
        self,
        *,
        agreement_score: float,
    ) -> UncertaintySignal | None:
        agreement_score = self._validate_metric(
            agreement_score,
            "agreement_score",
        )

        if not 0.0 <= agreement_score <= 1.0:
            raise ValueError(
                "agreement_score must be between 0 and 1."
            )

        conflict_score = 1.0 - agreement_score

        if conflict_score < self.evidence_conflict_threshold:
            return None

        ratio = (
            conflict_score
            / self.evidence_conflict_threshold
        )

        return UncertaintySignal(
            signal_type=UncertaintyType.EVIDENCE_CONFLICT,
            severity=self._severity(ratio),
            score=conflict_score,
            metric="evidence_conflict",
            observed_value=agreement_score,
            threshold=self.evidence_conflict_threshold,
            description=(
                "Available evidence contains a material "
                "level of disagreement."
            ),
        )

    def detect(
        self,
        *,
        confidence: tuple[float, float] | None = None,
        regime_distance: float | None = None,
        performance: tuple[float, float] | None = None,
        evidence_agreement: float | None = None,
    ) -> UncertaintyReport:
        signals: list[UncertaintySignal] = []

        if confidence is not None:
            if (
                not isinstance(confidence, tuple)
                or len(confidence) != 2
            ):
                raise ValueError(
                    "confidence must be a "
                    "(current, target) tuple."
                )

            signal = self.detect_confidence_gap(
                confidence=confidence[0],
                target_confidence=confidence[1],
            )

            if signal is not None:
                signals.append(signal)

        if regime_distance is not None:
            signal = self.detect_regime_change(
                regime_distance=regime_distance,
            )

            if signal is not None:
                signals.append(signal)

        if performance is not None:
            if (
                not isinstance(performance, tuple)
                or len(performance) != 2
            ):
                raise ValueError(
                    "performance must be an "
                    "(observed, expected) tuple."
                )

            signal = self.detect_performance_gap(
                observed_performance=performance[0],
                expected_performance=performance[1],
            )

            if signal is not None:
                signals.append(signal)

        if evidence_agreement is not None:
            signal = self.detect_evidence_conflict(
                agreement_score=evidence_agreement,
            )

            if signal is not None:
                signals.append(signal)

        signals.sort(
            key=lambda signal: (
                -{
                    Severity.LOW: 0,
                    Severity.MEDIUM: 1,
                    Severity.HIGH: 2,
                    Severity.CRITICAL: 3,
                }[signal.severity],
                signal.signal_type.value,
            )
        )

        return UncertaintyReport(
            signals=tuple(signals),
        )


__all__ = [
    "Severity",
    "UncertaintyDetector",
    "UncertaintyReport",
    "UncertaintySignal",
    "UncertaintyType",
]