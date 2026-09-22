from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Dict, Mapping, Optional, Tuple


class OutcomeAttributionError(ValueError):
    """Outcome attribution or meta-learning error."""


class OutcomeLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class AttributionRecord:
    outcome_id: str
    realized_outcome: float
    baseline_outcome: float
    signal_contributions: Mapping[str, float]
    label: OutcomeLabel
    fingerprint: str


@dataclass(frozen=True)
class LearningUpdate:
    signal_id: str
    previous_weight: float
    updated_weight: float
    observation_count: int


@dataclass
class OutcomeAttributionMetaLearning:
    """
    Research-only outcome attribution and meta-learning service.

    It attributes observed outcomes to registered research signals
    and updates signal weights conservatively using a bounded rule.
    """

    VERSION = "v7.0.11"
    RESEARCH_ONLY = True
    REAL_MONEY_EXECUTION = False
    AUTOMATIC_TRADING = False

    learning_rate: float = 0.10
    _records: list[AttributionRecord] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _weights: Dict[str, float] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _observations: Dict[str, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not isfinite(self.learning_rate)
            or not 0.0 < self.learning_rate <= 1.0
        ):
            raise OutcomeAttributionError(
                "learning_rate must be between 0 and 1."
            )

    @staticmethod
    def _validate_number(
        value: float,
        name: str,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            raise OutcomeAttributionError(
                f"{name} must be a finite number."
            )

    @staticmethod
    def _validate_text(
        value: str,
        name: str,
    ) -> None:
        if not isinstance(value, str) or not value.strip():
            raise OutcomeAttributionError(
                f"{name} must be a non-empty string."
            )

    @staticmethod
    def _fingerprint(
        outcome_id: str,
        realized_outcome: float,
        contributions: Mapping[str, float],
    ) -> str:
        payload = (
            outcome_id,
            round(realized_outcome, 12),
            tuple(sorted(contributions.items())),
        )
        return f"outcome-{abs(hash(payload)):016x}"

    @staticmethod
    def _label(
        excess_outcome: float,
    ) -> OutcomeLabel:
        if excess_outcome > 0:
            return OutcomeLabel.POSITIVE

        if excess_outcome < 0:
            return OutcomeLabel.NEGATIVE

        return OutcomeLabel.NEUTRAL

    def attribute(
        self,
        outcome_id: str,
        signal_scores: Mapping[str, float],
        realized_outcome: float,
        *,
        baseline_outcome: float = 0.0,
    ) -> AttributionRecord:
        self._validate_text(outcome_id, "outcome_id")

        if not isinstance(signal_scores, Mapping):
            raise OutcomeAttributionError(
                "signal_scores must be a mapping."
            )

        if not signal_scores:
            raise OutcomeAttributionError(
                "signal_scores cannot be empty."
            )

        self._validate_number(
            realized_outcome,
            "realized_outcome",
        )
        self._validate_number(
            baseline_outcome,
            "baseline_outcome",
        )

        if any(
            not isinstance(signal_id, str)
            or not signal_id.strip()
            for signal_id in signal_scores
        ):
            raise OutcomeAttributionError(
                "signal IDs must be non-empty strings."
            )

        for score in signal_scores.values():
            self._validate_number(score, "signal score")

        if any(
            record.outcome_id == outcome_id
            for record in self._records
        ):
            raise OutcomeAttributionError(
                f"Duplicate outcome ID: {outcome_id}"
            )

        total_abs_score = sum(
            abs(score)
            for score in signal_scores.values()
        )

        if total_abs_score == 0:
            equal_share = (
                realized_outcome - baseline_outcome
            ) / len(signal_scores)

            contributions = {
                signal_id: equal_share
                for signal_id in signal_scores
            }
        else:
            excess_outcome = (
                realized_outcome - baseline_outcome
            )

            contributions = {
                signal_id: (
                    excess_outcome
                    * score
                    / total_abs_score
                )
                for signal_id, score in signal_scores.items()
            }

        excess_outcome = (
            realized_outcome - baseline_outcome
        )

        record = AttributionRecord(
            outcome_id=outcome_id,
            realized_outcome=realized_outcome,
            baseline_outcome=baseline_outcome,
            signal_contributions=contributions,
            label=self._label(excess_outcome),
            fingerprint=self._fingerprint(
                outcome_id,
                realized_outcome,
                contributions,
            ),
        )

        self._records.append(record)

        for signal_id, contribution in contributions.items():
            previous_weight = self._weights.get(
                signal_id,
                1.0,
            )
            performance = max(
                -1.0,
                min(1.0, contribution),
            )

            updated_weight = (
                (1.0 - self.learning_rate)
                * previous_weight
                + self.learning_rate
                * (1.0 + performance)
            )

            self._weights[signal_id] = max(
                0.0,
                min(2.0, updated_weight),
            )

            self._observations[signal_id] = (
                self._observations.get(signal_id, 0)
                + 1
            )

        return record

    def weight(self, signal_id: str) -> float:
        self._validate_text(signal_id, "signal_id")
        return self._weights.get(signal_id, 1.0)

    def learning_updates(self) -> Tuple[LearningUpdate, ...]:
        return tuple(
            LearningUpdate(
                signal_id=signal_id,
                previous_weight=1.0,
                updated_weight=weight,
                observation_count=(
                    self._observations[signal_id]
                ),
            )
            for signal_id, weight
            in sorted(self._weights.items())
        )

    def records(self) -> Tuple[AttributionRecord, ...]:
        return tuple(self._records)

    def summary(self) -> Dict[str, object]:
        return {
            "component": (
                "outcome_attribution_meta_learning"
            ),
            "version": self.VERSION,
            "outcome_count": len(self._records),
            "signal_count": len(self._weights),
            "research_only": self.RESEARCH_ONLY,
            "real_money_execution": (
                self.REAL_MONEY_EXECUTION
            ),
            "automatic_trading": (
                self.AUTOMATIC_TRADING
            ),
            "meta_learning_enabled": True,
        }

    def snapshot(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "learning_rate": self.learning_rate,
            "outcome_count": len(self._records),
            "weights": dict(self._weights),
            "observations": dict(self._observations),
            "research_only": self.RESEARCH_ONLY,
            "real_money_execution": (
                self.REAL_MONEY_EXECUTION
            ),
        }