"""
Agent Tournament System - Scoring Engine
=========================================

v5.0.6 of the Multi-Agent Financial Research Platform.

Scores blind tournament submissions using:

- prediction quality
- calibration
- consistency
- latency
- cost

The engine produces normalized metrics and a composite score
without requiring access to the real competitor identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence

from .blind_evaluator import BlindSubmission
from .models import EvaluationMetrics


class ScoringError(RuntimeError):
    """Raised when tournament scoring cannot be completed."""


@dataclass(frozen=True)
class ScoringWeights:
    """
    Weights used for the composite tournament score.

    Higher values give the corresponding metric more influence.
    """

    prediction: float = 0.50
    calibration: float = 0.20
    consistency: float = 0.15
    latency: float = 0.10
    cost: float = 0.05

    def __post_init__(self) -> None:
        values = (
            self.prediction,
            self.calibration,
            self.consistency,
            self.latency,
            self.cost,
        )

        if any(value < 0.0 for value in values):
            raise ValueError("Scoring weights cannot be negative.")

        total = sum(values)

        if total <= 0.0:
            raise ValueError(
                "At least one scoring weight must be greater than zero."
            )

    def normalized(self) -> "ScoringWeights":
        """Return weights normalized to sum to 1.0."""

        total = (
            self.prediction
            + self.calibration
            + self.consistency
            + self.latency
            + self.cost
        )

        return ScoringWeights(
            prediction=self.prediction / total,
            calibration=self.calibration / total,
            consistency=self.consistency / total,
            latency=self.latency / total,
            cost=self.cost / total,
        )


@dataclass(frozen=True)
class ScoredObservation:
    """
    One evaluated blind prediction.

    actual_outcome and predicted_outcome are normalized to [0, 1].
    """

    submission: BlindSubmission
    actual_outcome: float
    predicted_outcome: float


class ScoringEngine:
    """
    Blind scoring engine for champion/challenger tournaments.

    The engine intentionally operates on blind IDs only.
    Competitor identities are revealed by the governance layer
    after scoring has completed.
    """

    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        latency_reference_ms: float = 5000.0,
        cost_reference_usd: float = 0.05,
    ) -> None:
        self.weights = (weights or ScoringWeights()).normalized()

        if latency_reference_ms <= 0.0:
            raise ValueError(
                "latency_reference_ms must be greater than zero."
            )

        if cost_reference_usd <= 0.0:
            raise ValueError(
                "cost_reference_usd must be greater than zero."
            )

        self.latency_reference_ms = latency_reference_ms
        self.cost_reference_usd = cost_reference_usd

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_probability(
        value: float,
        name: str,
    ) -> float:
        value = float(value)

        if not 0.0 <= value <= 1.0:
            raise ScoringError(
                f"{name} must be between 0.0 and 1.0, got {value}."
            )

        return value

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    @classmethod
    def prediction_quality(
        cls,
        predicted_outcome: float,
        actual_outcome: float,
    ) -> float:
        """
        Prediction quality using absolute normalized error.

        Perfect prediction = 1.0
        Maximum error = 0.0
        """

        predicted = cls._validate_probability(
            predicted_outcome,
            "predicted_outcome",
        )

        actual = cls._validate_probability(
            actual_outcome,
            "actual_outcome",
        )

        return 1.0 - abs(predicted - actual)

    @classmethod
    def calibration_score(
        cls,
        confidence: float,
        predicted_outcome: float,
        actual_outcome: float,
    ) -> float:
        """
        Measure whether confidence is justified by prediction accuracy.

        High confidence is rewarded only when prediction error is low.
        """

        confidence = cls._validate_probability(
            confidence,
            "confidence",
        )

        quality = cls.prediction_quality(
            predicted_outcome,
            actual_outcome,
        )

        calibration_error = abs(confidence - quality)

        return max(0.0, 1.0 - calibration_error)

    @staticmethod
    def consistency_score(
        prediction_scores: Sequence[float],
    ) -> float:
        """
        Estimate consistency across multiple observations.

        Lower dispersion produces a higher score.
        """

        if not prediction_scores:
            return 0.0

        if len(prediction_scores) == 1:
            return 1.0

        values = [float(value) for value in prediction_scores]

        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ScoringError(
                "prediction_scores must be between 0.0 and 1.0."
            )

        avg = mean(values)

        variance = mean(
            (value - avg) ** 2
            for value in values
        )

        normalized_variance = min(1.0, variance / 0.25)

        return max(0.0, 1.0 - normalized_variance)

    def latency_score(
        self,
        latency_ms: float,
    ) -> float:
        """
        Convert latency into a bounded [0, 1] score.

        Exponential decay avoids hard latency thresholds.
        """

        latency_ms = float(latency_ms)

        if latency_ms < 0.0:
            raise ScoringError("latency_ms cannot be negative.")

        return exp(
            -latency_ms / self.latency_reference_ms
        )

    def cost_score(
        self,
        cost_usd: float,
    ) -> float:
        """
        Convert execution cost into a bounded [0, 1] score.
        """

        cost_usd = float(cost_usd)

        if cost_usd < 0.0:
            raise ScoringError("cost_usd cannot be negative.")

        return exp(
            -cost_usd / self.cost_reference_usd
        )

    # ------------------------------------------------------------------
    # Observation scoring
    # ------------------------------------------------------------------

    def score_observation(
        self,
        observation: ScoredObservation,
    ) -> Dict[str, float]:
        """
        Score one blind tournament observation.
        """

        submission = observation.submission

        prediction = self.prediction_quality(
            observation.predicted_outcome,
            observation.actual_outcome,
        )

        calibration = self.calibration_score(
            submission.confidence,
            observation.predicted_outcome,
            observation.actual_outcome,
        )

        latency = self.latency_score(
            submission.latency_ms
        )

        cost = self.cost_score(
            submission.cost_usd
        )

        return {
            "prediction": prediction,
            "calibration": calibration,
            "latency": latency,
            "cost": cost,
        }

    # ------------------------------------------------------------------
    # Competitor aggregation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        observations: Iterable[ScoredObservation],
    ) -> EvaluationMetrics:
        """
        Aggregate multiple observations for one blind competitor.
        """

        observations_list: List[ScoredObservation] = list(
            observations
        )

        if not observations_list:
            raise ScoringError(
                "At least one observation is required."
            )

        blind_ids = {
            observation.submission.blind_id
            for observation in observations_list
        }

        if len(blind_ids) != 1:
            raise ScoringError(
                "evaluate() accepts observations from exactly "
                "one blind competitor."
            )

        scored = [
            self.score_observation(observation)
            for observation in observations_list
        ]

        prediction_scores = [
            item["prediction"]
            for item in scored
        ]

        calibration_scores = [
            item["calibration"]
            for item in scored
        ]

        latency_scores = [
            item["latency"]
            for item in scored
        ]

        cost_scores = [
            item["cost"]
            for item in scored
        ]

        prediction = mean(prediction_scores)
        calibration = mean(calibration_scores)

        consistency = self.consistency_score(
            prediction_scores
        )

        latency = mean(latency_scores)
        cost = mean(cost_scores)

        composite = (
            prediction * self.weights.prediction
            + calibration * self.weights.calibration
            + consistency * self.weights.consistency
            + latency * self.weights.latency
            + cost * self.weights.cost
        )

        return EvaluationMetrics(
            prediction_score=prediction,
            calibration_score=calibration,
            consistency_score=consistency,
            latency_ms=mean(
                observation.submission.latency_ms
                for observation in observations_list
            ),
            cost_usd=mean(
                observation.submission.cost_usd
                for observation in observations_list
            ),
            composite_score=composite,
            sample_count=len(observations_list),
            extra_metrics={
                "latency_normalized_score": latency,
                "cost_normalized_score": cost,
            },
        )


__all__ = [
    "ScoredObservation",
    "ScoringEngine",
    "ScoringError",
    "ScoringWeights",
]