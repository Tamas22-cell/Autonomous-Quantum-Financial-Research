from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Sequence

from research_control_center.models import (
    LeaderboardEntry,
    utc_now,
)


class LeaderboardConfig:
    """
    Scoring configuration for the v5.0.10 Agent & Model Leaderboard.
    """

    def __init__(
        self,
        prediction_quality_weight: float = 0.45,
        calibration_weight: float = 0.25,
        consistency_weight: float = 0.20,
        efficiency_weight: float = 0.10,
        latency_reference_ms: float = 1000.0,
        cost_reference: float = 1.0,
    ) -> None:
        weights = (
            prediction_quality_weight,
            calibration_weight,
            consistency_weight,
            efficiency_weight,
        )

        for weight in weights:
            if isinstance(weight, bool) or not isinstance(
                weight, (int, float)
            ):
                raise TypeError("weights must be numeric")

            if weight < 0.0 or weight > 1.0:
                raise ValueError(
                    "weights must be between 0 and 1"
                )

        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError(
                "leaderboard weights must sum to 1.0"
            )

        if latency_reference_ms <= 0:
            raise ValueError(
                "latency_reference_ms must be positive"
            )

        if cost_reference <= 0:
            raise ValueError(
                "cost_reference must be positive"
            )

        self.prediction_quality_weight = float(
            prediction_quality_weight
        )
        self.calibration_weight = float(
            calibration_weight
        )
        self.consistency_weight = float(
            consistency_weight
        )
        self.efficiency_weight = float(
            efficiency_weight
        )

        self.latency_reference_ms = float(
            latency_reference_ms
        )
        self.cost_reference = float(cost_reference)


class LeaderboardEngine:
    """
    Research Control Center v5.0.10
    Agent & Model Leaderboard.

    Supports:
    - agent/model registration
    - asset x regime x horizon segmentation
    - prediction quality
    - calibration
    - consistency
    - latency
    - cost efficiency
    - composite scoring
    - ranking
    - filtered leaderboards
    - participant performance matrices
    - audit history

    This is a research evaluation component only.
    It cannot authorize trading or portfolio execution.
    """

    VALID_PARTICIPANT_TYPES = {
        "AGENT",
        "MODEL",
    }

    def __init__(
        self,
        config: Optional[LeaderboardConfig] = None,
    ) -> None:
        self.config = config or LeaderboardConfig()

        self._entries: Dict[
            str,
            LeaderboardEntry,
        ] = {}

        self._audit_log: List[dict] = []

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _validate_unit_score(
        value: Optional[float],
        name: str,
    ) -> Optional[float]:
        if value is None:
            return None

        if isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise TypeError(
                f"{name} must be numeric"
            )

        value = float(value)

        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1"
            )

        return value

    @staticmethod
    def _validate_non_negative(
        value: Optional[float],
        name: str,
    ) -> Optional[float]:
        if value is None:
            return None

        if isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise TypeError(
                f"{name} must be numeric"
            )

        value = float(value)

        if value < 0.0:
            raise ValueError(
                f"{name} cannot be negative"
            )

        return value

    @staticmethod
    def _normalize_participant_type(
        participant_type: str,
    ) -> str:
        if not isinstance(participant_type, str):
            raise TypeError(
                "participant_type must be a string"
            )

        normalized = participant_type.strip().upper()

        if normalized not in (
            LeaderboardEngine.VALID_PARTICIPANT_TYPES
        ):
            raise ValueError(
                "participant_type must be AGENT or MODEL"
            )

        return normalized

    @staticmethod
    def _validate_context(
        asset: str,
        regime: str,
        horizon: str,
    ) -> None:
        for name, value in (
            ("asset", asset),
            ("regime", regime),
            ("horizon", horizon),
        ):
            if not isinstance(value, str):
                raise TypeError(
                    f"{name} must be a string"
                )

            if not value.strip():
                raise ValueError(
                    f"{name} cannot be empty"
                )

    def _audit(
        self,
        event: str,
        entry: LeaderboardEntry,
    ) -> None:
        self._audit_log.append(
            {
                "timestamp": utc_now(),
                "event": event,
                "entry_id": entry.entry_id,
                "participant_name": (
                    entry.participant_name
                ),
                "participant_type": (
                    entry.participant_type
                ),
                "asset": entry.asset,
                "regime": entry.regime,
                "horizon": entry.horizon,
                "score": entry.score,
            }
        )

    # =========================================================
    # EFFICIENCY
    # =========================================================

    def calculate_efficiency_score(
        self,
        latency_ms: Optional[float],
        cost: Optional[float],
    ) -> float:
        latency_ms = self._validate_non_negative(
            latency_ms,
            "latency_ms",
        )

        cost = self._validate_non_negative(
            cost,
            "cost",
        )

        if latency_ms is None and cost is None:
            return 0.5

        components: List[float] = []

        if latency_ms is not None:
            latency_score = 1.0 / (
                1.0
                + (
                    latency_ms
                    / self.config.latency_reference_ms
                )
            )

            components.append(latency_score)

        if cost is not None:
            cost_score = 1.0 / (
                1.0
                + (
                    cost
                    / self.config.cost_reference
                )
            )

            components.append(cost_score)

        return sum(components) / len(components)

    # =========================================================
    # COMPOSITE SCORE
    # =========================================================

    def calculate_score(
        self,
        prediction_quality: Optional[float],
        calibration_score: Optional[float],
        consistency_score: Optional[float],
        latency_ms: Optional[float],
        cost: Optional[float],
    ) -> float:
        prediction_quality = (
            self._validate_unit_score(
                prediction_quality,
                "prediction_quality",
            )
        )

        calibration_score = (
            self._validate_unit_score(
                calibration_score,
                "calibration_score",
            )
        )

        consistency_score = (
            self._validate_unit_score(
                consistency_score,
                "consistency_score",
            )
        )

        efficiency = (
            self.calculate_efficiency_score(
                latency_ms,
                cost,
            )
        )

        prediction_quality = (
            prediction_quality
            if prediction_quality is not None
            else 0.0
        )

        calibration_score = (
            calibration_score
            if calibration_score is not None
            else 0.0
        )

        consistency_score = (
            consistency_score
            if consistency_score is not None
            else 0.0
        )

        score = (
            prediction_quality
            * self.config.prediction_quality_weight
            + calibration_score
            * self.config.calibration_weight
            + consistency_score
            * self.config.consistency_weight
            + efficiency
            * self.config.efficiency_weight
        )

        return max(
            0.0,
            min(1.0, float(score)),
        )

    # =========================================================
    # REGISTRATION
    # =========================================================

    def add_entry(
        self,
        participant_name: str,
        participant_type: str,
        asset: str,
        regime: str,
        horizon: str,
        *,
        prediction_quality: Optional[
            float
        ] = None,
        calibration_score: Optional[
            float
        ] = None,
        consistency_score: Optional[
            float
        ] = None,
        latency_ms: Optional[float] = None,
        cost: Optional[float] = None,
        observations: int = 0,
        metadata: Optional[dict] = None,
    ) -> LeaderboardEntry:
        if not isinstance(
            participant_name,
            str,
        ):
            raise TypeError(
                "participant_name must be a string"
            )

        participant_name = (
            participant_name.strip()
        )

        if not participant_name:
            raise ValueError(
                "participant_name cannot be empty"
            )

        participant_type = (
            self._normalize_participant_type(
                participant_type
            )
        )

        self._validate_context(
            asset,
            regime,
            horizon,
        )

        if isinstance(observations, bool):
            raise TypeError(
                "observations must be an integer"
            )

        if not isinstance(observations, int):
            raise TypeError(
                "observations must be an integer"
            )

        if observations < 0:
            raise ValueError(
                "observations cannot be negative"
            )

        prediction_quality = (
            self._validate_unit_score(
                prediction_quality,
                "prediction_quality",
            )
        )

        calibration_score = (
            self._validate_unit_score(
                calibration_score,
                "calibration_score",
            )
        )

        consistency_score = (
            self._validate_unit_score(
                consistency_score,
                "consistency_score",
            )
        )

        latency_ms = (
            self._validate_non_negative(
                latency_ms,
                "latency_ms",
            )
        )

        cost = self._validate_non_negative(
            cost,
            "cost",
        )

        score = self.calculate_score(
            prediction_quality,
            calibration_score,
            consistency_score,
            latency_ms,
            cost,
        )

        entry = LeaderboardEntry(
            participant_name=participant_name,
            participant_type=participant_type,
            asset=asset.strip(),
            regime=regime.strip(),
            horizon=horizon.strip(),
            score=score,
            prediction_quality=(
                prediction_quality
            ),
            calibration_score=(
                calibration_score
            ),
            consistency_score=(
                consistency_score
            ),
            latency_ms=latency_ms,
            cost=cost,
            observations=observations,
            metadata={
                **deepcopy(metadata or {}),
                "engine": (
                    "v5.0.10-leaderboard"
                ),
                "research_only": True,
                "real_money_execution": False,
            },
        )

        self._entries[
            entry.entry_id
        ] = entry

        self._audit(
            "LEADERBOARD_ENTRY_ADDED",
            entry,
        )

        return deepcopy(entry)

    # =========================================================
    # READ / FILTER
    # =========================================================

    def get(
        self,
        entry_id: str,
    ) -> LeaderboardEntry:
        if entry_id not in self._entries:
            raise KeyError(
                f"unknown entry_id: {entry_id}"
            )

        return deepcopy(
            self._entries[entry_id]
        )

    def list_entries(
        self,
    ) -> List[LeaderboardEntry]:
        return [
            deepcopy(entry)
            for entry in self._entries.values()
        ]

    def filter_entries(
        self,
        *,
        asset: Optional[str] = None,
        regime: Optional[str] = None,
        horizon: Optional[str] = None,
        participant_type: Optional[
            str
        ] = None,
        participant_name: Optional[
            str
        ] = None,
    ) -> List[LeaderboardEntry]:
        normalized_type = None

        if participant_type is not None:
            normalized_type = (
                self._normalize_participant_type(
                    participant_type
                )
            )

        results = []

        for entry in self._entries.values():
            if (
                asset is not None
                and entry.asset != asset
            ):
                continue

            if (
                regime is not None
                and entry.regime != regime
            ):
                continue

            if (
                horizon is not None
                and entry.horizon != horizon
            ):
                continue

            if (
                normalized_type is not None
                and entry.participant_type
                != normalized_type
            ):
                continue

            if (
                participant_name is not None
                and entry.participant_name
                != participant_name
            ):
                continue

            results.append(
                deepcopy(entry)
            )

        return results

    # =========================================================
    # RANKING
    # =========================================================

    def rank(
        self,
        *,
        asset: Optional[str] = None,
        regime: Optional[str] = None,
        horizon: Optional[str] = None,
        participant_type: Optional[
            str
        ] = None,
    ) -> List[LeaderboardEntry]:
        entries = self.filter_entries(
            asset=asset,
            regime=regime,
            horizon=horizon,
            participant_type=participant_type,
        )

        entries.sort(
            key=lambda entry: (
                entry.score,
                entry.prediction_quality
                if entry.prediction_quality
                is not None
                else -1.0,
                entry.observations,
            ),
            reverse=True,
        )

        for index, entry in enumerate(
            entries,
            start=1,
        ):
            entry.rank = index

        return entries

    def top(
        self,
        limit: int = 10,
        **filters,
    ) -> List[LeaderboardEntry]:
        if isinstance(limit, bool):
            raise TypeError(
                "limit must be an integer"
            )

        if not isinstance(limit, int):
            raise TypeError(
                "limit must be an integer"
            )

        if limit <= 0:
            raise ValueError(
                "limit must be positive"
            )

        return self.rank(
            **filters
        )[:limit]

    # =========================================================
    # PERFORMANCE MATRIX
    # =========================================================

    def performance_matrix(
        self,
        participant_name: str,
    ) -> dict:
        entries = self.filter_entries(
            participant_name=participant_name
        )

        matrix = {}

        for entry in entries:
            key = (
                entry.asset,
                entry.regime,
                entry.horizon,
            )

            matrix[key] = {
                "score": entry.score,
                "prediction_quality": (
                    entry.prediction_quality
                ),
                "calibration_score": (
                    entry.calibration_score
                ),
                "consistency_score": (
                    entry.consistency_score
                ),
                "latency_ms": (
                    entry.latency_ms
                ),
                "cost": entry.cost,
                "observations": (
                    entry.observations
                ),
            }

        return matrix

    # =========================================================
    # PARTICIPANT SUMMARY
    # =========================================================

    def participant_summary(
        self,
        participant_name: str,
    ) -> dict:
        entries = self.filter_entries(
            participant_name=participant_name
        )

        if not entries:
            raise KeyError(
                f"participant not found: "
                f"{participant_name}"
            )

        average_score = (
            sum(
                entry.score
                for entry in entries
            )
            / len(entries)
        )

        total_observations = sum(
            entry.observations
            for entry in entries
        )

        best = max(
            entries,
            key=lambda entry: entry.score,
        )

        worst = min(
            entries,
            key=lambda entry: entry.score,
        )

        return {
            "participant_name": (
                participant_name
            ),
            "participant_type": (
                entries[0].participant_type
            ),
            "average_score": (
                average_score
            ),
            "total_observations": (
                total_observations
            ),
            "contexts": len(entries),
            "best_context": {
                "asset": best.asset,
                "regime": best.regime,
                "horizon": best.horizon,
                "score": best.score,
            },
            "worst_context": {
                "asset": worst.asset,
                "regime": worst.regime,
                "horizon": worst.horizon,
                "score": worst.score,
            },
            "research_only": True,
            "real_money_execution": False,
        }

    # =========================================================
    # GLOBAL SNAPSHOT
    # =========================================================

    def snapshot(self) -> dict:
        entries = self.list_entries()

        agents = [
            entry
            for entry in entries
            if entry.participant_type
            == "AGENT"
        ]

        models = [
            entry
            for entry in entries
            if entry.participant_type
            == "MODEL"
        ]

        contexts = {
            (
                entry.asset,
                entry.regime,
                entry.horizon,
            )
            for entry in entries
        }

        return {
            "entries": entries,
            "total_entries": len(entries),
            "agent_entries": len(agents),
            "model_entries": len(models),
            "contexts": len(contexts),
            "ranking": self.rank(),
            "generated_at": utc_now(),
            "research_only": True,
            "real_money_execution": False,
        }

    # =========================================================
    # AUDIT
    # =========================================================

    def audit_log(
        self,
    ) -> List[dict]:
        return deepcopy(
            self._audit_log
        )