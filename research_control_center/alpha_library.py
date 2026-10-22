from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

from research_control_center.models import (
    AlphaRecord,
    AlphaStatus,
    RobustnessStatus,
    utc_now,
)


class AlphaLibraryConfig:
    """
    Configuration for the v5.0.10 Alpha Library.
    """

    def __init__(
        self,
        active_validation_threshold: float = 0.65,
        active_robustness_threshold: float = 0.60,
        decay_warning_threshold: float = 0.50,
        retirement_decay_threshold: float = 0.80,
        weak_robustness_threshold: float = 0.40,
        strong_robustness_threshold: float = 0.70,
    ) -> None:
        values = {
            "active_validation_threshold": active_validation_threshold,
            "active_robustness_threshold": active_robustness_threshold,
            "decay_warning_threshold": decay_warning_threshold,
            "retirement_decay_threshold": retirement_decay_threshold,
            "weak_robustness_threshold": weak_robustness_threshold,
            "strong_robustness_threshold": strong_robustness_threshold,
        }

        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise TypeError(
                    f"{name} must be numeric"
                )

            if value < 0.0 or value > 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )

        if (
            decay_warning_threshold
            >= retirement_decay_threshold
        ):
            raise ValueError(
                "decay_warning_threshold must be lower "
                "than retirement_decay_threshold"
            )

        if (
            weak_robustness_threshold
            >= strong_robustness_threshold
        ):
            raise ValueError(
                "weak_robustness_threshold must be lower "
                "than strong_robustness_threshold"
            )

        self.active_validation_threshold = float(
            active_validation_threshold
        )
        self.active_robustness_threshold = float(
            active_robustness_threshold
        )
        self.decay_warning_threshold = float(
            decay_warning_threshold
        )
        self.retirement_decay_threshold = float(
            retirement_decay_threshold
        )
        self.weak_robustness_threshold = float(
            weak_robustness_threshold
        )
        self.strong_robustness_threshold = float(
            strong_robustness_threshold
        )


class AlphaLibrary:
    """
    Research Control Center v5.0.10 Alpha Library.

    Responsibilities:
    - register alpha candidates
    - manage alpha lifecycle
    - track validation and OOS performance
    - track realized performance
    - measure alpha decay
    - classify robustness
    - activate / decay / retire / reject alpha
    - rank alpha candidates
    - filter by asset, regime, horizon and status
    - preserve complete audit history

    Research-only component.
    No real-money execution is allowed.
    """

    def __init__(
        self,
        config: Optional[AlphaLibraryConfig] = None,
    ) -> None:
        self.config = config or AlphaLibraryConfig()

        self._alphas: Dict[str, AlphaRecord] = {}
        self._audit_log: List[dict] = []

    # =========================================================
    # VALIDATION HELPERS
    # =========================================================

    @staticmethod
    def _validate_required_string(
        value: str,
        name: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                f"{name} must be a string"
            )

        value = value.strip()

        if not value:
            raise ValueError(
                f"{name} cannot be empty"
            )

        return value

    @staticmethod
    def _validate_optional_score(
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
    def _validate_optional_number(
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

        return float(value)

    def _get_internal(
        self,
        alpha_id: str,
    ) -> AlphaRecord:
        alpha_id = self._validate_required_string(
            alpha_id,
            "alpha_id",
        )

        try:
            return self._alphas[alpha_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown alpha_id: {alpha_id}"
            ) from exc

    def _audit(
        self,
        event: str,
        alpha: AlphaRecord,
        details: Optional[dict] = None,
    ) -> None:
        self._audit_log.append(
            {
                "timestamp": utc_now(),
                "event": event,
                "alpha_id": alpha.alpha_id,
                "name": alpha.name,
                "asset": alpha.asset,
                "regime": alpha.regime,
                "horizon": alpha.horizon,
                "status": alpha.status.value,
                "robustness_status": (
                    alpha.robustness_status.value
                ),
                "details": deepcopy(
                    details or {}
                ),
            }
        )

    # =========================================================
    # REGISTRATION
    # =========================================================

    def register(
        self,
        record: AlphaRecord,
    ) -> AlphaRecord:
        if not isinstance(record, AlphaRecord):
            raise TypeError(
                "record must be an AlphaRecord"
            )

        record.name = self._validate_required_string(
            record.name,
            "name",
        )

        record.asset = self._validate_required_string(
            record.asset,
            "asset",
        )

        record.horizon = (
            self._validate_required_string(
                record.horizon,
                "horizon",
            )
        )

        if (
            record.regime is not None
            and not isinstance(record.regime, str)
        ):
            raise TypeError(
                "regime must be a string or None"
            )

        if record.alpha_id in self._alphas:
            raise ValueError(
                f"alpha already registered: "
                f"{record.alpha_id}"
            )

        stored = deepcopy(record)

        stored.metadata = {
            **stored.metadata,
            "library": "v5.0.10-alpha-library",
            "research_only": True,
            "real_money_execution": False,
        }

        self._alphas[
            stored.alpha_id
        ] = stored

        self._audit(
            "ALPHA_REGISTERED",
            stored,
        )

        return deepcopy(stored)

    def create(
        self,
        name: str,
        asset: str,
        horizon: str,
        *,
        regime: Optional[str] = None,
        expected_return: Optional[float] = None,
        source_strategy_id: Optional[str] = None,
        source_hypothesis_id: Optional[str] = None,
        source_experiment_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> AlphaRecord:
        name = self._validate_required_string(
            name,
            "name",
        )

        asset = self._validate_required_string(
            asset,
            "asset",
        )

        horizon = self._validate_required_string(
            horizon,
            "horizon",
        )

        expected_return = (
            self._validate_optional_number(
                expected_return,
                "expected_return",
            )
        )

        record = AlphaRecord(
            name=name,
            asset=asset,
            regime=regime,
            horizon=horizon,
            expected_return=expected_return,
            source_strategy_id=source_strategy_id,
            source_hypothesis_id=source_hypothesis_id,
            source_experiment_id=source_experiment_id,
            metadata=deepcopy(metadata or {}),
        )

        return self.register(record)

    # =========================================================
    # READ
    # =========================================================

    def get(
        self,
        alpha_id: str,
    ) -> AlphaRecord:
        return deepcopy(
            self._get_internal(alpha_id)
        )

    def list_all(self) -> List[AlphaRecord]:
        return [
            deepcopy(alpha)
            for alpha in self._alphas.values()
        ]

    def count(self) -> int:
        return len(self._alphas)

    # =========================================================
    # ROBUSTNESS
    # =========================================================

    def classify_robustness(
        self,
        robustness_score: float,
    ) -> RobustnessStatus:
        score = self._validate_optional_score(
            robustness_score,
            "robustness_score",
        )

        if score is None:
            return RobustnessStatus.UNKNOWN

        if (
            score
            < self.config.weak_robustness_threshold
        ):
            return RobustnessStatus.WEAK

        if (
            score
            >= self.config.strong_robustness_threshold
        ):
            return RobustnessStatus.STRONG

        return RobustnessStatus.MODERATE

    # =========================================================
    # VALIDATION
    # =========================================================

    def update_validation(
        self,
        alpha_id: str,
        *,
        validation_score: float,
        out_of_sample_score: Optional[
            float
        ] = None,
        robustness_score: Optional[
            float
        ] = None,
    ) -> AlphaRecord:
        validation_score = (
            self._validate_optional_score(
                validation_score,
                "validation_score",
            )
        )

        out_of_sample_score = (
            self._validate_optional_score(
                out_of_sample_score,
                "out_of_sample_score",
            )
        )

        robustness_score = (
            self._validate_optional_score(
                robustness_score,
                "robustness_score",
            )
        )

        alpha = self._get_internal(alpha_id)

        alpha.validation_score = (
            validation_score
        )

        alpha.out_of_sample_score = (
            out_of_sample_score
        )

        if robustness_score is not None:
            alpha.robustness_score = (
                robustness_score
            )

            alpha.robustness_status = (
                self.classify_robustness(
                    robustness_score
                )
            )

        alpha.status = AlphaStatus.VALIDATING
        alpha.updated_at = utc_now()

        self._audit(
            "ALPHA_VALIDATION_UPDATED",
            alpha,
            {
                "validation_score": (
                    validation_score
                ),
                "out_of_sample_score": (
                    out_of_sample_score
                ),
                "robustness_score": (
                    robustness_score
                ),
            },
        )

        return deepcopy(alpha)

    # =========================================================
    # PERFORMANCE
    # =========================================================

    def update_performance(
        self,
        alpha_id: str,
        *,
        realized_return: Optional[
            float
        ] = None,
        sharpe_ratio: Optional[
            float
        ] = None,
        sortino_ratio: Optional[
            float
        ] = None,
        max_drawdown: Optional[
            float
        ] = None,
    ) -> AlphaRecord:
        realized_return = (
            self._validate_optional_number(
                realized_return,
                "realized_return",
            )
        )

        sharpe_ratio = (
            self._validate_optional_number(
                sharpe_ratio,
                "sharpe_ratio",
            )
        )

        sortino_ratio = (
            self._validate_optional_number(
                sortino_ratio,
                "sortino_ratio",
            )
        )

        max_drawdown = (
            self._validate_optional_number(
                max_drawdown,
                "max_drawdown",
            )
        )

        alpha = self._get_internal(alpha_id)

        if realized_return is not None:
            alpha.realized_return = (
                realized_return
            )

        if sharpe_ratio is not None:
            alpha.sharpe_ratio = sharpe_ratio

        if sortino_ratio is not None:
            alpha.sortino_ratio = (
                sortino_ratio
            )

        if max_drawdown is not None:
            alpha.max_drawdown = (
                max_drawdown
            )

        alpha.updated_at = utc_now()

        self._audit(
            "ALPHA_PERFORMANCE_UPDATED",
            alpha,
            {
                "realized_return": (
                    realized_return
                ),
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
            },
        )

        return deepcopy(alpha)

    # =========================================================
    # DECAY
    # =========================================================

    def update_decay(
        self,
        alpha_id: str,
        decay_score: float,
    ) -> AlphaRecord:
        decay_score = (
            self._validate_optional_score(
                decay_score,
                "decay_score",
            )
        )

        alpha = self._get_internal(alpha_id)

        alpha.decay_score = decay_score
        alpha.updated_at = utc_now()

        if (
            decay_score
            >= self.config.retirement_decay_threshold
        ):
            alpha.status = AlphaStatus.RETIRED

        elif (
            decay_score
            >= self.config.decay_warning_threshold
        ):
            alpha.status = AlphaStatus.DECAYING

        self._audit(
            "ALPHA_DECAY_UPDATED",
            alpha,
            {
                "decay_score": decay_score,
            },
        )

        return deepcopy(alpha)

    # =========================================================
    # LIFECYCLE
    # =========================================================

    def evaluate_lifecycle(
        self,
        alpha_id: str,
    ) -> AlphaRecord:
        alpha = self._get_internal(alpha_id)

        old_status = alpha.status

        if (
            alpha.status
            in {
                AlphaStatus.RETIRED,
                AlphaStatus.REJECTED,
            }
        ):
            return deepcopy(alpha)

        if (
            alpha.decay_score
            >= self.config.retirement_decay_threshold
        ):
            alpha.status = AlphaStatus.RETIRED

        elif (
            alpha.decay_score
            >= self.config.decay_warning_threshold
        ):
            alpha.status = AlphaStatus.DECAYING

        elif (
            alpha.validation_score is not None
            and alpha.validation_score
            >= self.config.active_validation_threshold
            and alpha.robustness_score
            >= self.config.active_robustness_threshold
        ):
            alpha.status = AlphaStatus.ACTIVE

        elif alpha.validation_score is not None:
            alpha.status = AlphaStatus.VALIDATING

        else:
            alpha.status = AlphaStatus.DISCOVERY

        alpha.updated_at = utc_now()

        self._audit(
            "ALPHA_LIFECYCLE_EVALUATED",
            alpha,
            {
                "old_status": old_status.value,
                "new_status": alpha.status.value,
            },
        )

        return deepcopy(alpha)

    def reject(
        self,
        alpha_id: str,
        reason: str,
    ) -> AlphaRecord:
        reason = self._validate_required_string(
            reason,
            "reason",
        )

        alpha = self._get_internal(alpha_id)

        alpha.status = AlphaStatus.REJECTED
        alpha.updated_at = utc_now()

        self._audit(
            "ALPHA_REJECTED",
            alpha,
            {
                "reason": reason,
            },
        )

        return deepcopy(alpha)

    def retire(
        self,
        alpha_id: str,
        reason: str,
    ) -> AlphaRecord:
        reason = self._validate_required_string(
            reason,
            "reason",
        )

        alpha = self._get_internal(alpha_id)

        alpha.status = AlphaStatus.RETIRED
        alpha.updated_at = utc_now()

        self._audit(
            "ALPHA_RETIRED",
            alpha,
            {
                "reason": reason,
            },
        )

        return deepcopy(alpha)

    # =========================================================
    # FILTERING
    # =========================================================

    def filter(
        self,
        *,
        asset: Optional[str] = None,
        regime: Optional[str] = None,
        horizon: Optional[str] = None,
        status: Optional[AlphaStatus] = None,
        robustness_status: Optional[
            RobustnessStatus
        ] = None,
    ) -> List[AlphaRecord]:
        if (
            status is not None
            and not isinstance(
                status,
                AlphaStatus,
            )
        ):
            raise TypeError(
                "status must be AlphaStatus"
            )

        if (
            robustness_status is not None
            and not isinstance(
                robustness_status,
                RobustnessStatus,
            )
        ):
            raise TypeError(
                "robustness_status must be "
                "RobustnessStatus"
            )

        results = []

        for alpha in self._alphas.values():
            if (
                asset is not None
                and alpha.asset != asset
            ):
                continue

            if (
                regime is not None
                and alpha.regime != regime
            ):
                continue

            if (
                horizon is not None
                and alpha.horizon != horizon
            ):
                continue

            if (
                status is not None
                and alpha.status != status
            ):
                continue

            if (
                robustness_status is not None
                and alpha.robustness_status
                != robustness_status
            ):
                continue

            results.append(
                deepcopy(alpha)
            )

        return results

    # =========================================================
    # RANKING
    # =========================================================

    @staticmethod
    def alpha_quality_score(
        alpha: AlphaRecord,
    ) -> float:
        validation = (
            alpha.validation_score
            if alpha.validation_score is not None
            else 0.0
        )

        oos = (
            alpha.out_of_sample_score
            if alpha.out_of_sample_score is not None
            else 0.0
        )

        robustness = alpha.robustness_score

        decay_penalty = alpha.decay_score

        score = (
            validation * 0.40
            + oos * 0.30
            + robustness * 0.30
        )

        score *= 1.0 - decay_penalty

        return max(
            0.0,
            min(1.0, float(score)),
        )

    def rank(
        self,
        *,
        asset: Optional[str] = None,
        regime: Optional[str] = None,
        horizon: Optional[str] = None,
        status: Optional[AlphaStatus] = None,
    ) -> List[AlphaRecord]:
        alphas = self.filter(
            asset=asset,
            regime=regime,
            horizon=horizon,
            status=status,
        )

        alphas.sort(
            key=self.alpha_quality_score,
            reverse=True,
        )

        return alphas

    def top(
        self,
        limit: int = 10,
        **filters,
    ) -> List[AlphaRecord]:
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
    # SUMMARY
    # =========================================================

    def summary(self) -> dict:
        alphas = self.list_all()

        status_counts = {
            status.value: 0
            for status in AlphaStatus
        }

        robustness_counts = {
            status.value: 0
            for status in RobustnessStatus
        }

        for alpha in alphas:
            status_counts[
                alpha.status.value
            ] += 1

            robustness_counts[
                alpha.robustness_status.value
            ] += 1

        active = [
            alpha
            for alpha in alphas
            if alpha.status
            == AlphaStatus.ACTIVE
        ]

        decaying = [
            alpha
            for alpha in alphas
            if alpha.status
            == AlphaStatus.DECAYING
        ]

        retired = [
            alpha
            for alpha in alphas
            if alpha.status
            == AlphaStatus.RETIRED
        ]

        return {
            "total_alphas": len(alphas),
            "active_alphas": len(active),
            "decaying_alphas": len(decaying),
            "retired_alphas": len(retired),
            "status_counts": status_counts,
            "robustness_counts": (
                robustness_counts
            ),
            "generated_at": utc_now(),
            "research_only": True,
            "real_money_execution": False,
        }

    # =========================================================
    # AUDIT
    # =========================================================

    def audit_log(
        self,
        alpha_id: Optional[str] = None,
    ) -> List[dict]:
        if alpha_id is None:
            return deepcopy(
                self._audit_log
            )

        return [
            deepcopy(event)
            for event in self._audit_log
            if event["alpha_id"] == alpha_id
        ]

    # =========================================================
    # SNAPSHOT
    # =========================================================

    def snapshot(self) -> dict:
        return {
            "alphas": self.list_all(),
            "ranking": self.rank(),
            "summary": self.summary(),
            "generated_at": utc_now(),
            "research_only": True,
            "real_money_execution": False,
        }