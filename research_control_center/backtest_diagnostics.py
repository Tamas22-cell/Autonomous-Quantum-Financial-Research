from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Dict, List, Optional

from research_control_center.models import (
    BacktestDiagnostic,
    utc_now,
)


class BacktestDiagnosticsConfig:
    """
    Configuration for Research Control Center v5.0.10
    Backtest / OOS Statistical Diagnostics.
    """

    def __init__(
        self,
        max_sharpe_degradation: float = 0.40,
        max_return_degradation: float = 0.50,
        minimum_robustness_score: float = 0.60,
        significance_level: float = 0.05,
        minimum_observations: int = 30,
    ) -> None:
        score_values = {
            "max_sharpe_degradation": max_sharpe_degradation,
            "max_return_degradation": max_return_degradation,
            "minimum_robustness_score": minimum_robustness_score,
            "significance_level": significance_level,
        }

        for name, value in score_values.items():
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

        if isinstance(minimum_observations, bool):
            raise TypeError(
                "minimum_observations must be an integer"
            )

        if not isinstance(minimum_observations, int):
            raise TypeError(
                "minimum_observations must be an integer"
            )

        if minimum_observations <= 0:
            raise ValueError(
                "minimum_observations must be positive"
            )

        self.max_sharpe_degradation = float(
            max_sharpe_degradation
        )
        self.max_return_degradation = float(
            max_return_degradation
        )
        self.minimum_robustness_score = float(
            minimum_robustness_score
        )
        self.significance_level = float(
            significance_level
        )
        self.minimum_observations = (
            minimum_observations
        )


class BacktestDiagnosticsEngine:
    """
    v5.0.10 Backtest / Out-of-Sample Diagnostics Engine.

    Responsibilities:
    - store strategy diagnostics
    - compare IS vs OOS performance
    - calculate return degradation
    - calculate Sharpe degradation
    - calculate generalization score
    - classify robustness
    - flag potential overfitting
    - evaluate statistical significance
    - compare multiple strategies
    - preserve audit history

    Research-only component.
    It does not authorize real-money execution.
    """

    def __init__(
        self,
        config: Optional[
            BacktestDiagnosticsConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or BacktestDiagnosticsConfig()
        )

        self._diagnostics: Dict[
            str,
            BacktestDiagnostic,
        ] = {}

        self._audit_log: List[dict] = []

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def _validate_strategy_id(
        strategy_id: str,
    ) -> str:
        if not isinstance(strategy_id, str):
            raise TypeError(
                "strategy_id must be a string"
            )

        strategy_id = strategy_id.strip()

        if not strategy_id:
            raise ValueError(
                "strategy_id cannot be empty"
            )

        return strategy_id

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

        value = float(value)

        if not isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

        return value

    @staticmethod
    def _validate_optional_unit_score(
        value: Optional[float],
        name: str,
    ) -> Optional[float]:
        value = (
            BacktestDiagnosticsEngine
            ._validate_optional_number(
                value,
                name,
            )
        )

        if value is None:
            return None

        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1"
            )

        return value

    @staticmethod
    def _validate_observations(
        observations: int,
    ) -> int:
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

        return observations

    def _get_internal(
        self,
        diagnostic_id: str,
    ) -> BacktestDiagnostic:
        if not isinstance(diagnostic_id, str):
            raise TypeError(
                "diagnostic_id must be a string"
            )

        diagnostic_id = diagnostic_id.strip()

        if not diagnostic_id:
            raise ValueError(
                "diagnostic_id cannot be empty"
            )

        try:
            return self._diagnostics[
                diagnostic_id
            ]
        except KeyError as exc:
            raise KeyError(
                f"unknown diagnostic_id: "
                f"{diagnostic_id}"
            ) from exc

    def _audit(
        self,
        event: str,
        diagnostic: BacktestDiagnostic,
        details: Optional[dict] = None,
    ) -> None:
        self._audit_log.append(
            {
                "timestamp": utc_now(),
                "event": event,
                "diagnostic_id": (
                    diagnostic.diagnostic_id
                ),
                "strategy_id": (
                    diagnostic.strategy_id
                ),
                "details": deepcopy(
                    details or {}
                ),
            }
        )

    # =========================================================
    # DEGRADATION
    # =========================================================

    @staticmethod
    def degradation(
        in_sample: Optional[float],
        out_of_sample: Optional[float],
    ) -> Optional[float]:
        if (
            in_sample is None
            or out_of_sample is None
        ):
            return None

        in_sample = float(in_sample)
        out_of_sample = float(out_of_sample)

        if in_sample == 0.0:
            if out_of_sample == 0.0:
                return 0.0

            return 1.0

        result = (
            in_sample - out_of_sample
        ) / abs(in_sample)

        return max(
            0.0,
            float(result),
        )

    # =========================================================
    # GENERALIZATION
    # =========================================================

    def generalization_score(
        self,
        diagnostic: BacktestDiagnostic,
    ) -> float:
        return_degradation = self.degradation(
            diagnostic.in_sample_return,
            diagnostic.out_of_sample_return,
        )

        sharpe_degradation = self.degradation(
            diagnostic.in_sample_sharpe,
            diagnostic.out_of_sample_sharpe,
        )

        components: List[float] = []

        if return_degradation is not None:
            components.append(
                max(
                    0.0,
                    1.0 - return_degradation,
                )
            )

        if sharpe_degradation is not None:
            components.append(
                max(
                    0.0,
                    1.0 - sharpe_degradation,
                )
            )

        if diagnostic.robustness_score is not None:
            components.append(
                diagnostic.robustness_score
            )

        if not components:
            return 0.0

        score = sum(components) / len(
            components
        )

        return max(
            0.0,
            min(1.0, float(score)),
        )

    # =========================================================
    # REGISTRATION
    # =========================================================

    def register(
        self,
        diagnostic: BacktestDiagnostic,
    ) -> BacktestDiagnostic:
        if not isinstance(
            diagnostic,
            BacktestDiagnostic,
        ):
            raise TypeError(
                "diagnostic must be "
                "BacktestDiagnostic"
            )

        diagnostic.strategy_id = (
            self._validate_strategy_id(
                diagnostic.strategy_id
            )
        )

        diagnostic.observations = (
            self._validate_observations(
                diagnostic.observations
            )
        )

        diagnostic.robustness_score = (
            self._validate_optional_unit_score(
                diagnostic.robustness_score,
                "robustness_score",
            )
        )

        diagnostic.p_value = (
            self._validate_optional_unit_score(
                diagnostic.p_value,
                "p_value",
            )
        )

        if (
            diagnostic.diagnostic_id
            in self._diagnostics
        ):
            raise ValueError(
                "diagnostic already registered: "
                f"{diagnostic.diagnostic_id}"
            )

        stored = deepcopy(diagnostic)

        stored.metadata = {
            **stored.metadata,
            "engine": (
                "v5.0.10-backtest-diagnostics"
            ),
            "research_only": True,
            "real_money_execution": False,
        }

        self._diagnostics[
            stored.diagnostic_id
        ] = stored

        self._audit(
            "DIAGNOSTIC_REGISTERED",
            stored,
        )

        return deepcopy(stored)

    def create(
        self,
        strategy_id: str,
        *,
        in_sample_return: Optional[
            float
        ] = None,
        out_of_sample_return: Optional[
            float
        ] = None,
        in_sample_sharpe: Optional[
            float
        ] = None,
        out_of_sample_sharpe: Optional[
            float
        ] = None,
        in_sample_drawdown: Optional[
            float
        ] = None,
        out_of_sample_drawdown: Optional[
            float
        ] = None,
        robustness_score: Optional[
            float
        ] = None,
        p_value: Optional[
            float
        ] = None,
        confidence_interval_low: Optional[
            float
        ] = None,
        confidence_interval_high: Optional[
            float
        ] = None,
        observations: int = 0,
        metadata: Optional[dict] = None,
    ) -> BacktestDiagnostic:
        strategy_id = (
            self._validate_strategy_id(
                strategy_id
            )
        )

        numeric_values = {
            "in_sample_return": in_sample_return,
            "out_of_sample_return": (
                out_of_sample_return
            ),
            "in_sample_sharpe": in_sample_sharpe,
            "out_of_sample_sharpe": (
                out_of_sample_sharpe
            ),
            "in_sample_drawdown": (
                in_sample_drawdown
            ),
            "out_of_sample_drawdown": (
                out_of_sample_drawdown
            ),
            "confidence_interval_low": (
                confidence_interval_low
            ),
            "confidence_interval_high": (
                confidence_interval_high
            ),
        }

        validated = {}

        for name, value in numeric_values.items():
            validated[name] = (
                self._validate_optional_number(
                    value,
                    name,
                )
            )

        robustness_score = (
            self._validate_optional_unit_score(
                robustness_score,
                "robustness_score",
            )
        )

        p_value = (
            self._validate_optional_unit_score(
                p_value,
                "p_value",
            )
        )

        observations = (
            self._validate_observations(
                observations
            )
        )

        low = validated[
            "confidence_interval_low"
        ]

        high = validated[
            "confidence_interval_high"
        ]

        if (
            low is not None
            and high is not None
            and low > high
        ):
            raise ValueError(
                "confidence_interval_low cannot "
                "exceed confidence_interval_high"
            )

        diagnostic = BacktestDiagnostic(
            strategy_id=strategy_id,
            in_sample_return=validated[
                "in_sample_return"
            ],
            out_of_sample_return=validated[
                "out_of_sample_return"
            ],
            in_sample_sharpe=validated[
                "in_sample_sharpe"
            ],
            out_of_sample_sharpe=validated[
                "out_of_sample_sharpe"
            ],
            in_sample_drawdown=validated[
                "in_sample_drawdown"
            ],
            out_of_sample_drawdown=validated[
                "out_of_sample_drawdown"
            ],
            robustness_score=robustness_score,
            p_value=p_value,
            confidence_interval_low=low,
            confidence_interval_high=high,
            observations=observations,
            metadata=deepcopy(metadata or {}),
        )

        return self.register(diagnostic)

    # =========================================================
    # READ
    # =========================================================

    def get(
        self,
        diagnostic_id: str,
    ) -> BacktestDiagnostic:
        return deepcopy(
            self._get_internal(
                diagnostic_id
            )
        )

    def list_all(
        self,
    ) -> List[BacktestDiagnostic]:
        return [
            deepcopy(diagnostic)
            for diagnostic
            in self._diagnostics.values()
        ]

    def count(self) -> int:
        return len(self._diagnostics)

    # =========================================================
    # STATISTICAL DIAGNOSTICS
    # =========================================================

    def is_statistically_significant(
        self,
        diagnostic: BacktestDiagnostic,
    ) -> bool:
        if diagnostic.p_value is None:
            return False

        if (
            diagnostic.observations
            < self.config.minimum_observations
        ):
            return False

        return (
            diagnostic.p_value
            <= self.config.significance_level
        )

    def has_positive_confidence_interval(
        self,
        diagnostic: BacktestDiagnostic,
    ) -> bool:
        if (
            diagnostic.confidence_interval_low
            is None
            or diagnostic.confidence_interval_high
            is None
        ):
            return False

        return (
            diagnostic.confidence_interval_low
            > 0.0
        )

    def potential_overfitting(
        self,
        diagnostic: BacktestDiagnostic,
    ) -> bool:
        return_degradation = self.degradation(
            diagnostic.in_sample_return,
            diagnostic.out_of_sample_return,
        )

        sharpe_degradation = self.degradation(
            diagnostic.in_sample_sharpe,
            diagnostic.out_of_sample_sharpe,
        )

        if (
            return_degradation is not None
            and return_degradation
            > self.config.max_return_degradation
        ):
            return True

        if (
            sharpe_degradation is not None
            and sharpe_degradation
            > self.config.max_sharpe_degradation
        ):
            return True

        if (
            diagnostic.robustness_score is not None
            and diagnostic.robustness_score
            < self.config.minimum_robustness_score
        ):
            return True

        return False

    # =========================================================
    # REPORT
    # =========================================================

    def diagnostic_report(
        self,
        diagnostic_id: str,
    ) -> dict:
        diagnostic = self._get_internal(
            diagnostic_id
        )

        return_degradation = self.degradation(
            diagnostic.in_sample_return,
            diagnostic.out_of_sample_return,
        )

        sharpe_degradation = self.degradation(
            diagnostic.in_sample_sharpe,
            diagnostic.out_of_sample_sharpe,
        )

        return {
            "diagnostic_id": (
                diagnostic.diagnostic_id
            ),
            "strategy_id": (
                diagnostic.strategy_id
            ),
            "in_sample_return": (
                diagnostic.in_sample_return
            ),
            "out_of_sample_return": (
                diagnostic.out_of_sample_return
            ),
            "return_degradation": (
                return_degradation
            ),
            "in_sample_sharpe": (
                diagnostic.in_sample_sharpe
            ),
            "out_of_sample_sharpe": (
                diagnostic.out_of_sample_sharpe
            ),
            "sharpe_degradation": (
                sharpe_degradation
            ),
            "in_sample_drawdown": (
                diagnostic.in_sample_drawdown
            ),
            "out_of_sample_drawdown": (
                diagnostic.out_of_sample_drawdown
            ),
            "robustness_score": (
                diagnostic.robustness_score
            ),
            "generalization_score": (
                self.generalization_score(
                    diagnostic
                )
            ),
            "p_value": diagnostic.p_value,
            "statistically_significant": (
                self.is_statistically_significant(
                    diagnostic
                )
            ),
            "positive_confidence_interval": (
                self.has_positive_confidence_interval(
                    diagnostic
                )
            ),
            "potential_overfitting": (
                self.potential_overfitting(
                    diagnostic
                )
            ),
            "observations": (
                diagnostic.observations
            ),
            "research_only": True,
            "real_money_execution": False,
        }

    # =========================================================
    # STRATEGY FILTER
    # =========================================================

    def by_strategy(
        self,
        strategy_id: str,
    ) -> List[BacktestDiagnostic]:
        strategy_id = (
            self._validate_strategy_id(
                strategy_id
            )
        )

        return [
            deepcopy(diagnostic)
            for diagnostic
            in self._diagnostics.values()
            if diagnostic.strategy_id
            == strategy_id
        ]

    # =========================================================
    # RANKING
    # =========================================================

    def rank(
        self,
    ) -> List[BacktestDiagnostic]:
        diagnostics = self.list_all()

        diagnostics.sort(
            key=self.generalization_score,
            reverse=True,
        )

        return diagnostics

    def top(
        self,
        limit: int = 10,
    ) -> List[BacktestDiagnostic]:
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

        return self.rank()[:limit]

    # =========================================================
    # COMPARISON
    # =========================================================

    def compare(
        self,
        diagnostic_id_a: str,
        diagnostic_id_b: str,
    ) -> dict:
        a = self._get_internal(
            diagnostic_id_a
        )

        b = self._get_internal(
            diagnostic_id_b
        )

        score_a = self.generalization_score(
            a
        )

        score_b = self.generalization_score(
            b
        )

        return {
            "diagnostic_a": (
                a.diagnostic_id
            ),
            "diagnostic_b": (
                b.diagnostic_id
            ),
            "strategy_a": a.strategy_id,
            "strategy_b": b.strategy_id,
            "generalization_score_a": (
                score_a
            ),
            "generalization_score_b": (
                score_b
            ),
            "generalization_delta_b_minus_a": (
                score_b - score_a
            ),
            "overfitting_a": (
                self.potential_overfitting(a)
            ),
            "overfitting_b": (
                self.potential_overfitting(b)
            ),
            "significant_a": (
                self.is_statistically_significant(
                    a
                )
            ),
            "significant_b": (
                self.is_statistically_significant(
                    b
                )
            ),
            "research_only": True,
            "real_money_execution": False,
        }

    # =========================================================
    # SUMMARY
    # =========================================================

    def summary(self) -> dict:
        diagnostics = self.list_all()

        overfit_count = sum(
            1
            for diagnostic in diagnostics
            if self.potential_overfitting(
                diagnostic
            )
        )

        significant_count = sum(
            1
            for diagnostic in diagnostics
            if self.is_statistically_significant(
                diagnostic
            )
        )

        robust_count = sum(
            1
            for diagnostic in diagnostics
            if (
                diagnostic.robustness_score
                is not None
                and diagnostic.robustness_score
                >= self.config.minimum_robustness_score
            )
        )

        if diagnostics:
            average_generalization = (
                sum(
                    self.generalization_score(
                        diagnostic
                    )
                    for diagnostic
                    in diagnostics
                )
                / len(diagnostics)
            )
        else:
            average_generalization = 0.0

        return {
            "total_diagnostics": (
                len(diagnostics)
            ),
            "potential_overfitting": (
                overfit_count
            ),
            "statistically_significant": (
                significant_count
            ),
            "robust_diagnostics": (
                robust_count
            ),
            "average_generalization_score": (
                average_generalization
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
        diagnostic_id: Optional[
            str
        ] = None,
    ) -> List[dict]:
        if diagnostic_id is None:
            return deepcopy(
                self._audit_log
            )

        return [
            deepcopy(event)
            for event in self._audit_log
            if event["diagnostic_id"]
            == diagnostic_id
        ]

    # =========================================================
    # SNAPSHOT
    # =========================================================

    def snapshot(self) -> dict:
        return {
            "diagnostics": self.list_all(),
            "ranking": self.rank(),
            "summary": self.summary(),
            "generated_at": utc_now(),
            "research_only": True,
            "real_money_execution": False,
        }