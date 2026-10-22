"""
Causal Research Engine - Diagnostic Engine
Version: v5.0.5

Provides structured diagnostics for:
- confounders
- information leakage
- temporal leakage
- causal research readiness

The engine keeps diagnostic evidence separate from predictive evidence.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

from causal_research_engine.models import (
    ConfounderCheck,
    LeakageCheck,
    TestOutcome,
)


class DiagnosticEngineError(Exception):
    """Base exception for diagnostic-engine errors."""


class CausalDiagnosticEngine:
    """
    Performs causal diagnostic checks.

    Responsibilities:
    - evaluate confounder checks
    - evaluate leakage checks
    - calculate diagnostic quality scores
    - identify critical causal risks
    """

    def evaluate_confounders(
        self,
        checks: Iterable[ConfounderCheck],
    ) -> list[ConfounderCheck]:
        """
        Normalize and return confounder diagnostics.
        """

        normalized: list[ConfounderCheck] = []

        for check in checks:
            if not isinstance(
                check,
                ConfounderCheck,
            ):
                raise DiagnosticEngineError(
                    "All confounder checks must be "
                    "ConfounderCheck instances."
                )

            normalized.append(
                deepcopy(check)
            )

        return normalized

    def evaluate_leakage(
        self,
        checks: Iterable[LeakageCheck],
    ) -> list[LeakageCheck]:
        """
        Normalize and return leakage diagnostics.
        """

        normalized: list[LeakageCheck] = []

        for check in checks:
            if not isinstance(
                check,
                LeakageCheck,
            ):
                raise DiagnosticEngineError(
                    "All leakage checks must be "
                    "LeakageCheck instances."
                )

            normalized.append(
                deepcopy(check)
            )

        return normalized

    def confounder_score(
        self,
        checks: Iterable[ConfounderCheck],
    ) -> float:
        """
        Calculate a confounder-quality score.

        1.0 = no detected confounding risk.
        0.0 = maximum detected confounding risk.
        """

        checks = list(checks)

        if not checks:
            return 1.0

        severities = [
            max(
                0.0,
                min(
                    1.0,
                    float(check.severity),
                ),
            )
            for check in checks
        ]

        return max(
            0.0,
            min(
                1.0,
                1.0 - max(severities),
            ),
        )

    def leakage_score(
        self,
        checks: Iterable[LeakageCheck],
    ) -> float:
        """
        Calculate a leakage-quality score.

        1.0 = no leakage risk.
        0.0 = maximum leakage risk.
        """

        checks = list(checks)

        if not checks:
            return 1.0

        risks = []

        for check in checks:
            severity = max(
                0.0,
                min(
                    1.0,
                    float(check.severity),
                ),
            )

            if check.leakage_detected:
                risks.append(severity)
            else:
                risks.append(0.0)

        return max(
            0.0,
            min(
                1.0,
                1.0 - max(risks),
            ),
        )

    def has_critical_confounder(
        self,
        checks: Iterable[ConfounderCheck],
        threshold: float = 0.80,
    ) -> bool:
        """
        Return True when a confounder exceeds the critical threshold.
        """

        threshold = max(
            0.0,
            min(1.0, float(threshold)),
        )

        return any(
            float(check.severity) >= threshold
            for check in checks
        )

    def has_critical_leakage(
        self,
        checks: Iterable[LeakageCheck],
        threshold: float = 0.80,
    ) -> bool:
        """
        Return True when leakage risk exceeds the critical threshold.
        """

        threshold = max(
            0.0,
            min(1.0, float(threshold)),
        )

        return any(
            check.leakage_detected
            and float(check.severity) >= threshold
            for check in checks
        )

    def diagnostic_summary(
        self,
        confounders: Iterable[ConfounderCheck],
        leakage: Iterable[LeakageCheck],
    ) -> dict[str, object]:
        """
        Produce an auditable diagnostic summary.
        """

        confounders = list(confounders)
        leakage = list(leakage)

        confounder_score = self.confounder_score(
            confounders
        )

        leakage_score = self.leakage_score(
            leakage
        )

        return {
            "confounder_score": confounder_score,
            "leakage_score": leakage_score,
            "critical_confounder": self.has_critical_confounder(
                confounders
            ),
            "critical_leakage": self.has_critical_leakage(
                leakage
            ),
            "confounder_count": len(confounders),
            "leakage_check_count": len(leakage),
            "leakage_detected_count": sum(
                1
                for check in leakage
                if check.leakage_detected
            ),
        }

    def is_causally_clean(
        self,
        confounders: Iterable[ConfounderCheck],
        leakage: Iterable[LeakageCheck],
        threshold: float = 0.80,
    ) -> bool:
        """
        Return True when no critical confounding or leakage
        risk is detected.
        """

        confounders = list(confounders)
        leakage = list(leakage)

        return not (
            self.has_critical_confounder(
                confounders,
                threshold,
            )
            or self.has_critical_leakage(
                leakage,
                threshold,
            )
        )