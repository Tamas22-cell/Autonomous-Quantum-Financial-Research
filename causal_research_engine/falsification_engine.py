"""
Causal Research Engine - Falsification Engine
Version: v5.0.5

Provides placebo, falsification and negative-control diagnostics.

The engine does not declare causality by itself.
It produces structured causal test results that can later be
combined by the evidence scoring layer.
"""

from __future__ import annotations

from typing import Optional

from causal_research_engine.models import (
    CausalTestResult,
    EvidenceType,
    TestOutcome,
)


class FalsificationEngineError(Exception):
    """Base exception for falsification-engine errors."""


class FalsificationEngine:
    """
    Executes structured falsification diagnostics.

    Supported tests:
    - placebo
    - falsification
    - negative control
    """

    def run_placebo_test(
        self,
        *,
        test_name: str = "placebo_test",
        statistic: Optional[float] = None,
        p_value: Optional[float] = None,
        effect_size: Optional[float] = None,
        confidence: float = 0.0,
        notes: str = "",
    ) -> CausalTestResult:
        """
        Evaluate a placebo test.

        A placebo test is considered passed when the placebo effect
        is statistically weak or absent.
        """

        outcome = self._evaluate_placebo(
            p_value=p_value,
            effect_size=effect_size,
        )

        return CausalTestResult(
            test_name=test_name,
            evidence_type=EvidenceType.PLACEBO,
            outcome=outcome,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence=confidence,
            notes=notes,
        )

    def run_falsification_test(
        self,
        *,
        test_name: str = "falsification_test",
        statistic: Optional[float] = None,
        p_value: Optional[float] = None,
        effect_size: Optional[float] = None,
        confidence: float = 0.0,
        notes: str = "",
    ) -> CausalTestResult:
        """
        Evaluate a falsification test.

        A falsification test passes when the proposed causal
        relationship fails to appear where it should not.
        """

        outcome = self._evaluate_falsification(
            p_value=p_value,
            effect_size=effect_size,
        )

        return CausalTestResult(
            test_name=test_name,
            evidence_type=EvidenceType.FALSIFICATION,
            outcome=outcome,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence=confidence,
            notes=notes,
        )

    def run_negative_control_test(
        self,
        *,
        test_name: str = "negative_control_test",
        statistic: Optional[float] = None,
        p_value: Optional[float] = None,
        effect_size: Optional[float] = None,
        confidence: float = 0.0,
        notes: str = "",
    ) -> CausalTestResult:
        """
        Evaluate a negative-control test.

        A negative control should not exhibit the hypothesized
        causal effect.
        """

        outcome = self._evaluate_negative_control(
            p_value=p_value,
            effect_size=effect_size,
        )

        return CausalTestResult(
            test_name=test_name,
            evidence_type=EvidenceType.NEGATIVE_CONTROL,
            outcome=outcome,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence=confidence,
            notes=notes,
        )

    def run_all(
        self,
        *,
        placebo: bool = True,
        falsification: bool = True,
        negative_control: bool = True,
    ) -> list[CausalTestResult]:
        """
        Create a structured collection of requested test slots.

        Tests without supplied statistics are marked inconclusive.
        """

        results: list[CausalTestResult] = []

        if placebo:
            results.append(
                self.run_placebo_test()
            )

        if falsification:
            results.append(
                self.run_falsification_test()
            )

        if negative_control:
            results.append(
                self.run_negative_control_test()
            )

        return results

    @staticmethod
    def _evaluate_placebo(
        *,
        p_value: Optional[float],
        effect_size: Optional[float],
    ) -> TestOutcome:
        if p_value is None and effect_size is None:
            return TestOutcome.INCONCLUSIVE

        if p_value is not None and p_value < 0.05:
            return TestOutcome.FAILED

        if (
            effect_size is not None
            and abs(effect_size) > 0.10
        ):
            return TestOutcome.FAILED

        return TestOutcome.PASSED

    @staticmethod
    def _evaluate_falsification(
        *,
        p_value: Optional[float],
        effect_size: Optional[float],
    ) -> TestOutcome:
        if p_value is None and effect_size is None:
            return TestOutcome.INCONCLUSIVE

        if p_value is not None and p_value < 0.05:
            return TestOutcome.FAILED

        if (
            effect_size is not None
            and abs(effect_size) > 0.10
        ):
            return TestOutcome.FAILED

        return TestOutcome.PASSED

    @staticmethod
    def _evaluate_negative_control(
        *,
        p_value: Optional[float],
        effect_size: Optional[float],
    ) -> TestOutcome:
        if p_value is None and effect_size is None:
            return TestOutcome.INCONCLUSIVE

        if p_value is not None and p_value < 0.05:
            return TestOutcome.FAILED

        if (
            effect_size is not None
            and abs(effect_size) > 0.10
        ):
            return TestOutcome.FAILED

        return TestOutcome.PASSED