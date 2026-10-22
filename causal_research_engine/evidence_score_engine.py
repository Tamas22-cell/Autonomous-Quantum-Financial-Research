"""
Causal Research Engine - Evidence Score Engine
Version: v5.0.5

Combines predictive and causal evidence without treating predictive
performance as proof of causality.

The engine produces a structured CausalEvidenceScore.
"""

from __future__ import annotations

from typing import Iterable, Optional

from causal_research_engine.models import (
    CausalEvidenceScore,
    CausalTestResult,
    EventStudyResult,
    TestOutcome,
)


class EvidenceScoreEngineError(Exception):
    """Base exception for evidence-score errors."""


class EvidenceScoreEngine:
    """
    Calculates a structured causal-evidence score.

    Predictive and causal evidence remain separate.
    """

    def calculate(
        self,
        *,
        predictive_score: float,
        causal_tests: Optional[
            Iterable[CausalTestResult]
        ] = None,
        event_studies: Optional[
            Iterable[EventStudyResult]
        ] = None,
        confounder_score: float = 1.0,
        leakage_score: float = 1.0,
        falsification_score: Optional[float] = None,
        confidence: float = 0.0,
        explanation: Optional[list[str]] = None,
    ) -> CausalEvidenceScore:

        causal_tests = list(causal_tests or [])
        event_studies = list(event_studies or [])

        causal_test_score = self._causal_test_score(
            causal_tests
        )

        event_evidence_score = self._event_evidence_score(
            event_studies
        )

        if falsification_score is None:
            falsification_score = self._falsification_score(
                causal_tests
            )

        causal_components = [
            causal_test_score,
            event_evidence_score,
            self._bound(confounder_score),
            self._bound(leakage_score),
            self._bound(falsification_score),
        ]

        causal_score = sum(
            causal_components
        ) / len(causal_components)

        generated_explanation = list(
            explanation or []
        )

        generated_explanation.append(
            f"Predictive score: "
            f"{self._bound(predictive_score):.3f}"
        )

        generated_explanation.append(
            f"Causal score: "
            f"{causal_score:.3f}"
        )

        generated_explanation.append(
            f"Confounder score: "
            f"{self._bound(confounder_score):.3f}"
        )

        generated_explanation.append(
            f"Leakage score: "
            f"{self._bound(leakage_score):.3f}"
        )

        generated_explanation.append(
            f"Falsification score: "
            f"{self._bound(falsification_score):.3f}"
        )

        return CausalEvidenceScore(
            predictive_score=self._bound(
                predictive_score
            ),
            causal_score=self._bound(
                causal_score
            ),
            confounder_score=self._bound(
                confounder_score
            ),
            leakage_score=self._bound(
                leakage_score
            ),
            falsification_score=self._bound(
                falsification_score
            ),
            event_evidence_score=self._bound(
                event_evidence_score
            ),
            confidence=self._bound(
                confidence
            ),
            explanation=generated_explanation,
        )

    def compare_predictive_vs_causal(
        self,
        score: CausalEvidenceScore,
    ) -> dict[str, float]:
        """
        Return predictive and causal evidence independently.
        """

        return {
            "predictive_score": score.predictive_score,
            "causal_score": score.causal_score,
            "difference": (
                score.predictive_score
                - score.causal_score
            ),
        }

    @staticmethod
    def _causal_test_score(
        tests: Iterable[CausalTestResult],
    ) -> float:

        tests = list(tests)

        if not tests:
            return 0.0

        values: list[float] = []

        for test in tests:
            if test.outcome == TestOutcome.PASSED:
                values.append(
                    max(
                        0.0,
                        min(1.0, test.confidence),
                    )
                    or 1.0
                )

            elif test.outcome == TestOutcome.WARNING:
                values.append(0.5)

            elif test.outcome == TestOutcome.FAILED:
                values.append(0.0)

            else:
                values.append(0.5)

        return sum(values) / len(values)

    @staticmethod
    def _falsification_score(
        tests: Iterable[CausalTestResult],
    ) -> float:

        falsification_tests = [
            test
            for test in tests
            if test.test_name
            and (
                "falsification"
                in test.test_name.casefold()
            )
            or (
                "placebo"
                in test.test_name.casefold()
            )
            or (
                "negative_control"
                in test.test_name.casefold()
            )
        ]

        if not falsification_tests:
            return 1.0

        values: list[float] = []

        for test in falsification_tests:
            if test.outcome == TestOutcome.PASSED:
                values.append(1.0)
            elif test.outcome == TestOutcome.WARNING:
                values.append(0.5)
            elif test.outcome == TestOutcome.FAILED:
                values.append(0.0)
            else:
                values.append(0.5)

        return sum(values) / len(values)

    @staticmethod
    def _event_evidence_score(
        studies: Iterable[EventStudyResult],
    ) -> float:

        studies = list(studies)

        if not studies:
            return 0.0

        values: list[float] = []

        for study in studies:
            confidence = max(
                0.0,
                min(
                    1.0,
                    float(study.confidence),
                ),
            )

            if study.abnormal_effect is None:
                values.append(0.5)
                continue

            if study.p_value is not None:
                if study.p_value < 0.05:
                    values.append(
                        max(
                            confidence,
                            0.5,
                        )
                    )
                else:
                    values.append(0.5)
            else:
                values.append(
                    max(
                        confidence,
                        0.5,
                    )
                )

        return sum(values) / len(values)

    @staticmethod
    def _bound(value: float) -> float:
        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )