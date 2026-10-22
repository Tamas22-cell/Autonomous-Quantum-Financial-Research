"""
Causal Research Engine - Main Orchestrator
Version: v5.0.5

Coordinates the complete causal research workflow:

Hypothesis
    -> Diagnostics
    -> Falsification
    -> Event Study
    -> Evidence Score
    -> Causal Research Report
"""

from __future__ import annotations

from typing import Iterable, Optional

from causal_research_engine.diagnostic_engine import (
    CausalDiagnosticEngine,
)
from causal_research_engine.event_study_engine import (
    EventStudyEngine,
)
from causal_research_engine.evidence_score_engine import (
    EvidenceScoreEngine,
)
from causal_research_engine.falsification_engine import (
    FalsificationEngine,
)
from causal_research_engine.hypothesis_engine import (
    CausalHypothesisEngine,
)
from causal_research_engine.models import (
    CausalEvidenceScore,
    CausalHypothesis,
    CausalResearchReport,
    CausalTestResult,
    ConfounderCheck,
    EventStudyResult,
    LeakageCheck,
)


class CausalEngineError(Exception):
    """Base exception for the causal orchestrator."""


class CausalEngine:
    """
    Main orchestrator for the v5.0.5 causal research pipeline.
    """

    def __init__(self) -> None:
        self.hypothesis_engine = CausalHypothesisEngine()
        self.diagnostic_engine = CausalDiagnosticEngine()
        self.falsification_engine = FalsificationEngine()
        self.event_study_engine = EventStudyEngine()
        self.evidence_score_engine = EvidenceScoreEngine()

    def create_hypothesis(
        self,
        *,
        hypothesis_id: str,
        cause: str,
        effect: str,
        description: str = "",
        expected_direction: Optional[str] = None,
        mechanism: Optional[str] = None,
        confounders: Optional[Iterable[str]] = None,
        controls: Optional[Iterable[str]] = None,
        metadata: Optional[dict] = None,
    ) -> CausalHypothesis:
        """
        Create a structured causal hypothesis.
        """

        return self.hypothesis_engine.create_hypothesis(
            hypothesis_id=hypothesis_id,
            cause=cause,
            effect=effect,
            description=description,
            expected_direction=expected_direction,
            mechanism=mechanism,
            confounders=confounders,
            controls=controls,
            metadata=metadata,
        )

    def run_research(
        self,
        *,
        hypothesis: CausalHypothesis,
        predictive_score: float,
        confounder_checks: Optional[
            Iterable[ConfounderCheck]
        ] = None,
        leakage_checks: Optional[
            Iterable[LeakageCheck]
        ] = None,
        causal_tests: Optional[
            Iterable[CausalTestResult]
        ] = None,
        event_studies: Optional[
            Iterable[EventStudyResult]
        ] = None,
        confidence: float = 0.0,
        notes: Optional[list[str]] = None,
    ) -> CausalResearchReport:
        """
        Run the complete causal research workflow.

        The supplied evidence is evaluated and combined into
        one CausalResearchReport.
        """

        if not isinstance(
            hypothesis,
            CausalHypothesis,
        ):
            raise CausalEngineError(
                "hypothesis must be a CausalHypothesis."
            )

        validation = (
            self.hypothesis_engine.validate_hypothesis(
                hypothesis
            )
        )

        if validation:
            raise CausalEngineError(
                "Invalid hypothesis: "
                + "; ".join(validation)
            )

        confounder_checks = list(
            confounder_checks or []
        )

        leakage_checks = list(
            leakage_checks or []
        )

        causal_tests = list(
            causal_tests or []
        )

        event_studies = list(
            event_studies or []
        )

        normalized_confounders = (
            self.diagnostic_engine.evaluate_confounders(
                confounder_checks
            )
        )

        normalized_leakage = (
            self.diagnostic_engine.evaluate_leakage(
                leakage_checks
            )
        )

        normalized_tests = list(
            causal_tests
        )

        normalized_events = list(
            event_studies
        )

        confounder_score = (
            self.diagnostic_engine.confounder_score(
                normalized_confounders
            )
        )

        leakage_score = (
            self.diagnostic_engine.leakage_score(
                normalized_leakage
            )
        )

        evidence_score = (
            self.evidence_score_engine.calculate(
                predictive_score=predictive_score,
                causal_tests=normalized_tests,
                event_studies=normalized_events,
                confounder_score=confounder_score,
                leakage_score=leakage_score,
                confidence=confidence,
                explanation=list(notes or []),
            )
        )

        warnings = self._build_warnings(
            normalized_confounders,
            normalized_leakage,
            evidence_score,
        )

        return CausalResearchReport(
            hypothesis=hypothesis,
            confounder_checks=normalized_confounders,
            leakage_checks=normalized_leakage,
            causal_tests=normalized_tests,
            event_studies=normalized_events,
            evidence_score=evidence_score,
            warnings=warnings,
        )

    def run_from_raw_event(
        self,
        *,
        hypothesis: CausalHypothesis,
        predictive_score: float,
        event_name: str,
        pre_event_values: Iterable[float],
        post_event_values: Iterable[float],
        confounder_checks: Optional[
            Iterable[ConfounderCheck]
        ] = None,
        leakage_checks: Optional[
            Iterable[LeakageCheck]
        ] = None,
        causal_tests: Optional[
            Iterable[CausalTestResult]
        ] = None,
        benchmark_effect: Optional[float] = None,
        p_value: Optional[float] = None,
        confidence: float = 0.0,
        notes: Optional[list[str]] = None,
    ) -> CausalResearchReport:
        """
        Run the pipeline while calculating the event study internally.
        """

        event_result = (
            self.event_study_engine.analyze(
                event_name=event_name,
                pre_event_values=pre_event_values,
                post_event_values=post_event_values,
                benchmark_effect=benchmark_effect,
                p_value=p_value,
                confidence=confidence,
            )
        )

        return self.run_research(
            hypothesis=hypothesis,
            predictive_score=predictive_score,
            confounder_checks=confounder_checks,
            leakage_checks=leakage_checks,
            causal_tests=causal_tests,
            event_studies=[event_result],
            confidence=confidence,
            notes=notes,
        )

    def start_testing(
        self,
        hypothesis: CausalHypothesis,
    ) -> CausalHypothesis:
        """
        Move a validated hypothesis into TESTING status.
        """

        try:
            return self.hypothesis_engine.start_testing(
                hypothesis
            )
        except Exception as exc:
            raise CausalEngineError(
                str(exc)
            ) from exc

    def diagnostic_summary(
        self,
        *,
        confounder_checks: Iterable[ConfounderCheck],
        leakage_checks: Iterable[LeakageCheck],
    ) -> dict:
        """
        Return the diagnostic-engine summary.
        """

        return self.diagnostic_engine.diagnostic_summary(
            confounder_checks,
            leakage_checks,
        )

    @staticmethod
    def _build_warnings(
        confounders: list[ConfounderCheck],
        leakage: list[LeakageCheck],
        evidence_score: CausalEvidenceScore,
    ) -> list[str]:
        warnings: list[str] = []

        if evidence_score.confounder_score < 0.50:
            warnings.append(
                "Material confounding risk detected."
            )

        if evidence_score.leakage_score < 0.50:
            warnings.append(
                "Material information or temporal leakage risk detected."
            )

        if evidence_score.falsification_score < 0.50:
            warnings.append(
                "Falsification diagnostics provide weak causal support."
            )

        if not confounders:
            warnings.append(
                "No confounder checks were supplied."
            )

        if not leakage:
            warnings.append(
                "No leakage checks were supplied."
            )

        return warnings