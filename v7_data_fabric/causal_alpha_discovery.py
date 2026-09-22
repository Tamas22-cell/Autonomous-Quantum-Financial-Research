from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Dict, List, Optional, Tuple


class CausalAlphaDiscoveryError(ValueError):
    """Causal research and alpha discovery error."""


class HypothesisStatus(str, Enum):
    PROPOSED = "PROPOSED"
    UNDER_REVIEW = "UNDER_REVIEW"
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    REJECTED = "REJECTED"


class EvidenceClass(str, Enum):
    CAUSAL = "CAUSAL"
    PREDICTIVE = "PREDICTIVE"
    DESCRIPTIVE = "DESCRIPTIVE"
    UNKNOWN = "UNKNOWN"


class CandidateStatus(str, Enum):
    PROPOSED = "PROPOSED"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    VALIDATION_REQUIRED = "VALIDATION_REQUIRED"
    REJECTED = "REJECTED"


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CausalAlphaDiscoveryError(
            f"{name} must be a non-empty string."
        )


def _require_score(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise CausalAlphaDiscoveryError(
            f"{name} must be a finite number between 0 and 1."
        )


@dataclass(frozen=True)
class CausalHypothesis:
    hypothesis_id: str
    statement: str
    status: HypothesisStatus = (
        HypothesisStatus.PROPOSED
    )
    research_question: str = ""
    target_variable: str = ""
    treatment_variable: str = ""
    controls: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _require_text(
            self.hypothesis_id,
            "hypothesis_id",
        )
        _require_text(self.statement, "statement")

        if not isinstance(
            self.status,
            HypothesisStatus,
        ):
            raise CausalAlphaDiscoveryError(
                "status must be a HypothesisStatus."
            )

        if not isinstance(self.controls, tuple):
            raise CausalAlphaDiscoveryError(
                "controls must be a tuple."
            )

        for control in self.controls:
            _require_text(control, "control")

        if not isinstance(self.metadata, dict):
            raise CausalAlphaDiscoveryError(
                "metadata must be a dictionary."
            )


@dataclass(frozen=True)
class MechanismMapping:
    mapping_id: str
    hypothesis_id: str
    mechanism: str
    intervention: str
    expected_effect: str
    market_variables: Tuple[str, ...] = ()
    assumptions: Tuple[str, ...] = ()
    confidence: float = 0.0

    def __post_init__(self) -> None:
        _require_text(self.mapping_id, "mapping_id")
        _require_text(
            self.hypothesis_id,
            "hypothesis_id",
        )
        _require_text(self.mechanism, "mechanism")
        _require_text(
            self.intervention,
            "intervention",
        )
        _require_text(
            self.expected_effect,
            "expected_effect",
        )
        _require_score(
            self.confidence,
            "confidence",
        )

        for name, values in (
            ("market_variables", self.market_variables),
            ("assumptions", self.assumptions),
        ):
            if not isinstance(values, tuple):
                raise CausalAlphaDiscoveryError(
                    f"{name} must be a tuple."
                )

            for value in values:
                _require_text(value, name[:-1])


@dataclass(frozen=True)
class AlphaCandidate:
    candidate_id: str
    hypothesis_id: str
    name: str
    signal_definition: str
    causal_rationale: str
    predictive_rationale: str = ""
    status: CandidateStatus = (
        CandidateStatus.PROPOSED
    )
    causal_evidence_score: float = 0.0
    predictive_evidence_score: float = 0.0
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        _require_text(
            self.hypothesis_id,
            "hypothesis_id",
        )
        _require_text(self.name, "name")
        _require_text(
            self.signal_definition,
            "signal_definition",
        )
        _require_text(
            self.causal_rationale,
            "causal_rationale",
        )

        if not isinstance(
            self.status,
            CandidateStatus,
        ):
            raise CausalAlphaDiscoveryError(
                "status must be a CandidateStatus."
            )

        _require_score(
            self.causal_evidence_score,
            "causal_evidence_score",
        )
        _require_score(
            self.predictive_evidence_score,
            "predictive_evidence_score",
        )

        if not isinstance(self.metadata, dict):
            raise CausalAlphaDiscoveryError(
                "metadata must be a dictionary."
            )


@dataclass(frozen=True)
class EvidenceAssessment:
    evidence_id: str
    evidence_class: EvidenceClass
    summary: str
    score: float
    supports_hypothesis: bool
    limitations: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.evidence_id, "evidence_id")
        _require_text(self.summary, "summary")

        if not isinstance(
            self.evidence_class,
            EvidenceClass,
        ):
            raise CausalAlphaDiscoveryError(
                "evidence_class must be an EvidenceClass."
            )

        _require_score(self.score, "score")

        if not isinstance(
            self.supports_hypothesis,
            bool,
        ):
            raise CausalAlphaDiscoveryError(
                "supports_hypothesis must be boolean."
            )

        if not isinstance(self.limitations, tuple):
            raise CausalAlphaDiscoveryError(
                "limitations must be a tuple."
            )


class CausalAlphaDiscovery:
    """
    v7.0.7 causal research and alpha discovery facade.

    This component separates causal evidence from predictive
    performance. It creates research candidates only; it does
    not approve alpha and does not execute real-money trades.
    """

    VERSION = "v7.0.7"
    RESEARCH_ONLY = True
    REAL_MONEY_EXECUTION = False
    AUTOMATIC_ALPHA_APPROVAL = False

    def __init__(self) -> None:
        self._hypotheses: Dict[
            str,
            CausalHypothesis,
        ] = {}
        self._mappings: Dict[
            str,
            MechanismMapping,
        ] = {}
        self._candidates: Dict[
            str,
            AlphaCandidate,
        ] = {}
        self._assessments: Dict[
            str,
            EvidenceAssessment,
        ] = {}

    @property
    def hypothesis_count(self) -> int:
        return len(self._hypotheses)

    @property
    def mapping_count(self) -> int:
        return len(self._mappings)

    @property
    def candidate_count(self) -> int:
        return len(self._candidates)

    @property
    def assessment_count(self) -> int:
        return len(self._assessments)

    def create_hypothesis(
        self,
        hypothesis: CausalHypothesis,
    ) -> CausalHypothesis:
        if not isinstance(
            hypothesis,
            CausalHypothesis,
        ):
            raise CausalAlphaDiscoveryError(
                "hypothesis must be a CausalHypothesis."
            )

        if hypothesis.hypothesis_id in self._hypotheses:
            raise CausalAlphaDiscoveryError(
                "Duplicate hypothesis ID: "
                f"{hypothesis.hypothesis_id}"
            )

        self._hypotheses[
            hypothesis.hypothesis_id
        ] = hypothesis

        return hypothesis

    def update_hypothesis_status(
        self,
        hypothesis_id: str,
        status: HypothesisStatus,
    ) -> CausalHypothesis:
        hypothesis = self._require_hypothesis(
            hypothesis_id
        )

        if not isinstance(
            status,
            HypothesisStatus,
        ):
            raise CausalAlphaDiscoveryError(
                "status must be a HypothesisStatus."
            )

        updated = CausalHypothesis(
            hypothesis_id=hypothesis.hypothesis_id,
            statement=hypothesis.statement,
            status=status,
            research_question=(
                hypothesis.research_question
            ),
            target_variable=(
                hypothesis.target_variable
            ),
            treatment_variable=(
                hypothesis.treatment_variable
            ),
            controls=hypothesis.controls,
            metadata=dict(hypothesis.metadata),
        )

        self._hypotheses[hypothesis_id] = updated
        return updated

    def map_mechanism(
        self,
        mapping: MechanismMapping,
    ) -> MechanismMapping:
        if not isinstance(
            mapping,
            MechanismMapping,
        ):
            raise CausalAlphaDiscoveryError(
                "mapping must be a MechanismMapping."
            )

        self._require_hypothesis(
            mapping.hypothesis_id
        )

        if mapping.mapping_id in self._mappings:
            raise CausalAlphaDiscoveryError(
                "Duplicate mapping ID: "
                f"{mapping.mapping_id}"
            )

        self._mappings[mapping.mapping_id] = mapping
        return mapping

    def generate_alpha_candidate(
        self,
        candidate: AlphaCandidate,
    ) -> AlphaCandidate:
        if not isinstance(
            candidate,
            AlphaCandidate,
        ):
            raise CausalAlphaDiscoveryError(
                "candidate must be an AlphaCandidate."
            )

        self._require_hypothesis(
            candidate.hypothesis_id
        )

        if candidate.candidate_id in self._candidates:
            raise CausalAlphaDiscoveryError(
                "Duplicate candidate ID: "
                f"{candidate.candidate_id}"
            )

        self._candidates[
            candidate.candidate_id
        ] = candidate

        return candidate

    def assess_evidence(
        self,
        assessment: EvidenceAssessment,
    ) -> EvidenceAssessment:
        if not isinstance(
            assessment,
            EvidenceAssessment,
        ):
            raise CausalAlphaDiscoveryError(
                "assessment must be an EvidenceAssessment."
            )

        if assessment.evidence_id in self._assessments:
            raise CausalAlphaDiscoveryError(
                "Duplicate evidence ID: "
                f"{assessment.evidence_id}"
            )

        self._assessments[
            assessment.evidence_id
        ] = assessment

        return assessment

    def classify_evidence(
        self,
        evidence_id: str,
    ) -> EvidenceClass:
        try:
            return self._assessments[
                evidence_id
            ].evidence_class
        except (KeyError, TypeError) as exc:
            raise CausalAlphaDiscoveryError(
                f"Unknown evidence ID: {evidence_id}"
            ) from exc

    def list_hypotheses(
        self,
    ) -> Tuple[CausalHypothesis, ...]:
        return tuple(self._hypotheses.values())

    def list_mappings(
        self,
    ) -> Tuple[MechanismMapping, ...]:
        return tuple(self._mappings.values())

    def list_candidates(
        self,
    ) -> Tuple[AlphaCandidate, ...]:
        return tuple(self._candidates.values())

    def list_assessments(
        self,
    ) -> Tuple[EvidenceAssessment, ...]:
        return tuple(self._assessments.values())

    def summary(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "hypothesis_count": self.hypothesis_count,
            "mapping_count": self.mapping_count,
            "candidate_count": self.candidate_count,
            "assessment_count": self.assessment_count,
            "research_only": (
                self.RESEARCH_ONLY
            ),
            "real_money_execution": (
                self.REAL_MONEY_EXECUTION
            ),
            "automatic_alpha_approval": (
                self.AUTOMATIC_ALPHA_APPROVAL
            ),
            "evidence_classes": [
                evidence_class.value
                for evidence_class in EvidenceClass
            ],
        }

    def _require_hypothesis(
        self,
        hypothesis_id: str,
    ) -> CausalHypothesis:
        try:
            return self._hypotheses[hypothesis_id]
        except (KeyError, TypeError) as exc:
            raise CausalAlphaDiscoveryError(
                f"Unknown hypothesis ID: {hypothesis_id}"
            ) from exc