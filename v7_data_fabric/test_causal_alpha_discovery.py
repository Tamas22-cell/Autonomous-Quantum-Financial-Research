from __future__ import annotations

import pytest

from v7_data_fabric.causal_alpha_discovery import (
    AlphaCandidate,
    CandidateStatus,
    CausalAlphaDiscovery,
    CausalAlphaDiscoveryError,
    CausalHypothesis,
    EvidenceAssessment,
    EvidenceClass,
    HypothesisStatus,
    MechanismMapping,
)


def _hypothesis() -> CausalHypothesis:
    return CausalHypothesis(
        hypothesis_id="h-001",
        statement=(
            "A tightening liquidity regime reduces "
            "high-beta asset performance."
        ),
        research_question=(
            "Does liquidity tightening affect high-beta assets?"
        ),
        target_variable="high_beta_return",
        treatment_variable="liquidity_tightening",
        controls=("volatility", "rates"),
    )


def test_version_and_safety_flags():
    engine = CausalAlphaDiscovery()

    assert engine.VERSION == "v7.0.7"
    assert engine.RESEARCH_ONLY is True
    assert engine.REAL_MONEY_EXECUTION is False
    assert engine.AUTOMATIC_ALPHA_APPROVAL is False


def test_causal_hypothesis_lifecycle():
    engine = CausalAlphaDiscovery()
    hypothesis = engine.create_hypothesis(
        _hypothesis()
    )

    assert hypothesis.status == (
        HypothesisStatus.PROPOSED
    )
    assert engine.hypothesis_count == 1

    updated = engine.update_hypothesis_status(
        "h-001",
        HypothesisStatus.UNDER_REVIEW,
    )

    assert updated.status == (
        HypothesisStatus.UNDER_REVIEW
    )
    assert engine.list_hypotheses()[0].status == (
        HypothesisStatus.UNDER_REVIEW
    )


def test_mechanism_and_intervention_mapping():
    engine = CausalAlphaDiscovery()
    engine.create_hypothesis(_hypothesis())

    mapping = engine.map_mechanism(
        MechanismMapping(
            mapping_id="m-001",
            hypothesis_id="h-001",
            mechanism=(
                "Higher funding costs reduce "
                "risk appetite."
            ),
            intervention="Liquidity tightening",
            expected_effect=(
                "Lower high-beta asset returns."
            ),
            market_variables=(
                "credit_spreads",
                "real_rates",
            ),
            assumptions=(
                "The transmission channel remains active",
            ),
            confidence=0.75,
        )
    )

    assert mapping.mapping_id == "m-001"
    assert engine.mapping_count == 1


def test_alpha_candidate_generation_is_research_only():
    engine = CausalAlphaDiscovery()
    engine.create_hypothesis(_hypothesis())

    candidate = engine.generate_alpha_candidate(
        AlphaCandidate(
            candidate_id="a-001",
            hypothesis_id="h-001",
            name="Liquidity Beta Filter",
            signal_definition=(
                "Reduce exposure when liquidity "
                "tightening exceeds threshold."
            ),
            causal_rationale=(
                "Funding pressure reduces risk appetite."
            ),
            predictive_rationale=(
                "Historical tightening periods "
                "show weaker high-beta returns."
            ),
            status=CandidateStatus.VALIDATION_REQUIRED,
            causal_evidence_score=0.55,
            predictive_evidence_score=0.70,
        )
    )

    assert candidate.status == (
        CandidateStatus.VALIDATION_REQUIRED
    )
    assert engine.candidate_count == 1
    assert engine.AUTOMATIC_ALPHA_APPROVAL is False


def test_causal_and_predictive_evidence_are_separated():
    engine = CausalAlphaDiscovery()

    causal = engine.assess_evidence(
        EvidenceAssessment(
            evidence_id="e-001",
            evidence_class=EvidenceClass.CAUSAL,
            summary=(
                "Intervention study supports "
                "the proposed mechanism."
            ),
            score=0.80,
            supports_hypothesis=True,
        )
    )

    predictive = engine.assess_evidence(
        EvidenceAssessment(
            evidence_id="e-002",
            evidence_class=EvidenceClass.PREDICTIVE,
            summary=(
                "Historical signal shows "
                "positive forward performance."
            ),
            score=0.72,
            supports_hypothesis=True,
        )
    )

    assert engine.classify_evidence(
        causal.evidence_id
    ) == EvidenceClass.CAUSAL

    assert engine.classify_evidence(
        predictive.evidence_id
    ) == EvidenceClass.PREDICTIVE

    assert engine.assessment_count == 2


def test_summary_contains_all_v707_components():
    engine = CausalAlphaDiscovery()
    summary = engine.summary()

    assert summary["version"] == "v7.0.7"
    assert summary["hypothesis_count"] == 0
    assert summary["mapping_count"] == 0
    assert summary["candidate_count"] == 0
    assert summary["assessment_count"] == 0
    assert summary["research_only"] is True
    assert summary["real_money_execution"] is False
    assert summary["automatic_alpha_approval"] is False


def test_unknown_hypothesis_is_rejected():
    engine = CausalAlphaDiscovery()

    with pytest.raises(CausalAlphaDiscoveryError):
        engine.map_mechanism(
            MechanismMapping(
                mapping_id="m-invalid",
                hypothesis_id="missing",
                mechanism="Unknown mechanism",
                intervention="Unknown intervention",
                expected_effect="Unknown effect",
                confidence=0.2,
            )
        )