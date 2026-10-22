from __future__ import annotations

from typing import Dict, List

import pytest

from research_benchmarking.benchmark_engine import (
    BenchmarkDecision,
    BenchmarkEngine,
    BenchmarkEvidenceSummary,
    BenchmarkStagePolicy,
)
from research_benchmarking.models import (
    AlphaVerdict,
    OverfittingRisk,
    StatisticalTestResult,
    ValidationConfig,
    ValidationResult,
    ValidationStage,
    ValidationStatus,
)


VERSION = "v6.0.2"


# ============================================================
# Helpers
# ============================================================


def make_config() -> ValidationConfig:
    return ValidationConfig(
        confidence_level=0.95,
        significance_threshold=0.05,
        bootstrap_samples=1000,
        monte_carlo_simulations=1000,
        walk_forward_windows=5,
        minimum_observations=30,
        random_seed=42,
        anti_overfitting_enabled=True,
        research_only=True,
        real_money_execution=False,
    )


def make_significance_test(
    *,
    p_value: float = 0.01,
    adjusted_p_value: float | None = None,
) -> StatisticalTestResult:
    return StatisticalTestResult(
        test_name="v6.0.2 alpha significance",
        statistic=3.25,
        p_value=p_value,
        significance_threshold=0.05,
        adjusted_p_value=adjusted_p_value,
        effect_size=0.65,
        null_hypothesis="No persistent alpha.",
        alternative_hypothesis="Persistent alpha exists.",
        metadata={
            "version": VERSION,
            "research_only": True,
        },
    )


def make_validation_result(
    stage: ValidationStage,
    *,
    score: float = 0.90,
    status: ValidationStatus = ValidationStatus.PASSED,
    significant: bool = False,
    diagnostics: Dict | None = None,
) -> ValidationResult:
    tests = []

    if significant:
        tests.append(
            make_significance_test()
        )

    return ValidationResult(
        stage=stage,
        status=status,
        score=score,
        message=f"{stage.value} validation completed.",
        statistical_tests=tests,
        diagnostics=diagnostics or {},
        metadata={
            "version": VERSION,
            "research_only": True,
            "real_money_execution": False,
        },
    )


def make_complete_pipeline(
    *,
    score: float = 0.90,
    overfitting_risk: OverfittingRisk = OverfittingRisk.LOW,
    overfitting_score: float = 0.10,
) -> List[ValidationResult]:
    results: List[ValidationResult] = []

    for stage in BenchmarkStagePolicy().required_stages:
        diagnostics = {}
        significant = False

        if (
            stage
            == ValidationStage.STATISTICAL_SIGNIFICANCE
        ):
            significant = True
            diagnostics = {
                "p_value": 0.01,
                "effect_size": 0.65,
                "confidence_interval_excludes_zero": True,
            }

        elif (
            stage
            == ValidationStage.MULTIPLE_TESTING
        ):
            diagnostics = {
                "method": "benjamini_hochberg",
                "discoveries_retained": True,
                "adjusted_significance": True,
            }

        elif (
            stage
            == ValidationStage.WALK_FORWARD
        ):
            diagnostics = {
                "positive_oos_ratio": 0.80,
                "stable_window_ratio": 0.80,
                "average_test_degradation": 0.15,
            }

        elif (
            stage
            == ValidationStage.OUT_OF_SAMPLE
        ):
            diagnostics = {
                "positive_oos_ratio": 0.85,
                "stable_segment_ratio": 0.80,
                "average_degradation": 0.15,
            }

        elif (
            stage
            == ValidationStage.BOOTSTRAP
        ):
            diagnostics = {
                "probability_positive": 0.97,
                "confidence_interval_excludes_zero": True,
            }

        elif (
            stage
            == ValidationStage.MONTE_CARLO
        ):
            diagnostics = {
                "empirical_p_value": 0.02,
                "positive_effect": True,
            }

        elif (
            stage
            == ValidationStage.ROBUSTNESS
        ):
            diagnostics = {
                "positive_survival_ratio": 0.90,
                "stable_scenario_ratio": 0.85,
            }

        elif (
            stage
            == ValidationStage.SENSITIVITY
        ):
            diagnostics = {
                "stable_parameter_ratio": 0.90,
                "average_stability_score": 0.85,
            }

        elif (
            stage
            == ValidationStage.REGIME_STABILITY
        ):
            diagnostics = {
                "stable_regime_ratio": 0.80,
                "positive_regime_ratio": 0.85,
            }

        elif (
            stage
            == ValidationStage.OVERFITTING_CONTROL
        ):
            diagnostics = {
                "risk_level": overfitting_risk.value,
                "overall_risk_score": overfitting_score,
                "generalization_acceptable": True,
                "selection_bias_controlled": True,
                "parameter_instability_controlled": True,
            }

        results.append(
            make_validation_result(
                stage,
                score=score,
                significant=significant,
                diagnostics=diagnostics,
            )
        )

    return results


def replace_stage(
    results: List[ValidationResult],
    replacement: ValidationResult,
) -> List[ValidationResult]:
    return [
        (
            replacement
            if result.stage == replacement.stage
            else result
        )
        for result in results
    ]


# ============================================================
# Architecture
# ============================================================


def test_e2e_version():
    assert VERSION == "v6.0.2"


def test_e2e_required_stage_count():
    policy = BenchmarkStagePolicy()

    assert len(policy.required_stages) == 10


def test_e2e_all_validation_layers_present():
    stages = set(
        BenchmarkStagePolicy().required_stages
    )

    expected = {
        ValidationStage.WALK_FORWARD,
        ValidationStage.OUT_OF_SAMPLE,
        ValidationStage.BOOTSTRAP,
        ValidationStage.MONTE_CARLO,
        ValidationStage.STATISTICAL_SIGNIFICANCE,
        ValidationStage.MULTIPLE_TESTING,
        ValidationStage.ROBUSTNESS,
        ValidationStage.SENSITIVITY,
        ValidationStage.REGIME_STABILITY,
        ValidationStage.OVERFITTING_CONTROL,
    }

    assert stages == expected


def test_e2e_research_only_configuration():
    config = make_config()

    assert config.research_only is True
    assert config.real_money_execution is False
    assert config.anti_overfitting_enabled is True


# ============================================================
# Complete robust-alpha pipeline
# ============================================================


def test_e2e_complete_pipeline_has_ten_results():
    results = make_complete_pipeline()

    assert len(results) == 10
    assert len(
        {
            result.stage
            for result in results
        }
    ) == 10


def test_e2e_complete_pipeline_all_passed():
    results = make_complete_pipeline()

    assert all(
        result.status
        == ValidationStatus.PASSED
        for result in results
    )


def test_e2e_complete_pipeline_summary():
    engine = BenchmarkEngine(
        make_config()
    )

    summary = engine.summarize(
        make_complete_pipeline()
    )

    assert isinstance(
        summary,
        BenchmarkEvidenceSummary,
    )

    assert summary.required_stage_count == 10
    assert summary.observed_required_stage_count == 10
    assert summary.passed_required_stage_count == 10
    assert summary.pass_ratio == 1.0
    assert summary.all_required_stages_present is True
    assert summary.all_required_stages_passed is True
    assert summary.missing_stages == ()
    assert summary.failed_stages == ()


def test_e2e_complete_pipeline_decision():
    engine = BenchmarkEngine(
        make_config()
    )

    decision = engine.decide(
        make_complete_pipeline(
            score=0.92,
            overfitting_risk=OverfittingRisk.LOW,
            overfitting_score=0.08,
        )
    )

    assert isinstance(
        decision,
        BenchmarkDecision,
    )

    assert (
        decision.verdict
        == AlphaVerdict.ROBUST_ALPHA
    )

    assert (
        decision.robust_alpha_supported
        is True
    )

    assert (
        decision.overfitting_risk
        == OverfittingRisk.LOW
    )

    assert decision.overall_score > 0.80
    assert decision.confidence_score > 0.80
    assert decision.evidence_quality_score > 0.80


def test_e2e_complete_analysis():
    engine = BenchmarkEngine(
        make_config()
    )

    analysis = engine.analyze(
        make_complete_pipeline(
            score=0.92
        )
    )

    assert analysis["version"] == VERSION

    assert isinstance(
        analysis["summary"],
        BenchmarkEvidenceSummary,
    )

    assert isinstance(
        analysis["decision"],
        BenchmarkDecision,
    )

    assert (
        analysis["decision"].verdict
        == AlphaVerdict.ROBUST_ALPHA
    )

    assert analysis["research_only"] is True
    assert analysis["real_money_execution"] is False

    assert len(
        analysis["summary_fingerprint"]
    ) == 64

    assert len(
        analysis["decision_fingerprint"]
    ) == 64


# ============================================================
# Walk-forward / OOS critical controls
# ============================================================


def test_e2e_failed_oos_blocks_robust_alpha():
    engine = BenchmarkEngine()

    results = make_complete_pipeline()

    results = replace_stage(
        results,
        make_validation_result(
            ValidationStage.OUT_OF_SAMPLE,
            score=0.20,
            status=ValidationStatus.FAILED,
        ),
    )

    decision = engine.decide(results)

    assert decision.robust_alpha_supported is False

    assert any(
        "out-of-sample" in reason.lower()
        for reason in decision.reasons
    )


def test_e2e_missing_oos_is_inconclusive():
    engine = BenchmarkEngine()

    results = [
        result
        for result in make_complete_pipeline()
        if (
            result.stage
            != ValidationStage.OUT_OF_SAMPLE
        )
    ]

    decision = engine.decide(results)

    assert (
        decision.verdict
        == AlphaVerdict.INCONCLUSIVE
    )

    assert decision.robust_alpha_supported is False


def test_e2e_walk_forward_failure_reduces_evidence():
    engine = BenchmarkEngine()

    results = make_complete_pipeline()

    results = replace_stage(
        results,
        make_validation_result(
            ValidationStage.WALK_FORWARD,
            score=0.20,
            status=ValidationStatus.FAILED,
        ),
    )

    summary = engine.summarize(results)

    assert summary.pass_ratio == pytest.approx(
        0.90
    )

    assert (
        ValidationStage.WALK_FORWARD
        in summary.failed_stages
    )


# ============================================================
# Statistical controls
# ============================================================


def test_e2e_significance_stage_contains_test():
    results = make_complete_pipeline()

    significance = next(
        result
        for result in results
        if (
            result.stage
            == ValidationStage.STATISTICAL_SIGNIFICANCE
        )
    )

    assert len(
        significance.statistical_tests
    ) == 1

    assert (
        significance.significant_test_count
        == 1
    )


def test_e2e_failed_significance_blocks_robust_alpha():
    engine = BenchmarkEngine()

    results = make_complete_pipeline()

    results = replace_stage(
        results,
        make_validation_result(
            ValidationStage.STATISTICAL_SIGNIFICANCE,
            score=0.20,
            status=ValidationStatus.FAILED,
            significant=False,
        ),
    )

    decision = engine.decide(results)

    assert decision.robust_alpha_supported is False

    assert any(
        "significance" in reason.lower()
        for reason in decision.reasons
    )


def test_e2e_multiple_testing_stage_present():
    results = make_complete_pipeline()

    stage = next(
        result
        for result in results
        if (
            result.stage
            == ValidationStage.MULTIPLE_TESTING
        )
    )

    assert (
        stage.status
        == ValidationStatus.PASSED
    )

    assert (
        stage.diagnostics[
            "adjusted_significance"
        ]
        is True
    )


# ============================================================
# Robustness / sensitivity / regime controls
# ============================================================


def test_e2e_robustness_evidence():
    results = make_complete_pipeline()

    result = next(
        item
        for item in results
        if (
            item.stage
            == ValidationStage.ROBUSTNESS
        )
    )

    assert (
        result.diagnostics[
            "positive_survival_ratio"
        ]
        == 0.90
    )

    assert (
        result.diagnostics[
            "stable_scenario_ratio"
        ]
        == 0.85
    )


def test_e2e_sensitivity_evidence():
    results = make_complete_pipeline()

    result = next(
        item
        for item in results
        if (
            item.stage
            == ValidationStage.SENSITIVITY
        )
    )

    assert (
        result.diagnostics[
            "stable_parameter_ratio"
        ]
        == 0.90
    )


def test_e2e_regime_stability_evidence():
    results = make_complete_pipeline()

    result = next(
        item
        for item in results
        if (
            item.stage
            == ValidationStage.REGIME_STABILITY
        )
    )

    assert (
        result.diagnostics[
            "stable_regime_ratio"
        ]
        == 0.80
    )

    assert (
        result.diagnostics[
            "positive_regime_ratio"
        ]
        == 0.85
    )


# ============================================================
# Anti-overfitting controls
# ============================================================


def test_e2e_low_overfitting_risk():
    engine = BenchmarkEngine()

    decision = engine.decide(
        make_complete_pipeline(
            overfitting_risk=OverfittingRisk.LOW,
            overfitting_score=0.10,
        )
    )

    assert (
        decision.overfitting_risk
        == OverfittingRisk.LOW
    )


def test_e2e_high_overfitting_risk_blocks_alpha():
    engine = BenchmarkEngine()

    results = make_complete_pipeline(
        overfitting_risk=OverfittingRisk.HIGH,
        overfitting_score=0.80,
    )

    results = replace_stage(
        results,
        make_validation_result(
            ValidationStage.OVERFITTING_CONTROL,
            score=0.20,
            status=ValidationStatus.FAILED,
            diagnostics={
                "risk_level": (
                    OverfittingRisk.HIGH.value
                ),
                "overall_risk_score": 0.80,
                "generalization_acceptable": False,
            },
        ),
    )

    decision = engine.decide(results)

    assert (
        decision.verdict
        == AlphaVerdict.OVERFIT_RISK
    )

    assert (
        decision.robust_alpha_supported
        is False
    )

    assert any(
        "overfitting" in reason.lower()
        for reason in decision.reasons
    )


def test_e2e_critical_overfitting_risk():
    engine = BenchmarkEngine()

    results = make_complete_pipeline(
        overfitting_risk=OverfittingRisk.CRITICAL,
        overfitting_score=0.95,
    )

    results = replace_stage(
        results,
        make_validation_result(
            ValidationStage.OVERFITTING_CONTROL,
            score=0.05,
            status=ValidationStatus.FAILED,
            diagnostics={
                "risk_level": (
                    OverfittingRisk.CRITICAL.value
                ),
                "overall_risk_score": 0.95,
            },
        ),
    )

    decision = engine.decide(results)

    assert (
        decision.overfitting_risk
        == OverfittingRisk.CRITICAL
    )

    assert (
        decision.verdict
        == AlphaVerdict.OVERFIT_RISK
    )


# ============================================================
# Missing / inconclusive evidence
# ============================================================


@pytest.mark.parametrize(
    "missing_stage",
    [
        ValidationStage.WALK_FORWARD,
        ValidationStage.BOOTSTRAP,
        ValidationStage.MONTE_CARLO,
        ValidationStage.ROBUSTNESS,
        ValidationStage.SENSITIVITY,
        ValidationStage.REGIME_STABILITY,
    ],
)
def test_e2e_missing_stage_is_inconclusive(
    missing_stage,
):
    engine = BenchmarkEngine()

    results = [
        result
        for result in make_complete_pipeline()
        if result.stage != missing_stage
    ]

    decision = engine.decide(results)

    assert (
        decision.verdict
        == AlphaVerdict.INCONCLUSIVE
    )

    assert (
        decision.robust_alpha_supported
        is False
    )


def test_e2e_inconclusive_stage():
    engine = BenchmarkEngine()

    results = make_complete_pipeline()

    results = replace_stage(
        results,
        make_validation_result(
            ValidationStage.MONTE_CARLO,
            score=0.50,
            status=ValidationStatus.INCONCLUSIVE,
        ),
    )

    summary = engine.summarize(results)

    assert (
        ValidationStage.MONTE_CARLO
        in summary.inconclusive_stages
    )

    assert (
        summary.all_required_stages_passed
        is False
    )


# ============================================================
# Weak evidence
# ============================================================


def test_e2e_weak_evidence():
    engine = BenchmarkEngine()

    results = make_complete_pipeline(
        score=0.70
    )

    failed_stages = {
        ValidationStage.BOOTSTRAP,
        ValidationStage.ROBUSTNESS,
        ValidationStage.SENSITIVITY,
    }

    replaced = []

    for result in results:
        if result.stage in failed_stages:
            replaced.append(
                make_validation_result(
                    result.stage,
                    score=0.20,
                    status=ValidationStatus.FAILED,
                )
            )
        else:
            replaced.append(result)

    decision = engine.decide(replaced)

    assert (
        decision.verdict
        == AlphaVerdict.WEAK_EVIDENCE
    )

    assert (
        decision.robust_alpha_supported
        is False
    )


# ============================================================
# Rejection
# ============================================================


def test_e2e_rejected_evidence():
    engine = BenchmarkEngine()

    results = make_complete_pipeline()

    failed_stages = {
        ValidationStage.WALK_FORWARD,
        ValidationStage.BOOTSTRAP,
        ValidationStage.MONTE_CARLO,
        ValidationStage.MULTIPLE_TESTING,
        ValidationStage.ROBUSTNESS,
        ValidationStage.SENSITIVITY,
    }

    replaced = []

    for result in results:
        if result.stage in failed_stages:
            replaced.append(
                make_validation_result(
                    result.stage,
                    score=0.20,
                    status=ValidationStatus.FAILED,
                )
            )
        else:
            replaced.append(result)

    decision = engine.decide(replaced)

    assert (
        decision.verdict
        == AlphaVerdict.REJECTED
    )

    assert (
        decision.robust_alpha_supported
        is False
    )


# ============================================================
# Determinism / reproducibility
# ============================================================


def test_e2e_summary_reproducible():
    engine = BenchmarkEngine(
        make_config()
    )

    results = make_complete_pipeline()

    first = engine.summarize(results)
    second = engine.summarize(results)

    assert (
        first.fingerprint
        == second.fingerprint
    )


def test_e2e_decision_reproducible():
    engine = BenchmarkEngine(
        make_config()
    )

    results = make_complete_pipeline()

    first = engine.decide(results)
    second = engine.decide(results)

    assert (
        first.fingerprint
        == second.fingerprint
    )


def test_e2e_analysis_reproducible():
    engine = BenchmarkEngine(
        make_config()
    )

    results = make_complete_pipeline()

    first = engine.analyze(results)
    second = engine.analyze(results)

    assert (
        first["summary_fingerprint"]
        == second["summary_fingerprint"]
    )

    assert (
        first["decision_fingerprint"]
        == second["decision_fingerprint"]
    )


# ============================================================
# Safety invariants
# ============================================================


def test_e2e_every_stage_research_only():
    results = make_complete_pipeline()

    for result in results:
        assert (
            result.metadata[
                "research_only"
            ]
            is True
        )

        assert (
            result.metadata[
                "real_money_execution"
            ]
            is False
        )


def test_e2e_engine_analysis_research_only():
    engine = BenchmarkEngine()

    analysis = engine.analyze(
        make_complete_pipeline()
    )

    assert (
        analysis["research_only"]
        is True
    )

    assert (
        analysis[
            "real_money_execution"
        ]
        is False
    )


def test_e2e_snapshot_safety():
    engine = BenchmarkEngine(
        make_config()
    )

    snapshot = engine.snapshot()

    assert (
        snapshot["research_only"]
        is True
    )

    assert (
        snapshot[
            "real_money_execution"
        ]
        is False
    )


# ============================================================
# Fingerprints / auditability
# ============================================================


def test_e2e_all_results_have_fingerprints():
    results = make_complete_pipeline()

    fingerprints = [
        result.fingerprint
        for result in results
    ]

    assert len(fingerprints) == 10

    assert all(
        len(fingerprint) == 64
        for fingerprint in fingerprints
    )


def test_e2e_summary_has_fingerprint():
    engine = BenchmarkEngine()

    summary = engine.summarize(
        make_complete_pipeline()
    )

    assert len(
        summary.fingerprint
    ) == 64


def test_e2e_decision_has_fingerprint():
    engine = BenchmarkEngine()

    decision = engine.decide(
        make_complete_pipeline()
    )

    assert len(
        decision.fingerprint
    ) == 64


# ============================================================
# Final institutional validation scenario
# ============================================================


def test_e2e_institutional_research_scenario():
    """
    Final v6.0.2 integration scenario.

    Simulates a research candidate that has survived all
    independent statistical and anti-overfitting controls.
    """

    config = make_config()

    engine = BenchmarkEngine(
        config
    )

    results = make_complete_pipeline(
        score=0.93,
        overfitting_risk=OverfittingRisk.LOW,
        overfitting_score=0.07,
    )

    analysis = engine.analyze(
        results
    )

    summary = analysis["summary"]
    decision = analysis["decision"]

    # Complete evidence coverage
    assert summary.required_stage_count == 10
    assert summary.observed_required_stage_count == 10
    assert summary.passed_required_stage_count == 10

    # No missing evidence
    assert summary.missing_stages == ()
    assert summary.failed_stages == ()
    assert summary.inconclusive_stages == ()

    # Full validation coverage
    assert summary.pass_ratio == 1.0
    assert summary.all_required_stages_present is True
    assert summary.all_required_stages_passed is True

    # Evidence quality
    assert summary.average_score > 0.90
    assert summary.minimum_score > 0.90

    # Anti-overfitting
    assert (
        decision.overfitting_risk
        == OverfittingRisk.LOW
    )

    # Final research verdict
    assert (
        decision.verdict
        == AlphaVerdict.ROBUST_ALPHA
    )

    assert (
        decision.robust_alpha_supported
        is True
    )

    # Strong aggregate evidence
    assert decision.overall_score > 0.85
    assert decision.confidence_score > 0.85
    assert decision.evidence_quality_score > 0.85

    # Deterministic audit fingerprints
    assert len(
        summary.fingerprint
    ) == 64

    assert len(
        decision.fingerprint
    ) == 64

    # Safety boundary
    assert analysis[
        "research_only"
    ] is True

    assert analysis[
        "real_money_execution"
    ] is False