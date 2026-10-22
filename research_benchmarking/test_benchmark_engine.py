from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from research_benchmarking.benchmark_engine import (
    VERSION,
    BenchmarkDecision,
    BenchmarkEngine,
    BenchmarkEngineError,
    BenchmarkEvidenceSummary,
    BenchmarkStagePolicy,
    StageEvidence,
)
from research_benchmarking.models import (
    AlphaVerdict,
    OverfittingRisk,
    StatisticalTestResult,
    ValidationConfig,
    ValidationConfigurationError,
    ValidationResult,
    ValidationStage,
    ValidationStatus,
)


# ============================================================
# Helpers
# ============================================================


def make_config() -> ValidationConfig:
    return ValidationConfig(
        minimum_observations=30,
        research_only=True,
        real_money_execution=False,
    )


def make_statistical_test(
    *,
    p_value: float = 0.01,
) -> StatisticalTestResult:
    return StatisticalTestResult(
        test_name="benchmark-test",
        statistic=2.5,
        p_value=p_value,
        significance_threshold=0.05,
        effect_size=0.5,
        null_hypothesis="No alpha.",
        alternative_hypothesis="Alpha exists.",
    )


def make_result(
    stage: ValidationStage,
    *,
    status: ValidationStatus = ValidationStatus.PASSED,
    score: float = 0.85,
    significant: bool = False,
    diagnostics=None,
) -> ValidationResult:
    tests = []

    if significant:
        tests = [
            make_statistical_test()
        ]

    return ValidationResult(
        stage=stage,
        status=status,
        score=score,
        message=f"{stage.value} validation",
        statistical_tests=tests,
        diagnostics=diagnostics or {},
        metadata={
            "research_only": True,
            "real_money_execution": False,
        },
    )


def make_required_results(
    *,
    score: float = 0.85,
    overfitting_risk: OverfittingRisk = OverfittingRisk.LOW,
    overfitting_score: float = 0.10,
):
    results = []

    for stage in BenchmarkStagePolicy().required_stages:
        diagnostics = {}

        significant = (
            stage
            == ValidationStage.STATISTICAL_SIGNIFICANCE
        )

        if (
            stage
            == ValidationStage.OVERFITTING_CONTROL
        ):
            diagnostics = {
                "risk_level": overfitting_risk.value,
                "overall_risk_score": overfitting_score,
            }

        results.append(
            make_result(
                stage,
                score=score,
                significant=significant,
                diagnostics=diagnostics,
            )
        )

    return results


def make_stage_evidence(
    stage: ValidationStage = ValidationStage.WALK_FORWARD,
    *,
    score: float = 0.8,
    passed: bool = True,
) -> StageEvidence:
    return StageEvidence(
        stage=stage,
        status=(
            ValidationStatus.PASSED
            if passed
            else ValidationStatus.FAILED
        ),
        score=score,
        passed=passed,
        significant=False,
        message="test",
        fingerprint="abc123",
    )


def make_summary() -> BenchmarkEvidenceSummary:
    evidence = [
        make_stage_evidence(
            ValidationStage.WALK_FORWARD
        )
    ]

    return BenchmarkEvidenceSummary(
        evidence=evidence,
        required_stage_count=1,
        observed_required_stage_count=1,
        passed_required_stage_count=1,
        missing_stages=[],
        failed_stages=[],
        inconclusive_stages=[],
        average_score=0.8,
        minimum_score=0.8,
        pass_ratio=1.0,
        all_required_stages_present=True,
        all_required_stages_passed=True,
    )


# ============================================================
# Version
# ============================================================


def test_version():
    assert VERSION == "v6.0.2"


def test_engine_version():
    assert BenchmarkEngine.VERSION == "v6.0.2"


# ============================================================
# BenchmarkStagePolicy
# ============================================================


def test_default_policy():
    policy = BenchmarkStagePolicy()

    assert len(policy.required_stages) == 10
    assert policy.minimum_stage_score == 0.50
    assert policy.minimum_pass_ratio == 0.80
    assert policy.require_oos_pass is True
    assert policy.require_overfitting_pass is True
    assert policy.require_significance_pass is True


def test_policy_contains_all_required_stages():
    policy = BenchmarkStagePolicy()

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

    assert set(policy.required_stages) == expected


def test_policy_required_stages_becomes_tuple():
    policy = BenchmarkStagePolicy(
        required_stages=[
            ValidationStage.WALK_FORWARD,
            ValidationStage.OUT_OF_SAMPLE,
        ]
    )

    assert isinstance(
        policy.required_stages,
        tuple,
    )


def test_policy_empty_stages():
    with pytest.raises(
        ValidationConfigurationError
    ):
        BenchmarkStagePolicy(
            required_stages=[]
        )


def test_policy_invalid_stage():
    with pytest.raises(
        ValidationConfigurationError
    ):
        BenchmarkStagePolicy(
            required_stages=["bad"]
        )


def test_policy_duplicate_stage():
    with pytest.raises(
        ValidationConfigurationError
    ):
        BenchmarkStagePolicy(
            required_stages=[
                ValidationStage.WALK_FORWARD,
                ValidationStage.WALK_FORWARD,
            ]
        )


@pytest.mark.parametrize(
    "field",
    [
        "minimum_stage_score",
        "minimum_pass_ratio",
    ],
)
@pytest.mark.parametrize(
    "value",
    [-0.01, 1.01, True, "bad"],
)
def test_policy_invalid_ratio(
    field,
    value,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        BenchmarkStagePolicy(
            **{field: value}
        )


@pytest.mark.parametrize(
    "field",
    [
        "require_oos_pass",
        "require_overfitting_pass",
        "require_significance_pass",
    ],
)
def test_policy_invalid_bool(field):
    with pytest.raises(
        ValidationConfigurationError
    ):
        BenchmarkStagePolicy(
            **{field: 1}
        )


def test_policy_fingerprint():
    first = BenchmarkStagePolicy()
    second = BenchmarkStagePolicy()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_policy_fingerprint_changes():
    first = BenchmarkStagePolicy(
        minimum_pass_ratio=0.80
    )
    second = BenchmarkStagePolicy(
        minimum_pass_ratio=0.90
    )

    assert first.fingerprint != second.fingerprint


# ============================================================
# StageEvidence
# ============================================================


def test_stage_evidence_creation():
    evidence = make_stage_evidence()

    assert (
        evidence.stage
        == ValidationStage.WALK_FORWARD
    )
    assert evidence.score == 0.8
    assert evidence.passed is True


def test_stage_evidence_frozen():
    evidence = make_stage_evidence()

    with pytest.raises(FrozenInstanceError):
        evidence.score = 0.5


@pytest.mark.parametrize(
    "score",
    [-0.01, 1.01, True, "bad"],
)
def test_stage_evidence_invalid_score(score):
    with pytest.raises(
        ValidationConfigurationError
    ):
        StageEvidence(
            stage=ValidationStage.WALK_FORWARD,
            status=ValidationStatus.PASSED,
            score=score,
            passed=True,
            significant=False,
            message="test",
            fingerprint="abc",
        )


def test_stage_evidence_invalid_stage():
    with pytest.raises(
        ValidationConfigurationError
    ):
        StageEvidence(
            stage="bad",
            status=ValidationStatus.PASSED,
            score=0.8,
            passed=True,
            significant=False,
            message="test",
            fingerprint="abc",
        )


def test_stage_evidence_invalid_status():
    with pytest.raises(
        ValidationConfigurationError
    ):
        StageEvidence(
            stage=ValidationStage.WALK_FORWARD,
            status="PASSED",
            score=0.8,
            passed=True,
            significant=False,
            message="test",
            fingerprint="abc",
        )


# ============================================================
# BenchmarkEvidenceSummary
# ============================================================


def test_summary_creation():
    summary = make_summary()

    assert summary.required_stage_count == 1
    assert summary.pass_ratio == 1.0
    assert summary.all_required_stages_present is True
    assert summary.all_required_stages_passed is True


def test_summary_frozen():
    summary = make_summary()

    with pytest.raises(FrozenInstanceError):
        summary.pass_ratio = 0.5


def test_summary_fingerprint():
    first = make_summary()
    second = make_summary()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_summary_empty_evidence():
    with pytest.raises(
        ValidationConfigurationError
    ):
        BenchmarkEvidenceSummary(
            evidence=[],
            required_stage_count=1,
            observed_required_stage_count=0,
            passed_required_stage_count=0,
            missing_stages=[
                ValidationStage.WALK_FORWARD
            ],
            failed_stages=[],
            inconclusive_stages=[],
            average_score=0.0,
            minimum_score=0.0,
            pass_ratio=0.0,
            all_required_stages_present=False,
            all_required_stages_passed=False,
        )


def test_summary_observed_exceeds_required():
    with pytest.raises(
        ValidationConfigurationError
    ):
        BenchmarkEvidenceSummary(
            evidence=[
                make_stage_evidence()
            ],
            required_stage_count=1,
            observed_required_stage_count=2,
            passed_required_stage_count=1,
            missing_stages=[],
            failed_stages=[],
            inconclusive_stages=[],
            average_score=0.8,
            minimum_score=0.8,
            pass_ratio=1.0,
            all_required_stages_present=True,
            all_required_stages_passed=True,
        )


def test_summary_passed_exceeds_observed():
    with pytest.raises(
        ValidationConfigurationError
    ):
        BenchmarkEvidenceSummary(
            evidence=[
                make_stage_evidence()
            ],
            required_stage_count=2,
            observed_required_stage_count=1,
            passed_required_stage_count=2,
            missing_stages=[
                ValidationStage.OUT_OF_SAMPLE
            ],
            failed_stages=[],
            inconclusive_stages=[],
            average_score=0.8,
            minimum_score=0.8,
            pass_ratio=0.5,
            all_required_stages_present=False,
            all_required_stages_passed=False,
        )


# ============================================================
# BenchmarkDecision
# ============================================================


def test_decision_creation():
    decision = BenchmarkDecision(
        verdict=AlphaVerdict.ROBUST_ALPHA,
        overall_score=0.9,
        confidence_score=0.9,
        evidence_quality_score=0.9,
        overfitting_risk=OverfittingRisk.LOW,
        robust_alpha_supported=True,
        reasons=["Strong evidence."],
    )

    assert (
        decision.verdict
        == AlphaVerdict.ROBUST_ALPHA
    )
    assert decision.robust_alpha_supported is True


def test_decision_frozen():
    decision = BenchmarkDecision(
        verdict=AlphaVerdict.WEAK_EVIDENCE,
        overall_score=0.5,
        confidence_score=0.5,
        evidence_quality_score=0.5,
        overfitting_risk=OverfittingRisk.MODERATE,
        robust_alpha_supported=False,
        reasons=[],
    )

    with pytest.raises(FrozenInstanceError):
        decision.overall_score = 0.8


@pytest.mark.parametrize(
    "field",
    [
        "overall_score",
        "confidence_score",
        "evidence_quality_score",
    ],
)
@pytest.mark.parametrize(
    "value",
    [-0.01, 1.01, True, "bad"],
)
def test_decision_invalid_score(
    field,
    value,
):
    kwargs = {
        "verdict": AlphaVerdict.WEAK_EVIDENCE,
        "overall_score": 0.5,
        "confidence_score": 0.5,
        "evidence_quality_score": 0.5,
        "overfitting_risk": OverfittingRisk.MODERATE,
        "robust_alpha_supported": False,
        "reasons": [],
    }

    kwargs[field] = value

    with pytest.raises(
        ValidationConfigurationError
    ):
        BenchmarkDecision(**kwargs)


def test_decision_fingerprint():
    first = BenchmarkDecision(
        verdict=AlphaVerdict.ROBUST_ALPHA,
        overall_score=0.9,
        confidence_score=0.9,
        evidence_quality_score=0.9,
        overfitting_risk=OverfittingRisk.LOW,
        robust_alpha_supported=True,
        reasons=["Strong evidence."],
    )

    second = BenchmarkDecision(
        verdict=AlphaVerdict.ROBUST_ALPHA,
        overall_score=0.9,
        confidence_score=0.9,
        evidence_quality_score=0.9,
        overfitting_risk=OverfittingRisk.LOW,
        robust_alpha_supported=True,
        reasons=["Strong evidence."],
    )

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


# ============================================================
# Engine construction
# ============================================================


def test_default_engine():
    engine = BenchmarkEngine()

    assert isinstance(
        engine.config,
        ValidationConfig,
    )
    assert isinstance(
        engine.policy,
        BenchmarkStagePolicy,
    )


def test_custom_engine():
    config = make_config()

    policy = BenchmarkStagePolicy(
        minimum_pass_ratio=0.90
    )

    engine = BenchmarkEngine(
        config,
        policy=policy,
    )

    assert engine.config is config
    assert engine.policy is policy


# ============================================================
# Result validation
# ============================================================


def test_validate_results():
    results = [
        make_result(
            ValidationStage.WALK_FORWARD
        )
    ]

    validated = BenchmarkEngine._validate_results(
        results
    )

    assert validated == results


def test_validate_results_empty():
    with pytest.raises(
        BenchmarkEngineError
    ):
        BenchmarkEngine._validate_results([])


def test_validate_results_invalid_type():
    with pytest.raises(
        ValidationConfigurationError
    ):
        BenchmarkEngine._validate_results(
            "bad"
        )


def test_validate_results_invalid_item():
    with pytest.raises(
        ValidationConfigurationError
    ):
        BenchmarkEngine._validate_results(
            ["bad"]
        )


def test_validate_results_duplicate_stage():
    with pytest.raises(
        BenchmarkEngineError
    ):
        BenchmarkEngine._validate_results(
            [
                make_result(
                    ValidationStage.WALK_FORWARD
                ),
                make_result(
                    ValidationStage.WALK_FORWARD
                ),
            ]
        )


# ============================================================
# Stage evidence conversion
# ============================================================


def test_build_stage_evidence_pass():
    engine = BenchmarkEngine()

    result = make_result(
        ValidationStage.WALK_FORWARD,
        score=0.85,
    )

    evidence = engine.build_stage_evidence(
        result
    )

    assert evidence.passed is True
    assert evidence.score == 0.85
    assert (
        evidence.status
        == ValidationStatus.PASSED
    )


def test_build_stage_evidence_score_below_threshold():
    engine = BenchmarkEngine()

    result = make_result(
        ValidationStage.WALK_FORWARD,
        score=0.40,
    )

    evidence = engine.build_stage_evidence(
        result
    )

    assert evidence.passed is False


def test_build_stage_evidence_failed_status():
    engine = BenchmarkEngine()

    result = make_result(
        ValidationStage.WALK_FORWARD,
        status=ValidationStatus.FAILED,
        score=0.9,
    )

    evidence = engine.build_stage_evidence(
        result
    )

    assert evidence.passed is False


def test_build_stage_evidence_significant():
    engine = BenchmarkEngine()

    result = make_result(
        ValidationStage.STATISTICAL_SIGNIFICANCE,
        significant=True,
    )

    evidence = engine.build_stage_evidence(
        result
    )

    assert evidence.significant is True


# ============================================================
# Summarize
# ============================================================


def test_summarize_complete_pass():
    engine = BenchmarkEngine()

    summary = engine.summarize(
        make_required_results()
    )

    assert summary.required_stage_count == 10
    assert summary.observed_required_stage_count == 10
    assert summary.passed_required_stage_count == 10
    assert summary.pass_ratio == 1.0
    assert summary.missing_stages == ()
    assert summary.failed_stages == ()
    assert summary.all_required_stages_present is True
    assert summary.all_required_stages_passed is True


def test_summarize_missing_stage():
    engine = BenchmarkEngine()

    results = make_required_results()

    results = [
        result
        for result in results
        if (
            result.stage
            != ValidationStage.BOOTSTRAP
        )
    ]

    summary = engine.summarize(
        results
    )

    assert (
        ValidationStage.BOOTSTRAP
        in summary.missing_stages
    )

    assert (
        summary.all_required_stages_present
        is False
    )


def test_summarize_failed_stage():
    engine = BenchmarkEngine()

    results = make_required_results()

    for index, result in enumerate(results):
        if (
            result.stage
            == ValidationStage.ROBUSTNESS
        ):
            results[index] = make_result(
                ValidationStage.ROBUSTNESS,
                status=ValidationStatus.FAILED,
                score=0.20,
            )

    summary = engine.summarize(
        results
    )

    assert (
        ValidationStage.ROBUSTNESS
        in summary.failed_stages
    )

    assert summary.pass_ratio == pytest.approx(
        0.9
    )


def test_summarize_inconclusive_stage():
    engine = BenchmarkEngine()

    results = make_required_results()

    for index, result in enumerate(results):
        if (
            result.stage
            == ValidationStage.MONTE_CARLO
        ):
            results[index] = make_result(
                ValidationStage.MONTE_CARLO,
                status=ValidationStatus.INCONCLUSIVE,
                score=0.50,
            )

    summary = engine.summarize(
        results
    )

    assert (
        ValidationStage.MONTE_CARLO
        in summary.inconclusive_stages
    )


# ============================================================
# Overfitting extraction
# ============================================================


def test_extract_overfitting_risk():
    results = make_required_results(
        overfitting_risk=OverfittingRisk.LOW,
        overfitting_score=0.15,
    )

    level, score = (
        BenchmarkEngine._extract_overfitting_risk(
            results
        )
    )

    assert level == OverfittingRisk.LOW
    assert score == 0.15


def test_extract_overfitting_missing():
    results = [
        make_result(
            ValidationStage.WALK_FORWARD
        )
    ]

    level, score = (
        BenchmarkEngine._extract_overfitting_risk(
            results
        )
    )

    assert level == OverfittingRisk.MODERATE
    assert score == 0.50


def test_extract_overfitting_invalid_level():
    result = make_result(
        ValidationStage.OVERFITTING_CONTROL,
        diagnostics={
            "risk_level": "INVALID",
            "overall_risk_score": 0.2,
        },
    )

    level, score = (
        BenchmarkEngine._extract_overfitting_risk(
            [result]
        )
    )

    assert level == OverfittingRisk.MODERATE
    assert score == 0.2


def test_extract_overfitting_invalid_score():
    result = make_result(
        ValidationStage.OVERFITTING_CONTROL,
        diagnostics={
            "risk_level": OverfittingRisk.LOW.value,
            "overall_risk_score": "bad",
        },
    )

    level, score = (
        BenchmarkEngine._extract_overfitting_risk(
            [result]
        )
    )

    assert level == OverfittingRisk.LOW
    assert score == 0.50


# ============================================================
# Critical controls
# ============================================================


def test_no_critical_failures():
    engine = BenchmarkEngine()

    summary = engine.summarize(
        make_required_results()
    )

    reasons = engine._critical_stage_failures(
        summary
    )

    assert reasons == []


def test_oos_critical_failure():
    engine = BenchmarkEngine()

    results = make_required_results()

    for index, result in enumerate(results):
        if (
            result.stage
            == ValidationStage.OUT_OF_SAMPLE
        ):
            results[index] = make_result(
                ValidationStage.OUT_OF_SAMPLE,
                status=ValidationStatus.FAILED,
                score=0.2,
            )

    summary = engine.summarize(results)

    reasons = engine._critical_stage_failures(
        summary
    )

    assert any(
        "out-of-sample" in reason.lower()
        for reason in reasons
    )


def test_overfitting_critical_failure():
    engine = BenchmarkEngine()

    results = make_required_results()

    for index, result in enumerate(results):
        if (
            result.stage
            == ValidationStage.OVERFITTING_CONTROL
        ):
            results[index] = make_result(
                ValidationStage.OVERFITTING_CONTROL,
                status=ValidationStatus.FAILED,
                score=0.2,
                diagnostics={
                    "risk_level": (
                        OverfittingRisk.HIGH.value
                    ),
                    "overall_risk_score": 0.8,
                },
            )

    summary = engine.summarize(results)

    reasons = engine._critical_stage_failures(
        summary
    )

    assert any(
        "overfitting" in reason.lower()
        for reason in reasons
    )


def test_significance_critical_failure():
    engine = BenchmarkEngine()

    results = make_required_results()

    for index, result in enumerate(results):
        if (
            result.stage
            == ValidationStage.STATISTICAL_SIGNIFICANCE
        ):
            results[index] = make_result(
                ValidationStage.STATISTICAL_SIGNIFICANCE,
                status=ValidationStatus.FAILED,
                score=0.2,
            )

    summary = engine.summarize(results)

    reasons = engine._critical_stage_failures(
        summary
    )

    assert any(
        "significance" in reason.lower()
        for reason in reasons
    )


# ============================================================
# Decision engine
# ============================================================


def test_decide_robust_alpha():
    engine = BenchmarkEngine()

    decision = engine.decide(
        make_required_results(
            score=0.90,
            overfitting_risk=OverfittingRisk.LOW,
            overfitting_score=0.10,
        )
    )

    assert (
        decision.verdict
        == AlphaVerdict.ROBUST_ALPHA
    )

    assert (
        decision.robust_alpha_supported
        is True
    )

    assert decision.overall_score >= 0.70


def test_decide_high_overfitting_risk():
    engine = BenchmarkEngine()

    results = make_required_results(
        score=0.90,
        overfitting_risk=OverfittingRisk.HIGH,
        overfitting_score=0.80,
    )

    for index, result in enumerate(results):
        if (
            result.stage
            == ValidationStage.OVERFITTING_CONTROL
        ):
            results[index] = make_result(
                ValidationStage.OVERFITTING_CONTROL,
                status=ValidationStatus.FAILED,
                score=0.20,
                diagnostics={
                    "risk_level": (
                        OverfittingRisk.HIGH.value
                    ),
                    "overall_risk_score": 0.80,
                },
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


def test_decide_missing_stage_inconclusive():
    engine = BenchmarkEngine()

    results = make_required_results()

    results = [
        result
        for result in results
        if (
            result.stage
            != ValidationStage.BOOTSTRAP
        )
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


def test_decide_weak_evidence():
    engine = BenchmarkEngine()

    results = make_required_results(
        score=0.70
    )

    noncritical_failures = {
        ValidationStage.BOOTSTRAP,
        ValidationStage.ROBUSTNESS,
        ValidationStage.SENSITIVITY,
    }

    replaced = []

    for result in results:
        if result.stage in noncritical_failures:
            replaced.append(
                make_result(
                    result.stage,
                    status=ValidationStatus.FAILED,
                    score=0.20,
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


def test_decide_rejected():
    engine = BenchmarkEngine()

    results = make_required_results(
        score=0.80
    )

    noncritical_failures = {
        ValidationStage.WALK_FORWARD,
        ValidationStage.BOOTSTRAP,
        ValidationStage.MONTE_CARLO,
        ValidationStage.MULTIPLE_TESTING,
        ValidationStage.ROBUSTNESS,
        ValidationStage.SENSITIVITY,
    }

    replaced = []

    for result in results:
        if result.stage in noncritical_failures:
            replaced.append(
                make_result(
                    result.stage,
                    status=ValidationStatus.FAILED,
                    score=0.20,
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
# Complete analysis
# ============================================================


def test_analyze():
    engine = BenchmarkEngine()

    result = engine.analyze(
        make_required_results(
            score=0.90
        )
    )

    assert result["version"] == "v6.0.2"

    assert isinstance(
        result["summary"],
        BenchmarkEvidenceSummary,
    )

    assert isinstance(
        result["decision"],
        BenchmarkDecision,
    )

    assert result[
        "research_only"
    ] is True

    assert result[
        "real_money_execution"
    ] is False

    assert len(
        result["summary_fingerprint"]
    ) == 64

    assert len(
        result["decision_fingerprint"]
    ) == 64


# ============================================================
# Determinism
# ============================================================


def test_summary_deterministic():
    engine = BenchmarkEngine()

    results = make_required_results()

    first = engine.summarize(results)
    second = engine.summarize(results)

    assert first.fingerprint == second.fingerprint


def test_decision_deterministic():
    engine = BenchmarkEngine()

    results = make_required_results()

    first = engine.decide(results)
    second = engine.decide(results)

    assert first.fingerprint == second.fingerprint


# ============================================================
# Snapshot
# ============================================================


def test_snapshot():
    engine = BenchmarkEngine(
        make_config()
    )

    snapshot = engine.snapshot()

    assert snapshot["version"] == "v6.0.2"

    assert len(
        snapshot["required_stages"]
    ) == 10

    assert snapshot[
        "minimum_stage_score"
    ] == 0.50

    assert snapshot[
        "minimum_pass_ratio"
    ] == 0.80

    assert snapshot[
        "critical_controls"
    ][
        "require_oos_pass"
    ] is True

    assert snapshot[
        "critical_controls"
    ][
        "require_overfitting_pass"
    ] is True

    assert snapshot[
        "critical_controls"
    ][
        "require_significance_pass"
    ] is True

    assert snapshot[
        "research_only"
    ] is True

    assert snapshot[
        "real_money_execution"
    ] is False

    assert len(
        snapshot["config_fingerprint"]
    ) == 64

    assert len(
        snapshot["policy_fingerprint"]
    ) == 64


# ============================================================
# Full v6.0.2 benchmark flow
# ============================================================


def test_complete_benchmark_flow():
    engine = BenchmarkEngine(
        make_config()
    )

    results = make_required_results(
        score=0.92,
        overfitting_risk=OverfittingRisk.LOW,
        overfitting_score=0.08,
    )

    analysis = engine.analyze(results)

    summary = analysis["summary"]
    decision = analysis["decision"]

    assert summary.required_stage_count == 10

    assert (
        summary.observed_required_stage_count
        == 10
    )

    assert (
        summary.passed_required_stage_count
        == 10
    )

    assert summary.pass_ratio == 1.0

    assert (
        summary.all_required_stages_present
        is True
    )

    assert (
        summary.all_required_stages_passed
        is True
    )

    assert (
        decision.verdict
        == AlphaVerdict.ROBUST_ALPHA
    )

    assert (
        decision.overfitting_risk
        == OverfittingRisk.LOW
    )

    assert (
        decision.robust_alpha_supported
        is True
    )

    assert decision.overall_score > 0.85
    assert decision.confidence_score > 0.85
    assert decision.evidence_quality_score > 0.85

    assert analysis[
        "research_only"
    ] is True

    assert analysis[
        "real_money_execution"
    ] is False

    assert len(summary.fingerprint) == 64
    assert len(decision.fingerprint) == 64