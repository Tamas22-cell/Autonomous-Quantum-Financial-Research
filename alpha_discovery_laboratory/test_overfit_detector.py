from __future__ import annotations

import pytest

from alpha_discovery_laboratory.models import (
    VERSION,
    CandidateStatus,
    PerformanceMetrics,
    RejectionReason,
    StrategyCandidate,
    StrategyDirection,
)
from alpha_discovery_laboratory.overfit_detector import (
    OverfitConfigurationError,
    OverfitDetector,
    OverfitEvidence,
    OverfitThresholds,
    ParameterPerturbation,
    RegimePerformance,
)


# ============================================================
# Helpers
# ============================================================


def make_candidate(
    name: str = "btc_alpha",
) -> StrategyCandidate:
    return StrategyCandidate(
        name=name,
        feature_ids=["feature_001"],
        combination_ids=[],
        direction=StrategyDirection.LONG,
        entry_rule={
            "operator": ">",
            "value": 1.0,
        },
        exit_rule={
            "operator": "<",
            "value": 0.0,
        },
        position_rule={
            "type": "risk_scaled",
        },
        parameters={
            "threshold": 1.0,
        },
        metadata={
            "asset": "BTC",
        },
    )


def make_metrics(
    *,
    total_return: float = 0.20,
    annualized_return: float = 0.18,
    volatility: float = 0.15,
    sharpe_ratio: float = 1.50,
    sortino_ratio: float = 1.80,
    calmar_ratio: float = 1.40,
    max_drawdown: float = 0.12,
    win_rate: float = 0.58,
    profit_factor: float = 1.40,
    turnover: float = 0.50,
    execution_cost: float = 0.01,
    trade_count: int = 100,
) -> PerformanceMetrics:
    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        turnover=turnover,
        execution_cost=execution_cost,
        trade_count=trade_count,
    )


def make_perturbations(
    scores=None,
):
    if scores is None:
        scores = [
            0.82,
            0.80,
            0.78,
        ]

    return [
        ParameterPerturbation(
            name=f"parameter_{index}",
            baseline_value=1.0,
            perturbed_value=(
                1.0 + index * 0.05
            ),
            performance_score=score,
        )
        for index, score in enumerate(
            scores,
            start=1,
        )
    ]


def make_regimes(
    sharpe_values=None,
):
    if sharpe_values is None:
        sharpe_values = [
            1.30,
            1.20,
            1.10,
        ]

    return [
        RegimePerformance(
            regime=f"regime_{index}",
            metrics=make_metrics(
                sharpe_ratio=sharpe,
            ),
        )
        for index, sharpe in enumerate(
            sharpe_values,
            start=1,
        )
    ]


def make_evidence(
    *,
    in_sample=None,
    out_of_sample=None,
    parameter_perturbations=None,
    regime_performance=None,
    bootstrap_scores=None,
    metadata=None,
) -> OverfitEvidence:
    if in_sample is None:
        in_sample = make_metrics(
            total_return=0.20,
            annualized_return=0.18,
            sharpe_ratio=1.50,
            max_drawdown=0.12,
        )

    if out_of_sample is None:
        out_of_sample = make_metrics(
            total_return=0.17,
            annualized_return=0.16,
            sharpe_ratio=1.25,
            max_drawdown=0.14,
        )

    if parameter_perturbations is None:
        parameter_perturbations = (
            make_perturbations()
        )

    if regime_performance is None:
        regime_performance = (
            make_regimes()
        )

    if bootstrap_scores is None:
        bootstrap_scores = [
            0.80,
            0.82,
            0.78,
            0.81,
        ]

    if metadata is None:
        metadata = {
            "experiment": "baseline",
        }

    return OverfitEvidence(
        in_sample=in_sample,
        out_of_sample=out_of_sample,
        parameter_perturbations=list(
            parameter_perturbations
        ),
        regime_performance=list(
            regime_performance
        ),
        bootstrap_scores=list(
            bootstrap_scores
        ),
        metadata=dict(metadata),
    )


def make_good_evidence() -> OverfitEvidence:
    return make_evidence()


def make_detector(
    thresholds: OverfitThresholds | None = None,
) -> OverfitDetector:
    return OverfitDetector(
        thresholds=thresholds
    )


# ============================================================
# Basic
# ============================================================


def test_version():
    assert OverfitDetector.VERSION == VERSION


def test_default_thresholds():
    thresholds = OverfitThresholds()

    assert (
        thresholds.max_sharpe_degradation
        == 0.60
    )
    assert (
        thresholds.max_return_degradation
        == 0.70
    )
    assert (
        thresholds.max_drawdown_increase
        == 0.50
    )
    assert (
        thresholds.max_parameter_sensitivity
        == 0.50
    )
    assert (
        thresholds.max_regime_dispersion
        == 0.60
    )
    assert (
        thresholds.min_oos_retention
        == 0.40
    )
    assert (
        thresholds.min_stability_score
        == 0.50
    )
    assert thresholds.min_regime_count == 2
    assert (
        thresholds.min_perturbation_count
        == 3
    )


def test_empty_summary():
    detector = make_detector()

    summary = detector.summary()

    assert summary["version"] == VERSION
    assert summary["evaluations"] == 0
    assert summary["accepted"] == 0
    assert summary["rejected"] == 0
    assert summary["overfit_detected"] == 0
    assert summary["unstable_detected"] == 0
    assert summary["audit_records"] == 0
    assert summary["research_only"] is True
    assert (
        summary["real_money_execution"]
        is False
    )


# ============================================================
# Threshold Validation
# ============================================================


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_sharpe_degradation", -0.1),
        ("max_sharpe_degradation", 1.1),
        ("max_return_degradation", -0.1),
        ("max_return_degradation", 1.1),
        ("max_drawdown_increase", -0.1),
        ("max_drawdown_increase", 1.1),
        ("max_parameter_sensitivity", -0.1),
        ("max_parameter_sensitivity", 1.1),
        ("max_regime_dispersion", -0.1),
        ("max_regime_dispersion", 1.1),
        ("min_oos_retention", -0.1),
        ("min_oos_retention", 1.1),
        ("min_stability_score", -0.1),
        ("min_stability_score", 1.1),
    ],
)
def test_invalid_bounded_thresholds(
    field,
    value,
):
    with pytest.raises(
        OverfitConfigurationError
    ):
        OverfitThresholds(
            **{
                field: value
            }
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_regime_count", 0),
        ("min_regime_count", -1),
        ("min_perturbation_count", 0),
        ("min_perturbation_count", -1),
    ],
)
def test_invalid_count_thresholds(
    field,
    value,
):
    with pytest.raises(
        OverfitConfigurationError
    ):
        OverfitThresholds(
            **{
                field: value
            }
        )


def test_custom_thresholds():
    thresholds = OverfitThresholds(
        max_sharpe_degradation=0.50,
        max_return_degradation=0.60,
        max_drawdown_increase=0.40,
        max_parameter_sensitivity=0.30,
        max_regime_dispersion=0.40,
        min_oos_retention=0.50,
        min_stability_score=0.60,
        min_regime_count=3,
        min_perturbation_count=4,
    )

    assert (
        thresholds.max_sharpe_degradation
        == 0.50
    )
    assert thresholds.min_regime_count == 3
    assert (
        thresholds.min_perturbation_count
        == 4
    )


# ============================================================
# Parameter Sensitivity
# ============================================================


def test_parameter_sensitivity_stable():
    detector = make_detector()

    value = detector.parameter_sensitivity(
        make_perturbations(
            [
                0.80,
                0.80,
                0.80,
            ]
        )
    )

    assert value == pytest.approx(0.0)


def test_parameter_sensitivity_non_negative():
    detector = make_detector()

    value = detector.parameter_sensitivity(
        make_perturbations()
    )

    assert value >= 0.0


def test_parameter_sensitivity_deterministic():
    detector = make_detector()

    perturbations = make_perturbations()

    first = detector.parameter_sensitivity(
        perturbations
    )
    second = detector.parameter_sensitivity(
        perturbations
    )

    assert first == second


def test_parameter_sensitivity_empty():
    detector = make_detector()

    value = detector.parameter_sensitivity(
        []
    )

    assert isinstance(value, float)
    assert value >= 0.0


# ============================================================
# Regime Dispersion
# ============================================================


def test_regime_dispersion_equal_regimes():
    detector = make_detector()

    value = detector.regime_dispersion(
        make_regimes(
            [
                1.20,
                1.20,
                1.20,
            ]
        )
    )

    assert value == pytest.approx(0.0)


def test_regime_dispersion_non_negative():
    detector = make_detector()

    value = detector.regime_dispersion(
        make_regimes()
    )

    assert value >= 0.0


def test_regime_dispersion_deterministic():
    detector = make_detector()

    regimes = make_regimes()

    first = detector.regime_dispersion(
        regimes
    )
    second = detector.regime_dispersion(
        regimes
    )

    assert first == second


def test_regime_dispersion_empty():
    detector = make_detector()

    value = detector.regime_dispersion(
        []
    )

    assert isinstance(value, float)
    assert value >= 0.0


# ============================================================
# Bootstrap Stability
# ============================================================


def test_bootstrap_stability_stable():
    detector = make_detector()

    score = detector.bootstrap_stability(
        [
            0.80,
            0.80,
            0.80,
        ]
    )

    assert 0.0 <= score <= 1.0


def test_bootstrap_stability_variable():
    detector = make_detector()

    score = detector.bootstrap_stability(
        [
            0.20,
            0.80,
            0.40,
            0.90,
        ]
    )

    assert 0.0 <= score <= 1.0


def test_bootstrap_stability_deterministic():
    detector = make_detector()

    values = [
        0.70,
        0.80,
        0.75,
    ]

    first = detector.bootstrap_stability(
        values
    )
    second = detector.bootstrap_stability(
        values
    )

    assert first == second


def test_bootstrap_stability_empty():
    detector = make_detector()

    score = detector.bootstrap_stability(
        []
    )

    assert 0.0 <= score <= 1.0


# ============================================================
# Successful Detection
# ============================================================


def test_good_evidence_not_overfit():
    detector = make_detector()
    candidate = make_candidate()

    result = detector.detect(
        candidate,
        make_good_evidence(),
    )

    assert result.overfit_detected is False


def test_good_evidence_not_unstable():
    detector = make_detector()
    candidate = make_candidate()

    result = detector.detect(
        candidate,
        make_good_evidence(),
    )

    assert result.unstable_detected is False


def test_good_evidence_accepted():
    detector = make_detector()
    candidate = make_candidate()

    result = detector.detect(
        candidate,
        make_good_evidence(),
    )

    assert result.accepted is True


def test_good_evidence_no_rejection_reason():
    detector = make_detector()
    candidate = make_candidate()

    result = detector.detect(
        candidate,
        make_good_evidence(),
    )

    assert result.rejection_reason is None


def test_detection_candidate_id():
    detector = make_detector()
    candidate = make_candidate()

    result = detector.detect(
        candidate,
        make_good_evidence(),
    )

    assert (
        result.candidate_id
        == candidate.candidate_id
    )


def test_detection_has_stability_score():
    detector = make_detector()
    candidate = make_candidate()

    result = detector.detect(
        candidate,
        make_good_evidence(),
    )

    assert (
        0.0
        <= result.stability_score
        <= 1.0
    )


def test_detection_has_oos_retention():
    detector = make_detector()
    candidate = make_candidate()

    result = detector.detect(
        candidate,
        make_good_evidence(),
    )

    assert result.oos_retention >= 0.0


def test_detection_fingerprint():
    detector = make_detector()
    candidate = make_candidate()

    result = detector.detect(
        candidate,
        make_good_evidence(),
    )

    assert isinstance(
        result.fingerprint,
        str,
    )
    assert result.fingerprint


# ============================================================
# Sharpe Degradation
# ============================================================


def test_severe_sharpe_degradation_detected():
    detector = make_detector()
    candidate = make_candidate()

    evidence = make_evidence(
        out_of_sample=make_metrics(
            total_return=0.17,
            annualized_return=0.16,
            sharpe_ratio=0.20,
            max_drawdown=0.14,
        )
    )

    result = detector.detect(
        candidate,
        evidence,
    )

    assert result.overfit_detected is True
    assert (
        "sharpe_degradation"
        in result.failed_checks
    )
    assert (
        result.rejection_reason
        == RejectionReason.OVERFIT
    )


def test_sharpe_degradation_metric():
    detector = make_detector()
    candidate = make_candidate()

    result = detector.detect(
        candidate,
        make_good_evidence(),
    )

    assert result.sharpe_degradation >= 0.0


# ============================================================
# Return Degradation
# ============================================================


def test_severe_return_degradation_detected():
    detector = make_detector()
    candidate = make_candidate()

    evidence = make_evidence(
        out_of_sample=make_metrics(
            total_return=0.02,
            annualized_return=0.02,
            sharpe_ratio=1.20,
            max_drawdown=0.14,
        )
    )

    result = detector.detect(
        candidate,
        evidence,
    )

    assert result.overfit_detected is True
    assert (
        "return_degradation"
        in result.failed_checks
    )
    assert (
        result.rejection_reason
        == RejectionReason.OVERFIT
    )


def test_return_degradation_metric():
    detector = make_detector()
    candidate = make_candidate()

    result = detector.detect(
        candidate,
        make_good_evidence(),
    )

    assert result.return_degradation >= 0.0


# ============================================================
# Drawdown Increase
# ============================================================


def test_excessive_drawdown_increase_detected():
    detector = make_detector()
    candidate = make_candidate()

    evidence = make_evidence(
        out_of_sample=make_metrics(
            total_return=0.17,
            annualized_return=0.16,
            sharpe_ratio=1.20,
            max_drawdown=0.30,
        )
    )

    result = detector.detect(
        candidate,
        evidence,
    )

    assert result.overfit_detected is False
    assert result.unstable_detected is True
    assert (
        "drawdown_increase"
        in result.failed_checks
    )
    assert (
        result.rejection_reason
        == RejectionReason.LOW_STABILITY
    )


def test_drawdown_increase_metric():
    detector = make_detector()
    candidate = make_candidate()

    result = detector.detect(
        candidate,
        make_good_evidence(),
    )

    assert result.drawdown_increase >= 0.0


# ============================================================
# Parameter Instability
# ============================================================


def test_parameter_instability_detected():
    detector = make_detector()
    candidate = make_candidate()

    evidence = make_evidence(
        parameter_perturbations=(
            make_perturbations(
                [
                    0.10,
                    0.90,
                    0.20,
                ]
            )
        )
    )

    result = detector.detect(
        candidate,
        evidence,
    )

    assert (
        result.parameter_sensitivity
        >= 0.0
    )

    assert (
        result.overfit_detected
        or result.unstable_detected
    )


def test_insufficient_perturbations_unstable():
    detector = make_detector()
    candidate = make_candidate()

    evidence = make_evidence(
        parameter_perturbations=[
            ParameterPerturbation(
                name="only_one",
                baseline_value=1.0,
                perturbed_value=1.1,
                performance_score=0.80,
            )
        ]
    )

    result = detector.detect(
        candidate,
        evidence,
    )

    assert result.unstable_detected is True
    assert (
        "insufficient_parameter_perturbations"
        in result.failed_checks
    )


# ============================================================
# Regime Instability
# ============================================================


def test_regime_instability_detected():
    detector = make_detector()
    candidate = make_candidate()

    evidence = make_evidence(
        regime_performance=(
            make_regimes(
                [
                    0.10,
                    2.00,
                    0.20,
                ]
            )
        )
    )

    result = detector.detect(
        candidate,
        evidence,
    )

    assert result.regime_dispersion >= 0.0

    assert (
        result.overfit_detected
        or result.unstable_detected
    )


def test_insufficient_regimes_unstable():
    detector = make_detector()
    candidate = make_candidate()

    evidence = make_evidence(
        regime_performance=[
            RegimePerformance(
                regime="risk_on",
                metrics=make_metrics(),
            )
        ]
    )

    result = detector.detect(
        candidate,
        evidence,
    )

    assert result.unstable_detected is True
    assert (
        "insufficient_regime_coverage"
        in result.failed_checks
    )


# ============================================================
# OOS Retention
# ============================================================


def test_low_oos_retention_detected():
    detector = make_detector()
    candidate = make_candidate()

    evidence = make_evidence(
        out_of_sample=make_metrics(
            total_return=0.02,
            annualized_return=0.02,
            sharpe_ratio=0.20,
            max_drawdown=0.14,
        )
    )

    result = detector.detect(
        candidate,
        evidence,
    )

    assert result.oos_retention < 0.40
    assert result.overfit_detected is True
    assert (
        "oos_retention"
        in result.failed_checks
    )
    assert result.accepted is False


# ============================================================
# detect() vs detect_and_apply()
# ============================================================


def test_detect_does_not_reject_candidate():
    detector = make_detector()
    candidate = make_candidate()

    evidence = make_evidence(
        out_of_sample=make_metrics(
            total_return=0.01,
            annualized_return=0.01,
            sharpe_ratio=0.10,
            max_drawdown=0.30,
        )
    )

    result = detector.detect(
        candidate,
        evidence,
    )

    assert result.accepted is False
    assert (
        candidate.status
        != CandidateStatus.REJECTED
    )


def test_detect_and_apply_rejects_candidate():
    detector = make_detector()
    candidate = make_candidate()

    evidence = make_evidence(
        out_of_sample=make_metrics(
            total_return=0.01,
            annualized_return=0.01,
            sharpe_ratio=0.10,
            max_drawdown=0.30,
        )
    )

    result = detector.detect_and_apply(
        candidate,
        evidence,
    )

    assert result.accepted is False
    assert candidate.rejected is True
    assert (
        candidate.status
        == CandidateStatus.REJECTED
    )


def test_detect_and_apply_sets_rejection_reason():
    detector = make_detector()
    candidate = make_candidate()

    evidence = make_evidence(
        out_of_sample=make_metrics(
            total_return=0.01,
            annualized_return=0.01,
            sharpe_ratio=0.10,
            max_drawdown=0.30,
        )
    )

    result = detector.detect_and_apply(
        candidate,
        evidence,
    )

    assert (
        candidate.rejection_reason
        == result.rejection_reason
    )


def test_detect_and_apply_good_candidate_not_rejected():
    detector = make_detector()
    candidate = make_candidate()

    result = detector.detect_and_apply(
        candidate,
        make_good_evidence(),
    )

    assert result.accepted is True
    assert candidate.rejected is False


# ============================================================
# History
# ============================================================


def test_history_empty():
    detector = make_detector()

    assert detector.history() == []


def test_history_after_detection():
    detector = make_detector()
    candidate = make_candidate()

    result = detector.detect(
        candidate,
        make_good_evidence(),
    )

    history = detector.history()

    assert len(history) == 1
    assert history[0] == result


def test_history_candidate_filter():
    detector = make_detector()

    first = make_candidate(
        "alpha_one"
    )
    second = make_candidate(
        "alpha_two"
    )

    first_result = detector.detect(
        first,
        make_good_evidence(),
    )

    detector.detect(
        second,
        make_good_evidence(),
    )

    history = detector.history(
        first.candidate_id
    )

    assert len(history) == 1
    assert history[0] == first_result


def test_history_multiple_candidates():
    detector = make_detector()

    detector.detect(
        make_candidate("alpha_one"),
        make_good_evidence(),
    )

    detector.detect(
        make_candidate("alpha_two"),
        make_good_evidence(),
    )

    assert len(
        detector.history()
    ) == 2


# ============================================================
# Latest
# ============================================================


def test_latest_unknown_candidate():
    detector = make_detector()

    assert (
        detector.latest(
            "alpha_missing"
        )
        is None
    )


def test_latest_candidate():
    detector = make_detector()
    candidate = make_candidate()

    first = detector.detect(
        candidate,
        make_good_evidence(),
    )

    assert (
        detector.latest(
            candidate.candidate_id
        )
        == first
    )


def test_latest_returns_newest():
    detector = make_detector()
    candidate = make_candidate()

    detector.detect(
        candidate,
        make_good_evidence(),
    )

    second_evidence = make_evidence(
        out_of_sample=make_metrics(
            total_return=0.02,
            annualized_return=0.02,
            sharpe_ratio=0.10,
            max_drawdown=0.30,
        )
    )

    second = detector.detect(
        candidate,
        second_evidence,
    )

    assert (
        detector.latest(
            candidate.candidate_id
        )
        == second
    )


# ============================================================
# Audit
# ============================================================


def test_audit_empty():
    detector = make_detector()

    assert detector.audit_log() == []


def test_audit_after_detection():
    detector = make_detector()
    candidate = make_candidate()

    detector.detect(
        candidate,
        make_good_evidence(),
    )

    audit = detector.audit_log()

    assert len(audit) == 1
    assert (
        audit[0].candidate_id
        == candidate.candidate_id
    )


def test_audit_after_multiple_detections():
    detector = make_detector()

    detector.detect(
        make_candidate("alpha_one"),
        make_good_evidence(),
    )

    detector.detect(
        make_candidate("alpha_two"),
        make_good_evidence(),
    )

    assert len(
        detector.audit_log()
    ) == 2


def test_audit_returns_copy():
    detector = make_detector()
    candidate = make_candidate()

    detector.detect(
        candidate,
        make_good_evidence(),
    )

    audit = detector.audit_log()
    audit.clear()

    assert len(
        detector.audit_log()
    ) == 1


# ============================================================
# Configuration Fingerprint
# ============================================================


def test_configuration_fingerprint():
    detector = make_detector()

    fingerprint = (
        detector.configuration_fingerprint()
    )

    assert isinstance(
        fingerprint,
        str,
    )
    assert fingerprint


def test_configuration_fingerprint_deterministic():
    first = make_detector()
    second = make_detector()

    assert (
        first.configuration_fingerprint()
        ==
        second.configuration_fingerprint()
    )


def test_configuration_fingerprint_changes():
    first = make_detector()

    second = make_detector(
        OverfitThresholds(
            max_sharpe_degradation=0.40
        )
    )

    assert (
        first.configuration_fingerprint()
        !=
        second.configuration_fingerprint()
    )


# ============================================================
# Summary
# ============================================================


def test_summary_after_accepted_detection():
    detector = make_detector()
    candidate = make_candidate()

    detector.detect(
        candidate,
        make_good_evidence(),
    )

    summary = detector.summary()

    assert summary["evaluations"] == 1
    assert summary["accepted"] == 1
    assert summary["rejected"] == 0
    assert summary["overfit_detected"] == 0
    assert summary["unstable_detected"] == 0
    assert summary["audit_records"] == 1


def test_summary_after_rejected_detection():
    detector = make_detector()
    candidate = make_candidate()

    evidence = make_evidence(
        out_of_sample=make_metrics(
            total_return=0.01,
            annualized_return=0.01,
            sharpe_ratio=0.10,
            max_drawdown=0.30,
        )
    )

    detector.detect(
        candidate,
        evidence,
    )

    summary = detector.summary()

    assert summary["evaluations"] == 1
    assert summary["rejected"] == 1
    assert summary["overfit_detected"] >= 1


def test_summary_multiple_evaluations():
    detector = make_detector()

    detector.detect(
        make_candidate("alpha_one"),
        make_good_evidence(),
    )

    detector.detect(
        make_candidate("alpha_two"),
        make_good_evidence(),
    )

    summary = detector.summary()

    assert summary["evaluations"] == 2
    assert summary["accepted"] == 2
    assert summary["audit_records"] == 2


def test_summary_configuration_fingerprint():
    detector = make_detector()

    summary = detector.summary()

    assert (
        summary[
            "configuration_fingerprint"
        ]
        ==
        detector.configuration_fingerprint()
    )


# ============================================================
# Evidence Metadata
# ============================================================


def test_evidence_metadata():
    evidence = make_good_evidence()

    assert (
        evidence.metadata["experiment"]
        == "baseline"
    )


def test_parameter_perturbation_metadata():
    perturbation = ParameterPerturbation(
        name="window",
        baseline_value=20.0,
        perturbed_value=25.0,
        performance_score=0.80,
        metadata={
            "source": "grid_search"
        },
    )

    assert (
        perturbation.metadata["source"]
        == "grid_search"
    )


def test_regime_performance_metadata():
    regime = RegimePerformance(
        regime="risk_on",
        metrics=make_metrics(),
        metadata={
            "period": "2025"
        },
    )

    assert regime.regime == "risk_on"
    assert (
        regime.metadata["period"]
        == "2025"
    )


# ============================================================
# Frozen Evidence
# ============================================================


def test_overfit_evidence_is_frozen():
    evidence = make_good_evidence()

    with pytest.raises(
        AttributeError
    ):
        evidence.out_of_sample = (
            make_metrics()
        )


# ============================================================
# Isolation
# ============================================================


def test_detectors_are_independent():
    first = make_detector()
    second = make_detector()

    first.detect(
        make_candidate(),
        make_good_evidence(),
    )

    assert len(
        first.history()
    ) == 1

    assert second.history() == []


# ============================================================
# Research Safety
# ============================================================


def test_research_safety_flags():
    detector = make_detector()

    summary = detector.summary()

    assert summary["research_only"] is True
    assert (
        summary["real_money_execution"]
        is False
    )