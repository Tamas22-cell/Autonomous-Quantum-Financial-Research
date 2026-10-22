from __future__ import annotations

import pytest

from alpha_discovery_laboratory.models import (
    VERSION,
    AlphaValidationError,
    CandidateStatus,
    PerformanceMetrics,
    RejectionReason,
    StrategyCandidate,
    StrategyDirection,
    ValidationStage,
    ValidationStatus,
)
from alpha_discovery_laboratory.validation_pipeline import (
    AlphaValidationPipeline,
    BacktestEvidence,
    CausalEvidence,
    OOSEvidence,
    RobustnessEvidence,
    ValidationConfigurationError,
    ValidationThresholds,
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


def make_backtest(
    **metric_overrides,
) -> BacktestEvidence:
    return BacktestEvidence(
        metrics=make_metrics(
            **metric_overrides
        )
    )


def make_causal(
    *,
    causal_score: float = 0.80,
    stable_effect: bool = True,
    confounding_detected: bool = False,
) -> CausalEvidence:
    return CausalEvidence(
        causal_score=causal_score,
        stable_effect=stable_effect,
        confounding_detected=(
            confounding_detected
        ),
    )


def make_oos(
    *,
    performance_retention: float = 0.80,
    **metric_overrides,
) -> OOSEvidence:
    return OOSEvidence(
        metrics=make_metrics(
            **metric_overrides
        ),
        performance_retention=(
            performance_retention
        ),
    )


def make_robustness(
    *,
    robustness_score: float = 0.80,
    parameter_sensitivity: float = 0.20,
    regime_stable: bool = True,
    perturbation_stable: bool = True,
) -> RobustnessEvidence:
    return RobustnessEvidence(
        robustness_score=robustness_score,
        parameter_sensitivity=(
            parameter_sensitivity
        ),
        regime_stable=regime_stable,
        perturbation_stable=(
            perturbation_stable
        ),
    )


def make_pipeline(
    thresholds: ValidationThresholds | None = None,
) -> AlphaValidationPipeline:
    return AlphaValidationPipeline(
        thresholds=thresholds
    )


def run_good_pipeline(
    pipeline: AlphaValidationPipeline,
    candidate: StrategyCandidate,
):
    return pipeline.run(
        candidate,
        backtest=make_backtest(),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(),
    )


# ============================================================
# Basic
# ============================================================


def test_version():
    assert (
        AlphaValidationPipeline.VERSION
        == VERSION
    )


def test_default_thresholds():
    thresholds = ValidationThresholds()

    assert thresholds.min_sharpe_ratio == 0.75
    assert thresholds.min_profit_factor == 1.05
    assert thresholds.max_drawdown == 0.30
    assert thresholds.min_trade_count == 20
    assert thresholds.min_causal_score == 0.50
    assert (
        thresholds.min_oos_sharpe_ratio
        == 0.40
    )
    assert thresholds.min_oos_retention == 0.40
    assert (
        thresholds.min_robustness_score
        == 0.50
    )
    assert (
        thresholds.max_parameter_sensitivity
        == 0.50
    )


def test_stage_order():
    assert (
        AlphaValidationPipeline.STAGE_ORDER
        == (
            ValidationStage.BACKTEST,
            ValidationStage.CAUSAL,
            ValidationStage.OUT_OF_SAMPLE,
            ValidationStage.ROBUSTNESS,
        )
    )


def test_pipeline_summary_empty():
    pipeline = make_pipeline()

    summary = pipeline.summary()

    assert summary["version"] == VERSION
    assert summary["pipeline_runs"] == 0
    assert summary["accepted"] == 0
    assert summary["rejected"] == 0
    assert summary["audit_records"] == 0
    assert summary["research_only"] is True
    assert (
        summary["real_money_execution"]
        is False
    )


# ============================================================
# Threshold Validation
# ============================================================


def test_negative_sharpe_threshold_allowed():
    thresholds = ValidationThresholds(
        min_sharpe_ratio=-1.0
    )

    assert (
        thresholds.min_sharpe_ratio
        == -1.0
    )


def test_invalid_max_drawdown():
    with pytest.raises(
        ValidationConfigurationError
    ):
        ValidationThresholds(
            max_drawdown=1.5
        )


def test_invalid_causal_threshold():
    with pytest.raises(
        ValidationConfigurationError
    ):
        ValidationThresholds(
            min_causal_score=1.5
        )


def test_invalid_oos_retention_threshold():
    with pytest.raises(
        ValidationConfigurationError
    ):
        ValidationThresholds(
            min_oos_retention=-0.1
        )


def test_invalid_robustness_threshold():
    with pytest.raises(
        ValidationConfigurationError
    ):
        ValidationThresholds(
            min_robustness_score=1.1
        )


def test_invalid_parameter_sensitivity_threshold():
    with pytest.raises(
        ValidationConfigurationError
    ):
        ValidationThresholds(
            max_parameter_sensitivity=1.1
        )


def test_invalid_trade_count():
    with pytest.raises(
        ValidationConfigurationError
    ):
        ValidationThresholds(
            min_trade_count=-1
        )


# ============================================================
# Evidence Validation
# ============================================================


def test_valid_causal_evidence():
    evidence = make_causal()

    assert evidence.causal_score == 0.80


def test_invalid_causal_score_low():
    with pytest.raises(
        AlphaValidationError
    ):
        make_causal(
            causal_score=-0.1
        )


def test_invalid_causal_score_high():
    with pytest.raises(
        AlphaValidationError
    ):
        make_causal(
            causal_score=1.1
        )


def test_valid_oos_evidence():
    evidence = make_oos()

    assert (
        evidence.performance_retention
        == 0.80
    )


def test_invalid_oos_retention():
    with pytest.raises(
        AlphaValidationError
    ):
        make_oos(
            performance_retention=-0.1
        )


def test_valid_robustness_evidence():
    evidence = make_robustness()

    assert (
        evidence.robustness_score
        == 0.80
    )
    assert (
        evidence.parameter_sensitivity
        == 0.20
    )


def test_invalid_robustness_score_low():
    with pytest.raises(
        AlphaValidationError
    ):
        make_robustness(
            robustness_score=-0.1
        )


def test_invalid_robustness_score_high():
    with pytest.raises(
        AlphaValidationError
    ):
        make_robustness(
            robustness_score=1.1
        )


def test_invalid_sensitivity_low():
    with pytest.raises(
        AlphaValidationError
    ):
        make_robustness(
            parameter_sensitivity=-0.1
        )


def test_invalid_sensitivity_high():
    with pytest.raises(
        AlphaValidationError
    ):
        make_robustness(
            parameter_sensitivity=1.1
        )


# ============================================================
# Successful Pipeline
# ============================================================


def test_full_pipeline_passes():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = run_good_pipeline(
        pipeline,
        candidate,
    )

    assert result.accepted is True
    assert result.rejection_reason is None
    assert len(result.results) == 4


def test_successful_candidate_accepted():
    pipeline = make_pipeline()
    candidate = make_candidate()

    run_good_pipeline(
        pipeline,
        candidate,
    )

    assert candidate.accepted is True
    assert (
        candidate.status
        == CandidateStatus.ACCEPTED
    )


def test_successful_stage_order():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = run_good_pipeline(
        pipeline,
        candidate,
    )

    stages = [
        item.stage
        for item in result.results
    ]

    assert stages == [
        ValidationStage.BACKTEST,
        ValidationStage.CAUSAL,
        ValidationStage.OUT_OF_SAMPLE,
        ValidationStage.ROBUSTNESS,
    ]


def test_all_successful_results_passed():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = run_good_pipeline(
        pipeline,
        candidate,
    )

    assert all(
        item.status
        == ValidationStatus.PASSED
        for item in result.results
    )


def test_passed_stage_count():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = run_good_pipeline(
        pipeline,
        candidate,
    )

    assert result.passed_stage_count == 4
    assert result.failed_stage_count == 0


# ============================================================
# Direct Stage Validation
# ============================================================


def test_validate_backtest_direct():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.validate_backtest(
        candidate,
        make_backtest(),
    )

    assert (
        result.stage
        == ValidationStage.BACKTEST
    )
    assert (
        result.status
        == ValidationStatus.PASSED
    )


def test_validate_causal_direct():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.validate_causal(
        candidate,
        make_causal(),
    )

    assert (
        result.stage
        == ValidationStage.CAUSAL
    )
    assert (
        result.status
        == ValidationStatus.PASSED
    )


def test_validate_oos_direct():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.validate_oos(
        candidate,
        make_oos(),
    )

    assert (
        result.stage
        == ValidationStage.OUT_OF_SAMPLE
    )
    assert (
        result.status
        == ValidationStatus.PASSED
    )


def test_validate_robustness_direct():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.validate_robustness(
        candidate,
        make_robustness(),
    )

    assert (
        result.stage
        == ValidationStage.ROBUSTNESS
    )
    assert (
        result.status
        == ValidationStatus.PASSED
    )


# ============================================================
# Backtest Validation
# ============================================================


def test_backtest_low_sharpe_rejected():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(
            sharpe_ratio=0.20
        ),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(),
    )

    assert result.accepted is False
    assert (
        result.rejection_reason
        == RejectionReason.LOW_SHARPE
    )
    assert candidate.rejected is True


def test_backtest_low_profit_factor_rejected():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(
            profit_factor=0.80
        ),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(),
    )

    assert result.accepted is False
    assert (
        result.rejection_reason
        == RejectionReason.NEGATIVE_EXPECTANCY
    )


def test_backtest_excessive_drawdown_rejected():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(
            max_drawdown=0.50
        ),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(),
    )

    assert (
        result.rejection_reason
        == RejectionReason.EXCESSIVE_DRAWDOWN
    )


def test_backtest_insufficient_trades_rejected():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(
            trade_count=5
        ),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(),
    )

    assert result.accepted is False
    assert (
        result.rejection_reason
        == RejectionReason.NEGATIVE_EXPECTANCY
    )


def test_backtest_data_leakage_rejected():
    pipeline = make_pipeline()
    candidate = make_candidate()

    backtest = BacktestEvidence(
        metrics=make_metrics(),
        data_leakage_detected=True,
    )

    result = pipeline.run(
        candidate,
        backtest=backtest,
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(),
    )

    assert result.accepted is False
    assert (
        result.rejection_reason
        == RejectionReason.DATA_LEAKAGE
    )


def test_backtest_lookahead_bias_rejected():
    pipeline = make_pipeline()
    candidate = make_candidate()

    backtest = BacktestEvidence(
        metrics=make_metrics(),
        lookahead_bias_detected=True,
    )

    result = pipeline.run(
        candidate,
        backtest=backtest,
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(),
    )

    assert result.accepted is False
    assert (
        result.rejection_reason
        == RejectionReason.LOOKAHEAD_BIAS
    )


def test_backtest_high_turnover_rejected():
    pipeline = make_pipeline(
        ValidationThresholds(
            max_turnover=1.0
        )
    )

    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(
            turnover=2.0
        ),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(),
    )

    assert (
        result.rejection_reason
        == RejectionReason.HIGH_TURNOVER
    )


def test_backtest_high_execution_cost_rejected():
    pipeline = make_pipeline(
        ValidationThresholds(
            max_execution_cost=0.05
        )
    )

    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(
            execution_cost=0.10
        ),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(),
    )

    assert (
        result.rejection_reason
        == RejectionReason.HIGH_EXECUTION_COST
    )


def test_backtest_failure_stops_pipeline():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(
            sharpe_ratio=0.10
        ),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(),
    )

    assert len(result.results) == 1
    assert (
        result.stopped_at
        == ValidationStage.BACKTEST
    )


# ============================================================
# Causal Validation
# ============================================================


def test_low_causal_score_rejected():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(),
        causal=make_causal(
            causal_score=0.20
        ),
        oos=make_oos(),
        robustness=make_robustness(),
    )

    assert result.accepted is False
    assert (
        result.rejection_reason
        == RejectionReason.CAUSAL_FAILURE
    )


def test_unstable_causal_effect_rejected():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(),
        causal=make_causal(
            stable_effect=False
        ),
        oos=make_oos(),
        robustness=make_robustness(),
    )

    assert (
        result.rejection_reason
        == RejectionReason.CAUSAL_FAILURE
    )


def test_confounding_rejected():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(),
        causal=make_causal(
            confounding_detected=True
        ),
        oos=make_oos(),
        robustness=make_robustness(),
    )

    assert (
        result.rejection_reason
        == RejectionReason.CAUSAL_FAILURE
    )


def test_causal_failure_stops_pipeline():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(),
        causal=make_causal(
            causal_score=0.10
        ),
        oos=make_oos(),
        robustness=make_robustness(),
    )

    assert len(result.results) == 2
    assert (
        result.stopped_at
        == ValidationStage.CAUSAL
    )


# ============================================================
# OOS Validation
# ============================================================


def test_low_oos_sharpe_rejected():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(),
        causal=make_causal(),
        oos=make_oos(
            sharpe_ratio=0.10
        ),
        robustness=make_robustness(),
    )

    assert result.accepted is False
    assert (
        result.rejection_reason
        == RejectionReason.OOS_FAILURE
    )


def test_low_oos_retention_rejected():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(),
        causal=make_causal(),
        oos=make_oos(
            performance_retention=0.10
        ),
        robustness=make_robustness(),
    )

    assert (
        result.rejection_reason
        == RejectionReason.OOS_FAILURE
    )


def test_oos_drawdown_rejected():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(),
        causal=make_causal(),
        oos=make_oos(
            max_drawdown=0.60
        ),
        robustness=make_robustness(),
    )

    assert (
        result.rejection_reason
        == RejectionReason.OOS_FAILURE
    )


def test_oos_failure_stops_pipeline():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(),
        causal=make_causal(),
        oos=make_oos(
            sharpe_ratio=0.10
        ),
        robustness=make_robustness(),
    )

    assert len(result.results) == 3
    assert (
        result.stopped_at
        == ValidationStage.OUT_OF_SAMPLE
    )


# ============================================================
# Robustness Validation
# ============================================================


def test_low_robustness_score_rejected():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(
            robustness_score=0.20
        ),
    )

    assert result.accepted is False
    assert (
        result.rejection_reason
        == RejectionReason.ROBUSTNESS_FAILURE
    )


def test_high_parameter_sensitivity_rejected():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(
            parameter_sensitivity=0.90
        ),
    )

    assert (
        result.rejection_reason
        == RejectionReason.OVERFIT
    )


def test_regime_instability_rejected():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(
            regime_stable=False
        ),
    )

    assert (
        result.rejection_reason
        == RejectionReason.REGIME_DEPENDENT
    )


def test_perturbation_instability_rejected():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(
            perturbation_stable=False
        ),
    )

    assert (
        result.rejection_reason
        == RejectionReason.ROBUSTNESS_FAILURE
    )


def test_robustness_failure_stops_at_robustness():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(
            robustness_score=0.10
        ),
    )

    assert len(result.results) == 4
    assert (
        result.stopped_at
        == ValidationStage.ROBUSTNESS
    )


# ============================================================
# Candidate State
# ============================================================


def test_failed_candidate_status_rejected():
    pipeline = make_pipeline()
    candidate = make_candidate()

    pipeline.run(
        candidate,
        backtest=make_backtest(
            sharpe_ratio=0.10
        ),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(),
    )

    assert (
        candidate.status
        == CandidateStatus.REJECTED
    )


def test_failed_candidate_has_reason():
    pipeline = make_pipeline()
    candidate = make_candidate()

    pipeline.run(
        candidate,
        backtest=make_backtest(
            sharpe_ratio=0.10
        ),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(),
    )

    assert (
        candidate.rejection_reason
        == RejectionReason.LOW_SHARPE
    )


# ============================================================
# History
# ============================================================


def test_history_empty():
    pipeline = make_pipeline()

    assert pipeline.history() == []


def test_history_after_run():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = run_good_pipeline(
        pipeline,
        candidate,
    )

    history = pipeline.history()

    assert len(history) == 1
    assert history[0] == result


def test_history_candidate_filter():
    pipeline = make_pipeline()

    first = make_candidate(
        "alpha_1"
    )
    second = make_candidate(
        "alpha_2"
    )

    first_result = run_good_pipeline(
        pipeline,
        first,
    )

    run_good_pipeline(
        pipeline,
        second,
    )

    history = pipeline.history(
        first.candidate_id
    )

    assert len(history) == 1
    assert history[0] == first_result


def test_latest_after_run():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = run_good_pipeline(
        pipeline,
        candidate,
    )

    latest = pipeline.latest(
        candidate.candidate_id
    )

    assert latest == result
    assert latest is not None
    assert len(latest.results) == 4
    assert (
        latest.results[-1].stage
        == ValidationStage.ROBUSTNESS
    )


def test_latest_unknown_candidate():
    pipeline = make_pipeline()

    assert (
        pipeline.latest(
            "alpha_missing"
        )
        is None
    )


def test_history_multiple_candidates():
    pipeline = make_pipeline()

    first = make_candidate(
        "alpha_1"
    )
    second = make_candidate(
        "alpha_2"
    )

    run_good_pipeline(
        pipeline,
        first,
    )
    run_good_pipeline(
        pipeline,
        second,
    )

    assert len(
        pipeline.history()
    ) == 2


# ============================================================
# Audit
# ============================================================


def test_audit_empty():
    pipeline = make_pipeline()

    assert pipeline.audit_log() == []


def test_audit_after_successful_run():
    pipeline = make_pipeline()
    candidate = make_candidate()

    run_good_pipeline(
        pipeline,
        candidate,
    )

    assert len(
        pipeline.audit_log()
    ) == 4


def test_audit_after_failed_backtest():
    pipeline = make_pipeline()
    candidate = make_candidate()

    pipeline.run(
        candidate,
        backtest=make_backtest(
            sharpe_ratio=0.10
        ),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(),
    )

    assert len(
        pipeline.audit_log()
    ) == 1


def test_audit_returns_copy():
    pipeline = make_pipeline()
    candidate = make_candidate()

    run_good_pipeline(
        pipeline,
        candidate,
    )

    audit = pipeline.audit_log()
    audit.clear()

    assert len(
        pipeline.audit_log()
    ) == 4


# ============================================================
# Fingerprints
# ============================================================


def test_configuration_fingerprint():
    pipeline = make_pipeline()

    fingerprint = (
        pipeline.configuration_fingerprint()
    )

    assert isinstance(
        fingerprint,
        str,
    )
    assert fingerprint


def test_configuration_fingerprint_deterministic():
    first = make_pipeline()
    second = make_pipeline()

    assert (
        first.configuration_fingerprint()
        ==
        second.configuration_fingerprint()
    )


def test_configuration_fingerprint_changes():
    first = make_pipeline()

    second = make_pipeline(
        ValidationThresholds(
            min_sharpe_ratio=1.25
        )
    )

    assert (
        first.configuration_fingerprint()
        !=
        second.configuration_fingerprint()
    )


def test_pipeline_result_fingerprint():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = run_good_pipeline(
        pipeline,
        candidate,
    )

    assert isinstance(
        result.fingerprint,
        str,
    )
    assert result.fingerprint


def test_validation_results_have_fingerprints():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = run_good_pipeline(
        pipeline,
        candidate,
    )

    assert all(
        isinstance(
            item.fingerprint,
            str,
        )
        and item.fingerprint
        for item in result.results
    )


# ============================================================
# Custom Thresholds
# ============================================================


def test_custom_sharpe_threshold_passes():
    pipeline = make_pipeline(
        ValidationThresholds(
            min_sharpe_ratio=0.25
        )
    )

    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(
            sharpe_ratio=0.50
        ),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(),
    )

    assert result.accepted is True


def test_custom_sharpe_threshold_rejects():
    pipeline = make_pipeline(
        ValidationThresholds(
            min_sharpe_ratio=2.0
        )
    )

    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(
            sharpe_ratio=1.5
        ),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(),
    )

    assert result.accepted is False
    assert (
        result.rejection_reason
        == RejectionReason.LOW_SHARPE
    )


# ============================================================
# Summary
# ============================================================


def test_summary_after_success():
    pipeline = make_pipeline()
    candidate = make_candidate()

    run_good_pipeline(
        pipeline,
        candidate,
    )

    summary = pipeline.summary()

    assert summary["version"] == VERSION
    assert summary["pipeline_runs"] == 1
    assert summary["accepted"] == 1
    assert summary["rejected"] == 0
    assert summary["audit_records"] == 4
    assert summary["research_only"] is True
    assert (
        summary["real_money_execution"]
        is False
    )


def test_summary_after_rejection():
    pipeline = make_pipeline()
    candidate = make_candidate()

    pipeline.run(
        candidate,
        backtest=make_backtest(
            sharpe_ratio=0.10
        ),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(),
    )

    summary = pipeline.summary()

    assert summary["pipeline_runs"] == 1
    assert summary["accepted"] == 0
    assert summary["rejected"] == 1
    assert summary["audit_records"] == 1


def test_summary_after_multiple_runs():
    pipeline = make_pipeline()

    first = make_candidate(
        "alpha_1"
    )
    second = make_candidate(
        "alpha_2"
    )

    run_good_pipeline(
        pipeline,
        first,
    )
    run_good_pipeline(
        pipeline,
        second,
    )

    summary = pipeline.summary()

    assert summary["pipeline_runs"] == 2
    assert summary["accepted"] == 2
    assert summary["rejected"] == 0
    assert summary["audit_records"] == 8


def test_summary_stage_order():
    pipeline = make_pipeline()

    summary = pipeline.summary()

    assert summary["stage_order"] == [
        "backtest",
        "causal",
        "out_of_sample",
        "robustness",
    ]


def test_summary_configuration_fingerprint():
    pipeline = make_pipeline()

    summary = pipeline.summary()

    assert (
        summary[
            "configuration_fingerprint"
        ]
        ==
        pipeline.configuration_fingerprint()
    )


# ============================================================
# Isolation
# ============================================================


def test_pipelines_are_independent():
    first = make_pipeline()
    second = make_pipeline()

    candidate = make_candidate()

    run_good_pipeline(
        first,
        candidate,
    )

    assert len(
        first.history()
    ) == 1

    assert second.history() == []


# ============================================================
# Metadata
# ============================================================


def test_pipeline_run_metadata():
    pipeline = make_pipeline()
    candidate = make_candidate()

    result = pipeline.run(
        candidate,
        backtest=make_backtest(),
        causal=make_causal(),
        oos=make_oos(),
        robustness=make_robustness(),
        metadata={
            "experiment": "exp_001",
        },
    )

    assert (
        result.metadata["experiment"]
        == "exp_001"
    )


# ============================================================
# Research Safety
# ============================================================


def test_summary_research_only():
    pipeline = make_pipeline()

    summary = pipeline.summary()

    assert summary["research_only"] is True
    assert (
        summary["real_money_execution"]
        is False
    )