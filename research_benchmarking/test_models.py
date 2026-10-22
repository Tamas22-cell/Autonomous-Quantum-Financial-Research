from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from research_benchmarking.models import (
    VERSION,
    AlphaVerdict,
    BenchmarkReport,
    ConfidenceInterval,
    MultipleTestingMethod,
    OverfittingDiagnostics,
    OverfittingRisk,
    PerformanceMetrics,
    RegimeType,
    RegimeValidationResult,
    SensitivityResult,
    SignificanceLevel,
    StatisticalTestResult,
    ValidationConfig,
    ValidationConfigurationError,
    ValidationResult,
    ValidationStage,
    ValidationStatus,
    canonical_json,
    stable_hash,
    utc_now,
)


# ============================================================
# Helpers
# ============================================================


def make_metrics(
    *,
    sharpe_ratio: float = 1.50,
    observations: int = 100,
) -> PerformanceMetrics:
    return PerformanceMetrics(
        total_return=0.20,
        annualized_return=0.15,
        volatility=0.10,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=1.80,
        max_drawdown=0.08,
        win_rate=0.57,
        profit_factor=1.45,
        alpha=0.06,
        beta=0.85,
        information_ratio=0.90,
        observations=observations,
    )


def make_validation_result(
    stage: ValidationStage,
    status: ValidationStatus = ValidationStatus.PASSED,
) -> ValidationResult:
    return ValidationResult(
        stage=stage,
        status=status,
        score=0.85,
        message="Validation completed.",
        metrics=make_metrics(),
    )


def all_required_results():
    return [
        make_validation_result(
            ValidationStage.WALK_FORWARD
        ),
        make_validation_result(
            ValidationStage.OUT_OF_SAMPLE
        ),
        make_validation_result(
            ValidationStage.BOOTSTRAP
        ),
        make_validation_result(
            ValidationStage.MONTE_CARLO
        ),
        make_validation_result(
            ValidationStage.STATISTICAL_SIGNIFICANCE
        ),
        make_validation_result(
            ValidationStage.MULTIPLE_TESTING
        ),
        make_validation_result(
            ValidationStage.ROBUSTNESS
        ),
        make_validation_result(
            ValidationStage.SENSITIVITY
        ),
        make_validation_result(
            ValidationStage.REGIME_STABILITY
        ),
        make_validation_result(
            ValidationStage.OVERFITTING_CONTROL
        ),
    ]


# ============================================================
# Version / helpers
# ============================================================


def test_version():
    assert VERSION == "v6.0.2"


def test_utc_now_is_timezone_aware():
    now = utc_now()

    assert now.tzinfo is not None
    assert now.utcoffset() is not None


def test_canonical_json_is_deterministic():
    first = canonical_json(
        {
            "b": 2,
            "a": 1,
        }
    )

    second = canonical_json(
        {
            "a": 1,
            "b": 2,
        }
    )

    assert first == second


def test_stable_hash_is_deterministic():
    first = stable_hash(
        {
            "asset": "BTC",
            "horizon": "7D",
        }
    )

    second = stable_hash(
        {
            "horizon": "7D",
            "asset": "BTC",
        }
    )

    assert first == second
    assert len(first) == 64


def test_stable_hash_changes_when_input_changes():
    first = stable_hash(
        {
            "asset": "BTC",
        }
    )

    second = stable_hash(
        {
            "asset": "ETH",
        }
    )

    assert first != second


# ============================================================
# Enums
# ============================================================


def test_alpha_verdict_values():
    assert (
        AlphaVerdict.ROBUST_ALPHA.value
        == "ROBUST_ALPHA"
    )

    assert (
        AlphaVerdict.OVERFIT_RISK.value
        == "OVERFIT_RISK"
    )


def test_validation_status_values():
    assert (
        ValidationStatus.PASSED.value
        == "PASSED"
    )

    assert (
        ValidationStatus.FAILED.value
        == "FAILED"
    )


def test_validation_stage_count():
    assert len(ValidationStage) == 10


def test_multiple_testing_methods():
    assert (
        MultipleTestingMethod.BONFERRONI.value
        == "BONFERRONI"
    )

    assert (
        MultipleTestingMethod.HOLM.value
        == "HOLM"
    )

    assert (
        MultipleTestingMethod.BENJAMINI_HOCHBERG.value
        == "BENJAMINI_HOCHBERG"
    )


def test_regime_type_values():
    assert RegimeType.BULL.value == "BULL"
    assert RegimeType.BEAR.value == "BEAR"
    assert RegimeType.MIXED.value == "MIXED"


# ============================================================
# ValidationConfig
# ============================================================


def test_default_validation_config():
    config = ValidationConfig()

    assert config.confidence_level == 0.95
    assert config.significance_threshold == 0.05
    assert config.bootstrap_samples == 1000
    assert config.monte_carlo_simulations == 1000
    assert config.walk_forward_windows == 5
    assert config.train_fraction == 0.70
    assert config.validation_fraction == 0.15
    assert config.test_fraction == 0.15
    assert config.minimum_observations == 30
    assert config.random_seed == 42

    assert (
        config.multiple_testing_method
        == MultipleTestingMethod.BENJAMINI_HOCHBERG
    )

    assert config.anti_overfitting_enabled is True
    assert config.research_only is True
    assert config.real_money_execution is False


def test_validation_config_is_frozen():
    config = ValidationConfig()

    with pytest.raises(FrozenInstanceError):
        config.bootstrap_samples = 500


def test_validation_config_fingerprint_is_stable():
    first = ValidationConfig(
        metadata={
            "asset": "BTC",
            "horizon": "7D",
        }
    )

    second = ValidationConfig(
        metadata={
            "horizon": "7D",
            "asset": "BTC",
        }
    )

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_validation_config_fingerprint_changes():
    first = ValidationConfig(
        random_seed=42
    )

    second = ValidationConfig(
        random_seed=43
    )

    assert first.fingerprint != second.fingerprint


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
    ],
)
def test_invalid_confidence_level(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        ValidationConfig(
            confidence_level=value
        )


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
    ],
)
def test_invalid_significance_threshold(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        ValidationConfig(
            significance_threshold=value
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_invalid_bootstrap_samples(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        ValidationConfig(
            bootstrap_samples=value
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_invalid_monte_carlo_simulations(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        ValidationConfig(
            monte_carlo_simulations=value
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        2.5,
    ],
)
def test_invalid_walk_forward_windows(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        ValidationConfig(
            walk_forward_windows=value
        )


def test_invalid_split_total():
    with pytest.raises(
        ValidationConfigurationError
    ):
        ValidationConfig(
            train_fraction=0.60,
            validation_fraction=0.20,
            test_fraction=0.10,
        )


def test_real_money_execution_rejected():
    with pytest.raises(
        ValidationConfigurationError
    ):
        ValidationConfig(
            real_money_execution=True
        )


@pytest.mark.parametrize(
    "value",
    [
        -1,
        True,
        2.5,
    ],
)
def test_invalid_random_seed(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        ValidationConfig(
            random_seed=value
        )


# ============================================================
# PerformanceMetrics
# ============================================================


def test_performance_metrics():
    metrics = make_metrics()

    assert metrics.total_return == 0.20
    assert metrics.sharpe_ratio == 1.50
    assert metrics.observations == 100
    assert len(metrics.fingerprint) == 64


def test_performance_metrics_is_frozen():
    metrics = make_metrics()

    with pytest.raises(FrozenInstanceError):
        metrics.sharpe_ratio = 2.0


def test_performance_metrics_fingerprint_stable():
    first = make_metrics()
    second = make_metrics()

    assert first.fingerprint == second.fingerprint


def test_performance_metrics_fingerprint_changes():
    first = make_metrics(
        sharpe_ratio=1.50
    )

    second = make_metrics(
        sharpe_ratio=1.60
    )

    assert first.fingerprint != second.fingerprint


@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        1.01,
    ],
)
def test_invalid_win_rate(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        PerformanceMetrics(
            win_rate=value
        )


def test_invalid_negative_volatility():
    with pytest.raises(
        ValidationConfigurationError
    ):
        PerformanceMetrics(
            volatility=-0.01
        )


def test_invalid_negative_drawdown():
    with pytest.raises(
        ValidationConfigurationError
    ):
        PerformanceMetrics(
            max_drawdown=-0.01
        )


def test_invalid_negative_profit_factor():
    with pytest.raises(
        ValidationConfigurationError
    ):
        PerformanceMetrics(
            profit_factor=-0.01
        )


def test_invalid_negative_observations():
    with pytest.raises(
        ValidationConfigurationError
    ):
        PerformanceMetrics(
            observations=-1
        )


# ============================================================
# ConfidenceInterval
# ============================================================


def test_confidence_interval():
    interval = ConfidenceInterval(
        lower=0.02,
        upper=0.08,
        estimate=0.05,
    )

    assert interval.width == pytest.approx(
        0.06
    )

    assert interval.contains_zero is False
    assert len(interval.fingerprint) == 64


def test_confidence_interval_contains_zero():
    interval = ConfidenceInterval(
        lower=-0.02,
        upper=0.08,
    )

    assert interval.contains_zero is True


def test_confidence_interval_zero_boundary():
    interval = ConfidenceInterval(
        lower=0.0,
        upper=0.08,
    )

    assert interval.contains_zero is True


def test_invalid_confidence_interval_bounds():
    with pytest.raises(
        ValidationConfigurationError
    ):
        ConfidenceInterval(
            lower=0.10,
            upper=0.05,
        )


def test_invalid_confidence_interval_level():
    with pytest.raises(
        ValidationConfigurationError
    ):
        ConfidenceInterval(
            lower=0.0,
            upper=1.0,
            confidence_level=1.0,
        )


# ============================================================
# StatisticalTestResult
# ============================================================


def test_statistical_test_significant():
    result = StatisticalTestResult(
        test_name="mean_return_test",
        statistic=2.50,
        p_value=0.02,
    )

    assert result.significant is True

    assert (
        result.significance_level
        == SignificanceLevel.MODERATE
    )


def test_statistical_test_strong():
    result = StatisticalTestResult(
        test_name="alpha_test",
        statistic=3.50,
        p_value=0.005,
    )

    assert (
        result.significance_level
        == SignificanceLevel.STRONG
    )


def test_statistical_test_weak():
    result = StatisticalTestResult(
        test_name="alpha_test",
        statistic=1.70,
        p_value=0.08,
    )

    assert result.significant is False

    assert (
        result.significance_level
        == SignificanceLevel.WEAK
    )


def test_statistical_test_not_significant():
    result = StatisticalTestResult(
        test_name="alpha_test",
        statistic=0.50,
        p_value=0.50,
    )

    assert result.significant is False

    assert (
        result.significance_level
        == SignificanceLevel.NOT_SIGNIFICANT
    )


def test_adjusted_p_value_used():
    result = StatisticalTestResult(
        test_name="multiple_test",
        statistic=2.80,
        p_value=0.01,
        adjusted_p_value=0.08,
    )

    assert result.effective_p_value == 0.08
    assert result.significant is False


def test_adjusted_p_value_can_remain_significant():
    result = StatisticalTestResult(
        test_name="multiple_test",
        statistic=3.10,
        p_value=0.005,
        adjusted_p_value=0.03,
    )

    assert result.effective_p_value == 0.03
    assert result.significant is True


def test_statistical_test_fingerprint():
    result = StatisticalTestResult(
        test_name="alpha_test",
        statistic=2.50,
        p_value=0.02,
    )

    assert len(result.fingerprint) == 64


def test_empty_statistical_test_name_rejected():
    with pytest.raises(
        ValidationConfigurationError
    ):
        StatisticalTestResult(
            test_name="",
            statistic=1.0,
            p_value=0.05,
        )


@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        1.01,
    ],
)
def test_invalid_p_value(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        StatisticalTestResult(
            test_name="test",
            statistic=1.0,
            p_value=value,
        )


# ============================================================
# ValidationResult
# ============================================================


def test_validation_result_passed():
    result = make_validation_result(
        ValidationStage.WALK_FORWARD
    )

    assert result.successful is True
    assert result.score == 0.85
    assert len(result.fingerprint) == 64


def test_validation_result_failed():
    result = make_validation_result(
        ValidationStage.OUT_OF_SAMPLE,
        ValidationStatus.FAILED,
    )

    assert result.successful is False


def test_validation_result_significant_test_count():
    result = ValidationResult(
        stage=(
            ValidationStage.STATISTICAL_SIGNIFICANCE
        ),
        status=ValidationStatus.PASSED,
        statistical_tests=[
            StatisticalTestResult(
                test_name="test_1",
                statistic=3.0,
                p_value=0.01,
            ),
            StatisticalTestResult(
                test_name="test_2",
                statistic=1.0,
                p_value=0.50,
            ),
        ],
    )

    assert (
        result.significant_test_count
        == 1
    )


@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        1.01,
    ],
)
def test_invalid_validation_score(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        ValidationResult(
            stage=ValidationStage.BOOTSTRAP,
            status=ValidationStatus.PASSED,
            score=value,
        )


# ============================================================
# RegimeValidationResult
# ============================================================


def test_regime_validation_result():
    result = RegimeValidationResult(
        regime=RegimeType.BULL,
        observations=100,
        metrics=make_metrics(),
        stable=True,
        score=0.90,
    )

    assert result.regime == RegimeType.BULL
    assert result.stable is True
    assert result.score == 0.90
    assert len(result.fingerprint) == 64


def test_invalid_regime_score():
    with pytest.raises(
        ValidationConfigurationError
    ):
        RegimeValidationResult(
            regime=RegimeType.BEAR,
            observations=50,
            metrics=make_metrics(),
            stable=False,
            score=1.10,
        )


def test_invalid_regime_observations():
    with pytest.raises(
        ValidationConfigurationError
    ):
        RegimeValidationResult(
            regime=RegimeType.BEAR,
            observations=-1,
            metrics=make_metrics(),
            stable=False,
            score=0.50,
        )


# ============================================================
# SensitivityResult
# ============================================================


def test_sensitivity_result():
    result = SensitivityResult(
        parameter_name="lookback",
        baseline_value=20.0,
        tested_values=[
            10.0,
            20.0,
            30.0,
        ],
        performance_values=[
            1.20,
            1.50,
            1.35,
        ],
        sensitivity_score=0.82,
        robust=True,
    )

    assert result.robust is True
    assert result.sensitivity_score == 0.82
    assert len(result.fingerprint) == 64


def test_empty_sensitivity_parameter_name():
    with pytest.raises(
        ValidationConfigurationError
    ):
        SensitivityResult(
            parameter_name="",
            baseline_value=20.0,
            tested_values=[10.0],
            performance_values=[1.0],
            sensitivity_score=0.50,
            robust=False,
        )


def test_empty_sensitivity_values():
    with pytest.raises(
        ValidationConfigurationError
    ):
        SensitivityResult(
            parameter_name="lookback",
            baseline_value=20.0,
            tested_values=[],
            performance_values=[],
            sensitivity_score=0.50,
            robust=False,
        )


def test_mismatched_sensitivity_values():
    with pytest.raises(
        ValidationConfigurationError
    ):
        SensitivityResult(
            parameter_name="lookback",
            baseline_value=20.0,
            tested_values=[
                10.0,
                20.0,
            ],
            performance_values=[
                1.0
            ],
            sensitivity_score=0.50,
            robust=False,
        )


def test_invalid_sensitivity_score():
    with pytest.raises(
        ValidationConfigurationError
    ):
        SensitivityResult(
            parameter_name="lookback",
            baseline_value=20.0,
            tested_values=[20.0],
            performance_values=[1.0],
            sensitivity_score=1.50,
            robust=False,
        )


# ============================================================
# OverfittingDiagnostics
# ============================================================


def test_overfitting_diagnostics():
    diagnostics = OverfittingDiagnostics(
        in_sample_score=1.50,
        out_of_sample_score=1.20,
        degradation_ratio=0.20,
        parameter_count=5,
        observation_count=500,
        multiple_testing_count=10,
        probability_of_backtest_overfitting=0.15,
        deflated_sharpe_ratio=1.10,
        risk=OverfittingRisk.LOW,
    )

    assert (
        diagnostics.risk
        == OverfittingRisk.LOW
    )

    assert (
        diagnostics.probability_of_backtest_overfitting
        == 0.15
    )

    assert len(diagnostics.fingerprint) == 64


def test_overfitting_diagnostics_is_frozen():
    diagnostics = OverfittingDiagnostics(
        in_sample_score=1.50,
        out_of_sample_score=1.20,
        degradation_ratio=0.20,
        parameter_count=5,
        observation_count=500,
    )

    with pytest.raises(FrozenInstanceError):
        diagnostics.risk = OverfittingRisk.HIGH


def test_invalid_probability_of_backtest_overfitting():
    with pytest.raises(
        ValidationConfigurationError
    ):
        OverfittingDiagnostics(
            in_sample_score=1.50,
            out_of_sample_score=1.20,
            degradation_ratio=0.20,
            parameter_count=5,
            observation_count=500,
            probability_of_backtest_overfitting=1.10,
        )


def test_invalid_multiple_testing_count():
    with pytest.raises(
        ValidationConfigurationError
    ):
        OverfittingDiagnostics(
            in_sample_score=1.50,
            out_of_sample_score=1.20,
            degradation_ratio=0.20,
            parameter_count=5,
            observation_count=500,
            multiple_testing_count=0,
        )


# ============================================================
# BenchmarkReport
# ============================================================


def test_benchmark_report_defaults():
    report = BenchmarkReport(
        research_id="research_001",
        strategy_name="BTC Alpha",
    )

    assert (
        report.verdict
        == AlphaVerdict.INCONCLUSIVE
    )

    assert report.overall_score == 0.0
    assert report.passed_stage_count == 0
    assert report.failed_stage_count == 0
    assert report.inconclusive_stage_count == 0

    assert (
        report.all_required_stages_passed
        is False
    )

    assert report.research_only is True
    assert report.real_money_execution is False


def test_benchmark_report_stage_counts():
    report = BenchmarkReport(
        research_id="research_001",
        strategy_name="BTC Alpha",
        validation_results=[
            make_validation_result(
                ValidationStage.WALK_FORWARD,
                ValidationStatus.PASSED,
            ),
            make_validation_result(
                ValidationStage.OUT_OF_SAMPLE,
                ValidationStatus.FAILED,
            ),
            make_validation_result(
                ValidationStage.BOOTSTRAP,
                ValidationStatus.INCONCLUSIVE,
            ),
        ],
    )

    assert report.passed_stage_count == 1
    assert report.failed_stage_count == 1
    assert report.inconclusive_stage_count == 1


def test_all_required_stages_passed():
    report = BenchmarkReport(
        research_id="research_001",
        strategy_name="BTC Alpha",
        validation_results=(
            all_required_results()
        ),
        verdict=AlphaVerdict.ROBUST_ALPHA,
        overall_score=0.90,
    )

    assert (
        report.all_required_stages_passed
        is True
    )

    assert report.passed_stage_count == 10


def test_required_stage_failure():
    results = all_required_results()

    results[0] = make_validation_result(
        ValidationStage.WALK_FORWARD,
        ValidationStatus.FAILED,
    )

    report = BenchmarkReport(
        research_id="research_001",
        strategy_name="BTC Alpha",
        validation_results=results,
    )

    assert (
        report.all_required_stages_passed
        is False
    )


def test_benchmark_report_summary():
    diagnostics = OverfittingDiagnostics(
        in_sample_score=1.50,
        out_of_sample_score=1.30,
        degradation_ratio=0.13,
        parameter_count=4,
        observation_count=1000,
        risk=OverfittingRisk.LOW,
    )

    report = BenchmarkReport(
        research_id="research_001",
        strategy_name="BTC Alpha",
        validation_results=(
            all_required_results()
        ),
        overfitting_diagnostics=diagnostics,
        verdict=AlphaVerdict.ROBUST_ALPHA,
        overall_score=0.92,
    )

    summary = report.summary()

    assert summary["version"] == "v6.0.2"

    assert (
        summary["research_id"]
        == "research_001"
    )

    assert (
        summary["strategy_name"]
        == "BTC Alpha"
    )

    assert (
        summary["verdict"]
        == "ROBUST_ALPHA"
    )

    assert (
        summary["overall_score"]
        == 0.92
    )

    assert (
        summary["validation_stage_count"]
        == 10
    )

    assert (
        summary["passed_stage_count"]
        == 10
    )

    assert (
        summary["failed_stage_count"]
        == 0
    )

    assert (
        summary[
            "all_required_stages_passed"
        ]
        is True
    )

    assert (
        summary["overfitting_risk"]
        == "LOW"
    )

    assert (
        summary["research_only"]
        is True
    )

    assert (
        summary["real_money_execution"]
        is False
    )

    assert len(
        summary["fingerprint"]
    ) == 64


def test_benchmark_report_fingerprint_stable():
    first = BenchmarkReport(
        research_id="research_001",
        strategy_name="BTC Alpha",
        validation_results=(
            all_required_results()
        ),
        verdict=AlphaVerdict.ROBUST_ALPHA,
        overall_score=0.90,
        metadata={
            "asset": "BTC",
            "horizon": "7D",
        },
    )

    second = BenchmarkReport(
        research_id="research_001",
        strategy_name="BTC Alpha",
        validation_results=(
            all_required_results()
        ),
        verdict=AlphaVerdict.ROBUST_ALPHA,
        overall_score=0.90,
        metadata={
            "horizon": "7D",
            "asset": "BTC",
        },
    )

    assert (
        first.fingerprint
        == second.fingerprint
    )


def test_benchmark_report_fingerprint_changes():
    first = BenchmarkReport(
        research_id="research_001",
        strategy_name="BTC Alpha",
        overall_score=0.80,
    )

    second = BenchmarkReport(
        research_id="research_001",
        strategy_name="BTC Alpha",
        overall_score=0.90,
    )

    assert (
        first.fingerprint
        != second.fingerprint
    )


def test_empty_research_id_rejected():
    with pytest.raises(
        ValidationConfigurationError
    ):
        BenchmarkReport(
            research_id="",
            strategy_name="BTC Alpha",
        )


def test_empty_strategy_name_rejected():
    with pytest.raises(
        ValidationConfigurationError
    ):
        BenchmarkReport(
            research_id="research_001",
            strategy_name="",
        )


@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        1.01,
    ],
)
def test_invalid_overall_score(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        BenchmarkReport(
            research_id="research_001",
            strategy_name="BTC Alpha",
            overall_score=value,
        )


def test_benchmark_report_real_money_rejected():
    with pytest.raises(
        ValidationConfigurationError
    ):
        BenchmarkReport(
            research_id="research_001",
            strategy_name="BTC Alpha",
            real_money_execution=True,
        )


# ============================================================
# Complete model integration
# ============================================================


def test_complete_model_integration():
    config = ValidationConfig(
        bootstrap_samples=2000,
        monte_carlo_simulations=2000,
        walk_forward_windows=6,
        metadata={
            "asset": "BTC",
            "horizon": "7D",
        },
    )

    metrics = make_metrics(
        sharpe_ratio=1.70,
        observations=1200,
    )

    confidence_interval = ConfidenceInterval(
        lower=0.02,
        upper=0.09,
        estimate=0.055,
        confidence_level=0.95,
        method="bootstrap",
    )

    statistical_test = StatisticalTestResult(
        test_name="alpha_significance",
        statistic=3.25,
        p_value=0.003,
        adjusted_p_value=0.015,
        effect_size=0.45,
        null_hypothesis="alpha <= 0",
        alternative_hypothesis="alpha > 0",
    )

    validation_results = (
        all_required_results()
    )

    validation_results[2] = ValidationResult(
        stage=ValidationStage.BOOTSTRAP,
        status=ValidationStatus.PASSED,
        score=0.91,
        metrics=metrics,
        confidence_interval=(
            confidence_interval
        ),
        statistical_tests=[
            statistical_test
        ],
        message=(
            "Bootstrap validation passed."
        ),
    )

    regime_results = [
        RegimeValidationResult(
            regime=RegimeType.BULL,
            observations=400,
            metrics=metrics,
            stable=True,
            score=0.90,
        ),
        RegimeValidationResult(
            regime=RegimeType.BEAR,
            observations=300,
            metrics=metrics,
            stable=True,
            score=0.82,
        ),
        RegimeValidationResult(
            regime=RegimeType.MIXED,
            observations=500,
            metrics=metrics,
            stable=True,
            score=0.86,
        ),
    ]

    sensitivity_results = [
        SensitivityResult(
            parameter_name="lookback",
            baseline_value=20.0,
            tested_values=[
                10.0,
                20.0,
                30.0,
            ],
            performance_values=[
                1.55,
                1.70,
                1.61,
            ],
            sensitivity_score=0.88,
            robust=True,
        )
    ]

    diagnostics = OverfittingDiagnostics(
        in_sample_score=1.85,
        out_of_sample_score=1.70,
        degradation_ratio=0.081,
        parameter_count=6,
        observation_count=1200,
        multiple_testing_count=12,
        probability_of_backtest_overfitting=0.12,
        deflated_sharpe_ratio=1.42,
        risk=OverfittingRisk.LOW,
        warnings=[],
    )

    report = BenchmarkReport(
        research_id="btc_alpha_001",
        strategy_name=(
            "BTC Multi-Agent Alpha"
        ),
        validation_results=(
            validation_results
        ),
        regime_results=regime_results,
        sensitivity_results=(
            sensitivity_results
        ),
        overfitting_diagnostics=diagnostics,
        verdict=AlphaVerdict.ROBUST_ALPHA,
        overall_score=0.91,
        metadata={
            "config_fingerprint": (
                config.fingerprint
            ),
            "version": VERSION,
        },
    )

    assert (
        report.verdict
        == AlphaVerdict.ROBUST_ALPHA
    )

    assert (
        report.all_required_stages_passed
        is True
    )

    assert report.passed_stage_count == 10

    assert (
        report.failed_stage_count
        == 0
    )

    assert (
        report.overfitting_diagnostics
        is not None
    )

    assert (
        report.overfitting_diagnostics.risk
        == OverfittingRisk.LOW
    )

    assert (
        report.regime_results[0].stable
        is True
    )

    assert (
        report.sensitivity_results[0].robust
        is True
    )

    assert (
        validation_results[
            2
        ].confidence_interval
        is not None
    )

    assert (
        validation_results[
            2
        ].confidence_interval.contains_zero
        is False
    )

    assert (
        validation_results[
            2
        ].statistical_tests[
            0
        ].significant
        is True
    )

    assert (
        report.research_only
        is True
    )

    assert (
        report.real_money_execution
        is False
    )

    assert len(
        report.fingerprint
    ) == 64

    assert len(
        config.fingerprint
    ) == 64