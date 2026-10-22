from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from research_benchmarking.models import (
    RegimeType,
    ValidationConfig,
    ValidationConfigurationError,
    ValidationStage,
    ValidationStatus,
)
from research_benchmarking.regime_stability import (
    VERSION,
    RegimeObservation,
    RegimeStatistics,
    RegimeStabilityEngine,
    RegimeStabilityError,
    RegimeStabilityReport,
    _validate_numeric,
    _validate_positive_integer,
    _validate_ratio,
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


def make_observation(
    regime: RegimeType = RegimeType.BULL,
    score: float = 0.5,
) -> RegimeObservation:
    return RegimeObservation(
        regime=regime,
        score=score,
    )


def make_statistics(
    regime: RegimeType = RegimeType.BULL,
    *,
    average_score: float = 0.50,
    stability_score: float = 0.85,
    stable: bool = True,
    sufficient_sample: bool = True,
) -> RegimeStatistics:
    return RegimeStatistics(
        regime=regime,
        observation_count=10,
        average_score=average_score,
        minimum_score=0.20,
        maximum_score=0.80,
        volatility=0.10,
        positive_ratio=0.90,
        stability_score=stability_score,
        sufficient_sample=sufficient_sample,
        stable=stable,
    )


def make_report(
    passed: bool = True,
) -> RegimeStabilityReport:
    results = [
        make_statistics(
            RegimeType.BULL,
            average_score=0.60,
            stability_score=0.90,
        ),
        make_statistics(
            RegimeType.BEAR,
            average_score=0.40,
            stability_score=0.80,
        ),
    ]

    return RegimeStabilityReport(
        regime_results=results,
        evaluated_regime_count=2,
        sufficient_regime_count=2,
        stable_regime_count=2,
        positive_regime_ratio=1.0,
        stable_regime_ratio=1.0,
        average_stability_score=0.85,
        cross_regime_dispersion=0.10,
        weakest_regime=RegimeType.BEAR,
        strongest_regime=RegimeType.BULL,
        passed=passed,
    )


# ============================================================
# Version
# ============================================================


def test_version():
    assert VERSION == "v6.0.2"


def test_engine_version():
    assert RegimeStabilityEngine.VERSION == "v6.0.2"


# ============================================================
# Numeric helpers
# ============================================================


@pytest.mark.parametrize(
    "value",
    [0, 1, -1, 0.5, -0.5],
)
def test_validate_numeric_valid(value):
    assert _validate_numeric(value, "value") == float(value)


@pytest.mark.parametrize(
    "value",
    [
        True,
        "bad",
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_validate_numeric_invalid(value):
    with pytest.raises(ValidationConfigurationError):
        _validate_numeric(value, "value")


@pytest.mark.parametrize(
    "value",
    [0.0, 0.25, 0.5, 0.75, 1.0],
)
def test_validate_ratio_valid(value):
    assert _validate_ratio(value, "value") == float(value)


@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        1.01,
        True,
        float("nan"),
        float("inf"),
    ],
)
def test_validate_ratio_invalid(value):
    with pytest.raises(ValidationConfigurationError):
        _validate_ratio(value, "value")


@pytest.mark.parametrize(
    "value",
    [1, 2, 10, 100],
)
def test_validate_positive_integer_valid(value):
    assert _validate_positive_integer(value, "value") == value


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        1.5,
        "1",
        None,
    ],
)
def test_validate_positive_integer_invalid(value):
    with pytest.raises(ValidationConfigurationError):
        _validate_positive_integer(value, "value")


# ============================================================
# RegimeObservation
# ============================================================


def test_observation_creation():
    observation = make_observation()

    assert observation.regime == RegimeType.BULL
    assert observation.score == 0.5
    assert observation.positive is True


def test_observation_negative():
    observation = make_observation(
        score=-0.1,
    )

    assert observation.positive is False


def test_observation_zero_not_positive():
    observation = make_observation(
        score=0.0,
    )

    assert observation.positive is False


def test_observation_frozen():
    observation = make_observation()

    with pytest.raises(FrozenInstanceError):
        observation.score = 1.0


def test_observation_fingerprint():
    first = make_observation()
    second = make_observation()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_observation_fingerprint_changes():
    first = make_observation(score=0.5)
    second = make_observation(score=0.6)

    assert first.fingerprint != second.fingerprint


def test_observation_invalid_regime():
    with pytest.raises(ValidationConfigurationError):
        RegimeObservation(
            regime="bull",
            score=0.5,
        )


@pytest.mark.parametrize(
    "score",
    [
        True,
        "bad",
        None,
        float("nan"),
        float("inf"),
    ],
)
def test_observation_invalid_score(score):
    with pytest.raises(ValidationConfigurationError):
        RegimeObservation(
            regime=RegimeType.BULL,
            score=score,
        )


# ============================================================
# RegimeStatistics
# ============================================================


def test_statistics_creation():
    result = make_statistics()

    assert result.regime == RegimeType.BULL
    assert result.observation_count == 10
    assert result.stable is True
    assert result.sufficient_sample is True


def test_statistics_frozen():
    result = make_statistics()

    with pytest.raises(FrozenInstanceError):
        result.stable = False


def test_statistics_fingerprint():
    first = make_statistics()
    second = make_statistics()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_statistics_invalid_regime():
    with pytest.raises(ValidationConfigurationError):
        RegimeStatistics(
            regime="bull",
            observation_count=10,
            average_score=0.5,
            minimum_score=0.1,
            maximum_score=0.9,
            volatility=0.1,
            positive_ratio=0.8,
            stability_score=0.8,
            sufficient_sample=True,
            stable=True,
        )


@pytest.mark.parametrize(
    "count",
    [0, -1, True, 1.5],
)
def test_statistics_invalid_count(count):
    with pytest.raises(ValidationConfigurationError):
        RegimeStatistics(
            regime=RegimeType.BULL,
            observation_count=count,
            average_score=0.5,
            minimum_score=0.1,
            maximum_score=0.9,
            volatility=0.1,
            positive_ratio=0.8,
            stability_score=0.8,
            sufficient_sample=True,
            stable=True,
        )


def test_statistics_negative_volatility():
    with pytest.raises(ValidationConfigurationError):
        RegimeStatistics(
            regime=RegimeType.BULL,
            observation_count=10,
            average_score=0.5,
            minimum_score=0.1,
            maximum_score=0.9,
            volatility=-0.1,
            positive_ratio=0.8,
            stability_score=0.8,
            sufficient_sample=True,
            stable=True,
        )


@pytest.mark.parametrize(
    "field",
    [
        "positive_ratio",
        "stability_score",
    ],
)
@pytest.mark.parametrize(
    "value",
    [-0.01, 1.01],
)
def test_statistics_invalid_ratio(field, value):
    kwargs = {
        "positive_ratio": 0.8,
        "stability_score": 0.8,
    }
    kwargs[field] = value

    with pytest.raises(ValidationConfigurationError):
        RegimeStatistics(
            regime=RegimeType.BULL,
            observation_count=10,
            average_score=0.5,
            minimum_score=0.1,
            maximum_score=0.9,
            volatility=0.1,
            sufficient_sample=True,
            stable=True,
            **kwargs,
        )


# ============================================================
# Report
# ============================================================


def test_report_creation():
    report = make_report()

    assert report.evaluated_regime_count == 2
    assert report.stable_regime_count == 2
    assert report.passed is True
    assert report.unstable_regimes == []
    assert report.insufficient_regimes == []


def test_report_frozen():
    report = make_report()

    with pytest.raises(FrozenInstanceError):
        report.passed = False


def test_report_fingerprint():
    first = make_report()
    second = make_report()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_report_empty_results():
    with pytest.raises(ValidationConfigurationError):
        RegimeStabilityReport(
            regime_results=[],
            evaluated_regime_count=1,
            sufficient_regime_count=1,
            stable_regime_count=1,
            positive_regime_ratio=1.0,
            stable_regime_ratio=1.0,
            average_stability_score=1.0,
            cross_regime_dispersion=0.0,
            weakest_regime=RegimeType.BULL,
            strongest_regime=RegimeType.BULL,
            passed=True,
        )


def test_report_invalid_result_type():
    with pytest.raises(ValidationConfigurationError):
        RegimeStabilityReport(
            regime_results=["bad"],
            evaluated_regime_count=1,
            sufficient_regime_count=1,
            stable_regime_count=1,
            positive_regime_ratio=1.0,
            stable_regime_ratio=1.0,
            average_stability_score=1.0,
            cross_regime_dispersion=0.0,
            weakest_regime=RegimeType.BULL,
            strongest_regime=RegimeType.BULL,
            passed=True,
        )


def test_report_unstable_and_insufficient():
    results = [
        make_statistics(
            RegimeType.BULL,
            stable=True,
            sufficient_sample=True,
        ),
        make_statistics(
            RegimeType.BEAR,
            stable=False,
            sufficient_sample=False,
        ),
    ]

    report = RegimeStabilityReport(
        regime_results=results,
        evaluated_regime_count=2,
        sufficient_regime_count=1,
        stable_regime_count=1,
        positive_regime_ratio=1.0,
        stable_regime_ratio=0.5,
        average_stability_score=0.7,
        cross_regime_dispersion=0.1,
        weakest_regime=RegimeType.BEAR,
        strongest_regime=RegimeType.BULL,
        passed=False,
    )

    assert report.unstable_regimes == [RegimeType.BEAR]
    assert report.insufficient_regimes == [RegimeType.BEAR]


# ============================================================
# Engine construction
# ============================================================


def test_default_engine():
    engine = RegimeStabilityEngine()

    assert engine.minimum_regime_observations == 5
    assert engine.minimum_positive_ratio == 0.60
    assert engine.minimum_regime_stability_score == 0.60
    assert engine.minimum_stable_regime_ratio == 0.60
    assert engine.maximum_cross_regime_dispersion == 1.00


def test_custom_engine():
    engine = RegimeStabilityEngine(
        make_config(),
        minimum_regime_observations=10,
        minimum_positive_ratio=0.70,
        minimum_regime_stability_score=0.70,
        minimum_stable_regime_ratio=0.75,
        maximum_cross_regime_dispersion=0.50,
    )

    assert engine.minimum_regime_observations == 10
    assert engine.minimum_positive_ratio == 0.70
    assert engine.minimum_regime_stability_score == 0.70
    assert engine.minimum_stable_regime_ratio == 0.75
    assert engine.maximum_cross_regime_dispersion == 0.50


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5],
)
def test_invalid_minimum_observations(value):
    with pytest.raises(ValidationConfigurationError):
        RegimeStabilityEngine(
            minimum_regime_observations=value
        )


@pytest.mark.parametrize(
    "field",
    [
        "minimum_positive_ratio",
        "minimum_regime_stability_score",
        "minimum_stable_regime_ratio",
    ],
)
@pytest.mark.parametrize(
    "value",
    [-0.01, 1.01, True, float("nan")],
)
def test_invalid_engine_ratio(field, value):
    with pytest.raises(ValidationConfigurationError):
        RegimeStabilityEngine(
            **{field: value}
        )


@pytest.mark.parametrize(
    "value",
    [-0.01, True, float("nan")],
)
def test_invalid_dispersion(value):
    with pytest.raises(ValidationConfigurationError):
        RegimeStabilityEngine(
            maximum_cross_regime_dispersion=value
        )


# ============================================================
# Sample standard deviation
# ============================================================


def test_sample_std_single():
    assert RegimeStabilityEngine._sample_std([1.0]) == 0.0


def test_sample_std_constant():
    assert RegimeStabilityEngine._sample_std(
        [1.0, 1.0, 1.0]
    ) == 0.0


def test_sample_std_variable():
    value = RegimeStabilityEngine._sample_std(
        [1.0, 2.0, 3.0]
    )

    assert value == pytest.approx(1.0)


# ============================================================
# Analyze regime
# ============================================================


def test_analyze_regime_stable():
    engine = RegimeStabilityEngine()

    result = engine.analyze_regime(
        RegimeType.BULL,
        [
            0.50,
            0.55,
            0.60,
            0.58,
            0.62,
            0.57,
        ],
    )

    assert result.regime == RegimeType.BULL
    assert result.observation_count == 6
    assert result.average_score > 0.0
    assert result.positive_ratio == 1.0
    assert result.sufficient_sample is True
    assert result.stable is True


def test_analyze_regime_negative():
    engine = RegimeStabilityEngine()

    result = engine.analyze_regime(
        RegimeType.BEAR,
        [
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
        ],
    )

    assert result.average_score < 0.0
    assert result.positive_ratio == 0.0
    assert result.stable is False


def test_analyze_regime_insufficient_sample():
    engine = RegimeStabilityEngine(
        minimum_regime_observations=5
    )

    result = engine.analyze_regime(
        RegimeType.BULL,
        [0.5, 0.6],
    )

    assert result.sufficient_sample is False
    assert result.stable is False


def test_analyze_regime_empty():
    engine = RegimeStabilityEngine()

    with pytest.raises(RegimeStabilityError):
        engine.analyze_regime(
            RegimeType.BULL,
            [],
        )


def test_analyze_regime_invalid_regime():
    engine = RegimeStabilityEngine()

    with pytest.raises(ValidationConfigurationError):
        engine.analyze_regime(
            "bull",
            [0.5] * 5,
        )


def test_analyze_regime_invalid_score():
    engine = RegimeStabilityEngine()

    with pytest.raises(ValidationConfigurationError):
        engine.analyze_regime(
            RegimeType.BULL,
            [0.5, 0.6, float("nan"), 0.7, 0.8],
        )


# ============================================================
# Aggregate
# ============================================================


def test_aggregate_pass():
    engine = RegimeStabilityEngine()

    bull = engine.analyze_regime(
        RegimeType.BULL,
        [0.50, 0.52, 0.55, 0.58, 0.60],
    )

    bear = engine.analyze_regime(
        RegimeType.BEAR,
        [0.30, 0.32, 0.35, 0.38, 0.40],
    )

    report = engine.aggregate(
        [bull, bear]
    )

    assert report.evaluated_regime_count == 2
    assert report.sufficient_regime_count == 2
    assert report.stable_regime_count == 2
    assert report.positive_regime_ratio == 1.0
    assert report.stable_regime_ratio == 1.0
    assert report.passed is True


def test_aggregate_identifies_weakest_and_strongest():
    engine = RegimeStabilityEngine()

    bull = engine.analyze_regime(
        RegimeType.BULL,
        [0.7] * 5,
    )

    bear = engine.analyze_regime(
        RegimeType.BEAR,
        [0.2] * 5,
    )

    report = engine.aggregate(
        [bull, bear]
    )

    assert report.strongest_regime == RegimeType.BULL
    assert report.weakest_regime == RegimeType.BEAR


def test_aggregate_fails_with_insufficient_regime():
    engine = RegimeStabilityEngine(
        minimum_regime_observations=5
    )

    bull = engine.analyze_regime(
        RegimeType.BULL,
        [0.5] * 5,
    )

    bear = engine.analyze_regime(
        RegimeType.BEAR,
        [0.4, 0.4],
    )

    report = engine.aggregate(
        [bull, bear]
    )

    assert report.sufficient_regime_count == 1
    assert report.passed is False


def test_aggregate_empty():
    engine = RegimeStabilityEngine()

    with pytest.raises(RegimeStabilityError):
        engine.aggregate([])


def test_aggregate_invalid_type():
    engine = RegimeStabilityEngine()

    with pytest.raises(ValidationConfigurationError):
        engine.aggregate(
            [
                make_statistics(),
                "bad",
            ]
        )


# ============================================================
# Mapping API
# ============================================================


def test_validate_regimes():
    engine = RegimeStabilityEngine()

    report = engine.validate_regimes(
        {
            RegimeType.BULL: [
                0.50,
                0.52,
                0.55,
                0.58,
                0.60,
            ],
            RegimeType.BEAR: [
                0.30,
                0.32,
                0.35,
                0.38,
                0.40,
            ],
        }
    )

    assert report.evaluated_regime_count == 2
    assert report.passed is True


def test_validate_regimes_empty():
    engine = RegimeStabilityEngine()

    with pytest.raises(RegimeStabilityError):
        engine.validate_regimes({})


# ============================================================
# Observation API
# ============================================================


def test_validate_observations():
    engine = RegimeStabilityEngine()

    observations = [
        RegimeObservation(
            RegimeType.BULL,
            score,
        )
        for score in [
            0.50,
            0.52,
            0.55,
            0.58,
            0.60,
        ]
    ]

    observations.extend(
        RegimeObservation(
            RegimeType.BEAR,
            score,
        )
        for score in [
            0.30,
            0.32,
            0.35,
            0.38,
            0.40,
        ]
    )

    report = engine.validate_observations(
        observations
    )

    assert report.evaluated_regime_count == 2
    assert report.passed is True


def test_validate_observations_empty():
    engine = RegimeStabilityEngine()

    with pytest.raises(RegimeStabilityError):
        engine.validate_observations([])


def test_validate_observations_invalid_type():
    engine = RegimeStabilityEngine()

    with pytest.raises(ValidationConfigurationError):
        engine.validate_observations(
            [
                make_observation(),
                "bad",
            ]
        )


# ============================================================
# Shared model adapter
# ============================================================


def test_shared_model_adapter_safe():
    engine = RegimeStabilityEngine()

    report = engine.validate_regimes(
        {
            RegimeType.BULL: [0.5] * 5,
        }
    )

    converted = (
        engine.to_regime_validation_results(
            report
        )
    )

    assert isinstance(converted, list)


# ============================================================
# Score
# ============================================================


def test_calculate_score_bounded():
    engine = RegimeStabilityEngine()

    report = make_report()

    score = engine._calculate_score(
        report
    )

    assert 0.0 <= score <= 1.0


def test_strong_regimes_high_score():
    engine = RegimeStabilityEngine()

    report = engine.validate_regimes(
        {
            RegimeType.BULL: [0.8] * 5,
            RegimeType.BEAR: [0.7] * 5,
            RegimeType.SIDEWAYS: [0.6] * 5,
        }
    )

    assert engine._calculate_score(report) > 0.90


# ============================================================
# ValidationResult adapter
# ============================================================


def test_to_validation_result_pass():
    engine = RegimeStabilityEngine()

    report = engine.validate_regimes(
        {
            RegimeType.BULL: [0.5] * 5,
            RegimeType.BEAR: [0.4] * 5,
        }
    )

    result = engine.to_validation_result(
        report
    )

    assert (
        result.stage
        == ValidationStage.REGIME_STABILITY
    )
    assert (
        result.status
        == ValidationStatus.PASSED
    )
    assert 0.0 <= result.score <= 1.0
    assert result.metadata[
        "research_only"
    ] is True
    assert result.metadata[
        "real_money_execution"
    ] is False


def test_to_validation_result_fail():
    engine = RegimeStabilityEngine()

    report = engine.validate_regimes(
        {
            RegimeType.BULL: [0.5] * 5,
            RegimeType.BEAR: [-0.5] * 5,
        }
    )

    result = engine.to_validation_result(
        report
    )

    assert (
        result.status
        == ValidationStatus.FAILED
    )


# ============================================================
# Determinism
# ============================================================


def test_regime_analysis_deterministic():
    engine = RegimeStabilityEngine()

    scores = [
        0.5,
        0.52,
        0.55,
        0.58,
        0.60,
    ]

    first = engine.analyze_regime(
        RegimeType.BULL,
        scores,
    )

    second = engine.analyze_regime(
        RegimeType.BULL,
        scores,
    )

    assert first.fingerprint == second.fingerprint


def test_report_deterministic():
    engine = RegimeStabilityEngine()

    data = {
        RegimeType.BULL: [0.5] * 5,
        RegimeType.BEAR: [0.4] * 5,
    }

    first = engine.validate_regimes(data)
    second = engine.validate_regimes(data)

    assert first.fingerprint == second.fingerprint


# ============================================================
# Snapshot
# ============================================================


def test_snapshot():
    engine = RegimeStabilityEngine(
        make_config(),
        minimum_regime_observations=10,
        minimum_positive_ratio=0.70,
        minimum_regime_stability_score=0.75,
        minimum_stable_regime_ratio=0.80,
        maximum_cross_regime_dispersion=0.50,
    )

    snapshot = engine.snapshot()

    assert snapshot["version"] == "v6.0.2"

    assert snapshot[
        "minimum_regime_observations"
    ] == 10

    assert snapshot[
        "minimum_positive_ratio"
    ] == 0.70

    assert snapshot[
        "minimum_regime_stability_score"
    ] == 0.75

    assert snapshot[
        "minimum_stable_regime_ratio"
    ] == 0.80

    assert snapshot[
        "maximum_cross_regime_dispersion"
    ] == 0.50

    assert snapshot["research_only"] is True
    assert snapshot[
        "real_money_execution"
    ] is False

    assert len(
        snapshot["config_fingerprint"]
    ) == 64


# ============================================================
# Complete multi-regime flow
# ============================================================


def test_complete_regime_stability_flow():
    engine = RegimeStabilityEngine(
        make_config(),
        minimum_regime_observations=5,
        minimum_positive_ratio=0.60,
        minimum_regime_stability_score=0.60,
        minimum_stable_regime_ratio=0.60,
        maximum_cross_regime_dispersion=1.0,
    )

    regime_scores = {
        RegimeType.BULL: [
            0.70,
            0.72,
            0.68,
            0.75,
            0.71,
            0.73,
        ],
        RegimeType.BEAR: [
            0.30,
            0.32,
            0.28,
            0.35,
            0.31,
            0.33,
        ],
        RegimeType.SIDEWAYS: [
            0.40,
            0.42,
            0.38,
            0.45,
            0.41,
            0.43,
        ],
        RegimeType.HIGH_VOLATILITY: [
            0.25,
            0.30,
            0.20,
            0.35,
            0.28,
            0.32,
        ],
        RegimeType.LOW_VOLATILITY: [
            0.55,
            0.58,
            0.52,
            0.60,
            0.56,
            0.57,
        ],
        RegimeType.RISK_ON: [
            0.65,
            0.68,
            0.62,
            0.70,
            0.66,
            0.67,
        ],
        RegimeType.RISK_OFF: [
            0.22,
            0.25,
            0.20,
            0.28,
            0.24,
            0.26,
        ],
        RegimeType.MIXED: [
            0.35,
            0.38,
            0.32,
            0.40,
            0.36,
            0.37,
        ],
    }

    report = engine.validate_regimes(
        regime_scores
    )

    assert report.evaluated_regime_count == 8
    assert report.sufficient_regime_count == 8
    assert report.stable_regime_count == 8
    assert report.positive_regime_ratio == 1.0
    assert report.stable_regime_ratio == 1.0
    assert report.average_stability_score > 0.80
    assert report.passed is True

    validation = engine.to_validation_result(
        report
    )

    assert (
        validation.stage
        == ValidationStage.REGIME_STABILITY
    )

    assert (
        validation.status
        == ValidationStatus.PASSED
    )

    assert validation.score > 0.80

    assert validation.metadata[
        "research_only"
    ] is True

    assert validation.metadata[
        "real_money_execution"
    ] is False

    assert len(report.fingerprint) == 64