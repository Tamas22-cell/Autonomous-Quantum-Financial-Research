from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from research_benchmarking.models import (
    PerformanceMetrics,
    ValidationConfig,
    ValidationConfigurationError,
    ValidationStage,
    ValidationStatus,
)
from research_benchmarking.walk_forward_validator import (
    VERSION,
    WalkForwardReport,
    WalkForwardValidationError,
    WalkForwardValidator,
    WalkForwardWindow,
    WalkForwardWindowResult,
)


# ============================================================
# Helpers
# ============================================================


def make_config(
    *,
    windows: int = 5,
    minimum_observations: int = 30,
) -> ValidationConfig:
    return ValidationConfig(
        walk_forward_windows=windows,
        minimum_observations=minimum_observations,
    )


def make_metrics(
    *,
    sharpe: float = 1.5,
    observations: int = 100,
) -> PerformanceMetrics:
    return PerformanceMetrics(
        total_return=0.20,
        annualized_return=0.15,
        volatility=0.10,
        sharpe_ratio=sharpe,
        sortino_ratio=1.80,
        max_drawdown=0.08,
        win_rate=0.57,
        profit_factor=1.45,
        alpha=0.05,
        beta=0.90,
        information_ratio=0.80,
        observations=observations,
    )


def make_window(
    window_id: int = 0,
) -> WalkForwardWindow:
    return WalkForwardWindow(
        window_id=window_id,
        train_start=0,
        train_end=100,
        validation_start=105,
        validation_end=120,
        test_start=125,
        test_end=150,
        embargo_size=5,
    )


def make_result(
    *,
    window_id: int = 0,
    train_score: float = 1.50,
    validation_score: float = 1.35,
    test_score: float = 1.25,
) -> WalkForwardWindowResult:
    return WalkForwardWindowResult(
        window=make_window(window_id),
        train_score=train_score,
        validation_score=validation_score,
        test_score=test_score,
    )


# ============================================================
# Version
# ============================================================


def test_version():
    assert VERSION == "v6.0.2"


# ============================================================
# WalkForwardWindow
# ============================================================


def test_window_creation():
    window = make_window()

    assert window.window_id == 0
    assert window.train_start == 0
    assert window.train_end == 100
    assert window.validation_start == 105
    assert window.validation_end == 120
    assert window.test_start == 125
    assert window.test_end == 150
    assert window.embargo_size == 5


def test_window_sizes():
    window = make_window()

    assert window.train_size == 100
    assert window.validation_size == 15
    assert window.test_size == 25
    assert window.total_size == 140


def test_window_fingerprint_stable():
    first = make_window()
    second = make_window()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_window_is_frozen():
    window = make_window()

    with pytest.raises(FrozenInstanceError):
        window.train_end = 200


@pytest.mark.parametrize(
    "field,value",
    [
        ("window_id", -1),
        ("train_start", -1),
        ("embargo_size", -1),
    ],
)
def test_window_negative_values_rejected(
    field,
    value,
):
    kwargs = {
        "window_id": 0,
        "train_start": 0,
        "train_end": 100,
        "validation_start": 105,
        "validation_end": 120,
        "test_start": 125,
        "test_end": 150,
        "embargo_size": 5,
    }

    kwargs[field] = value

    with pytest.raises(
        ValidationConfigurationError
    ):
        WalkForwardWindow(**kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("window_id", True),
        ("train_start", 1.5),
        ("train_end", "100"),
        ("embargo_size", 2.5),
    ],
)
def test_window_non_integer_values_rejected(
    field,
    value,
):
    kwargs = {
        "window_id": 0,
        "train_start": 0,
        "train_end": 100,
        "validation_start": 105,
        "validation_end": 120,
        "test_start": 125,
        "test_end": 150,
        "embargo_size": 5,
    }

    kwargs[field] = value

    with pytest.raises(
        ValidationConfigurationError
    ):
        WalkForwardWindow(**kwargs)


def test_invalid_train_range():
    with pytest.raises(
        ValidationConfigurationError
    ):
        WalkForwardWindow(
            window_id=0,
            train_start=100,
            train_end=100,
            validation_start=105,
            validation_end=120,
            test_start=125,
            test_end=150,
        )


def test_training_validation_overlap_rejected():
    with pytest.raises(
        ValidationConfigurationError
    ):
        WalkForwardWindow(
            window_id=0,
            train_start=0,
            train_end=100,
            validation_start=99,
            validation_end=120,
            test_start=125,
            test_end=150,
        )


def test_invalid_validation_range():
    with pytest.raises(
        ValidationConfigurationError
    ):
        WalkForwardWindow(
            window_id=0,
            train_start=0,
            train_end=100,
            validation_start=105,
            validation_end=105,
            test_start=125,
            test_end=150,
        )


def test_validation_test_overlap_rejected():
    with pytest.raises(
        ValidationConfigurationError
    ):
        WalkForwardWindow(
            window_id=0,
            train_start=0,
            train_end=100,
            validation_start=105,
            validation_end=120,
            test_start=119,
            test_end=150,
        )


def test_invalid_test_range():
    with pytest.raises(
        ValidationConfigurationError
    ):
        WalkForwardWindow(
            window_id=0,
            train_start=0,
            train_end=100,
            validation_start=105,
            validation_end=120,
            test_start=125,
            test_end=125,
        )


# ============================================================
# WalkForwardWindowResult
# ============================================================


def test_window_result_creation():
    result = make_result()

    assert result.train_score == 1.50
    assert result.validation_score == 1.35
    assert result.test_score == 1.25
    assert result.positive_oos is True


def test_validation_degradation():
    result = make_result(
        train_score=2.0,
        validation_score=1.5,
    )

    assert result.validation_degradation == pytest.approx(
        0.25
    )


def test_test_degradation():
    result = make_result(
        train_score=2.0,
        test_score=1.0,
    )

    assert result.test_degradation == pytest.approx(
        0.50
    )


def test_zero_train_score_degradation():
    result = make_result(
        train_score=0.0,
        validation_score=-1.0,
        test_score=-1.0,
    )

    assert result.validation_degradation == 0.0
    assert result.test_degradation == 0.0


def test_negative_test_score_not_positive_oos():
    result = make_result(
        test_score=-0.10
    )

    assert result.positive_oos is False


def test_zero_test_score_not_positive_oos():
    result = make_result(
        test_score=0.0
    )

    assert result.positive_oos is False


def test_window_result_with_metrics():
    metrics = make_metrics()

    result = WalkForwardWindowResult(
        window=make_window(),
        train_score=1.50,
        validation_score=1.40,
        test_score=1.30,
        train_metrics=metrics,
        validation_metrics=metrics,
        test_metrics=metrics,
    )

    assert result.train_metrics == metrics
    assert result.validation_metrics == metrics
    assert result.test_metrics == metrics


def test_window_result_fingerprint_stable():
    first = make_result()
    second = make_result()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("train_score", float("nan")),
        ("validation_score", float("inf")),
        ("test_score", float("-inf")),
        ("test_score", True),
    ],
)
def test_invalid_window_scores_rejected(
    field,
    value,
):
    kwargs = {
        "window": make_window(),
        "train_score": 1.50,
        "validation_score": 1.30,
        "test_score": 1.20,
    }

    kwargs[field] = value

    with pytest.raises(
        ValidationConfigurationError
    ):
        WalkForwardWindowResult(**kwargs)


# ============================================================
# WalkForwardValidator construction
# ============================================================


def test_default_validator():
    validator = WalkForwardValidator()

    assert validator.embargo_size == 0
    assert validator.expanding_window is True
    assert validator.maximum_degradation == 0.50
    assert validator.minimum_positive_oos_ratio == 0.60
    assert validator.minimum_stable_window_ratio == 0.60


def test_custom_validator():
    validator = WalkForwardValidator(
        make_config(windows=4),
        embargo_size=2,
        expanding_window=False,
        maximum_degradation=0.40,
        minimum_positive_oos_ratio=0.75,
        minimum_stable_window_ratio=0.70,
    )

    assert validator.embargo_size == 2
    assert validator.expanding_window is False
    assert validator.maximum_degradation == 0.40
    assert validator.minimum_positive_oos_ratio == 0.75
    assert validator.minimum_stable_window_ratio == 0.70


@pytest.mark.parametrize(
    "value",
    [
        -1,
        True,
        1.5,
    ],
)
def test_invalid_embargo(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        WalkForwardValidator(
            embargo_size=value
        )


def test_negative_maximum_degradation():
    with pytest.raises(
        ValidationConfigurationError
    ):
        WalkForwardValidator(
            maximum_degradation=-0.1
        )


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        1.1,
    ],
)
def test_invalid_positive_oos_ratio(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        WalkForwardValidator(
            minimum_positive_oos_ratio=value
        )


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        1.1,
    ],
)
def test_invalid_stable_window_ratio(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        WalkForwardValidator(
            minimum_stable_window_ratio=value
        )


# ============================================================
# Window construction
# ============================================================


def test_build_windows_count():
    validator = WalkForwardValidator(
        make_config(windows=5)
    )

    windows = validator.build_windows(1000)

    assert len(windows) == 5


def test_build_windows_are_ordered():
    validator = WalkForwardValidator(
        make_config(windows=5)
    )

    windows = validator.build_windows(1000)

    for index, window in enumerate(windows):
        assert window.window_id == index

        assert (
            window.train_start
            < window.train_end
        )

        assert (
            window.validation_start
            >= window.train_end
        )

        assert (
            window.validation_end
            <= window.test_start
        )

        assert (
            window.test_start
            < window.test_end
        )


def test_expanding_windows_start_at_zero():
    validator = WalkForwardValidator(
        make_config(windows=4),
        expanding_window=True,
    )

    windows = validator.build_windows(1000)

    assert all(
        window.train_start == 0
        for window in windows
    )


def test_expanding_train_size_increases():
    validator = WalkForwardValidator(
        make_config(windows=4),
        expanding_window=True,
    )

    windows = validator.build_windows(1000)

    sizes = [
        window.train_size
        for window in windows
    ]

    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


def test_rolling_windows_move_train_start():
    validator = WalkForwardValidator(
        make_config(windows=4),
        expanding_window=False,
    )

    windows = validator.build_windows(1000)

    starts = [
        window.train_start
        for window in windows
    ]

    assert starts[0] == 0
    assert starts[-1] > starts[0]


def test_last_window_ends_at_observation_count():
    validator = WalkForwardValidator(
        make_config(windows=5)
    )

    windows = validator.build_windows(1000)

    assert windows[-1].test_end == 1000


def test_embargo_applied():
    validator = WalkForwardValidator(
        make_config(windows=5),
        embargo_size=3,
    )

    windows = validator.build_windows(1000)

    for window in windows:
        assert (
            window.validation_start
            - window.train_end
            == 3
        )

        assert (
            window.test_start
            - window.validation_end
            == 3
        )


def test_insufficient_observations():
    validator = WalkForwardValidator(
        make_config(
            minimum_observations=100
        )
    )

    with pytest.raises(
        WalkForwardValidationError
    ):
        validator.build_windows(99)


def test_boolean_observation_count_rejected():
    validator = WalkForwardValidator()

    with pytest.raises(
        ValidationConfigurationError
    ):
        validator.build_windows(True)


def test_float_observation_count_rejected():
    validator = WalkForwardValidator()

    with pytest.raises(
        ValidationConfigurationError
    ):
        validator.build_windows(100.5)


def test_excessive_embargo_rejected():
    validator = WalkForwardValidator(
        make_config(windows=5),
        embargo_size=20,
    )

    with pytest.raises(
        WalkForwardValidationError
    ):
        validator.build_windows(100)


# ============================================================
# Sequence splitting
# ============================================================


def test_split_sequence():
    values = list(range(200))
    window = make_window()

    train, validation, test = (
        WalkForwardValidator.split_sequence(
            values,
            window,
        )
    )

    assert train == list(range(0, 100))
    assert validation == list(range(105, 120))
    assert test == list(range(125, 150))


def test_split_sequence_sizes():
    values = list(range(200))
    window = make_window()

    train, validation, test = (
        WalkForwardValidator.split_sequence(
            values,
            window,
        )
    )

    assert len(train) == window.train_size
    assert (
        len(validation)
        == window.validation_size
    )
    assert len(test) == window.test_size


def test_short_sequence_rejected():
    window = make_window()

    with pytest.raises(
        WalkForwardValidationError
    ):
        WalkForwardValidator.split_sequence(
            list(range(149)),
            window,
        )


# ============================================================
# create_window_result
# ============================================================


def test_create_window_result():
    validator = WalkForwardValidator()
    window = make_window()
    metrics = make_metrics()

    result = validator.create_window_result(
        window=window,
        train_score=1.50,
        validation_score=1.40,
        test_score=1.30,
        train_metrics=metrics,
        validation_metrics=metrics,
        test_metrics=metrics,
        metadata={"asset": "BTC"},
    )

    assert result.window == window
    assert result.train_score == 1.50
    assert result.validation_score == 1.40
    assert result.test_score == 1.30
    assert result.metadata["asset"] == "BTC"


# ============================================================
# Aggregation
# ============================================================


def test_aggregate_passing_results():
    validator = WalkForwardValidator()

    results = [
        make_result(
            window_id=0,
            train_score=1.50,
            validation_score=1.40,
            test_score=1.30,
        ),
        make_result(
            window_id=1,
            train_score=1.60,
            validation_score=1.45,
            test_score=1.35,
        ),
        make_result(
            window_id=2,
            train_score=1.40,
            validation_score=1.30,
            test_score=1.20,
        ),
    ]

    report = validator.aggregate(results)

    assert report.passed is True
    assert report.window_count == 3
    assert report.positive_oos_ratio == 1.0
    assert report.stable_window_ratio == 1.0


def test_aggregate_average_scores():
    validator = WalkForwardValidator()

    results = [
        make_result(
            window_id=0,
            train_score=2.0,
            validation_score=1.5,
            test_score=1.0,
        ),
        make_result(
            window_id=1,
            train_score=4.0,
            validation_score=3.0,
            test_score=2.0,
        ),
    ]

    report = validator.aggregate(results)

    assert (
        report.average_train_score
        == pytest.approx(3.0)
    )

    assert (
        report.average_validation_score
        == pytest.approx(2.25)
    )

    assert (
        report.average_test_score
        == pytest.approx(1.50)
    )

    assert (
        report.average_test_degradation
        == pytest.approx(0.50)
    )


def test_aggregate_negative_oos_fails():
    validator = WalkForwardValidator()

    results = [
        make_result(
            window_id=0,
            test_score=-0.10,
        ),
        make_result(
            window_id=1,
            test_score=-0.20,
        ),
        make_result(
            window_id=2,
            test_score=0.10,
        ),
    ]

    report = validator.aggregate(results)

    assert report.passed is False

    assert (
        report.positive_oos_ratio
        == pytest.approx(1 / 3)
    )


def test_aggregate_instability_fails():
    validator = WalkForwardValidator(
        maximum_degradation=0.20
    )

    results = [
        make_result(
            window_id=0,
            train_score=2.0,
            test_score=0.5,
        ),
        make_result(
            window_id=1,
            train_score=2.0,
            test_score=0.6,
        ),
        make_result(
            window_id=2,
            train_score=2.0,
            test_score=0.7,
        ),
    ]

    report = validator.aggregate(results)

    assert report.passed is False
    assert report.stable_window_ratio == 0.0


def test_empty_aggregate_rejected():
    validator = WalkForwardValidator()

    with pytest.raises(
        WalkForwardValidationError
    ):
        validator.aggregate([])


def test_report_fingerprint_stable():
    validator = WalkForwardValidator()

    results = [
        make_result(window_id=0),
        make_result(window_id=1),
    ]

    first = validator.aggregate(results)
    second = validator.aggregate(results)

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_report_requires_windows():
    with pytest.raises(
        ValidationConfigurationError
    ):
        WalkForwardReport(
            windows=[],
            average_train_score=1.0,
            average_validation_score=1.0,
            average_test_score=1.0,
            average_test_degradation=0.0,
            positive_oos_ratio=1.0,
            stable_window_ratio=1.0,
            passed=True,
        )


# ============================================================
# ValidationResult adapter
# ============================================================


def test_to_validation_result_pass():
    validator = WalkForwardValidator()

    report = validator.aggregate(
        [
            make_result(window_id=0),
            make_result(window_id=1),
            make_result(window_id=2),
        ]
    )

    result = validator.to_validation_result(
        report
    )

    assert (
        result.stage
        == ValidationStage.WALK_FORWARD
    )

    assert (
        result.status
        == ValidationStatus.PASSED
    )

    assert 0.0 <= result.score <= 1.0

    assert (
        result.metadata[
            "real_money_execution"
        ]
        is False
    )


def test_to_validation_result_fail():
    validator = WalkForwardValidator()

    report = validator.aggregate(
        [
            make_result(
                window_id=0,
                test_score=-1.0,
            ),
            make_result(
                window_id=1,
                test_score=-1.0,
            ),
        ]
    )

    result = validator.to_validation_result(
        report
    )

    assert (
        result.status
        == ValidationStatus.FAILED
    )

    assert 0.0 <= result.score <= 1.0


# ============================================================
# validate_scores
# ============================================================


def test_validate_scores():
    validator = WalkForwardValidator(
        make_config(windows=3)
    )

    report = validator.validate_scores(
        observation_count=1000,
        scores=[
            (1.50, 1.40, 1.30),
            (1.60, 1.50, 1.40),
            (1.40, 1.30, 1.20),
        ],
    )

    assert report.window_count == 3
    assert report.passed is True


def test_validate_scores_count_mismatch():
    validator = WalkForwardValidator(
        make_config(windows=3)
    )

    with pytest.raises(
        WalkForwardValidationError
    ):
        validator.validate_scores(
            observation_count=1000,
            scores=[
                (1.50, 1.40, 1.30),
            ],
        )


def test_validate_scores_invalid_tuple():
    validator = WalkForwardValidator(
        make_config(windows=1)
    )

    with pytest.raises(
        WalkForwardValidationError
    ):
        validator.validate_scores(
            observation_count=1000,
            scores=[
                (1.50, 1.40),
            ],
        )


# ============================================================
# Snapshot
# ============================================================


def test_snapshot():
    validator = WalkForwardValidator(
        make_config(windows=6),
        embargo_size=4,
        expanding_window=True,
        maximum_degradation=0.35,
        minimum_positive_oos_ratio=0.70,
        minimum_stable_window_ratio=0.75,
    )

    snapshot = validator.snapshot()

    assert snapshot["version"] == "v6.0.2"

    assert (
        snapshot["walk_forward_windows"]
        == 6
    )

    assert snapshot["embargo_size"] == 4

    assert (
        snapshot["expanding_window"]
        is True
    )

    assert (
        snapshot["maximum_degradation"]
        == 0.35
    )

    assert (
        snapshot[
            "minimum_positive_oos_ratio"
        ]
        == 0.70
    )

    assert (
        snapshot[
            "minimum_stable_window_ratio"
        ]
        == 0.75
    )

    assert snapshot["research_only"] is True

    assert (
        snapshot["real_money_execution"]
        is False
    )

    assert len(
        snapshot["config_fingerprint"]
    ) == 64


# ============================================================
# End-to-end walk-forward module test
# ============================================================


def test_walk_forward_complete_flow():
    config = ValidationConfig(
        walk_forward_windows=5,
        minimum_observations=100,
    )

    validator = WalkForwardValidator(
        config,
        embargo_size=2,
        expanding_window=True,
        maximum_degradation=0.50,
        minimum_positive_oos_ratio=0.60,
        minimum_stable_window_ratio=0.60,
    )

    windows = validator.build_windows(
        2000
    )

    assert len(windows) == 5

    results = []

    scores = [
        (1.80, 1.65, 1.55),
        (1.75, 1.60, 1.50),
        (1.70, 1.58, 1.48),
        (1.65, 1.52, 1.42),
        (1.60, 1.48, 1.38),
    ]

    metrics = make_metrics(
        sharpe=1.50,
        observations=300,
    )

    for window, (
        train_score,
        validation_score,
        test_score,
    ) in zip(
        windows,
        scores,
    ):
        result = (
            validator.create_window_result(
                window=window,
                train_score=train_score,
                validation_score=(
                    validation_score
                ),
                test_score=test_score,
                train_metrics=metrics,
                validation_metrics=metrics,
                test_metrics=metrics,
                metadata={
                    "asset": "BTC",
                    "horizon": "7D",
                },
            )
        )

        results.append(result)

    report = validator.aggregate(
        results
    )

    validation_result = (
        validator.to_validation_result(
            report
        )
    )

    assert report.window_count == 5
    assert report.passed is True

    assert (
        report.positive_oos_ratio
        == 1.0
    )

    assert (
        report.stable_window_ratio
        == 1.0
    )

    assert (
        validation_result.stage
        == ValidationStage.WALK_FORWARD
    )

    assert (
        validation_result.status
        == ValidationStatus.PASSED
    )

    assert (
        validation_result.score
        is not None
    )

    assert (
        validation_result.score
        > 0.80
    )

    assert (
        validation_result.metadata[
            "research_only"
        ]
        is True
    )

    assert (
        validation_result.metadata[
            "real_money_execution"
        ]
        is False
    )

    assert len(report.fingerprint) == 64