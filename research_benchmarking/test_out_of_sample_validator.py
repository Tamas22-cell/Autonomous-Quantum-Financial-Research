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
from research_benchmarking.out_of_sample_validator import (
    VERSION,
    OutOfSampleReport,
    OutOfSampleSegment,
    OutOfSampleSegmentResult,
    OutOfSampleValidationError,
    OutOfSampleValidator,
)


# ============================================================
# Helpers
# ============================================================


def make_config(
    *,
    minimum_observations: int = 30,
    test_fraction: float = 0.15,
) -> ValidationConfig:
    remaining = 1.0 - test_fraction

    return ValidationConfig(
        train_fraction=remaining - 0.15,
        validation_fraction=0.15,
        test_fraction=test_fraction,
        minimum_observations=minimum_observations,
    )


def make_metrics(
    *,
    sharpe: float = 1.50,
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


def make_segment(
    segment_id: int = 0,
) -> OutOfSampleSegment:
    return OutOfSampleSegment(
        segment_id=segment_id,
        start=100,
        end=150,
        label=f"OOS_{segment_id + 1}",
    )


def make_result(
    *,
    segment_id: int = 0,
    in_sample_score: float = 1.50,
    out_of_sample_score: float = 1.25,
    benchmark_score: float | None = None,
) -> OutOfSampleSegmentResult:
    return OutOfSampleSegmentResult(
        segment=make_segment(segment_id),
        in_sample_score=in_sample_score,
        out_of_sample_score=out_of_sample_score,
        benchmark_score=benchmark_score,
    )


# ============================================================
# Version
# ============================================================


def test_version():
    assert VERSION == "v6.0.2"


# ============================================================
# OutOfSampleSegment
# ============================================================


def test_segment_creation():
    segment = make_segment()

    assert segment.segment_id == 0
    assert segment.start == 100
    assert segment.end == 150
    assert segment.label == "OOS_1"


def test_segment_size():
    segment = make_segment()

    assert segment.size == 50


def test_segment_is_frozen():
    segment = make_segment()

    with pytest.raises(FrozenInstanceError):
        segment.end = 200


def test_segment_fingerprint_stable():
    first = make_segment()
    second = make_segment()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("segment_id", True),
        ("segment_id", 1.5),
        ("start", "100"),
        ("end", 150.5),
    ],
)
def test_segment_integer_validation(
    field,
    value,
):
    kwargs = {
        "segment_id": 0,
        "start": 100,
        "end": 150,
        "label": "OOS",
    }

    kwargs[field] = value

    with pytest.raises(
        ValidationConfigurationError
    ):
        OutOfSampleSegment(**kwargs)


def test_negative_segment_id_rejected():
    with pytest.raises(
        ValidationConfigurationError
    ):
        OutOfSampleSegment(
            segment_id=-1,
            start=100,
            end=150,
        )


def test_negative_start_rejected():
    with pytest.raises(
        ValidationConfigurationError
    ):
        OutOfSampleSegment(
            segment_id=0,
            start=-1,
            end=150,
        )


def test_invalid_segment_range():
    with pytest.raises(
        ValidationConfigurationError
    ):
        OutOfSampleSegment(
            segment_id=0,
            start=100,
            end=100,
        )


def test_empty_segment_label():
    with pytest.raises(
        ValidationConfigurationError
    ):
        OutOfSampleSegment(
            segment_id=0,
            start=100,
            end=150,
            label="",
        )


# ============================================================
# OutOfSampleSegmentResult
# ============================================================


def test_segment_result_creation():
    result = make_result()

    assert result.in_sample_score == 1.50
    assert result.out_of_sample_score == 1.25
    assert result.positive_oos is True
    assert result.benchmark_score is None
    assert result.beats_benchmark is None


def test_degradation_ratio():
    result = make_result(
        in_sample_score=2.0,
        out_of_sample_score=1.0,
    )

    assert result.degradation_ratio == pytest.approx(
        0.50
    )


def test_negative_degradation_when_oos_improves():
    result = make_result(
        in_sample_score=1.0,
        out_of_sample_score=1.5,
    )

    assert result.degradation_ratio == pytest.approx(
        -0.50
    )


def test_zero_in_sample_degradation():
    result = make_result(
        in_sample_score=0.0,
        out_of_sample_score=1.0,
    )

    assert result.degradation_ratio == 0.0


def test_negative_oos_not_positive():
    result = make_result(
        out_of_sample_score=-0.1
    )

    assert result.positive_oos is False


def test_zero_oos_not_positive():
    result = make_result(
        out_of_sample_score=0.0
    )

    assert result.positive_oos is False


def test_beats_benchmark():
    result = make_result(
        out_of_sample_score=1.25,
        benchmark_score=1.00,
    )

    assert result.beats_benchmark is True
    assert result.benchmark_excess == pytest.approx(
        0.25
    )


def test_does_not_beat_benchmark():
    result = make_result(
        out_of_sample_score=0.80,
        benchmark_score=1.00,
    )

    assert result.beats_benchmark is False
    assert result.benchmark_excess == pytest.approx(
        -0.20
    )


def test_equal_benchmark_not_win():
    result = make_result(
        out_of_sample_score=1.00,
        benchmark_score=1.00,
    )

    assert result.beats_benchmark is False


def test_segment_result_with_metrics():
    metrics = make_metrics()

    result = OutOfSampleSegmentResult(
        segment=make_segment(),
        in_sample_score=1.50,
        out_of_sample_score=1.25,
        benchmark_score=1.00,
        in_sample_metrics=metrics,
        out_of_sample_metrics=metrics,
        benchmark_metrics=metrics,
    )

    assert result.in_sample_metrics == metrics
    assert result.out_of_sample_metrics == metrics
    assert result.benchmark_metrics == metrics


def test_segment_result_fingerprint():
    first = make_result(
        benchmark_score=1.0
    )

    second = make_result(
        benchmark_score=1.0
    )

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("in_sample_score", True),
        ("in_sample_score", float("nan")),
        ("out_of_sample_score", float("inf")),
        ("out_of_sample_score", float("-inf")),
        ("benchmark_score", float("nan")),
    ],
)
def test_invalid_segment_result_scores(
    field,
    value,
):
    kwargs = {
        "segment": make_segment(),
        "in_sample_score": 1.50,
        "out_of_sample_score": 1.20,
        "benchmark_score": 1.00,
    }

    kwargs[field] = value

    with pytest.raises(
        ValidationConfigurationError
    ):
        OutOfSampleSegmentResult(**kwargs)


# ============================================================
# Validator construction
# ============================================================


def test_default_validator():
    validator = OutOfSampleValidator()

    assert validator.maximum_degradation == 0.50

    assert (
        validator.minimum_positive_oos_ratio
        == 0.60
    )

    assert (
        validator.minimum_stable_segment_ratio
        == 0.60
    )

    assert (
        validator.minimum_benchmark_win_ratio
        == 0.50
    )

    assert (
        validator.require_benchmark_outperformance
        is False
    )


def test_custom_validator():
    validator = OutOfSampleValidator(
        maximum_degradation=0.30,
        minimum_positive_oos_ratio=0.75,
        minimum_stable_segment_ratio=0.80,
        minimum_benchmark_win_ratio=0.70,
        require_benchmark_outperformance=True,
    )

    assert validator.maximum_degradation == 0.30
    assert validator.minimum_positive_oos_ratio == 0.75
    assert validator.minimum_stable_segment_ratio == 0.80
    assert validator.minimum_benchmark_win_ratio == 0.70

    assert (
        validator.require_benchmark_outperformance
        is True
    )


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        float("nan"),
        float("inf"),
        True,
    ],
)
def test_invalid_maximum_degradation(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        OutOfSampleValidator(
            maximum_degradation=value
        )


@pytest.mark.parametrize(
    "parameter",
    [
        "minimum_positive_oos_ratio",
        "minimum_stable_segment_ratio",
        "minimum_benchmark_win_ratio",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        1.1,
    ],
)
def test_invalid_ratio_parameters(
    parameter,
    value,
):
    kwargs = {
        parameter: value
    }

    with pytest.raises(
        ValidationConfigurationError
    ):
        OutOfSampleValidator(**kwargs)


# ============================================================
# Holdout construction
# ============================================================


def test_build_holdout_segment():
    validator = OutOfSampleValidator()

    segment = validator.build_holdout_segment(
        1000
    )

    assert segment.segment_id == 0
    assert segment.start == 850
    assert segment.end == 1000
    assert segment.size == 150
    assert segment.label == "FINAL_HOLDOUT"


def test_holdout_respects_custom_fraction():
    validator = OutOfSampleValidator(
        make_config(
            test_fraction=0.20
        )
    )

    segment = validator.build_holdout_segment(
        1000
    )

    assert segment.start == 800
    assert segment.end == 1000
    assert segment.size == 200


def test_holdout_insufficient_observations():
    validator = OutOfSampleValidator(
        make_config(
            minimum_observations=100
        )
    )

    with pytest.raises(
        OutOfSampleValidationError
    ):
        validator.build_holdout_segment(
            99
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        100.5,
        "100",
    ],
)
def test_invalid_holdout_observation_count(value):
    validator = OutOfSampleValidator()

    with pytest.raises(
        ValidationConfigurationError
    ):
        validator.build_holdout_segment(
            value
        )


# ============================================================
# Multiple segment construction
# ============================================================


def test_build_segments_count():
    validator = OutOfSampleValidator()

    segments = validator.build_segments(
        1000,
        5,
    )

    assert len(segments) == 5


def test_build_segments_cover_oos_region():
    validator = OutOfSampleValidator()

    segments = validator.build_segments(
        1000,
        5,
    )

    assert segments[0].start == 850
    assert segments[-1].end == 1000


def test_build_segments_are_contiguous():
    validator = OutOfSampleValidator()

    segments = validator.build_segments(
        1000,
        5,
    )

    for previous, current in zip(
        segments,
        segments[1:],
    ):
        assert previous.end == current.start


def test_build_segments_have_unique_ids():
    validator = OutOfSampleValidator()

    segments = validator.build_segments(
        1000,
        5,
    )

    assert [
        segment.segment_id
        for segment in segments
    ] == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_build_segments_labels():
    validator = OutOfSampleValidator()

    segments = validator.build_segments(
        1000,
        3,
    )

    assert [
        segment.label
        for segment in segments
    ] == [
        "OOS_1",
        "OOS_2",
        "OOS_3",
    ]


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        2.5,
    ],
)
def test_invalid_segment_count(value):
    validator = OutOfSampleValidator()

    with pytest.raises(
        ValidationConfigurationError
    ):
        validator.build_segments(
            1000,
            value,
        )


def test_build_segments_insufficient_observations():
    validator = OutOfSampleValidator(
        make_config(
            minimum_observations=100
        )
    )

    with pytest.raises(
        OutOfSampleValidationError
    ):
        validator.build_segments(
            99,
            3,
        )


# ============================================================
# Segment extraction
# ============================================================


def test_extract_segment():
    values = list(range(200))
    segment = make_segment()

    extracted = (
        OutOfSampleValidator.extract_segment(
            values,
            segment,
        )
    )

    assert extracted == list(
        range(100, 150)
    )


def test_extract_segment_size():
    values = list(range(200))
    segment = make_segment()

    extracted = (
        OutOfSampleValidator.extract_segment(
            values,
            segment,
        )
    )

    assert len(extracted) == segment.size


def test_extract_segment_short_sequence():
    segment = make_segment()

    with pytest.raises(
        OutOfSampleValidationError
    ):
        OutOfSampleValidator.extract_segment(
            list(range(149)),
            segment,
        )


# ============================================================
# create_segment_result
# ============================================================


def test_create_segment_result():
    validator = OutOfSampleValidator()
    segment = make_segment()
    metrics = make_metrics()

    result = validator.create_segment_result(
        segment=segment,
        in_sample_score=1.50,
        out_of_sample_score=1.25,
        benchmark_score=1.00,
        in_sample_metrics=metrics,
        out_of_sample_metrics=metrics,
        benchmark_metrics=metrics,
        metadata={
            "asset": "BTC",
        },
    )

    assert result.segment == segment
    assert result.in_sample_score == 1.50
    assert result.out_of_sample_score == 1.25
    assert result.benchmark_score == 1.00
    assert result.metadata["asset"] == "BTC"


# ============================================================
# Aggregation
# ============================================================


def test_aggregate_passing_results():
    validator = OutOfSampleValidator()

    results = [
        make_result(
            segment_id=0,
            in_sample_score=1.50,
            out_of_sample_score=1.30,
        ),
        make_result(
            segment_id=1,
            in_sample_score=1.60,
            out_of_sample_score=1.40,
        ),
        make_result(
            segment_id=2,
            in_sample_score=1.40,
            out_of_sample_score=1.20,
        ),
    ]

    report = validator.aggregate(results)

    assert report.passed is True
    assert report.segment_count == 3
    assert report.positive_oos_ratio == 1.0
    assert report.stable_segment_ratio == 1.0


def test_aggregate_average_scores():
    validator = OutOfSampleValidator()

    results = [
        make_result(
            segment_id=0,
            in_sample_score=2.0,
            out_of_sample_score=1.0,
        ),
        make_result(
            segment_id=1,
            in_sample_score=4.0,
            out_of_sample_score=2.0,
        ),
    ]

    report = validator.aggregate(results)

    assert (
        report.average_in_sample_score
        == pytest.approx(3.0)
    )

    assert (
        report.average_out_of_sample_score
        == pytest.approx(1.5)
    )

    assert (
        report.average_degradation_ratio
        == pytest.approx(0.50)
    )


def test_aggregate_negative_oos_fails():
    validator = OutOfSampleValidator()

    results = [
        make_result(
            segment_id=0,
            out_of_sample_score=-0.10,
        ),
        make_result(
            segment_id=1,
            out_of_sample_score=-0.20,
        ),
        make_result(
            segment_id=2,
            out_of_sample_score=0.10,
        ),
    ]

    report = validator.aggregate(results)

    assert report.passed is False

    assert (
        report.positive_oos_ratio
        == pytest.approx(1 / 3)
    )


def test_aggregate_instability_fails():
    validator = OutOfSampleValidator(
        maximum_degradation=0.20
    )

    results = [
        make_result(
            segment_id=0,
            in_sample_score=2.0,
            out_of_sample_score=0.5,
        ),
        make_result(
            segment_id=1,
            in_sample_score=2.0,
            out_of_sample_score=0.6,
        ),
        make_result(
            segment_id=2,
            in_sample_score=2.0,
            out_of_sample_score=0.7,
        ),
    ]

    report = validator.aggregate(results)

    assert report.passed is False
    assert report.stable_segment_ratio == 0.0


def test_benchmark_win_ratio():
    validator = OutOfSampleValidator()

    results = [
        make_result(
            segment_id=0,
            out_of_sample_score=1.20,
            benchmark_score=1.00,
        ),
        make_result(
            segment_id=1,
            out_of_sample_score=0.80,
            benchmark_score=1.00,
        ),
        make_result(
            segment_id=2,
            out_of_sample_score=1.30,
            benchmark_score=1.00,
        ),
    ]

    report = validator.aggregate(results)

    assert (
        report.benchmark_comparison_count
        == 3
    )

    assert (
        report.benchmark_win_ratio
        == pytest.approx(2 / 3)
    )


def test_no_benchmark_results():
    validator = OutOfSampleValidator()

    report = validator.aggregate(
        [
            make_result(segment_id=0),
            make_result(segment_id=1),
        ]
    )

    assert (
        report.benchmark_comparison_count
        == 0
    )

    assert report.benchmark_win_ratio is None


def test_required_benchmark_passes():
    validator = OutOfSampleValidator(
        require_benchmark_outperformance=True,
        minimum_benchmark_win_ratio=0.50,
    )

    results = [
        make_result(
            segment_id=0,
            out_of_sample_score=1.30,
            benchmark_score=1.00,
        ),
        make_result(
            segment_id=1,
            out_of_sample_score=1.20,
            benchmark_score=1.00,
        ),
    ]

    report = validator.aggregate(results)

    assert report.passed is True


def test_required_benchmark_fails():
    validator = OutOfSampleValidator(
        require_benchmark_outperformance=True,
        minimum_benchmark_win_ratio=0.75,
    )

    results = [
        make_result(
            segment_id=0,
            out_of_sample_score=1.20,
            benchmark_score=1.00,
        ),
        make_result(
            segment_id=1,
            out_of_sample_score=0.90,
            benchmark_score=1.00,
        ),
    ]

    report = validator.aggregate(results)

    assert report.benchmark_win_ratio == 0.50
    assert report.passed is False


def test_required_benchmark_missing_fails():
    validator = OutOfSampleValidator(
        require_benchmark_outperformance=True
    )

    report = validator.aggregate(
        [
            make_result(segment_id=0),
            make_result(segment_id=1),
        ]
    )

    assert report.benchmark_win_ratio is None
    assert report.passed is False


def test_empty_aggregate_rejected():
    validator = OutOfSampleValidator()

    with pytest.raises(
        OutOfSampleValidationError
    ):
        validator.aggregate([])


def test_report_fingerprint_stable():
    validator = OutOfSampleValidator()

    results = [
        make_result(segment_id=0),
        make_result(segment_id=1),
    ]

    first = validator.aggregate(results)
    second = validator.aggregate(results)

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_report_requires_segments():
    with pytest.raises(
        ValidationConfigurationError
    ):
        OutOfSampleReport(
            segments=[],
            average_in_sample_score=1.0,
            average_out_of_sample_score=1.0,
            average_degradation_ratio=0.0,
            positive_oos_ratio=1.0,
            stable_segment_ratio=1.0,
            benchmark_comparison_count=0,
            benchmark_win_ratio=None,
            passed=True,
        )


# ============================================================
# ValidationResult adapter
# ============================================================


def test_to_validation_result_pass():
    validator = OutOfSampleValidator()

    report = validator.aggregate(
        [
            make_result(segment_id=0),
            make_result(segment_id=1),
            make_result(segment_id=2),
        ]
    )

    result = validator.to_validation_result(
        report
    )

    assert (
        result.stage
        == ValidationStage.OUT_OF_SAMPLE
    )

    assert (
        result.status
        == ValidationStatus.PASSED
    )

    assert result.score is not None
    assert 0.0 <= result.score <= 1.0

    assert (
        result.metadata[
            "real_money_execution"
        ]
        is False
    )


def test_to_validation_result_fail():
    validator = OutOfSampleValidator()

    report = validator.aggregate(
        [
            make_result(
                segment_id=0,
                out_of_sample_score=-1.0,
            ),
            make_result(
                segment_id=1,
                out_of_sample_score=-1.0,
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

    assert result.score is not None
    assert 0.0 <= result.score <= 1.0


# ============================================================
# validate_scores
# ============================================================


def test_validate_scores():
    validator = OutOfSampleValidator()

    report = validator.validate_scores(
        observation_count=1000,
        in_sample_score=1.50,
        out_of_sample_scores=[
            1.30,
            1.25,
            1.20,
        ],
    )

    assert report.segment_count == 3
    assert report.passed is True


def test_validate_scores_with_benchmark():
    validator = OutOfSampleValidator(
        require_benchmark_outperformance=True
    )

    report = validator.validate_scores(
        observation_count=1000,
        in_sample_score=1.50,
        out_of_sample_scores=[
            1.30,
            1.25,
            1.20,
        ],
        benchmark_scores=[
            1.00,
            1.00,
            1.00,
        ],
    )

    assert report.segment_count == 3
    assert report.benchmark_win_ratio == 1.0
    assert report.passed is True


def test_validate_scores_empty():
    validator = OutOfSampleValidator()

    with pytest.raises(
        OutOfSampleValidationError
    ):
        validator.validate_scores(
            observation_count=1000,
            in_sample_score=1.50,
            out_of_sample_scores=[],
        )


def test_validate_scores_benchmark_mismatch():
    validator = OutOfSampleValidator()

    with pytest.raises(
        OutOfSampleValidationError
    ):
        validator.validate_scores(
            observation_count=1000,
            in_sample_score=1.50,
            out_of_sample_scores=[
                1.20,
                1.10,
            ],
            benchmark_scores=[
                1.00,
            ],
        )


# ============================================================
# Snapshot
# ============================================================


def test_snapshot():
    validator = OutOfSampleValidator(
        maximum_degradation=0.35,
        minimum_positive_oos_ratio=0.70,
        minimum_stable_segment_ratio=0.75,
        minimum_benchmark_win_ratio=0.60,
        require_benchmark_outperformance=True,
    )

    snapshot = validator.snapshot()

    assert snapshot["version"] == "v6.0.2"

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
            "minimum_stable_segment_ratio"
        ]
        == 0.75
    )

    assert (
        snapshot[
            "minimum_benchmark_win_ratio"
        ]
        == 0.60
    )

    assert (
        snapshot[
            "require_benchmark_outperformance"
        ]
        is True
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
# Complete OOS flow
# ============================================================


def test_complete_out_of_sample_flow():
    config = ValidationConfig(
        minimum_observations=100,
    )

    validator = OutOfSampleValidator(
        config,
        maximum_degradation=0.40,
        minimum_positive_oos_ratio=0.75,
        minimum_stable_segment_ratio=0.75,
        minimum_benchmark_win_ratio=0.60,
        require_benchmark_outperformance=True,
    )

    segments = validator.build_segments(
        2000,
        5,
    )

    assert len(segments) == 5

    metrics = make_metrics(
        sharpe=1.45,
        observations=300,
    )

    oos_scores = [
        1.45,
        1.40,
        1.35,
        1.30,
        1.25,
    ]

    benchmark_scores = [
        0.95,
        1.00,
        0.90,
        0.85,
        0.80,
    ]

    results = []

    for index, segment in enumerate(
        segments
    ):
        result = validator.create_segment_result(
            segment=segment,
            in_sample_score=1.60,
            out_of_sample_score=(
                oos_scores[index]
            ),
            benchmark_score=(
                benchmark_scores[index]
            ),
            in_sample_metrics=metrics,
            out_of_sample_metrics=metrics,
            benchmark_metrics=metrics,
            metadata={
                "asset": "BTC",
                "horizon": "7D",
            },
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

    assert report.segment_count == 5
    assert report.passed is True

    assert report.positive_oos_ratio == 1.0
    assert report.stable_segment_ratio == 1.0
    assert report.benchmark_win_ratio == 1.0

    assert (
        report.average_out_of_sample_score
        == pytest.approx(1.35)
    )

    assert (
        validation_result.stage
        == ValidationStage.OUT_OF_SAMPLE
    )

    assert (
        validation_result.status
        == ValidationStatus.PASSED
    )

    assert validation_result.score is not None
    assert validation_result.score > 0.80

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