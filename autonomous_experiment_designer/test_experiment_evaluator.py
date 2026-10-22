import pytest

from autonomous_experiment_designer.experiment_designer import (
    ExperimentDesigner,
)
from autonomous_experiment_designer.experiment_evaluator import (
    EvaluationStatus,
    ExperimentEvaluator,
    Recommendation,
)
from autonomous_experiment_designer.experiment_executor import (
    ExecutionStatus,
    ExecutionResult,
    ExperimentExecutor,
)
from autonomous_experiment_designer.uncertainty_detector import (
    UncertaintyDetector,
)


@pytest.fixture
def plan():
    detector = UncertaintyDetector()
    designer = ExperimentDesigner()

    report = detector.detect(
        regime_distance=0.60,
    )

    result = designer.design(
        report,
        experiment_id="EXP-EVAL-001",
    )

    assert result is not None
    return result


@pytest.fixture
def successful_execution(plan):
    executor = ExperimentExecutor()

    return executor.execute(
        plan,
        approved=True,
        runner=lambda _: {
            "sharpe": 1.50,
            "return": 0.20,
            "drawdown": 0.10,
        },
        execution_id="EXEC-EVAL-001",
    )


@pytest.fixture
def evaluator():
    return ExperimentEvaluator()


def test_default_evaluator_configuration(evaluator):
    assert evaluator.minimum_confidence == 0.70
    assert evaluator.minimum_validation_score == 0.70
    assert evaluator.minimum_improvement == 0.0


def test_successful_execution_is_validated(
    evaluator,
    successful_execution,
):
    result = evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.00,
            "return": 0.10,
            "drawdown": 0.20,
        },
        confidence=0.90,
        validation_score=0.90,
        evidence=("backtest", "walk_forward"),
    )

    assert result.status == EvaluationStatus.VALIDATED
    assert result.recommendation == Recommendation.ACCEPT


def test_validated_result_preserves_execution_identity(
    evaluator,
    successful_execution,
):
    result = evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.00,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    assert (
        result.execution_id
        == successful_execution.metadata.execution_id
    )
    assert (
        result.experiment_id
        == successful_execution.metadata.experiment_id
    )


def test_metrics_are_calculated_correctly(
    evaluator,
    successful_execution,
):
    result = evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.00,
            "return": 0.10,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    assert len(result.metrics) == 2

    sharpe = next(
        metric
        for metric in result.metrics
        if metric.name == "sharpe"
    )

    assert sharpe.baseline == pytest.approx(1.00)
    assert sharpe.observed == pytest.approx(1.50)
    assert sharpe.improvement == pytest.approx(0.50)
    assert sharpe.relative_change == pytest.approx(0.50)


def test_metrics_are_sorted_by_name(
    evaluator,
    successful_execution,
):
    result = evaluator.evaluate(
        successful_execution,
        baseline={
            "return": 0.10,
            "sharpe": 1.00,
            "drawdown": 0.20,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    names = [
        metric.name
        for metric in result.metrics
    ]

    assert names == sorted(names)


def test_low_validation_score_is_inconclusive(
    evaluator,
    successful_execution,
):
    result = evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.00,
        },
        confidence=0.90,
        validation_score=0.50,
    )

    assert result.status == EvaluationStatus.INCONCLUSIVE
    assert result.recommendation == Recommendation.REPEAT


def test_low_confidence_is_inconclusive(
    evaluator,
    successful_execution,
):
    result = evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.00,
        },
        confidence=0.50,
        validation_score=0.90,
    )

    assert result.status == EvaluationStatus.INCONCLUSIVE
    assert (
        result.recommendation
        == Recommendation.INVESTIGATE
    )


def test_insufficient_improvement_is_contradicted(
    evaluator,
    successful_execution,
):
    result = evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.50,
            "return": 0.20,
            "drawdown": 0.10,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    assert result.status == EvaluationStatus.CONTRADICTED
    assert result.recommendation == Recommendation.REJECT


def test_failed_execution_is_not_validated(
    evaluator,
    plan,
):
    executor = ExperimentExecutor()

    execution = executor.execute(
        plan,
        approved=False,
        runner=lambda _: {
            "sharpe": 2.0,
        },
        execution_id="EXEC-REJECTED",
    )

    result = evaluator.evaluate(
        execution,
        baseline={
            "sharpe": 1.0,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    assert result.status == EvaluationStatus.FAILED
    assert result.recommendation == Recommendation.REPEAT
    assert result.metrics == ()


def test_failed_runner_is_not_validated(
    evaluator,
    plan,
):
    executor = ExperimentExecutor()

    execution = executor.execute(
        plan,
        approved=True,
        runner=lambda _: (
            (_ for _ in ()).throw(
                RuntimeError("experiment failure")
            )
        ),
        execution_id="EXEC-FAILED",
    )

    assert (
        execution.metadata.status
        == ExecutionStatus.FAILED
    )

    result = evaluator.evaluate(
        execution,
        baseline={
            "sharpe": 1.0,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    assert result.status == EvaluationStatus.FAILED
    assert result.recommendation == Recommendation.REPEAT


def test_evidence_is_preserved(
    evaluator,
    successful_execution,
):
    evidence = (
        "macro_data",
        "market_data",
        "walk_forward_test",
    )

    result = evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.0,
        },
        confidence=0.90,
        validation_score=0.90,
        evidence=evidence,
    )

    assert result.evidence == evidence


def test_evidence_is_optional(
    evaluator,
    successful_execution,
):
    result = evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.0,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    assert result.evidence == ()


def test_zero_baseline_is_handled(
    evaluator,
    successful_execution,
):
    result = evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 0.0,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    metric = result.metrics[0]

    assert metric.baseline == 0.0
    assert metric.observed == pytest.approx(1.50)
    assert metric.improvement == pytest.approx(1.50)
    assert metric.relative_change == pytest.approx(1.50)


def test_zero_baseline_and_zero_observed_have_zero_relative_change(
    evaluator,
    plan,
):
    executor = ExperimentExecutor()

    execution = executor.execute(
        plan,
        approved=True,
        runner=lambda _: {
            "metric": 0.0,
        },
        execution_id="EXEC-ZERO",
    )

    result = evaluator.evaluate(
        execution,
        baseline={
            "metric": 0.0,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    metric = result.metrics[0]

    assert metric.improvement == 0.0
    assert metric.relative_change == 0.0


def test_only_common_metrics_are_evaluated(
    evaluator,
    successful_execution,
):
    result = evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.0,
            "unknown_metric": 10.0,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    names = [
        metric.name
        for metric in result.metrics
    ]

    assert names == ["sharpe"]


def test_missing_common_metric_is_rejected(
    evaluator,
    successful_execution,
):
    with pytest.raises(ValueError):
        evaluator.evaluate(
            successful_execution,
            baseline={
                "unknown_metric": 10.0,
            },
            confidence=0.90,
            validation_score=0.90,
        )


def test_empty_baseline_is_rejected(
    evaluator,
    successful_execution,
):
    with pytest.raises(ValueError):
        evaluator.evaluate(
            successful_execution,
            baseline={},
            confidence=0.90,
            validation_score=0.90,
        )


def test_empty_observed_output_is_rejected(
    evaluator,
    plan,
):
    executor = ExperimentExecutor()

    execution = executor.execute(
        plan,
        approved=True,
        runner=lambda _: {},
        execution_id="EXEC-EMPTY-OUTPUT",
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(
            execution,
            baseline={
                "sharpe": 1.0,
            },
            confidence=0.90,
            validation_score=0.90,
        )


def test_invalid_execution_type_is_rejected(
    evaluator,
):
    with pytest.raises(TypeError):
        evaluator.evaluate(
            "not-an-execution",
            baseline={
                "sharpe": 1.0,
            },
            confidence=0.90,
            validation_score=0.90,
        )


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        True,
        "0.8",
    ],
)
def test_invalid_confidence_is_rejected(
    evaluator,
    successful_execution,
    value,
):
    with pytest.raises(ValueError):
        evaluator.evaluate(
            successful_execution,
            baseline={
                "sharpe": 1.0,
            },
            confidence=value,
            validation_score=0.90,
        )


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        True,
        "0.8",
    ],
)
def test_invalid_validation_score_is_rejected(
    evaluator,
    successful_execution,
    value,
):
    with pytest.raises(ValueError):
        evaluator.evaluate(
            successful_execution,
            baseline={
                "sharpe": 1.0,
            },
            confidence=0.90,
            validation_score=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        float("nan"),
        float("inf"),
        True,
        "0.1",
    ],
)
def test_invalid_minimum_improvement_is_rejected(value):
    with pytest.raises(ValueError):
        ExperimentEvaluator(
            minimum_improvement=value
        )


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        True,
        "0.8",
    ],
)
def test_invalid_minimum_confidence_is_rejected(value):
    with pytest.raises(ValueError):
        ExperimentEvaluator(
            minimum_confidence=value
        )


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        True,
        "0.8",
    ],
)
def test_invalid_minimum_validation_score_is_rejected(value):
    with pytest.raises(ValueError):
        ExperimentEvaluator(
            minimum_validation_score=value
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        float("nan"),
        float("inf"),
        True,
        "0.1",
    ],
)
def test_invalid_relative_change_epsilon_is_rejected(value):
    with pytest.raises(ValueError):
        ExperimentEvaluator(
            relative_change_epsilon=value
        )


def test_invalid_baseline_type_is_rejected(
    evaluator,
    successful_execution,
):
    with pytest.raises(TypeError):
        evaluator.evaluate(
            successful_execution,
            baseline=["invalid"],
            confidence=0.90,
            validation_score=0.90,
        )


def test_invalid_baseline_metric_name_is_rejected(
    evaluator,
    successful_execution,
):
    with pytest.raises(ValueError):
        evaluator.evaluate(
            successful_execution,
            baseline={
                "": 1.0,
            },
            confidence=0.90,
            validation_score=0.90,
        )


def test_invalid_baseline_metric_value_is_rejected(
    evaluator,
    successful_execution,
):
    with pytest.raises(ValueError):
        evaluator.evaluate(
            successful_execution,
            baseline={
                "sharpe": float("nan"),
            },
            confidence=0.90,
            validation_score=0.90,
        )


def test_invalid_evidence_string_is_rejected(
    evaluator,
    successful_execution,
):
    with pytest.raises(ValueError):
        evaluator.evaluate(
            successful_execution,
            baseline={
                "sharpe": 1.0,
            },
            confidence=0.90,
            validation_score=0.90,
            evidence="invalid",
        )


def test_empty_evidence_item_is_rejected(
    evaluator,
    successful_execution,
):
    with pytest.raises(ValueError):
        evaluator.evaluate(
            successful_execution,
            baseline={
                "sharpe": 1.0,
            },
            confidence=0.90,
            validation_score=0.90,
            evidence=(
                "valid",
                "",
            ),
        )


def test_custom_thresholds_are_respected(
    successful_execution,
):
    evaluator = ExperimentEvaluator(
        minimum_confidence=0.90,
        minimum_validation_score=0.90,
        minimum_improvement=0.40,
    )

    result = evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.00,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    assert result.status == EvaluationStatus.VALIDATED
    assert result.recommendation == Recommendation.ACCEPT


def test_custom_improvement_threshold_can_reject(
    successful_execution,
):
    evaluator = ExperimentEvaluator(
        minimum_improvement=0.60,
    )

    result = evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.00,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    assert result.status == EvaluationStatus.CONTRADICTED
    assert result.recommendation == Recommendation.REJECT


def test_boundary_confidence_is_accepted(
    evaluator,
    successful_execution,
):
    result = evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.00,
        },
        confidence=0.70,
        validation_score=0.70,
    )

    assert result.status == EvaluationStatus.VALIDATED


def test_boundary_improvement_is_accepted(
    successful_execution,
):
    evaluator = ExperimentEvaluator(
        minimum_improvement=0.50,
    )

    result = evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.00,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    assert result.status == EvaluationStatus.VALIDATED


def test_evaluation_is_immutable(
    evaluator,
    successful_execution,
):
    result = evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.00,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    with pytest.raises(AttributeError):
        result.status = EvaluationStatus.FAILED


def test_metric_is_immutable(
    evaluator,
    successful_execution,
):
    result = evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.00,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    metric = result.metrics[0]

    with pytest.raises(AttributeError):
        metric.observed = 99.0


def test_confidence_and_validation_are_preserved(
    evaluator,
    successful_execution,
):
    result = evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.00,
        },
        confidence=0.83,
        validation_score=0.91,
    )

    assert result.confidence == pytest.approx(0.83)
    assert result.validation_score == pytest.approx(0.91)


def test_rationale_is_nonempty(
    evaluator,
    successful_execution,
):
    result = evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.00,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    assert result.rationale.strip()


def test_original_execution_is_not_modified(
    evaluator,
    successful_execution,
):
    original_status = successful_execution.metadata.status
    original_output = dict(successful_execution.output)

    evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.00,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    assert (
        successful_execution.metadata.status
        == original_status
    )
    assert dict(successful_execution.output) == original_output