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
    ExperimentExecutor,
)
from autonomous_experiment_designer.research_task_generator import (
    ResearchTaskGenerator,
    ResearchTaskPriority,
    ResearchTaskType,
)
from autonomous_experiment_designer.uncertainty_detector import (
    UncertaintyDetector,
)


@pytest.fixture
def plan():
    report = UncertaintyDetector().detect(
        regime_distance=0.60,
    )

    result = ExperimentDesigner().design(
        report,
        experiment_id="EXP-TASK-001",
    )

    assert result is not None
    return result


@pytest.fixture
def successful_execution(plan):
    return ExperimentExecutor().execute(
        plan,
        approved=True,
        runner=lambda _: {
            "sharpe": 1.50,
            "return": 0.20,
        },
        execution_id="EXEC-TASK-001",
    )


@pytest.fixture
def evaluator():
    return ExperimentEvaluator()


@pytest.fixture
def validated_evaluation(
    evaluator,
    successful_execution,
):
    return evaluator.evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.00,
            "return": 0.10,
        },
        confidence=0.90,
        validation_score=0.90,
        evidence=(
            "backtest",
            "walk_forward",
        ),
    )


@pytest.fixture
def generator():
    return ResearchTaskGenerator()


def test_default_generator_configuration(generator):
    assert generator._default_evidence == ()


def test_generate_validated_follow_up_task(
    generator,
    validated_evaluation,
):
    task = generator.generate(
        validated_evaluation,
        task_id="TASK-001",
    )

    assert task.task_id == "TASK-001"
    assert (
        task.experiment_id
        == validated_evaluation.experiment_id
    )
    assert (
        task.task_type
        == ResearchTaskType.FOLLOW_UP_RESEARCH
    )
    assert task.priority == ResearchTaskPriority.LOW


def test_validated_task_requires_human_approval(
    generator,
    validated_evaluation,
):
    task = generator.generate(
        validated_evaluation,
        task_id="TASK-002",
    )

    assert task.required_approval is True


def test_validated_task_has_nonempty_objective(
    generator,
    validated_evaluation,
):
    task = generator.generate(
        validated_evaluation,
        task_id="TASK-003",
    )

    assert task.objective.strip()


def test_validated_task_preserves_evaluation_evidence(
    generator,
    validated_evaluation,
):
    task = generator.generate(
        validated_evaluation,
        task_id="TASK-004",
    )

    assert task.source_evidence == (
        "backtest",
        "walk_forward",
    )


def test_default_evidence_is_combined(
    validated_evaluation,
):
    generator = ResearchTaskGenerator(
        default_evidence=(
            "market_data",
            "macro_data",
        )
    )

    task = generator.generate(
        validated_evaluation,
        task_id="TASK-005",
    )

    assert task.source_evidence == (
        "market_data",
        "macro_data",
        "backtest",
        "walk_forward",
    )


def test_duplicate_evidence_is_removed(
    validated_evaluation,
):
    generator = ResearchTaskGenerator(
        default_evidence=(
            "backtest",
            "market_data",
        )
    )

    task = generator.generate(
        validated_evaluation,
        task_id="TASK-006",
    )

    assert task.source_evidence == (
        "backtest",
        "market_data",
        "walk_forward",
    )


def test_failed_evaluation_creates_repeat_task(
    generator,
    plan,
):
    execution = ExperimentExecutor().execute(
        plan,
        approved=False,
        runner=lambda _: {
            "sharpe": 2.0,
        },
        execution_id="EXEC-FAILED-TASK",
    )

    evaluation = ExperimentEvaluator().evaluate(
        execution,
        baseline={
            "sharpe": 1.0,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    assert evaluation.status == EvaluationStatus.FAILED

    task = generator.generate(
        evaluation,
        task_id="TASK-007",
    )

    assert (
        task.task_type
        == ResearchTaskType.REPEAT_EXPERIMENT
    )
    assert task.priority == ResearchTaskPriority.CRITICAL
    assert task.required_approval is True


def test_repeat_recommendation_creates_repeat_task(
    generator,
    successful_execution,
):
    evaluation = ExperimentEvaluator().evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.00,
        },
        confidence=0.90,
        validation_score=0.50,
    )

    assert (
        evaluation.recommendation
        == Recommendation.REPEAT
    )

    task = generator.generate(
        evaluation,
        task_id="TASK-008",
    )

    assert (
        task.task_type
        == ResearchTaskType.IMPROVE_VALIDATION
    )
    assert task.priority == ResearchTaskPriority.HIGH


def test_investigate_recommendation_creates_confidence_task(
    generator,
    successful_execution,
):
    evaluation = ExperimentEvaluator().evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.00,
        },
        confidence=0.50,
        validation_score=0.90,
    )

    assert (
        evaluation.recommendation
        == Recommendation.INVESTIGATE
    )

    task = generator.generate(
        evaluation,
        task_id="TASK-009",
    )

    assert (
        task.task_type
        == ResearchTaskType.INVESTIGATE_CONFIDENCE
    )
    assert task.priority == ResearchTaskPriority.HIGH


def test_reject_recommendation_creates_performance_task(
    generator,
    successful_execution,
):
    evaluation = ExperimentEvaluator().evaluate(
        successful_execution,
        baseline={
            "sharpe": 1.50,
            "return": 0.20,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    assert (
        evaluation.recommendation
        == Recommendation.REJECT
    )

    task = generator.generate(
        evaluation,
        task_id="TASK-010",
    )

    assert (
        task.task_type
        == ResearchTaskType.INVESTIGATE_PERFORMANCE
    )
    assert task.priority == ResearchTaskPriority.MEDIUM


def test_rationale_is_preserved(
    generator,
    validated_evaluation,
):
    task = generator.generate(
        validated_evaluation,
        task_id="TASK-011",
    )

    assert (
        task.rationale
        == validated_evaluation.rationale
    )


def test_task_is_immutable(
    generator,
    validated_evaluation,
):
    task = generator.generate(
        validated_evaluation,
        task_id="TASK-012",
    )

    with pytest.raises(AttributeError):
        task.priority = ResearchTaskPriority.CRITICAL


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        None,
        123,
    ],
)
def test_invalid_task_id_is_rejected(
    generator,
    validated_evaluation,
    value,
):
    with pytest.raises(ValueError):
        generator.generate(
            validated_evaluation,
            task_id=value,
        )


def test_invalid_evaluation_is_rejected(
    generator,
):
    with pytest.raises(TypeError):
        generator.generate(
            "invalid",
            task_id="TASK-013",
        )


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        None,
        123,
    ],
)
def test_invalid_default_evidence_is_rejected(value):
    with pytest.raises(ValueError):
        ResearchTaskGenerator(
            default_evidence=value
        )


def test_duplicate_default_evidence_is_rejected():
    with pytest.raises(ValueError):
        ResearchTaskGenerator(
            default_evidence=(
                "backtest",
                "backtest",
            )
        )


def test_string_default_evidence_is_rejected():
    with pytest.raises(ValueError):
        ResearchTaskGenerator(
            default_evidence="backtest"
        )


def test_generate_many_creates_requested_tasks(
    generator,
    validated_evaluation,
):
    tasks = generator.generate_many(
        validated_evaluation,
        task_ids=(
            "TASK-014",
            "TASK-015",
            "TASK-016",
        ),
    )

    assert len(tasks) == 3

    assert [
        task.task_id
        for task in tasks
    ] == [
        "TASK-014",
        "TASK-015",
        "TASK-016",
    ]


def test_generate_many_preserves_order(
    generator,
    validated_evaluation,
):
    tasks = generator.generate_many(
        validated_evaluation,
        task_ids=(
            "Z-TASK",
            "A-TASK",
            "M-TASK",
        ),
    )

    assert [
        task.task_id
        for task in tasks
    ] == [
        "Z-TASK",
        "A-TASK",
        "M-TASK",
    ]


def test_generate_many_rejects_empty_ids(
    generator,
    validated_evaluation,
):
    with pytest.raises(ValueError):
        generator.generate_many(
            validated_evaluation,
            task_ids=(),
        )


def test_generate_many_rejects_string_ids(
    generator,
    validated_evaluation,
):
    with pytest.raises(ValueError):
        generator.generate_many(
            validated_evaluation,
            task_ids="TASK-001",
        )


def test_generate_many_rejects_duplicate_ids(
    generator,
    validated_evaluation,
):
    with pytest.raises(ValueError):
        generator.generate_many(
            validated_evaluation,
            task_ids=(
                "TASK-017",
                "TASK-017",
            ),
        )


def test_generate_many_rejects_invalid_id(
    generator,
    validated_evaluation,
):
    with pytest.raises(ValueError):
        generator.generate_many(
            validated_evaluation,
            task_ids=(
                "TASK-018",
                "",
            ),
        )


def test_objective_matches_task_type(
    generator,
    validated_evaluation,
):
    task = generator.generate(
        validated_evaluation,
        task_id="TASK-019",
    )

    assert (
        "follow-up research"
        in task.objective.lower()
    )


def test_source_evidence_is_tuple(
    generator,
    validated_evaluation,
):
    task = generator.generate(
        validated_evaluation,
        task_id="TASK-020",
    )

    assert isinstance(
        task.source_evidence,
        tuple,
    )


def test_task_contains_experiment_identity(
    generator,
    validated_evaluation,
):
    task = generator.generate(
        validated_evaluation,
        task_id="TASK-021",
    )

    assert task.experiment_id == "EXP-TASK-001"


def test_generator_does_not_modify_evaluation(
    generator,
    validated_evaluation,
):
    original_status = validated_evaluation.status
    original_evidence = validated_evaluation.evidence

    generator.generate(
        validated_evaluation,
        task_id="TASK-022",
    )

    assert validated_evaluation.status == original_status
    assert (
        validated_evaluation.evidence
        == original_evidence
    )