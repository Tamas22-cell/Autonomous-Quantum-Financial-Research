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
    ExperimentExecutor,
)
from autonomous_experiment_designer.research_task_generator import (
    ResearchTaskGenerator,
    ResearchTaskPriority,
    ResearchTaskType,
)
from autonomous_experiment_designer.uncertainty_detector import (
    Severity,
    UncertaintyDetector,
)


def test_v6_0_6_end_to_end_autonomous_research_cycle():
    # ---------------------------------------------------------
    # 1. UNCERTAINTY DETECTION
    # ---------------------------------------------------------
    detector = UncertaintyDetector()

    uncertainty = detector.detect(
        confidence=(0.60, 0.90),
        regime_distance=0.80,
        evidence_agreement=0.50,
    )

    assert uncertainty.has_uncertainty is True
    assert len(uncertainty.signals) > 0
    assert uncertainty.highest_severity in (
        Severity.MEDIUM,
        Severity.HIGH,
        Severity.CRITICAL,
    )

    # ---------------------------------------------------------
    # 2. EXPERIMENT DESIGN
    # ---------------------------------------------------------
    designer = ExperimentDesigner()

    plan = designer.design(
        uncertainty,
        experiment_id="EXP-V6-0-6-E2E",
    )

    assert plan is not None
    assert (
        plan.experiment_id
        == "EXP-V6-0-6-E2E"
    )
    assert plan.requires_human_approval is True
    assert len(plan.datasets) > 0
    assert len(plan.agents) > 0
    assert len(plan.models) > 0

    # ---------------------------------------------------------
    # 3. PREPARE EXECUTION
    # ---------------------------------------------------------
    executor = ExperimentExecutor()

    prepared = executor.prepare(
        plan,
        execution_id="EXEC-V6-0-6-E2E",
    )

    assert (
        prepared.metadata.status
        == ExecutionStatus.PENDING_APPROVAL
    )
    assert prepared.success is False

    # ---------------------------------------------------------
    # 4. HUMAN APPROVAL GATE
    # ---------------------------------------------------------
    approved_execution = executor.execute(
        plan,
        approved=True,
        runner=lambda _: {
            "sharpe": 1.50,
            "return": 0.20,
            "drawdown": 0.10,
        },
        execution_id="EXEC-V6-0-6-E2E",
    )

    assert approved_execution.success is True
    assert (
        approved_execution.metadata.status
        == ExecutionStatus.COMPLETED
    )

    # ---------------------------------------------------------
    # 5. EXPERIMENT EVALUATION
    # ---------------------------------------------------------
    evaluator = ExperimentEvaluator()

    evaluation = evaluator.evaluate(
        approved_execution,
        baseline={
            "sharpe": 1.00,
            "return": 0.10,
            "drawdown": 0.20,
        },
        confidence=0.90,
        validation_score=0.90,
        evidence=(
            "historical_data",
            "walk_forward_validation",
            "regime_test",
        ),
    )

    assert (
        evaluation.status
        == EvaluationStatus.VALIDATED
    )
    assert (
        evaluation.recommendation
        == Recommendation.ACCEPT
    )
    assert evaluation.confidence == 0.90
    assert evaluation.validation_score == 0.90
    assert len(evaluation.metrics) == 3

    # ---------------------------------------------------------
    # 6. NEXT RESEARCH TASK GENERATION
    # ---------------------------------------------------------
    task_generator = ResearchTaskGenerator(
        default_evidence=(
            "research_history",
        )
    )

    task = task_generator.generate(
        evaluation,
        task_id="TASK-V6-0-6-E2E",
    )

    assert (
        task.task_type
        == ResearchTaskType.FOLLOW_UP_RESEARCH
    )
    assert (
        task.priority
        == ResearchTaskPriority.LOW
    )
    assert task.required_approval is True
    assert (
        task.experiment_id
        == plan.experiment_id
    )
    assert task.source_evidence == (
        "research_history",
        "historical_data",
        "walk_forward_validation",
        "regime_test",
    )
    assert task.objective.strip()
    assert task.rationale.strip()


def test_v6_0_6_human_gate_blocks_execution():
    detector = UncertaintyDetector()

    uncertainty = detector.detect(
        regime_distance=0.80,
    )

    plan = ExperimentDesigner().design(
        uncertainty,
        experiment_id="EXP-HUMAN-GATE",
    )

    assert plan is not None
    assert plan.requires_human_approval is True

    executor = ExperimentExecutor()

    called = False

    def forbidden_runner(_):
        nonlocal called
        called = True

        return {
            "sharpe": 999.0,
        }

    result = executor.execute(
        plan,
        approved=False,
        runner=forbidden_runner,
        execution_id="EXEC-HUMAN-GATE",
    )

    assert called is False
    assert result.success is False
    assert (
        result.metadata.status
        == ExecutionStatus.REJECTED
    )
    assert result.output == {}
    assert (
        result.error
        == "Human approval was not granted."
    )


def test_v6_0_6_failed_execution_creates_repeat_task():
    detector = UncertaintyDetector()

    uncertainty = detector.detect(
        regime_distance=0.80,
    )

    plan = ExperimentDesigner().design(
        uncertainty,
        experiment_id="EXP-FAILED-E2E",
    )

    assert plan is not None

    executor = ExperimentExecutor()

    def failing_runner(_):
        raise RuntimeError(
            "controlled experiment failure"
        )

    execution = executor.execute(
        plan,
        approved=True,
        runner=failing_runner,
        execution_id="EXEC-FAILED-E2E",
    )

    assert execution.success is False
    assert (
        execution.metadata.status
        == ExecutionStatus.FAILED
    )

    evaluation = ExperimentEvaluator().evaluate(
        execution,
        baseline={
            "sharpe": 1.00,
        },
        confidence=0.90,
        validation_score=0.90,
    )

    assert (
        evaluation.status
        == EvaluationStatus.FAILED
    )
    assert (
        evaluation.recommendation
        == Recommendation.REPEAT
    )

    task = ResearchTaskGenerator().generate(
        evaluation,
        task_id="TASK-FAILED-E2E",
    )

    assert (
        task.task_type
        == ResearchTaskType.REPEAT_EXPERIMENT
    )
    assert (
        task.priority
        == ResearchTaskPriority.CRITICAL
    )
    assert task.required_approval is True