import pytest

from autonomous_experiment_designer.experiment_designer import (
    ExperimentDesigner,
)
from autonomous_experiment_designer.experiment_executor import (
    ExecutionStatus,
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
        experiment_id="EXP-EXEC-001",
    )

    assert result is not None
    return result


def test_executor_with_empty_registries(plan):
    executor = ExperimentExecutor()

    prepared = executor.prepare(
        plan,
        execution_id="EXEC-001",
    )

    assert prepared.metadata.execution_id == "EXEC-001"
    assert prepared.metadata.experiment_id == plan.experiment_id
    assert (
        prepared.metadata.status
        == ExecutionStatus.PENDING_APPROVAL
    )
    assert prepared.success is False
    assert prepared.error is None


def test_prepare_requires_valid_plan():
    executor = ExperimentExecutor()

    with pytest.raises(TypeError):
        executor.prepare(
            "not-a-plan",
            execution_id="EXEC-INVALID",
        )


def test_prepare_generates_execution_id(plan):
    executor = ExperimentExecutor()

    result = executor.prepare(plan)

    assert result.metadata.execution_id.startswith("EXEC-")
    assert result.metadata.experiment_id == plan.experiment_id


def test_prepare_preserves_resources(plan):
    executor = ExperimentExecutor()

    result = executor.prepare(
        plan,
        execution_id="EXEC-RESOURCES",
    )

    assert result.metadata.datasets == plan.datasets
    assert result.metadata.agents == plan.agents
    assert result.metadata.models == plan.models


def test_prepare_requires_approval_state(plan):
    executor = ExperimentExecutor()

    result = executor.prepare(
        plan,
        execution_id="EXEC-APPROVAL",
    )

    assert (
        result.metadata.status
        == ExecutionStatus.PENDING_APPROVAL
    )


def test_rejected_execution_does_not_call_runner(plan):
    executor = ExperimentExecutor()

    called = False

    def runner(_plan):
        nonlocal called
        called = True
        return {"value": 123}

    result = executor.execute(
        plan,
        approved=False,
        runner=runner,
        execution_id="EXEC-REJECTED",
    )

    assert called is False
    assert result.success is False
    assert (
        result.metadata.status
        == ExecutionStatus.REJECTED
    )
    assert result.output == {}
    assert result.error == (
        "Human approval was not granted."
    )


def test_approved_execution_calls_runner(plan):
    executor = ExperimentExecutor()

    called = []

    def runner(received_plan):
        called.append(received_plan)
        return {
            "metric": 0.84,
            "status": "success",
        }

    result = executor.execute(
        plan,
        approved=True,
        runner=runner,
        execution_id="EXEC-APPROVED",
    )

    assert len(called) == 1
    assert called[0] is plan
    assert result.success is True
    assert (
        result.metadata.status
        == ExecutionStatus.COMPLETED
    )


def test_completed_execution_preserves_output(plan):
    executor = ExperimentExecutor()

    output = {
        "accuracy": 0.91,
        "sharpe": 1.42,
        "observations": 250,
    }

    result = executor.execute(
        plan,
        approved=True,
        runner=lambda _: output,
        execution_id="EXEC-OUTPUT",
    )

    assert result.success is True
    assert dict(result.output) == output
    assert result.error is None


def test_execution_records_timestamps(plan):
    executor = ExperimentExecutor()

    result = executor.execute(
        plan,
        approved=True,
        runner=lambda _: {"ok": True},
        execution_id="EXEC-TIME",
    )

    assert result.metadata.started_at is not None
    assert result.metadata.completed_at is not None
    assert (
        result.metadata.completed_at
        >= result.metadata.started_at
    )
    assert result.metadata.started_at.tzinfo is not None
    assert result.metadata.completed_at.tzinfo is not None


def test_runner_failure_is_captured(plan):
    executor = ExperimentExecutor()

    def failing_runner(_):
        raise RuntimeError("experiment failed")

    result = executor.execute(
        plan,
        approved=True,
        runner=failing_runner,
        execution_id="EXEC-FAILED",
    )

    assert result.success is False
    assert (
        result.metadata.status
        == ExecutionStatus.FAILED
    )
    assert result.output == {}
    assert result.error == "experiment failed"


def test_failed_runner_does_not_raise_to_caller(plan):
    executor = ExperimentExecutor()

    def failing_runner(_):
        raise ValueError("controlled failure")

    result = executor.execute(
        plan,
        approved=True,
        runner=failing_runner,
        execution_id="EXEC-CONTROLLED",
    )

    assert result.success is False
    assert result.error == "controlled failure"


def test_non_mapping_runner_output_fails_execution(plan):
    executor = ExperimentExecutor()

    result = executor.execute(
        plan,
        approved=True,
        runner=lambda _: ["invalid"],
        execution_id="EXEC-BAD-OUTPUT",
    )

    assert result.success is False
    assert (
        result.metadata.status
        == ExecutionStatus.FAILED
    )
    assert "mapping" in result.error


def test_dataset_registry_is_enforced(plan):
    executor = ExperimentExecutor(
        dataset_registry=("unrelated_dataset",),
    )

    with pytest.raises(ValueError, match="datasets"):
        executor.prepare(
            plan,
            execution_id="EXEC-DATASET",
        )


def test_agent_registry_is_enforced(plan):
    executor = ExperimentExecutor(
        agent_registry=("unrelated_agent",),
    )

    with pytest.raises(ValueError, match="agents"):
        executor.prepare(
            plan,
            execution_id="EXEC-AGENT",
        )


def test_model_registry_is_enforced(plan):
    executor = ExperimentExecutor(
        model_registry=("unrelated_model",),
    )

    with pytest.raises(ValueError, match="models"):
        executor.prepare(
            plan,
            execution_id="EXEC-MODEL",
        )


def test_matching_registries_are_accepted(plan):
    executor = ExperimentExecutor(
        dataset_registry=plan.datasets,
        agent_registry=plan.agents,
        model_registry=plan.models,
    )

    result = executor.prepare(
        plan,
        execution_id="EXEC-MATCH",
    )

    assert (
        result.metadata.status
        == ExecutionStatus.PENDING_APPROVAL
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
def test_invalid_execution_id_is_rejected(
    plan,
    value,
):
    executor = ExperimentExecutor()

    with pytest.raises(ValueError):
        executor.prepare(
            plan,
            execution_id=value,
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
def test_invalid_execution_id_is_rejected_by_execute(
    plan,
    value,
):
    executor = ExperimentExecutor()

    with pytest.raises(ValueError):
        executor.execute(
            plan,
            approved=True,
            runner=lambda _: {"ok": True},
            execution_id=value,
        )


def test_invalid_approval_type_is_rejected(plan):
    executor = ExperimentExecutor()

    with pytest.raises(TypeError):
        executor.execute(
            plan,
            approved="yes",
            runner=lambda _: {"ok": True},
            execution_id="EXEC-TYPE",
        )


def test_invalid_runner_is_rejected(plan):
    executor = ExperimentExecutor()

    with pytest.raises(TypeError):
        executor.execute(
            plan,
            approved=True,
            runner="not-callable",
            execution_id="EXEC-RUNNER",
        )


@pytest.mark.parametrize(
    "field",
    [
        "dataset_registry",
        "agent_registry",
        "model_registry",
    ],
)
def test_registry_string_is_rejected(field):
    kwargs = {
        "dataset_registry": (),
        "agent_registry": (),
        "model_registry": (),
    }

    kwargs[field] = "invalid"

    with pytest.raises(ValueError):
        ExperimentExecutor(**kwargs)


def test_duplicate_dataset_registry_is_rejected():
    with pytest.raises(ValueError):
        ExperimentExecutor(
            dataset_registry=(
                "dataset_a",
                "dataset_a",
            )
        )


def test_duplicate_agent_registry_is_rejected():
    with pytest.raises(ValueError):
        ExperimentExecutor(
            agent_registry=(
                "agent_a",
                "agent_a",
            )
        )


def test_duplicate_model_registry_is_rejected():
    with pytest.raises(ValueError):
        ExperimentExecutor(
            model_registry=(
                "model_a",
                "model_a",
            )
        )


def test_empty_resource_names_are_rejected():
    with pytest.raises(ValueError):
        ExperimentExecutor(
            dataset_registry=(
                "dataset_a",
                "",
            )
        )


def test_plan_is_not_modified_by_prepare(plan):
    original_datasets = plan.datasets
    original_agents = plan.agents
    original_models = plan.models

    executor = ExperimentExecutor()

    executor.prepare(
        plan,
        execution_id="EXEC-NO-MUTATION",
    )

    assert plan.datasets == original_datasets
    assert plan.agents == original_agents
    assert plan.models == original_models


def test_plan_is_not_modified_by_execute(plan):
    original_id = plan.experiment_id
    original_datasets = plan.datasets
    original_agents = plan.agents
    original_models = plan.models

    executor = ExperimentExecutor()

    executor.execute(
        plan,
        approved=True,
        runner=lambda _: {"ok": True},
        execution_id="EXEC-NO-MUTATION-2",
    )

    assert plan.experiment_id == original_id
    assert plan.datasets == original_datasets
    assert plan.agents == original_agents
    assert plan.models == original_models


def test_execution_result_is_immutable(plan):
    executor = ExperimentExecutor()

    result = executor.execute(
        plan,
        approved=True,
        runner=lambda _: {"ok": True},
        execution_id="EXEC-IMMUTABLE",
    )

    with pytest.raises(AttributeError):
        result.success = False


def test_execution_metadata_is_immutable(plan):
    executor = ExperimentExecutor()

    result = executor.execute(
        plan,
        approved=True,
        runner=lambda _: {"ok": True},
        execution_id="EXEC-META-IMMUTABLE",
    )

    with pytest.raises(AttributeError):
        result.metadata.status = ExecutionStatus.FAILED


def test_rejected_execution_has_completion_timestamp(plan):
    executor = ExperimentExecutor()

    result = executor.execute(
        plan,
        approved=False,
        runner=lambda _: {"should": "not-run"},
        execution_id="EXEC-REJECT-TIME",
    )

    assert result.metadata.started_at is not None
    assert result.metadata.completed_at is not None
    assert (
        result.metadata.completed_at
        >= result.metadata.started_at
    )


def test_failed_execution_has_completion_timestamp(plan):
    executor = ExperimentExecutor()

    def failing_runner(_):
        raise RuntimeError("failure")

    result = executor.execute(
        plan,
        approved=True,
        runner=failing_runner,
        execution_id="EXEC-FAIL-TIME",
    )

    assert result.metadata.started_at is not None
    assert result.metadata.completed_at is not None
    assert (
        result.metadata.completed_at
        >= result.metadata.started_at
    )


def test_successful_execution_has_no_error(plan):
    executor = ExperimentExecutor()

    result = executor.execute(
        plan,
        approved=True,
        runner=lambda _: {"success": True},
        execution_id="EXEC-NO-ERROR",
    )

    assert result.success is True
    assert result.error is None


def test_rejected_execution_has_empty_output(plan):
    executor = ExperimentExecutor()

    result = executor.execute(
        plan,
        approved=False,
        runner=lambda _: {"success": True},
        execution_id="EXEC-EMPTY",
    )

    assert result.output == {}


def test_execution_preserves_experiment_id(plan):
    executor = ExperimentExecutor()

    result = executor.execute(
        plan,
        approved=True,
        runner=lambda _: {"ok": True},
        execution_id="EXEC-ID",
    )

    assert (
        result.metadata.experiment_id
        == plan.experiment_id
    )