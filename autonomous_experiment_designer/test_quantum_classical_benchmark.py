import pytest

from autonomous_experiment_designer.quantum_classical_benchmark import (
    BenchmarkConclusion,
    BenchmarkStatus,
    QuantumClassicalBenchmark,
    QuantumClassicalBenchmarkResult,
)


def classical_runner(problem_definition, dataset):
    return {
        "method": "classical",
        "problem": problem_definition,
        "value": 1.0,
        "dataset_size": len(dataset),
    }


def quantum_runner(problem_definition, dataset):
    return {
        "method": "quantum",
        "problem": problem_definition,
        "value": 1.2,
        "dataset_size": len(dataset),
    }


def metric_function(output):
    return output["value"]


def test_default_configuration():
    benchmark = QuantumClassicalBenchmark()

    assert benchmark.metric_name == "score"
    assert benchmark.tolerance == 1e-9
    assert benchmark.require_human_approval is True


def test_custom_configuration():
    benchmark = QuantumClassicalBenchmark(
        metric_name="sharpe",
        tolerance=0.01,
        require_human_approval=False,
    )

    assert benchmark.metric_name == "sharpe"
    assert benchmark.tolerance == 0.01
    assert benchmark.require_human_approval is False


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        " ",
        123,
        True,
    ],
)
def test_invalid_metric_name_is_rejected(value):
    with pytest.raises(ValueError):
        QuantumClassicalBenchmark(
            metric_name=value
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "invalid",
        True,
        float("nan"),
        float("inf"),
        -0.1,
    ],
)
def test_invalid_tolerance_is_rejected(value):
    with pytest.raises(ValueError):
        QuantumClassicalBenchmark(
            tolerance=value
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        "true",
    ],
)
def test_invalid_human_approval_flag_is_rejected(value):
    with pytest.raises(ValueError):
        QuantumClassicalBenchmark(
            require_human_approval=value
        )


def test_completed_comparison():
    benchmark = QuantumClassicalBenchmark()

    result = benchmark.compare(
        "Portfolio optimization benchmark.",
        (
            "NVDA",
            "MSFT",
            "META",
        ),
        classical_runner,
        quantum_runner,
        metric_function,
    )

    assert isinstance(
        result,
        QuantumClassicalBenchmarkResult,
    )

    assert (
        result.status
        == BenchmarkStatus.COMPLETED
    )

    assert result.classical_metric == 1.0
    assert result.quantum_metric == 1.2
    assert result.performance_delta == pytest.approx(
        0.2
    )

    assert result.relative_improvement == pytest.approx(
        0.2
    )

    assert (
        result.conclusion
        == BenchmarkConclusion.QUANTUM_BETTER
    )


def test_classical_better_is_detected():
    def classical(problem, dataset):
        return 2.0

    def quantum(problem, dataset):
        return 1.0

    result = QuantumClassicalBenchmark().compare(
        "Optimization benchmark.",
        (1, 2, 3),
        classical,
        quantum,
        lambda output: output,
    )

    assert (
        result.conclusion
        == BenchmarkConclusion.CLASSICAL_BETTER
    )

    assert result.performance_delta == -1.0
    assert result.relative_improvement == -0.5


def test_equivalent_result_is_detected():
    def classical(problem, dataset):
        return 1.0

    def quantum(problem, dataset):
        return 1.0 + 1e-10

    result = QuantumClassicalBenchmark(
        tolerance=1e-9
    ).compare(
        "Equivalent benchmark.",
        (1, 2),
        classical,
        quantum,
        lambda output: output,
    )

    assert (
        result.conclusion
        == BenchmarkConclusion.EQUIVALENT
    )


def test_tolerance_controls_equivalence():
    def classical(problem, dataset):
        return 1.0

    def quantum(problem, dataset):
        return 1.005

    result = QuantumClassicalBenchmark(
        tolerance=0.01
    ).compare(
        "Tolerance benchmark.",
        (1, 2),
        classical,
        quantum,
        lambda output: output,
    )

    assert (
        result.conclusion
        == BenchmarkConclusion.EQUIVALENT
    )


def test_same_problem_definition_is_used():
    received = []

    def classical(problem, dataset):
        received.append(
            ("classical", problem, dataset)
        )
        return 1.0

    def quantum(problem, dataset):
        received.append(
            ("quantum", problem, dataset)
        )
        return 1.1

    problem = "Same optimization problem."
    dataset = ("A", "B", "C")

    result = QuantumClassicalBenchmark().compare(
        problem,
        dataset,
        classical,
        quantum,
        lambda output: output,
    )

    assert (
        result.status
        == BenchmarkStatus.COMPLETED
    )

    assert received[0][1] == problem
    assert received[1][1] == problem


def test_same_dataset_is_used():
    received = []

    def classical(problem, dataset):
        received.append(dataset)
        return 1.0

    def quantum(problem, dataset):
        received.append(dataset)
        return 1.1

    dataset = ("A", "B", "C")

    QuantumClassicalBenchmark().compare(
        "Dataset comparison.",
        dataset,
        classical,
        quantum,
        lambda output: output,
    )

    assert received[0] == received[1]
    assert received[0] == dataset


def test_dataset_is_normalized_to_tuple():
    received_types = []

    def classical(problem, dataset):
        received_types.append(type(dataset))
        return 1.0

    def quantum(problem, dataset):
        received_types.append(type(dataset))
        return 1.1

    QuantumClassicalBenchmark().compare(
        "Tuple normalization.",
        ["A", "B", "C"],
        classical,
        quantum,
        lambda output: output,
    )

    assert received_types == [
        tuple,
        tuple,
    ]


def test_problem_definition_is_normalized():
    received = []

    def classical(problem, dataset):
        received.append(problem)
        return 1.0

    def quantum(problem, dataset):
        received.append(problem)
        return 1.1

    QuantumClassicalBenchmark().compare(
        "  benchmark problem  ",
        (1, 2),
        classical,
        quantum,
        lambda output: output,
    )

    assert received == [
        "benchmark problem",
        "benchmark problem",
    ]


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "",
        " ",
        123,
    ],
)
def test_invalid_problem_definition_is_rejected(
    problem,
):
    with pytest.raises(ValueError):
        QuantumClassicalBenchmark().compare(
            problem,
            (1, 2),
            classical_runner,
            quantum_runner,
            metric_function,
        )


@pytest.mark.parametrize(
    "dataset",
    [
        None,
        "invalid",
        b"invalid",
        123,
    ],
)
def test_invalid_dataset_is_rejected(dataset):
    with pytest.raises(ValueError):
        QuantumClassicalBenchmark().compare(
            "Valid problem.",
            dataset,
            classical_runner,
            quantum_runner,
            metric_function,
        )


@pytest.mark.parametrize(
    "runner",
    [
        None,
        "invalid",
        123,
    ],
)
def test_invalid_classical_runner_is_rejected(
    runner,
):
    with pytest.raises(ValueError):
        QuantumClassicalBenchmark().compare(
            "Valid problem.",
            (1, 2),
            runner,
            quantum_runner,
            metric_function,
        )


@pytest.mark.parametrize(
    "runner",
    [
        None,
        "invalid",
        123,
    ],
)
def test_invalid_quantum_runner_is_rejected(
    runner,
):
    with pytest.raises(ValueError):
        QuantumClassicalBenchmark().compare(
            "Valid problem.",
            (1, 2),
            classical_runner,
            runner,
            metric_function,
        )


@pytest.mark.parametrize(
    "metric",
    [
        None,
        "invalid",
        123,
    ],
)
def test_invalid_metric_function_is_rejected(
    metric,
):
    with pytest.raises(ValueError):
        QuantumClassicalBenchmark().compare(
            "Valid problem.",
            (1, 2),
            classical_runner,
            quantum_runner,
            metric,
        )


def test_classical_runner_failure_is_captured():
    def failing_classical(problem, dataset):
        raise RuntimeError(
            "controlled classical failure"
        )

    result = QuantumClassicalBenchmark().compare(
        "Failure benchmark.",
        (1, 2),
        failing_classical,
        quantum_runner,
        metric_function,
    )

    assert (
        result.status
        == BenchmarkStatus.FAILED
    )

    assert result.error is not None
    assert (
        "controlled classical failure"
        in result.error
    )

    assert result.classical_metric is None
    assert result.quantum_metric is None


def test_quantum_runner_failure_is_captured():
    def failing_quantum(problem, dataset):
        raise RuntimeError(
            "controlled quantum failure"
        )

    result = QuantumClassicalBenchmark().compare(
        "Failure benchmark.",
        (1, 2),
        classical_runner,
        failing_quantum,
        metric_function,
    )

    assert (
        result.status
        == BenchmarkStatus.FAILED
    )

    assert result.error is not None
    assert (
        "controlled quantum failure"
        in result.error
    )


def test_classical_metric_failure_is_captured():
    result = QuantumClassicalBenchmark().compare(
        "Metric failure.",
        (1, 2),
        classical_runner,
        quantum_runner,
        lambda output: 1 / 0,
    )

    assert (
        result.status
        == BenchmarkStatus.FAILED
    )

    assert result.error is not None


def test_quantum_metric_failure_is_captured():
    def metric(output):
        if output["method"] == "quantum":
            raise RuntimeError(
                "quantum metric failure"
            )

        return output["value"]

    result = QuantumClassicalBenchmark().compare(
        "Metric failure.",
        (1, 2),
        classical_runner,
        quantum_runner,
        metric,
    )

    assert (
        result.status
        == BenchmarkStatus.FAILED
    )

    assert (
        "quantum metric failure"
        in result.error
    )


@pytest.mark.parametrize(
    "classical_value",
    [
        None,
        "invalid",
        True,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_invalid_classical_metric_is_captured(
    classical_value,
):
    result = QuantumClassicalBenchmark().compare(
        "Invalid metric.",
        (1, 2),
        lambda problem, dataset: classical_value,
        lambda problem, dataset: 1.0,
        lambda output: output,
    )

    assert (
        result.status
        == BenchmarkStatus.FAILED
    )

    assert result.error is not None


@pytest.mark.parametrize(
    "quantum_value",
    [
        None,
        "invalid",
        True,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_invalid_quantum_metric_is_captured(
    quantum_value,
):
    result = QuantumClassicalBenchmark().compare(
        "Invalid metric.",
        (1, 2),
        lambda problem, dataset: 1.0,
        lambda problem, dataset: quantum_value,
        lambda output: output,
    )

    assert (
        result.status
        == BenchmarkStatus.FAILED
    )

    assert result.error is not None


def test_relative_improvement_handles_zero_baseline():
    result = QuantumClassicalBenchmark().compare(
        "Zero baseline.",
        (1, 2),
        lambda problem, dataset: 0.0,
        lambda problem, dataset: 1.0,
        lambda output: output,
    )

    assert result.classical_metric == 0.0
    assert result.quantum_metric == 1.0
    assert result.performance_delta == 1.0
    assert result.relative_improvement is None

    assert (
        result.conclusion
        == BenchmarkConclusion.QUANTUM_BETTER
    )


def test_negative_metrics_are_supported():
    result = QuantumClassicalBenchmark().compare(
        "Negative metric.",
        (1, 2),
        lambda problem, dataset: -2.0,
        lambda problem, dataset: -1.0,
        lambda output: output,
    )

    assert result.classical_metric == -2.0
    assert result.quantum_metric == -1.0
    assert result.performance_delta == 1.0


def test_outputs_are_preserved():
    classical_output = {
        "method": "classical",
        "result": [1, 2, 3],
    }

    quantum_output = {
        "method": "quantum",
        "result": [4, 5, 6],
    }

    result = QuantumClassicalBenchmark().compare(
        "Output preservation.",
        (1, 2),
        lambda problem, dataset: classical_output,
        lambda problem, dataset: quantum_output,
        lambda output: 1.0,
    )

    assert (
        result.classical_output
        is classical_output
    )

    assert (
        result.quantum_output
        is quantum_output
    )


def test_metadata_contains_metric_name():
    result = QuantumClassicalBenchmark(
        metric_name="accuracy"
    ).compare(
        "Metadata benchmark.",
        (1, 2),
        lambda problem, dataset: 1.0,
        lambda problem, dataset: 1.1,
        lambda output: output,
    )

    assert (
        result.metadata["metric_name"]
        == "accuracy"
    )


def test_metadata_contains_tolerance():
    result = QuantumClassicalBenchmark(
        tolerance=0.005
    ).compare(
        "Metadata benchmark.",
        (1, 2),
        lambda problem, dataset: 1.0,
        lambda problem, dataset: 1.1,
        lambda output: output,
    )

    assert (
        result.metadata["tolerance"]
        == 0.005
    )


def test_metadata_contains_dataset_size():
    result = QuantumClassicalBenchmark().compare(
        "Metadata benchmark.",
        (1, 2, 3, 4),
        lambda problem, dataset: 1.0,
        lambda problem, dataset: 1.1,
        lambda output: output,
    )

    assert (
        result.metadata["dataset_size"]
        == 4
    )


def test_metadata_records_same_problem():
    result = QuantumClassicalBenchmark().compare(
        "Same problem.",
        (1, 2),
        classical_runner,
        quantum_runner,
        metric_function,
    )

    assert (
        result.metadata[
            "same_problem_definition"
        ]
        is True
    )


def test_metadata_records_same_dataset():
    result = QuantumClassicalBenchmark().compare(
        "Same dataset.",
        (1, 2),
        classical_runner,
        quantum_runner,
        metric_function,
    )

    assert (
        result.metadata["same_dataset"]
        is True
    )


def test_human_approval_requirement_is_recorded():
    result = QuantumClassicalBenchmark(
        require_human_approval=True
    ).compare(
        "Approval benchmark.",
        (1, 2),
        classical_runner,
        quantum_runner,
        metric_function,
    )

    assert (
        result.metadata[
            "human_approval_required"
        ]
        is True
    )


def test_human_approval_can_be_disabled():
    result = QuantumClassicalBenchmark(
        require_human_approval=False
    ).compare(
        "Approval benchmark.",
        (1, 2),
        classical_runner,
        quantum_runner,
        metric_function,
    )

    assert (
        result.metadata[
            "human_approval_required"
        ]
        is False
    )


def test_result_is_immutable():
    result = QuantumClassicalBenchmark().compare(
        "Immutable result.",
        (1, 2),
        classical_runner,
        quantum_runner,
        metric_function,
    )

    with pytest.raises(AttributeError):
        result.status = BenchmarkStatus.FAILED


def test_result_contains_expected_fields():
    result = QuantumClassicalBenchmark().compare(
        "Fields benchmark.",
        (1, 2),
        classical_runner,
        quantum_runner,
        metric_function,
    )

    assert result.status is not None
    assert result.classical_metric is not None
    assert result.quantum_metric is not None
    assert result.performance_delta is not None
    assert result.relative_improvement is not None
    assert result.conclusion is not None
    assert result.classical_output is not None
    assert result.quantum_output is not None
    assert result.metadata is not None
    assert result.error is None


def test_benchmark_is_repeatable():
    benchmark = QuantumClassicalBenchmark()

    first = benchmark.compare(
        "Repeatable benchmark.",
        (1, 2, 3),
        classical_runner,
        quantum_runner,
        metric_function,
    )

    second = benchmark.compare(
        "Repeatable benchmark.",
        (1, 2, 3),
        classical_runner,
        quantum_runner,
        metric_function,
    )

    assert first == second


def test_quantum_advantage_is_based_on_measured_metric():
    result = QuantumClassicalBenchmark().compare(
        "Measured comparison.",
        (1, 2),
        lambda problem, dataset: 10.0,
        lambda problem, dataset: 12.0,
        lambda output: output,
    )

    assert result.performance_delta == 2.0
    assert result.relative_improvement == 0.2

    assert (
        result.conclusion
        == BenchmarkConclusion.QUANTUM_BETTER
    )


def test_classical_advantage_is_based_on_measured_metric():
    result = QuantumClassicalBenchmark().compare(
        "Measured comparison.",
        (1, 2),
        lambda problem, dataset: 12.0,
        lambda problem, dataset: 10.0,
        lambda output: output,
    )

    assert result.performance_delta == -2.0
    assert result.relative_improvement == pytest.approx(
        -2.0 / 12.0
    )

    assert (
        result.conclusion
        == BenchmarkConclusion.CLASSICAL_BETTER
    )


def test_equivalent_metrics_are_reported():
    result = QuantumClassicalBenchmark().compare(
        "Equivalent comparison.",
        (1, 2),
        lambda problem, dataset: 5.0,
        lambda problem, dataset: 5.0,
        lambda output: output,
    )

    assert result.performance_delta == 0.0
    assert result.relative_improvement == 0.0

    assert (
        result.conclusion
        == BenchmarkConclusion.EQUIVALENT
    )


def test_dataset_is_not_mutated():
    dataset = [
        "A",
        "B",
        "C",
    ]

    original = list(dataset)

    QuantumClassicalBenchmark().compare(
        "No mutation.",
        dataset,
        classical_runner,
        quantum_runner,
        metric_function,
    )

    assert dataset == original


def test_problem_definition_is_not_mutated():
    problem = "  Original problem  "

    received = []

    def classical(problem_definition, dataset):
        received.append(problem_definition)
        return 1.0

    def quantum(problem_definition, dataset):
        received.append(problem_definition)
        return 1.0

    QuantumClassicalBenchmark().compare(
        problem,
        (1, 2),
        classical,
        quantum,
        lambda output: output,
    )

    assert problem == "  Original problem  "
    assert received == [
        "Original problem",
        "Original problem",
    ]


def test_empty_dataset_is_allowed_as_sequence():
    result = QuantumClassicalBenchmark().compare(
        "Empty dataset benchmark.",
        (),
        lambda problem, dataset: 1.0,
        lambda problem, dataset: 1.1,
        lambda output: output,
    )

    assert (
        result.status
        == BenchmarkStatus.COMPLETED
    )

    assert (
        result.metadata["dataset_size"]
        == 0
    )


def test_result_error_is_none_on_success():
    result = QuantumClassicalBenchmark().compare(
        "Successful benchmark.",
        (1, 2),
        classical_runner,
        quantum_runner,
        metric_function,
    )

    assert result.error is None


def test_failed_result_contains_metadata():
    result = QuantumClassicalBenchmark().compare(
        "Failed benchmark.",
        (1, 2),
        lambda problem, dataset: 1.0,
        lambda problem, dataset: 1.1,
        lambda output: "invalid",
    )

    assert (
        result.status
        == BenchmarkStatus.FAILED
    )

    assert (
        result.metadata["metric_name"]
        == "score"
    )

    assert (
        result.metadata["dataset_size"]
        == 2
    )


def test_custom_metric_name_is_preserved():
    benchmark = QuantumClassicalBenchmark(
        metric_name="return"
    )

    result = benchmark.compare(
        "Return benchmark.",
        (1, 2),
        lambda problem, dataset: 1.0,
        lambda problem, dataset: 1.2,
        lambda output: output,
    )

    assert (
        result.metadata["metric_name"]
        == "return"
    )


def test_relative_improvement_uses_absolute_baseline():
    result = QuantumClassicalBenchmark().compare(
        "Negative baseline.",
        (1, 2),
        lambda problem, dataset: -4.0,
        lambda problem, dataset: -2.0,
        lambda output: output,
    )

    assert result.performance_delta == 2.0
    assert result.relative_improvement == 0.5


def test_zero_delta_has_zero_relative_improvement():
    result = QuantumClassicalBenchmark().compare(
        "Zero delta.",
        (1, 2),
        lambda problem, dataset: 7.0,
        lambda problem, dataset: 7.0,
        lambda output: output,
    )

    assert result.performance_delta == 0.0
    assert result.relative_improvement == 0.0


def test_metric_values_are_converted_to_float():
    result = QuantumClassicalBenchmark().compare(
        "Numeric conversion.",
        (1, 2),
        lambda problem, dataset: 1,
        lambda problem, dataset: 2,
        lambda output: output,
    )

    assert isinstance(
        result.classical_metric,
        float,
    )

    assert isinstance(
        result.quantum_metric,
        float,
    )


def test_benchmark_result_is_reproducible_with_same_inputs():
    def classical(problem, dataset):
        return sum(dataset)

    def quantum(problem, dataset):
        return sum(dataset) + 2

    metric = lambda output: output

    benchmark = QuantumClassicalBenchmark()

    first = benchmark.compare(
        "Reproducibility benchmark.",
        (1, 2, 3),
        classical,
        quantum,
        metric,
    )

    second = benchmark.compare(
        "Reproducibility benchmark.",
        (1, 2, 3),
        classical,
        quantum,
        metric,
    )

    assert first == second


def test_classical_and_quantum_outputs_can_be_arbitrary_objects():
    class Output:
        def __init__(self, value):
            self.value = value

    classical = Output(1.0)
    quantum = Output(2.0)

    result = QuantumClassicalBenchmark().compare(
        "Object output benchmark.",
        (1, 2),
        lambda problem, dataset: classical,
        lambda problem, dataset: quantum,
        lambda output: output.value,
    )

    assert result.classical_output is classical
    assert result.quantum_output is quantum
    assert result.classical_metric == 1.0
    assert result.quantum_metric == 2.0


def test_large_metric_values_are_supported():
    result = QuantumClassicalBenchmark().compare(
        "Large metric benchmark.",
        (1, 2),
        lambda problem, dataset: 1_000_000.0,
        lambda problem, dataset: 1_500_000.0,
        lambda output: output,
    )

    assert result.performance_delta == 500_000.0
    assert result.relative_improvement == 0.5


def test_small_metric_difference_is_detected():
    result = QuantumClassicalBenchmark(
        tolerance=1e-12
    ).compare(
        "Precision benchmark.",
        (1, 2),
        lambda problem, dataset: 1.0,
        lambda problem, dataset: 1.0000001,
        lambda output: output,
    )

    assert (
        result.conclusion
        == BenchmarkConclusion.QUANTUM_BETTER
    )


def test_zero_tolerance_requires_exact_equivalence():
    result = QuantumClassicalBenchmark(
        tolerance=0.0
    ).compare(
        "Exact comparison.",
        (1, 2),
        lambda problem, dataset: 1.0,
        lambda problem, dataset: 1.0000001,
        lambda output: output,
    )

    assert (
        result.conclusion
        == BenchmarkConclusion.QUANTUM_BETTER
    )