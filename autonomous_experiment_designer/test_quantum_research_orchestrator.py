import pytest

from autonomous_experiment_designer.quantum_research_orchestrator import (
    QuantumResearchMethod,
    QuantumResearchOrchestrator,
    QuantumResearchResult,
    QuantumResearchStatus,
)


def quantum_runner(problem, dataset):
    return {
        "method": "quantum",
        "problem": problem,
        "dataset": tuple(dataset),
        "score": 1.2,
    }


def classical_runner(problem, dataset):
    return {
        "method": "classical",
        "problem": problem,
        "dataset": tuple(dataset),
        "score": 1.0,
    }


def benchmark_runner(
    problem,
    dataset,
    classical_result,
    quantum_result,
):
    return {
        "problem": problem,
        "dataset_size": len(dataset),
        "classical_score": classical_result["score"],
        "quantum_score": quantum_result["score"],
        "delta": (
            quantum_result["score"]
            - classical_result["score"]
        ),
    }


def test_default_configuration():
    orchestrator = QuantumResearchOrchestrator()

    assert orchestrator.require_human_approval is True


def test_human_approval_can_be_disabled():
    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    assert orchestrator.require_human_approval is False


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        " ",
        123,
    ],
)
def test_invalid_experiment_id_is_rejected(value):
    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    with pytest.raises(ValueError):
        orchestrator.run(
            experiment_id=value,
            method=QuantumResearchMethod.QAOA,
            problem_definition="Optimization.",
            dataset=(1, 2),
            quantum_runner=quantum_runner,
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        " ",
        123,
    ],
)
def test_invalid_problem_definition_is_rejected(value):
    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    with pytest.raises(ValueError):
        orchestrator.run(
            experiment_id="EXP-001",
            method=QuantumResearchMethod.QAOA,
            problem_definition=value,
            dataset=(1, 2),
            quantum_runner=quantum_runner,
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "QAOA",
        "invalid",
        123,
    ],
)
def test_invalid_method_is_rejected(value):
    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    with pytest.raises(ValueError):
        orchestrator.run(
            experiment_id="EXP-001",
            method=value,
            problem_definition="Optimization.",
            dataset=(1, 2),
            quantum_runner=quantum_runner,
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
    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    with pytest.raises(ValueError):
        orchestrator.run(
            experiment_id="EXP-001",
            method=QuantumResearchMethod.QAOA,
            problem_definition="Optimization.",
            dataset=dataset,
            quantum_runner=quantum_runner,
        )


def test_dataset_is_normalized_to_tuple():
    received = []

    def quantum(problem, dataset):
        received.append(type(dataset))
        return "quantum-result"

    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    result = orchestrator.run(
        experiment_id="EXP-002",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Optimization.",
        dataset=[1, 2, 3],
        quantum_runner=quantum,
    )

    assert (
        result.status
        == QuantumResearchStatus.COMPLETED
    )

    assert received == [tuple]


def test_problem_definition_is_trimmed():
    received = []

    def quantum(problem, dataset):
        received.append(problem)
        return "quantum-result"

    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    orchestrator.run(
        experiment_id="EXP-003",
        method=QuantumResearchMethod.VQE,
        problem_definition="  VQE problem  ",
        dataset=(1, 2),
        quantum_runner=quantum,
    )

    assert received == ["VQE problem"]


def test_human_gate_blocks_execution():
    calls = []

    def quantum(problem, dataset):
        calls.append("quantum")
        return "result"

    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=True
    )

    result = orchestrator.run(
        experiment_id="EXP-GATE-001",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Approval test.",
        dataset=(1, 2),
        quantum_runner=quantum,
        human_approval=False,
    )

    assert (
        result.status
        == QuantumResearchStatus.BLOCKED
    )

    assert calls == []
    assert result.quantum_result is None
    assert result.classical_result is None
    assert result.benchmark_result is None

    assert (
        result.metadata["execution_started"]
        is False
    )


def test_human_gate_allows_approved_execution():
    calls = []

    def quantum(problem, dataset):
        calls.append("quantum")
        return "quantum-result"

    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=True
    )

    result = orchestrator.run(
        experiment_id="EXP-GATE-002",
        method=QuantumResearchMethod.VQE,
        problem_definition="Approved test.",
        dataset=(1, 2),
        quantum_runner=quantum,
        human_approval=True,
    )

    assert (
        result.status
        == QuantumResearchStatus.COMPLETED
    )

    assert calls == ["quantum"]
    assert result.quantum_result == "quantum-result"


def test_approval_type_is_validated():
    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    with pytest.raises(ValueError):
        orchestrator.run(
            experiment_id="EXP-004",
            method=QuantumResearchMethod.QAOA,
            problem_definition="Approval.",
            dataset=(1, 2),
            quantum_runner=quantum_runner,
            human_approval="yes",
        )


@pytest.mark.parametrize(
    "runner",
    [
        None,
        "invalid",
        123,
    ],
)
def test_invalid_quantum_runner_is_rejected(runner):
    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    with pytest.raises(ValueError):
        orchestrator.run(
            experiment_id="EXP-005",
            method=QuantumResearchMethod.QAOA,
            problem_definition="Runner.",
            dataset=(1, 2),
            quantum_runner=runner,
        )


@pytest.mark.parametrize(
    "runner",
    [
        "invalid",
        123,
    ],
)
def test_invalid_classical_runner_is_rejected(runner):
    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    with pytest.raises(ValueError):
        orchestrator.run(
            experiment_id="EXP-006",
            method=QuantumResearchMethod.QAOA,
            problem_definition="Runner.",
            dataset=(1, 2),
            quantum_runner=quantum_runner,
            classical_runner=runner,
        )


@pytest.mark.parametrize(
    "runner",
    [
        "invalid",
        123,
    ],
)
def test_invalid_benchmark_runner_is_rejected(runner):
    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    with pytest.raises(ValueError):
        orchestrator.run(
            experiment_id="EXP-007",
            method=QuantumResearchMethod.QAOA,
            problem_definition="Runner.",
            dataset=(1, 2),
            quantum_runner=quantum_runner,
            benchmark_runner=runner,
        )


def test_qaoa_method_is_preserved():
    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    result = orchestrator.run(
        experiment_id="QAOA-001",
        method=QuantumResearchMethod.QAOA,
        problem_definition="QAOA benchmark.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
    )

    assert result.method == QuantumResearchMethod.QAOA


def test_vqe_method_is_preserved():
    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    result = orchestrator.run(
        experiment_id="VQE-001",
        method=QuantumResearchMethod.VQE,
        problem_definition="VQE benchmark.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
    )

    assert result.method == QuantumResearchMethod.VQE


def test_quantum_kernel_method_is_preserved():
    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    result = orchestrator.run(
        experiment_id="KERNEL-001",
        method=QuantumResearchMethod.QUANTUM_KERNEL,
        problem_definition="Kernel benchmark.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
    )

    assert (
        result.method
        == QuantumResearchMethod.QUANTUM_KERNEL
    )


def test_qmc_method_is_preserved():
    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    result = orchestrator.run(
        experiment_id="QMC-001",
        method=QuantumResearchMethod.QUANTUM_MONTE_CARLO,
        problem_definition="QMC benchmark.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
    )

    assert (
        result.method
        == QuantumResearchMethod.QUANTUM_MONTE_CARLO
    )


def test_quantum_runner_is_executed():
    calls = []

    def quantum(problem, dataset):
        calls.append(
            (problem, dataset)
        )
        return {"value": 42}

    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    result = orchestrator.run(
        experiment_id="EXP-008",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Execution test.",
        dataset=(1, 2, 3),
        quantum_runner=quantum,
    )

    assert calls == [
        (
            "Execution test.",
            (1, 2, 3),
        )
    ]

    assert result.quantum_result == {
        "value": 42
    }


def test_classical_runner_is_optional():
    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    result = orchestrator.run(
        experiment_id="EXP-009",
        method=QuantumResearchMethod.QAOA,
        problem_definition="No baseline.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
    )

    assert (
        result.status
        == QuantumResearchStatus.COMPLETED
    )

    assert result.classical_result is None
    assert result.benchmark_result is None


def test_classical_baseline_is_executed():
    calls = []

    def classical(problem, dataset):
        calls.append(
            (problem, dataset)
        )
        return "classical-result"

    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    result = orchestrator.run(
        experiment_id="EXP-010",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Baseline.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
        classical_runner=classical,
    )

    assert calls == [
        (
            "Baseline.",
            (1, 2),
        )
    ]

    assert (
        result.classical_result
        == "classical-result"
    )


def test_benchmark_is_executed():
    calls = []

    def benchmark(
        problem,
        dataset,
        classical,
        quantum,
    ):
        calls.append(
            (
                problem,
                dataset,
                classical,
                quantum,
            )
        )

        return {
            "delta": 0.2
        }

    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    result = orchestrator.run(
        experiment_id="EXP-011",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Benchmark.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
        classical_runner=classical_runner,
        benchmark_runner=benchmark,
    )

    assert len(calls) == 1

    assert calls[0][0] == "Benchmark."
    assert calls[0][1] == (1, 2)

    assert (
        result.benchmark_result
        == {"delta": 0.2}
    )


def test_benchmark_requires_classical_runner():
    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    result = orchestrator.run(
        experiment_id="EXP-012",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Invalid benchmark.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
        benchmark_runner=benchmark_runner,
    )

    assert (
        result.status
        == QuantumResearchStatus.FAILED
    )

    assert result.error is not None
    assert (
        "classical_runner"
        in result.error
    )


def test_runner_none_output_is_captured():
    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-013",
        method=QuantumResearchMethod.QAOA,
        problem_definition="None output.",
        dataset=(1, 2),
        quantum_runner=lambda problem, dataset: None,
    )

    assert (
        result.status
        == QuantumResearchStatus.FAILED
    )

    assert result.error is not None
    assert (
        "quantum_runner returned None"
        in result.error
    )


def test_classical_none_output_is_captured():
    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-014",
        method=QuantumResearchMethod.QAOA,
        problem_definition="None output.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
        classical_runner=lambda problem, dataset: None,
    )

    assert (
        result.status
        == QuantumResearchStatus.FAILED
    )

    assert result.error is not None


def test_benchmark_none_output_is_captured():
    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-015",
        method=QuantumResearchMethod.QAOA,
        problem_definition="None benchmark.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
        classical_runner=classical_runner,
        benchmark_runner=lambda *args: None,
    )

    assert (
        result.status
        == QuantumResearchStatus.FAILED
    )

    assert result.error is not None


def test_runner_exception_is_captured():
    def failing_quantum(problem, dataset):
        raise RuntimeError(
            "controlled quantum failure"
        )

    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-016",
        method=QuantumResearchMethod.VQE,
        problem_definition="Failure.",
        dataset=(1, 2),
        quantum_runner=failing_quantum,
    )

    assert (
        result.status
        == QuantumResearchStatus.FAILED
    )

    assert (
        "controlled quantum failure"
        in result.error
    )


def test_classical_failure_is_captured():
    def failing_classical(problem, dataset):
        raise RuntimeError(
            "controlled classical failure"
        )

    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-017",
        method=QuantumResearchMethod.VQE,
        problem_definition="Failure.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
        classical_runner=failing_classical,
    )

    assert (
        result.status
        == QuantumResearchStatus.FAILED
    )

    assert (
        "controlled classical failure"
        in result.error
    )


def test_benchmark_failure_is_captured():
    def failing_benchmark(
        problem,
        dataset,
        classical,
        quantum,
    ):
        raise RuntimeError(
            "controlled benchmark failure"
        )

    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-018",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Failure.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
        classical_runner=classical_runner,
        benchmark_runner=failing_benchmark,
    )

    assert (
        result.status
        == QuantumResearchStatus.FAILED
    )

    assert (
        "controlled benchmark failure"
        in result.error
    )


def test_metadata_is_preserved():
    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-019",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Metadata.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
        metadata={
            "source": "v6.0.7",
            "research_mode": "quantum",
        },
    )

    assert (
        result.metadata["source"]
        == "v6.0.7"
    )

    assert (
        result.metadata["research_mode"]
        == "quantum"
    )


def test_metadata_contains_dataset_size():
    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-020",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Metadata.",
        dataset=(1, 2, 3, 4),
        quantum_runner=quantum_runner,
    )

    assert (
        result.metadata["dataset_size"]
        == 4
    )


def test_metadata_records_human_approval():
    result = QuantumResearchOrchestrator(
        require_human_approval=True
    ).run(
        experiment_id="EXP-021",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Approval.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
        human_approval=True,
    )

    assert (
        result.metadata["human_approval_required"]
        is True
    )

    assert (
        result.metadata["human_approved"]
        is True
    )


def test_metadata_records_classical_execution():
    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-022",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Baseline.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
        classical_runner=classical_runner,
    )

    assert (
        result.metadata[
            "classical_baseline_executed"
        ]
        is True
    )


def test_metadata_records_benchmark_execution():
    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-023",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Benchmark.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
        classical_runner=classical_runner,
        benchmark_runner=benchmark_runner,
    )

    assert (
        result.metadata[
            "benchmark_executed"
        ]
        is True
    )


def test_same_problem_definition_is_recorded():
    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-024",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Same problem.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
    )

    assert (
        result.metadata[
            "same_problem_definition"
        ]
        is True
    )


def test_same_dataset_is_recorded():
    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-025",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Same dataset.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
    )

    assert (
        result.metadata["same_dataset"]
        is True
    )


def test_execution_started_is_true_on_success():
    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-026",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Execution.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
    )

    assert (
        result.metadata["execution_started"]
        is True
    )


def test_blocked_execution_does_not_start():
    result = QuantumResearchOrchestrator(
        require_human_approval=True
    ).run(
        experiment_id="EXP-027",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Blocked.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
        human_approval=False,
    )

    assert (
        result.metadata["execution_started"]
        is False
    )


def test_result_is_immutable():
    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-028",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Immutable.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
    )

    assert isinstance(
        result,
        QuantumResearchResult,
    )

    with pytest.raises(AttributeError):
        result.status = QuantumResearchStatus.FAILED


def test_experiment_id_is_preserved():
    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-PRESERVE-001",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Identity.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
    )

    assert (
        result.experiment_id
        == "EXP-PRESERVE-001"
    )


def test_problem_definition_is_preserved():
    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-029",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Preserve this problem.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
    )

    assert (
        result.problem_definition
        == "Preserve this problem."
    )


def test_quantum_result_is_preserved():
    quantum = {
        "algorithm": "QAOA",
        "score": 0.95,
    }

    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-030",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Result.",
        dataset=(1, 2),
        quantum_runner=lambda problem, dataset: quantum,
    )

    assert result.quantum_result is quantum


def test_classical_result_is_preserved():
    classical = {
        "algorithm": "classical",
        "score": 0.90,
    }

    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-031",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Result.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
        classical_runner=lambda problem, dataset: classical,
    )

    assert result.classical_result is classical


def test_benchmark_result_is_preserved():
    benchmark = {
        "quantum_advantage": 0.05
    }

    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-032",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Result.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
        classical_runner=classical_runner,
        benchmark_runner=lambda *args: benchmark,
    )

    assert result.benchmark_result is benchmark


def test_full_research_cycle():
    result = QuantumResearchOrchestrator(
        require_human_approval=True
    ).run(
        experiment_id="EXP-FULL-001",
        method=QuantumResearchMethod.QAOA,
        problem_definition=(
            "Portfolio optimization."
        ),
        dataset=(
            "NVDA",
            "MSFT",
            "META",
        ),
        quantum_runner=quantum_runner,
        classical_runner=classical_runner,
        benchmark_runner=benchmark_runner,
        human_approval=True,
        metadata={
            "version": "v6.0.7",
        },
    )

    assert (
        result.status
        == QuantumResearchStatus.COMPLETED
    )

    assert result.experiment_id == "EXP-FULL-001"

    assert (
        result.method
        == QuantumResearchMethod.QAOA
    )

    assert (
        result.problem_definition
        == "Portfolio optimization."
    )

    assert result.quantum_result is not None
    assert result.classical_result is not None
    assert result.benchmark_result is not None

    assert (
        result.metadata["version"]
        == "v6.0.7"
    )

    assert (
        result.metadata[
            "classical_baseline_executed"
        ]
        is True
    )

    assert (
        result.metadata["benchmark_executed"]
        is True
    )


def test_all_quantum_methods_can_execute():
    for method in QuantumResearchMethod:
        result = QuantumResearchOrchestrator(
            require_human_approval=False
        ).run(
            experiment_id=(
                f"EXP-{method.value}"
            ),
            method=method,
            problem_definition=(
                f"{method.value} experiment."
            ),
            dataset=(1, 2),
            quantum_runner=quantum_runner,
        )

        assert (
            result.status
            == QuantumResearchStatus.COMPLETED
        )

        assert result.method == method


def test_repeated_execution_is_deterministic():
    orchestrator = QuantumResearchOrchestrator(
        require_human_approval=False
    )

    first = orchestrator.run(
        experiment_id="EXP-REPEAT-001",
        method=QuantumResearchMethod.VQE,
        problem_definition="Repeatability.",
        dataset=(1, 2, 3),
        quantum_runner=quantum_runner,
        classical_runner=classical_runner,
        benchmark_runner=benchmark_runner,
    )

    second = orchestrator.run(
        experiment_id="EXP-REPEAT-001",
        method=QuantumResearchMethod.VQE,
        problem_definition="Repeatability.",
        dataset=(1, 2, 3),
        quantum_runner=quantum_runner,
        classical_runner=classical_runner,
        benchmark_runner=benchmark_runner,
    )

    assert first == second


def test_classical_and_quantum_receive_same_dataset():
    received = []

    def quantum(problem, dataset):
        received.append(("quantum", dataset))
        return "q"

    def classical(problem, dataset):
        received.append(("classical", dataset))
        return "c"

    QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-033",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Same data.",
        dataset=[1, 2, 3],
        quantum_runner=quantum,
        classical_runner=classical,
    )

    assert received[0][1] == received[1][1]
    assert received[0][1] == (1, 2, 3)


def test_benchmark_receives_both_outputs():
    received = {}

    def benchmark(
        problem,
        dataset,
        classical,
        quantum,
    ):
        received["classical"] = classical
        received["quantum"] = quantum
        return "benchmark"

    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-034",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Benchmark inputs.",
        dataset=(1, 2),
        quantum_runner=lambda problem, dataset: "Q",
        classical_runner=lambda problem, dataset: "C",
        benchmark_runner=benchmark,
    )

    assert received["classical"] == "C"
    assert received["quantum"] == "Q"
    assert result.benchmark_result == "benchmark"


def test_blocked_result_has_no_outputs():
    result = QuantumResearchOrchestrator(
        require_human_approval=True
    ).run(
        experiment_id="EXP-035",
        method=QuantumResearchMethod.VQE,
        problem_definition="Blocked outputs.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
        classical_runner=classical_runner,
        benchmark_runner=benchmark_runner,
        human_approval=False,
    )

    assert result.quantum_result is None
    assert result.classical_result is None
    assert result.benchmark_result is None
    assert result.error is not None


def test_failed_result_has_no_completed_outputs():
    def failing_quantum(problem, dataset):
        raise RuntimeError("failure")

    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-036",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Failed outputs.",
        dataset=(1, 2),
        quantum_runner=failing_quantum,
        classical_runner=classical_runner,
        benchmark_runner=benchmark_runner,
    )

    assert (
        result.status
        == QuantumResearchStatus.FAILED
    )

    assert result.quantum_result is None
    assert result.classical_result is None
    assert result.benchmark_result is None


def test_optional_benchmark_without_classical_is_failed():
    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-037",
        method=QuantumResearchMethod.QUANTUM_KERNEL,
        problem_definition="Invalid benchmark.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
        benchmark_runner=benchmark_runner,
    )

    assert (
        result.status
        == QuantumResearchStatus.FAILED
    )

    assert (
        result.error
        == (
            "benchmark_runner requires "
            "a classical_runner."
        )
    )


def test_metadata_is_not_required():
    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-038",
        method=QuantumResearchMethod.QAOA,
        problem_definition="No metadata.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
    )

    assert result.status == QuantumResearchStatus.COMPLETED
    assert result.metadata is not None


def test_empty_metadata_is_supported():
    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-039",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Empty metadata.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
        metadata={},
    )

    assert result.status == QuantumResearchStatus.COMPLETED


def test_metadata_is_copied():
    metadata = {
        "source": "test",
    }

    result = QuantumResearchOrchestrator(
        require_human_approval=False
    ).run(
        experiment_id="EXP-040",
        method=QuantumResearchMethod.QAOA,
        problem_definition="Metadata copy.",
        dataset=(1, 2),
        quantum_runner=quantum_runner,
        metadata=metadata,
    )

    metadata["source"] = "changed"

    assert (
        result.metadata["source"]
        == "test"
    )