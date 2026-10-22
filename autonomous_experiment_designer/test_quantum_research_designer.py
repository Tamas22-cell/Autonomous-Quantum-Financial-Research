import pytest

from autonomous_experiment_designer.quantum_research_designer import (
    QuantumExperimentType,
    QuantumPriority,
    QuantumResearchDesigner,
)


@pytest.fixture
def designer():
    return QuantumResearchDesigner()


def test_default_configuration(designer):
    assert designer.datasets == (
        "synthetic_benchmark",
        "historical_market_data",
    )

    assert designer.metrics == (
        "objective_value",
        "solution_quality",
        "runtime",
        "convergence",
        "resource_usage",
    )

    assert designer.validation_tests == (
        "classical_baseline",
        "out_of_sample_validation",
        "repeated_runs",
        "noise_sensitivity",
    )


def test_qaoa_plan(designer):
    plan = designer.design(
        experiment_id="Q-001",
        experiment_type=QuantumExperimentType.QAOA,
        problem_definition="Portfolio optimization with fixed constraints.",
    )

    assert plan.experiment_id == "Q-001"
    assert (
        plan.experiment_type
        == QuantumExperimentType.QAOA
    )

    assert plan.priority == QuantumPriority.HIGH
    assert plan.classical_baseline_required is True
    assert plan.quantum_method_required is True
    assert plan.hybrid_method_allowed is False
    assert plan.requires_human_approval is True

    assert "QAOA" in plan.algorithms
    assert "parameter_optimization" in plan.algorithms


def test_vqe_plan(designer):
    plan = designer.design(
        experiment_id="Q-VQE-001",
        experiment_type=QuantumExperimentType.VQE,
        problem_definition="Variational optimization benchmark.",
    )

    assert (
        plan.experiment_type
        == QuantumExperimentType.VQE
    )

    assert plan.priority == QuantumPriority.HIGH
    assert "VQE" in plan.algorithms
    assert "variational_optimizer" in plan.algorithms


def test_quantum_kernel_plan(designer):
    plan = designer.design(
        experiment_id="Q-KERNEL-001",
        experiment_type=QuantumExperimentType.QUANTUM_KERNEL,
        problem_definition="Financial classification problem.",
    )

    assert (
        plan.experiment_type
        == QuantumExperimentType.QUANTUM_KERNEL
    )

    assert plan.priority == QuantumPriority.MEDIUM
    assert "quantum_kernel" in plan.algorithms
    assert "classical_svm" in plan.algorithms


def test_quantum_ml_plan(designer):
    plan = designer.design(
        experiment_id="QML-001",
        experiment_type=QuantumExperimentType.QUANTUM_ML,
        problem_definition="Quantum machine learning classification.",
    )

    assert (
        plan.experiment_type
        == QuantumExperimentType.QUANTUM_ML
    )

    assert "quantum_feature_map" in plan.algorithms
    assert "quantum_neural_network" in plan.algorithms


def test_quantum_monte_carlo_plan(designer):
    plan = designer.design(
        experiment_id="QMC-001",
        experiment_type=QuantumExperimentType.QUANTUM_MONTE_CARLO,
        problem_definition="Monte Carlo estimation benchmark.",
    )

    assert (
        plan.experiment_type
        == QuantumExperimentType.QUANTUM_MONTE_CARLO
    )

    assert "quantum_monte_carlo" in plan.algorithms
    assert "amplitude_estimation" in plan.algorithms


def test_hybrid_plan(designer):
    plan = designer.design(
        experiment_id="HYBRID-001",
        experiment_type=QuantumExperimentType.HYBRID,
        problem_definition="Hybrid classical quantum optimization.",
    )

    assert (
        plan.experiment_type
        == QuantumExperimentType.HYBRID
    )

    assert plan.priority == QuantumPriority.HIGH
    assert plan.classical_baseline_required is True
    assert plan.quantum_method_required is True
    assert plan.hybrid_method_allowed is True


def test_classical_quantum_benchmark_plan(designer):
    plan = designer.design(
        experiment_id="BENCH-001",
        experiment_type=(
            QuantumExperimentType.CLASSICAL_QUANTUM_BENCHMARK
        ),
        problem_definition=(
            "Compare classical and quantum optimization "
            "under identical constraints."
        ),
    )

    assert (
        plan.experiment_type
        == QuantumExperimentType.CLASSICAL_QUANTUM_BENCHMARK
    )

    assert plan.priority == QuantumPriority.HIGH
    assert plan.classical_baseline_required is True
    assert plan.quantum_method_required is True
    assert plan.hybrid_method_allowed is True

    assert "classical_baseline" in plan.algorithms
    assert "quantum_method" in plan.algorithms


def test_default_datasets_are_present(designer):
    plan = designer.design(
        experiment_id="DATA-001",
        experiment_type=QuantumExperimentType.QAOA,
        problem_definition="Test dataset selection.",
    )

    assert plan.datasets == designer.datasets


def test_custom_datasets_are_used(designer):
    plan = designer.design(
        experiment_id="DATA-002",
        experiment_type=QuantumExperimentType.QAOA,
        problem_definition="Custom dataset experiment.",
        datasets=(
            "portfolio_dataset",
            "market_regime_dataset",
        ),
    )

    assert plan.datasets == (
        "portfolio_dataset",
        "market_regime_dataset",
    )


def test_custom_algorithms_are_used(designer):
    plan = designer.design(
        experiment_id="ALG-001",
        experiment_type=QuantumExperimentType.QAOA,
        problem_definition="Custom QAOA algorithm.",
        algorithms=(
            "QAOA",
            "COBYLA",
            "SPSA",
        ),
    )

    assert plan.algorithms == (
        "QAOA",
        "COBYLA",
        "SPSA",
    )


def test_custom_metrics_are_used(designer):
    plan = designer.design(
        experiment_id="METRIC-001",
        experiment_type=QuantumExperimentType.VQE,
        problem_definition="Custom metric experiment.",
        metrics=(
            "energy",
            "fidelity",
            "runtime",
        ),
    )

    assert plan.metrics == (
        "energy",
        "fidelity",
        "runtime",
    )


def test_custom_validation_tests_are_used(designer):
    plan = designer.design(
        experiment_id="VAL-001",
        experiment_type=QuantumExperimentType.VQE,
        problem_definition="Custom validation experiment.",
        validation_tests=(
            "noise_test",
            "repeated_runs",
            "classical_comparison",
        ),
    )

    assert plan.validation_tests == (
        "noise_test",
        "repeated_runs",
        "classical_comparison",
    )


def test_problem_definition_is_preserved(designer):
    definition = (
        "Optimize a constrained portfolio with identical "
        "classical and quantum objective functions."
    )

    plan = designer.design(
        experiment_id="PROBLEM-001",
        experiment_type=(
            QuantumExperimentType.CLASSICAL_QUANTUM_BENCHMARK
        ),
        problem_definition=definition,
    )

    assert plan.problem_definition == definition


def test_objective_is_nonempty_for_all_experiment_types(
    designer,
):
    for experiment_type in QuantumExperimentType:
        plan = designer.design(
            experiment_id=f"OBJ-{experiment_type.value}",
            experiment_type=experiment_type,
            problem_definition="Generic quantum research problem.",
        )

        assert plan.objective.strip()


def test_plan_is_immutable(designer):
    plan = designer.design(
        experiment_id="IMMUTABLE-001",
        experiment_type=QuantumExperimentType.QAOA,
        problem_definition="Immutability test.",
    )

    with pytest.raises(AttributeError):
        plan.priority = QuantumPriority.CRITICAL


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        None,
        123,
    ],
)
def test_invalid_experiment_id_is_rejected(
    designer,
    value,
):
    with pytest.raises(ValueError):
        designer.design(
            experiment_id=value,
            experiment_type=QuantumExperimentType.QAOA,
            problem_definition="Invalid ID test.",
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
def test_invalid_problem_definition_is_rejected(
    designer,
    value,
):
    with pytest.raises(ValueError):
        designer.design(
            experiment_id="VALID-ID",
            experiment_type=QuantumExperimentType.QAOA,
            problem_definition=value,
        )


def test_invalid_experiment_type_is_rejected(
    designer,
):
    with pytest.raises(ValueError):
        designer.design(
            experiment_id="INVALID-TYPE",
            experiment_type="QAOA",
            problem_definition="Invalid type test.",
        )


@pytest.mark.parametrize(
    "field",
    [
        "datasets",
        "algorithms",
        "metrics",
        "validation_tests",
    ],
)
def test_empty_custom_configuration_is_rejected(
    designer,
    field,
):
    kwargs = {
        field: (),
    }

    with pytest.raises(ValueError):
        designer.design(
            experiment_id="EMPTY-CONFIG",
            experiment_type=QuantumExperimentType.QAOA,
            problem_definition="Empty configuration test.",
            **kwargs,
        )


@pytest.mark.parametrize(
    "field",
    [
        "datasets",
        "algorithms",
        "metrics",
        "validation_tests",
    ],
)
def test_string_custom_configuration_is_rejected(
    designer,
    field,
):
    kwargs = {
        field: "invalid",
    }

    with pytest.raises(ValueError):
        designer.design(
            experiment_id="STRING-CONFIG",
            experiment_type=QuantumExperimentType.QAOA,
            problem_definition="String configuration test.",
            **kwargs,
        )


def test_duplicate_datasets_are_rejected(designer):
    with pytest.raises(ValueError):
        designer.design(
            experiment_id="DUP-DATA",
            experiment_type=QuantumExperimentType.QAOA,
            problem_definition="Duplicate dataset test.",
            datasets=(
                "market",
                "market",
            ),
        )


def test_duplicate_algorithms_are_rejected(designer):
    with pytest.raises(ValueError):
        designer.design(
            experiment_id="DUP-ALG",
            experiment_type=QuantumExperimentType.QAOA,
            problem_definition="Duplicate algorithm test.",
            algorithms=(
                "QAOA",
                "QAOA",
            ),
        )


def test_duplicate_metrics_are_rejected(designer):
    with pytest.raises(ValueError):
        designer.design(
            experiment_id="DUP-METRIC",
            experiment_type=QuantumExperimentType.VQE,
            problem_definition="Duplicate metric test.",
            metrics=(
                "energy",
                "energy",
            ),
        )


def test_duplicate_validation_tests_are_rejected(designer):
    with pytest.raises(ValueError):
        designer.design(
            experiment_id="DUP-VAL",
            experiment_type=QuantumExperimentType.VQE,
            problem_definition="Duplicate validation test.",
            validation_tests=(
                "noise",
                "noise",
            ),
        )


def test_empty_name_inside_dataset_configuration_is_rejected(
    designer,
):
    with pytest.raises(ValueError):
        designer.design(
            experiment_id="EMPTY-NAME",
            experiment_type=QuantumExperimentType.QAOA,
            problem_definition="Empty dataset name test.",
            datasets=(
                "valid_dataset",
                "",
            ),
        )


def test_empty_name_inside_algorithm_configuration_is_rejected(
    designer,
):
    with pytest.raises(ValueError):
        designer.design(
            experiment_id="EMPTY-ALG-NAME",
            experiment_type=QuantumExperimentType.QAOA,
            problem_definition="Empty algorithm name test.",
            algorithms=(
                "QAOA",
                " ",
            ),
        )


def test_empty_name_inside_metric_configuration_is_rejected(
    designer,
):
    with pytest.raises(ValueError):
        designer.design(
            experiment_id="EMPTY-METRIC-NAME",
            experiment_type=QuantumExperimentType.VQE,
            problem_definition="Empty metric name test.",
            metrics=(
                "energy",
                "",
            ),
        )


def test_empty_name_inside_validation_configuration_is_rejected(
    designer,
):
    with pytest.raises(ValueError):
        designer.design(
            experiment_id="EMPTY-VAL-NAME",
            experiment_type=QuantumExperimentType.VQE,
            problem_definition="Empty validation name test.",
            validation_tests=(
                "noise_test",
                " ",
            ),
        )


def test_plan_requires_human_approval(designer):
    for experiment_type in QuantumExperimentType:
        plan = designer.design(
            experiment_id=f"APPROVAL-{experiment_type.value}",
            experiment_type=experiment_type,
            problem_definition="Approval gate test.",
        )

        assert plan.requires_human_approval is True


def test_benchmark_always_requires_classical_baseline(
    designer,
):
    plan = designer.design(
        experiment_id="BASELINE-001",
        experiment_type=(
            QuantumExperimentType.CLASSICAL_QUANTUM_BENCHMARK
        ),
        problem_definition="Baseline requirement test.",
    )

    assert plan.classical_baseline_required is True


def test_qaoa_does_not_allow_hybrid_by_default(
    designer,
):
    plan = designer.design(
        experiment_id="QAOA-HYBRID",
        experiment_type=QuantumExperimentType.QAOA,
        problem_definition="QAOA hybrid flag test.",
    )

    assert plan.hybrid_method_allowed is False


def test_hybrid_experiment_allows_hybrid_method(
    designer,
):
    plan = designer.design(
        experiment_id="HYBRID-FLAG",
        experiment_type=QuantumExperimentType.HYBRID,
        problem_definition="Hybrid flag test.",
    )

    assert plan.hybrid_method_allowed is True


def test_all_plans_have_metrics(designer):
    for experiment_type in QuantumExperimentType:
        plan = designer.design(
            experiment_id=f"METRICS-{experiment_type.value}",
            experiment_type=experiment_type,
            problem_definition="Metric availability test.",
        )

        assert len(plan.metrics) > 0


def test_all_plans_have_validation_tests(designer):
    for experiment_type in QuantumExperimentType:
        plan = designer.design(
            experiment_id=f"VALIDATION-{experiment_type.value}",
            experiment_type=experiment_type,
            problem_definition="Validation availability test.",
        )

        assert len(plan.validation_tests) > 0


def test_all_plans_have_datasets(designer):
    for experiment_type in QuantumExperimentType:
        plan = designer.design(
            experiment_id=f"DATASETS-{experiment_type.value}",
            experiment_type=experiment_type,
            problem_definition="Dataset availability test.",
        )

        assert len(plan.datasets) > 0


def test_plan_contains_at_least_one_algorithm(designer):
    for experiment_type in QuantumExperimentType:
        plan = designer.design(
            experiment_id=f"ALGORITHMS-{experiment_type.value}",
            experiment_type=experiment_type,
            problem_definition="Algorithm availability test.",
        )

        assert len(plan.algorithms) > 0


def test_default_algorithm_mapping_is_deterministic(
    designer,
):
    first = designer.design(
        experiment_id="DETERMINISTIC-1",
        experiment_type=QuantumExperimentType.QAOA,
        problem_definition="Deterministic test.",
    )

    second = designer.design(
        experiment_id="DETERMINISTIC-2",
        experiment_type=QuantumExperimentType.QAOA,
        problem_definition="Deterministic test.",
    )

    assert first.algorithms == second.algorithms