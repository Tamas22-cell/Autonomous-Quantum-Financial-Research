from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from quantum_portfolio_lab.qaoa_solver import (
    IsingQUBOModel,
    QAOAPortfolioSolver,
    QAOASolverConfig,
    QAOASolverError,
)
from quantum_portfolio_lab.qubo_builder import (
    PortfolioQUBOBuilder,
    PortfolioSelectionQUBORequest,
)
from quantum_portfolio_lab.qubo_models import (
    QuantumExecutionBackend,
    QuantumSolutionStatus,
    QuantumSolverType,
)


# ============================================================
# Test helpers
# ============================================================


def build_three_asset_problem():
    """Create a deterministic three-asset QUBO test problem."""

    request = PortfolioSelectionQUBORequest(
        asset_symbols=("AAA", "BBB", "CCC"),
        expected_returns=np.array(
            [0.12, 0.08, 0.05],
            dtype=float,
        ),
        covariance_matrix=np.array(
            [
                [0.040, 0.010, 0.005],
                [0.010, 0.030, 0.004],
                [0.005, 0.004, 0.020],
            ],
            dtype=float,
        ),
        target_cardinality=2,
        risk_aversion=1.0,
        return_weight=1.0,
        budget_penalty=2.0,
    )

    return PortfolioQUBOBuilder.build(request)


def build_two_asset_problem():
    """Create a very small problem for real QAOA execution."""

    request = PortfolioSelectionQUBORequest(
        asset_symbols=("AAA", "BBB"),
        expected_returns=np.array(
            [0.15, 0.05],
            dtype=float,
        ),
        covariance_matrix=np.array(
            [
                [0.02, 0.00],
                [0.00, 0.02],
            ],
            dtype=float,
        ),
        target_cardinality=1,
        risk_aversion=1.0,
        return_weight=1.0,
        budget_penalty=3.0,
    )

    return PortfolioQUBOBuilder.build(request)


# ============================================================
# Configuration tests
# ============================================================


def test_default_config():
    config = QAOASolverConfig()

    assert config.reps == 1
    assert config.shots == 2048
    assert config.maxiter == 100
    assert config.seed == 42
    assert config.optimization_level == 1


def test_custom_config():
    config = QAOASolverConfig(
        reps=2,
        shots=512,
        maxiter=25,
        seed=123,
        optimization_level=0,
    )

    assert config.reps == 2
    assert config.shots == 512
    assert config.maxiter == 25
    assert config.seed == 123
    assert config.optimization_level == 0


@pytest.mark.parametrize(
    "value",
    [0, -1, -10],
)
def test_invalid_reps(value):
    with pytest.raises(QAOASolverError):
        QAOASolverConfig(reps=value)


@pytest.mark.parametrize(
    "value",
    [0, -1, -100],
)
def test_invalid_shots(value):
    with pytest.raises(QAOASolverError):
        QAOASolverConfig(shots=value)


@pytest.mark.parametrize(
    "value",
    [0, -1, -100],
)
def test_invalid_maxiter(value):
    with pytest.raises(QAOASolverError):
        QAOASolverConfig(maxiter=value)


@pytest.mark.parametrize(
    "value",
    [-1, 4, 10],
)
def test_invalid_optimization_level(value):
    with pytest.raises(QAOASolverError):
        QAOASolverConfig(
            optimization_level=value
        )


# ============================================================
# Ising model tests
# ============================================================


def test_ising_model_creation():
    model = IsingQUBOModel(
        linear=(-1.0, 0.5),
        quadratic={(0, 1): 0.25},
        constant=2.0,
    )

    assert model.linear == (-1.0, 0.5)
    assert model.quadratic[(0, 1)] == pytest.approx(
        0.25
    )
    assert model.constant == pytest.approx(2.0)


def test_ising_model_rejects_empty_linear():
    with pytest.raises(QAOASolverError):
        IsingQUBOModel(
            linear=(),
            quadratic={},
            constant=0.0,
        )


def test_ising_model_rejects_nan_linear():
    with pytest.raises(QAOASolverError):
        IsingQUBOModel(
            linear=(1.0, np.nan),
            quadratic={},
            constant=0.0,
        )


def test_ising_model_rejects_infinite_constant():
    with pytest.raises(QAOASolverError):
        IsingQUBOModel(
            linear=(1.0,),
            quadratic={},
            constant=np.inf,
        )


def test_ising_model_rejects_reversed_pair():
    with pytest.raises(QAOASolverError):
        IsingQUBOModel(
            linear=(1.0, 1.0),
            quadratic={(1, 0): 0.5},
            constant=0.0,
        )


def test_ising_model_rejects_same_pair_index():
    with pytest.raises(QAOASolverError):
        IsingQUBOModel(
            linear=(1.0,),
            quadratic={(0, 0): 0.5},
            constant=0.0,
        )


def test_ising_model_rejects_nan_quadratic():
    with pytest.raises(QAOASolverError):
        IsingQUBOModel(
            linear=(1.0, 1.0),
            quadratic={(0, 1): np.nan},
            constant=0.0,
        )


# ============================================================
# QUBO -> Ising mathematical equivalence
# ============================================================


def test_qubo_to_ising_variable_count():
    problem = build_three_asset_problem()

    model = QAOAPortfolioSolver.qubo_to_ising(
        problem
    )

    assert len(model.linear) == problem.variable_count


def test_qubo_to_ising_coefficients_are_finite():
    problem = build_three_asset_problem()

    model = QAOAPortfolioSolver.qubo_to_ising(
        problem
    )

    assert np.isfinite(model.constant)

    assert all(
        np.isfinite(value)
        for value in model.linear
    )

    assert all(
        np.isfinite(value)
        for value in model.quadratic.values()
    )


def test_all_binary_states_preserve_qubo_energy():
    """
    Critical correctness test.

    Every possible binary portfolio must have the same
    objective under the original QUBO and Ising mapping.
    """

    problem = build_three_asset_problem()

    model = QAOAPortfolioSolver.qubo_to_ising(
        problem
    )

    for state in product(
        (0, 1),
        repeat=problem.variable_count,
    ):
        qubo_energy = problem.energy(state)

        ising_energy = (
            QAOAPortfolioSolver.ising_energy(
                model=model,
                binary_vector=state,
            )
        )

        assert ising_energy == pytest.approx(
            qubo_energy,
            abs=1e-10,
        )


def test_zero_state_energy_equivalence():
    problem = build_three_asset_problem()

    model = QAOAPortfolioSolver.qubo_to_ising(
        problem
    )

    state = (0, 0, 0)

    assert QAOAPortfolioSolver.ising_energy(
        model=model,
        binary_vector=state,
    ) == pytest.approx(
        problem.energy(state),
        abs=1e-10,
    )


def test_all_one_state_energy_equivalence():
    problem = build_three_asset_problem()

    model = QAOAPortfolioSolver.qubo_to_ising(
        problem
    )

    state = (1, 1, 1)

    assert QAOAPortfolioSolver.ising_energy(
        model=model,
        binary_vector=state,
    ) == pytest.approx(
        problem.energy(state),
        abs=1e-10,
    )


def test_ising_energy_rejects_wrong_length():
    problem = build_three_asset_problem()

    model = QAOAPortfolioSolver.qubo_to_ising(
        problem
    )

    with pytest.raises(QAOASolverError):
        QAOAPortfolioSolver.ising_energy(
            model=model,
            binary_vector=(1, 0),
        )


def test_ising_energy_rejects_nonbinary_vector():
    problem = build_three_asset_problem()

    model = QAOAPortfolioSolver.qubo_to_ising(
        problem
    )

    with pytest.raises(QAOASolverError):
        QAOAPortfolioSolver.ising_energy(
            model=model,
            binary_vector=(1, 2, 0),
        )


# ============================================================
# Qiskit basis / bit ordering
# ============================================================


@pytest.mark.parametrize(
    ("basis_index", "expected"),
    [
        (0, (0, 0, 0)),
        (1, (1, 0, 0)),
        (2, (0, 1, 0)),
        (3, (1, 1, 0)),
        (4, (0, 0, 1)),
        (7, (1, 1, 1)),
    ],
)
def test_basis_index_to_binary_solution(
    basis_index,
    expected,
):
    result = (
        QAOAPortfolioSolver
        .basis_index_to_binary_solution(
            basis_index=basis_index,
            variable_count=3,
        )
    )

    assert result == expected


def test_basis_index_rejects_negative():
    with pytest.raises(QAOASolverError):
        QAOAPortfolioSolver.basis_index_to_binary_solution(
            basis_index=-1,
            variable_count=3,
        )


def test_basis_index_rejects_out_of_range():
    with pytest.raises(QAOASolverError):
        QAOAPortfolioSolver.basis_index_to_binary_solution(
            basis_index=8,
            variable_count=3,
        )


def test_basis_index_rejects_zero_variables():
    with pytest.raises(QAOASolverError):
        QAOAPortfolioSolver.basis_index_to_binary_solution(
            basis_index=0,
            variable_count=0,
        )


@pytest.mark.parametrize(
    ("bitstring", "expected"),
    [
        ("000", (0, 0, 0)),
        ("001", (1, 0, 0)),
        ("010", (0, 1, 0)),
        ("100", (0, 0, 1)),
        ("101", (1, 0, 1)),
        ("111", (1, 1, 1)),
    ],
)
def test_bitstring_to_binary_solution(
    bitstring,
    expected,
):
    result = (
        QAOAPortfolioSolver
        .bitstring_to_binary_solution(
            bitstring=bitstring,
            variable_count=3,
        )
    )

    assert result == expected


def test_bitstring_with_spaces():
    result = (
        QAOAPortfolioSolver
        .bitstring_to_binary_solution(
            bitstring="0 1 1",
            variable_count=3,
        )
    )

    assert result == (1, 1, 0)


def test_bitstring_rejects_wrong_length():
    with pytest.raises(QAOASolverError):
        QAOAPortfolioSolver.bitstring_to_binary_solution(
            bitstring="01",
            variable_count=3,
        )


def test_bitstring_rejects_invalid_character():
    with pytest.raises(QAOASolverError):
        QAOAPortfolioSolver.bitstring_to_binary_solution(
            bitstring="0A1",
            variable_count=3,
        )


def test_binary_solution_to_bitstring():
    result = (
        QAOAPortfolioSolver
        .binary_solution_to_bitstring(
            (1, 0, 0)
        )
    )

    assert result == "001"


def test_binary_solution_round_trip():
    original = (1, 0, 1, 1)

    bitstring = (
        QAOAPortfolioSolver
        .binary_solution_to_bitstring(
            original
        )
    )

    restored = (
        QAOAPortfolioSolver
        .bitstring_to_binary_solution(
            bitstring=bitstring,
            variable_count=4,
        )
    )

    assert restored == original


def test_binary_solution_rejects_empty():
    with pytest.raises(QAOASolverError):
        QAOAPortfolioSolver.binary_solution_to_bitstring(
            ()
        )


def test_binary_solution_rejects_nonbinary():
    with pytest.raises(QAOASolverError):
        QAOAPortfolioSolver.binary_solution_to_bitstring(
            (1, 2, 0)
        )


# ============================================================
# Statevector expected-energy tests
# ============================================================


def test_expected_energy_zero_basis_state():
    problem = build_two_asset_problem()

    statevector = np.array(
        [1.0, 0.0, 0.0, 0.0],
        dtype=complex,
    )

    result = (
        QAOAPortfolioSolver
        ._expected_qubo_energy(
            problem=problem,
            statevector=statevector,
        )
    )

    assert result == pytest.approx(
        problem.energy((0, 0))
    )


def test_expected_energy_basis_state_one():
    problem = build_two_asset_problem()

    statevector = np.array(
        [0.0, 1.0, 0.0, 0.0],
        dtype=complex,
    )

    result = (
        QAOAPortfolioSolver
        ._expected_qubo_energy(
            problem=problem,
            statevector=statevector,
        )
    )

    assert result == pytest.approx(
        problem.energy((1, 0))
    )


def test_expected_energy_uniform_state():
    problem = build_two_asset_problem()

    statevector = np.ones(
        4,
        dtype=complex,
    ) / 2.0

    result = (
        QAOAPortfolioSolver
        ._expected_qubo_energy(
            problem=problem,
            statevector=statevector,
        )
    )

    manual = np.mean(
        [
            problem.energy(state)
            for state in product(
                (0, 1),
                repeat=2,
            )
        ]
    )

    assert result == pytest.approx(
        manual,
        abs=1e-10,
    )


def test_expected_energy_rejects_wrong_size():
    problem = build_two_asset_problem()

    with pytest.raises(QAOASolverError):
        QAOAPortfolioSolver._expected_qubo_energy(
            problem=problem,
            statevector=np.array(
                [1.0, 0.0],
                dtype=complex,
            ),
        )


# ============================================================
# Measurement selection
# ============================================================


def test_best_measured_state_uses_lowest_energy():
    problem = build_two_asset_problem()

    counts = {
        "00": 100,
        "01": 10,
        "10": 20,
        "11": 200,
    }

    state = (
        QAOAPortfolioSolver
        ._best_measured_state(
            problem=problem,
            counts=counts,
        )
    )

    measured_states = [
        QAOAPortfolioSolver
        .bitstring_to_binary_solution(
            bitstring=bitstring,
            variable_count=2,
        )
        for bitstring in counts
    ]

    minimum_energy = min(
        problem.energy(candidate)
        for candidate in measured_states
    )

    assert problem.energy(
        state
    ) == pytest.approx(
        minimum_energy
    )


def test_best_measured_state_rejects_empty_counts():
    problem = build_two_asset_problem()

    with pytest.raises(QAOASolverError):
        QAOAPortfolioSolver._best_measured_state(
            problem=problem,
            counts={},
        )


def test_best_measured_state_rejects_negative_count():
    problem = build_two_asset_problem()

    with pytest.raises(QAOASolverError):
        QAOAPortfolioSolver._best_measured_state(
            problem=problem,
            counts={"00": -1},
        )


# ============================================================
# Portfolio decoding
# ============================================================


def test_selected_assets():
    problem = build_three_asset_problem()

    assets = (
        QAOAPortfolioSolver
        ._selected_assets(
            problem=problem,
            binary_solution=(1, 0, 1),
        )
    )

    assert assets == ("AAA", "CCC")


def test_equal_weights():
    problem = build_three_asset_problem()

    weights = (
        QAOAPortfolioSolver
        ._derive_equal_weights(
            problem=problem,
            binary_solution=(1, 0, 1),
        )
    )

    assert weights["AAA"] == pytest.approx(0.5)
    assert weights["BBB"] == pytest.approx(0.0)
    assert weights["CCC"] == pytest.approx(0.5)

    assert sum(
        weights.values()
    ) == pytest.approx(1.0)


def test_zero_selected_assets_have_zero_weights():
    problem = build_three_asset_problem()

    weights = (
        QAOAPortfolioSolver
        ._derive_equal_weights(
            problem=problem,
            binary_solution=(0, 0, 0),
        )
    )

    assert all(
        value == 0.0
        for value in weights.values()
    )


# ============================================================
# Constraint diagnostics
# ============================================================


def test_cardinality_constraint_satisfied():
    problem = build_three_asset_problem()

    diagnostics = (
        QAOAPortfolioSolver
        ._constraint_diagnostics(
            problem=problem,
            binary_solution=(1, 1, 0),
        )
    )

    assert len(diagnostics) == 1

    diagnostic = diagnostics[0]

    assert diagnostic.name == "cardinality"
    assert diagnostic.satisfied is True
    assert diagnostic.violation == pytest.approx(
        0.0
    )
    assert diagnostic.penalty == pytest.approx(
        0.0
    )


def test_cardinality_constraint_violation():
    problem = build_three_asset_problem()

    diagnostics = (
        QAOAPortfolioSolver
        ._constraint_diagnostics(
            problem=problem,
            binary_solution=(1, 0, 0),
        )
    )

    diagnostic = diagnostics[0]

    assert diagnostic.satisfied is False
    assert diagnostic.violation == pytest.approx(
        1.0
    )
    assert diagnostic.penalty > 0.0


# ============================================================
# REAL QAOA + AerSimulator integration tests
# ============================================================


@pytest.fixture(scope="module")
def qaoa_solution():
    """
    Run one real QAOA experiment and reuse the result.

    This keeps the test suite fast while still validating
    the complete simulator-first quantum workflow.
    """

    problem = build_two_asset_problem()

    config = QAOASolverConfig(
        reps=1,
        shots=256,
        maxiter=15,
        seed=42,
        optimization_level=0,
    )

    solver = QAOAPortfolioSolver(
        config
    )

    solution = solver.solve(
        problem
    )

    return problem, config, solution


def test_qaoa_returns_completed_solution(
    qaoa_solution,
):
    _, _, solution = qaoa_solution

    assert solution.status == (
        QuantumSolutionStatus.COMPLETED
    )

    assert solution.solver_type == (
        QuantumSolverType.QAOA
    )

    assert solution.backend == (
        QuantumExecutionBackend.SIMULATOR
    )


def test_qaoa_binary_solution_is_valid(
    qaoa_solution,
):
    problem, _, solution = qaoa_solution

    assert len(
        solution.binary_solution
    ) == problem.variable_count

    assert all(
        value in (0, 1)
        for value in solution.binary_solution
    )


def test_qaoa_objective_matches_same_qubo(
    qaoa_solution,
):
    """
    Critical fair-comparison test.

    QAOA's reported objective is evaluated using the exact
    same PortfolioQUBOProblem used by the classical solver.
    """

    problem, _, solution = qaoa_solution

    expected = problem.energy(
        solution.binary_solution
    )

    assert solution.objective_value == pytest.approx(
        expected,
        abs=1e-10,
    )


def test_qaoa_runtime_is_nonnegative(
    qaoa_solution,
):
    _, _, solution = qaoa_solution

    assert solution.runtime_seconds >= 0.0


def test_qaoa_metadata(
    qaoa_solution,
):
    problem, config, solution = qaoa_solution

    metadata = solution.metadata

    assert metadata["algorithm"] == "qaoa"

    assert (
        metadata["execution_mode"]
        == "local_simulator"
    )

    assert metadata["simulator_first"] is True
    assert metadata["hardware_execution"] is False

    assert metadata["reps"] == config.reps
    assert metadata["shots"] == config.shots
    assert metadata["maxiter"] == config.maxiter
    assert metadata["seed"] == config.seed

    assert metadata["variable_count"] == (
        problem.variable_count
    )


def test_qaoa_measurement_counts(
    qaoa_solution,
):
    _, config, solution = qaoa_solution

    counts = solution.metadata[
        "measurement_counts"
    ]

    assert isinstance(counts, dict)
    assert counts

    assert sum(
        counts.values()
    ) == config.shots


def test_qaoa_selected_assets_match_binary_solution(
    qaoa_solution,
):
    problem, _, solution = qaoa_solution

    expected_assets = tuple(
        symbol
        for symbol, value in zip(
            problem.asset_symbols,
            solution.binary_solution,
        )
        if value == 1
    )

    assert solution.selected_assets == (
        expected_assets
    )


def test_qaoa_constraint_diagnostics_exist(
    qaoa_solution,
):
    _, _, solution = qaoa_solution

    assert len(
        solution.constraint_diagnostics
    ) == 1

    assert (
        solution.constraint_diagnostics[0].name
        == "cardinality"
    )


def test_qaoa_optimizer_value_is_finite(
    qaoa_solution,
):
    _, _, solution = qaoa_solution

    assert np.isfinite(
        solution.metadata[
            "optimizer_value"
        ]
    )


def test_qaoa_best_probability_is_valid(
    qaoa_solution,
):
    _, _, solution = qaoa_solution

    probability = solution.metadata[
        "best_probability"
    ]

    assert 0.0 <= probability <= 1.0


# ============================================================
# Solver construction
# ============================================================


def test_solver_default_config():
    solver = QAOAPortfolioSolver()

    assert isinstance(
        solver.config,
        QAOASolverConfig,
    )

    assert solver.config.reps == 1


def test_solver_custom_config():
    config = QAOASolverConfig(
        reps=2,
        shots=64,
        maxiter=5,
        seed=99,
        optimization_level=0,
    )

    solver = QAOAPortfolioSolver(
        config
    )

    assert solver.config is config