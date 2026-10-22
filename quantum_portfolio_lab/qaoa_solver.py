"""
QAOA solver for the v5.0.8 Quantum Portfolio Lab.

Environment targeted by this implementation:

- Qiskit 2.x
- qiskit-algorithms 0.4.x
- qiskit-aer 0.17.x

The solver uses the same PortfolioQUBOProblem representation
as the classical exact solver, enabling fair comparison of:

- objective value
- solution quality
- runtime
- constraint violations
- stability
- scaling

Workflow
--------

PortfolioQUBOProblem
        |
        v
QUBO -> Ising Hamiltonian
        |
        v
QAOA circuit
        |
        v
Classical parameter optimization
        |
        v
Simulator sampling
        |
        v
Best measured binary portfolio

The implementation is simulator-first. Hardware execution is
intentionally not enabled automatically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .qubo_models import (
    PortfolioQUBOProblem,
    QUBOConstraintDiagnostic,
    QuantumExecutionBackend,
    QuantumPortfolioModelError,
    QuantumPortfolioSolution,
    QuantumSolutionStatus,
    QuantumSolverType,
)


class QAOASolverError(QuantumPortfolioModelError):
    """
    Raised when QAOA portfolio optimization fails.
    """


@dataclass(frozen=True)
class QAOASolverConfig:
    """
    Configuration for simulator-first QAOA experiments.

    Parameters
    ----------
    reps:
        QAOA circuit depth p.

    shots:
        Number of measurements used to estimate the final
        solution distribution.

    maxiter:
        Maximum COBYLA optimizer iterations.

    seed:
        Random seed used for reproducibility.

    optimization_level:
        Qiskit transpiler optimization level.

    """

    reps: int = 1
    shots: int = 2048
    maxiter: int = 100
    seed: int = 42
    optimization_level: int = 1

    def __post_init__(self) -> None:
        if self.reps <= 0:
            raise QAOASolverError(
                "reps must be positive."
            )

        if self.shots <= 0:
            raise QAOASolverError(
                "shots must be positive."
            )

        if self.maxiter <= 0:
            raise QAOASolverError(
                "maxiter must be positive."
            )

        if self.optimization_level not in (
            0,
            1,
            2,
            3,
        ):
            raise QAOASolverError(
                "optimization_level must be "
                "0, 1, 2 or 3."
            )


@dataclass(frozen=True)
class IsingQUBOModel:
    """
    Ising representation of a QUBO.

    For binary variables:

        x_i = (1 - Z_i) / 2

    the QUBO objective

        x^T Q x + offset

    is converted to:

        constant
        + sum_i h_i Z_i
        + sum_{i<j} J_ij Z_i Z_j
    """

    linear: tuple[float, ...]
    quadratic: Mapping[
        tuple[int, int],
        float,
    ]
    constant: float

    def __post_init__(self) -> None:
        if not self.linear:
            raise QAOASolverError(
                "Ising model must contain "
                "at least one variable."
            )

        if not all(
            np.isfinite(value)
            for value in self.linear
        ):
            raise QAOASolverError(
                "Ising linear coefficients "
                "must be finite."
            )

        if not np.isfinite(
            self.constant
        ):
            raise QAOASolverError(
                "Ising constant must be finite."
            )

        for pair, value in (
            self.quadratic.items()
        ):
            if len(pair) != 2:
                raise QAOASolverError(
                    "Ising quadratic key must "
                    "contain two indices."
                )

            first, second = pair

            if first < 0 or second < 0:
                raise QAOASolverError(
                    "Ising indices cannot be negative."
                )

            if first >= second:
                raise QAOASolverError(
                    "Ising quadratic indices must "
                    "satisfy i < j."
                )

            if not np.isfinite(value):
                raise QAOASolverError(
                    "Ising quadratic coefficients "
                    "must be finite."
                )


class QAOAPortfolioSolver:
    """
    Simulator-first QAOA solver for portfolio QUBO problems.
    """

    def __init__(
        self,
        config: Optional[
            QAOASolverConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or QAOASolverConfig()
        )

    # ==========================================================
    # Public API
    # ==========================================================

    def solve(
        self,
        problem: PortfolioQUBOProblem,
    ) -> QuantumPortfolioSolution:
        """
        Solve a portfolio QUBO using QAOA on a local simulator.
        """

        self._validate_problem(
            problem
        )

        start_time = time.perf_counter()

        try:
            result = (
                self._run_qaoa(
                    problem
                )
            )

        except QAOASolverError:
            raise

        except Exception as exc:
            raise QAOASolverError(
                "QAOA execution failed: "
                f"{exc}"
            ) from exc

        runtime_seconds = (
            time.perf_counter()
            - start_time
        )

        binary_solution = (
            result[
                "binary_solution"
            ]
        )

        objective_value = float(
            problem.energy(
                binary_solution
            )
        )

        selected_assets = (
            self._selected_assets(
                problem=problem,
                binary_solution=(
                    binary_solution
                ),
            )
        )

        weights = (
            self._derive_equal_weights(
                problem=problem,
                binary_solution=(
                    binary_solution
                ),
            )
        )

        diagnostics = (
            self._constraint_diagnostics(
                problem=problem,
                binary_solution=(
                    binary_solution
                ),
            )
        )

        metadata = {
            "algorithm": "qaoa",
            "execution_mode": (
                "local_simulator"
            ),
            "simulator_first": True,
            "hardware_execution": False,
            "reps": self.config.reps,
            "shots": self.config.shots,
            "maxiter": self.config.maxiter,
            "seed": self.config.seed,
            "optimization_level": (
                self.config.optimization_level
            ),
            "variable_count": (
                problem.variable_count
            ),
            "best_probability": (
                result.get(
                    "best_probability"
                )
            ),
            "best_count": (
                result.get(
                    "best_count"
                )
            ),
            "optimizer_value": (
                result.get(
                    "optimizer_value"
                )
            ),
            "optimal_parameters": (
                result.get(
                    "optimal_parameters"
                )
            ),
            "measurement_counts": (
                result.get(
                    "measurement_counts",
                    {},
                )
            ),
            "ising_constant": (
                result.get(
                    "ising_constant"
                )
            ),
        }

        return QuantumPortfolioSolution(
            solver_type=(
                QuantumSolverType.QAOA
            ),
            backend=(
                QuantumExecutionBackend.SIMULATOR
            ),
            status=(
                QuantumSolutionStatus.COMPLETED
            ),
            binary_solution=(
                binary_solution
            ),
            objective_value=(
                objective_value
            ),
            selected_assets=(
                selected_assets
            ),
            weights=weights,
            runtime_seconds=float(
                runtime_seconds
            ),
            constraint_diagnostics=(
                diagnostics
            ),
            metadata=metadata,
        )

    # ==========================================================
    # QAOA execution
    # ==========================================================

    def _run_qaoa(
        self,
        problem: PortfolioQUBOProblem,
    ) -> dict[str, Any]:
        """
        Execute QAOA using Qiskit circuits and AerSimulator.

        Qiskit Algorithms is used for the classical optimizer,
        while Qiskit circuit primitives and Aer provide the
        simulator-first quantum workflow.
        """

        try:
            from qiskit import (
                QuantumCircuit,
                transpile,
            )

            from qiskit.circuit import (
                Parameter,
            )

            from qiskit_aer import (
                AerSimulator,
            )

            from qiskit_algorithms.optimizers import (
                COBYLA,
            )

        except ImportError as exc:
            raise QAOASolverError(
                "QAOA dependencies are not installed. "
                "Required packages: qiskit, "
                "qiskit-algorithms and qiskit-aer."
            ) from exc

        ising = self.qubo_to_ising(
            problem
        )

        qubit_count = (
            problem.variable_count
        )

        gamma_parameters = [
            Parameter(
                f"gamma_{layer}"
            )
            for layer in range(
                self.config.reps
            )
        ]

        beta_parameters = [
            Parameter(
                f"beta_{layer}"
            )
            for layer in range(
                self.config.reps
            )
        ]

        circuit = QuantumCircuit(
            qubit_count
        )

        circuit.h(
            range(
                qubit_count
            )
        )

        for layer in range(
            self.config.reps
        ):
            gamma = (
                gamma_parameters[
                    layer
                ]
            )

            beta = (
                beta_parameters[
                    layer
                ]
            )

            self._apply_cost_layer(
                circuit=circuit,
                ising=ising,
                gamma=gamma,
            )

            self._apply_mixer_layer(
                circuit=circuit,
                beta=beta,
                qubit_count=(
                    qubit_count
                ),
            )

        simulator = AerSimulator(
            seed_simulator=(
                self.config.seed
            )
        )

        transpiled = transpile(
            circuit,
            simulator,
            optimization_level=(
                self.config.optimization_level
            ),
            seed_transpiler=(
                self.config.seed
            ),
        )

        parameter_order = (
            gamma_parameters
            + beta_parameters
        )

        optimizer = COBYLA(
            maxiter=(
                self.config.maxiter
            )
        )

        rng = np.random.default_rng(
            self.config.seed
        )

        initial_point = (
            rng.uniform(
                low=0.0,
                high=np.pi,
                size=len(
                    parameter_order
                ),
            )
        )

        def objective(
            parameter_values: np.ndarray,
        ) -> float:
            bindings = {
                parameter: float(value)
                for parameter, value
                in zip(
                    parameter_order,
                    parameter_values,
                )
            }

            statevector_circuit = (
                transpiled.assign_parameters(
                    bindings,
                    inplace=False,
                )
            )

            statevector_circuit.save_statevector()

            result = simulator.run(
                statevector_circuit
            ).result()

            statevector = np.asarray(
                result.get_statevector(
                    statevector_circuit
                ),
                dtype=complex,
            )

            return (
                self._expected_qubo_energy(
                    problem=problem,
                    statevector=(
                        statevector
                    ),
                )
            )

        optimizer_result = (
            optimizer.minimize(
                fun=objective,
                x0=initial_point,
            )
        )

        optimal_parameters = np.asarray(
            optimizer_result.x,
            dtype=float,
        )

        bindings = {
            parameter: float(value)
            for parameter, value
            in zip(
                parameter_order,
                optimal_parameters,
            )
        }

        measurement_circuit = (
            transpiled.assign_parameters(
                bindings,
                inplace=False,
            )
        )

        measurement_circuit.measure_all()

        measurement_result = (
            simulator.run(
                measurement_circuit,
                shots=self.config.shots,
            ).result()
        )

        raw_counts = (
            measurement_result.get_counts()
        )

        counts = {
            str(bitstring): int(count)
            for bitstring, count
            in raw_counts.items()
        }

        if not counts:
            raise QAOASolverError(
                "QAOA produced no measurement counts."
            )

        best_state = (
            self._best_measured_state(
                problem=problem,
                counts=counts,
            )
        )

        best_bitstring = (
            self.binary_solution_to_bitstring(
                best_state
            )
        )

        best_count = int(
            counts.get(
                best_bitstring,
                0,
            )
        )

        best_probability = float(
            best_count
            / self.config.shots
        )

        return {
            "binary_solution": (
                best_state
            ),
            "best_probability": (
                best_probability
            ),
            "best_count": (
                best_count
            ),
            "optimizer_value": float(
                optimizer_result.fun
            ),
            "optimal_parameters": tuple(
                float(value)
                for value
                in optimal_parameters
            ),
            "measurement_counts": (
                counts
            ),
            "ising_constant": float(
                ising.constant
            ),
        }

    # ==========================================================
    # QUBO -> Ising
    # ==========================================================

    @staticmethod
    def qubo_to_ising(
        problem: PortfolioQUBOProblem,
    ) -> IsingQUBOModel:
        """
        Convert symmetric QUBO matrix to Ising coefficients.

        Binary-to-spin mapping:

            x_i = (1 - z_i) / 2

        with z_i in {-1, +1}.

        The QUBO representation is:

            x^T Q x + offset

        where symmetric off-diagonal Q entries are counted
        twice by x^T Q x.
        """

        if not isinstance(
            problem,
            PortfolioQUBOProblem,
        ):
            raise QAOASolverError(
                "problem must be a PortfolioQUBOProblem."
            )

        variable_count = (
            problem.variable_count
        )

        if variable_count == 0:
            raise QAOASolverError(
                "QUBO problem must contain "
                "at least one variable."
            )

        matrix = np.asarray(
            problem.qubo_matrix,
            dtype=float,
        )

        linear = np.zeros(
            variable_count,
            dtype=float,
        )

        quadratic: dict[
            tuple[int, int],
            float,
        ] = {}

        constant = float(
            problem.offset
        )

        for index in range(
            variable_count
        ):
            diagonal = float(
                matrix[
                    index,
                    index,
                ]
            )

            constant += (
                diagonal
                / 2.0
            )

            linear[index] -= (
                diagonal
                / 2.0
            )

        for first in range(
            variable_count
        ):
            for second in range(
                first + 1,
                variable_count,
            ):
                symmetric_coefficient = float(
                    matrix[
                        first,
                        second,
                    ]
                    + matrix[
                        second,
                        first,
                    ]
                )

                if abs(
                    symmetric_coefficient
                ) <= 1e-15:
                    continue

                constant += (
                    symmetric_coefficient
                    / 4.0
                )

                linear[first] -= (
                    symmetric_coefficient
                    / 4.0
                )

                linear[second] -= (
                    symmetric_coefficient
                    / 4.0
                )

                quadratic[
                    (
                        first,
                        second,
                    )
                ] = (
                    symmetric_coefficient
                    / 4.0
                )

        return IsingQUBOModel(
            linear=tuple(
                float(value)
                for value
                in linear
            ),
            quadratic=quadratic,
            constant=float(
                constant
            ),
        )

    @staticmethod
    def ising_energy(
        *,
        model: IsingQUBOModel,
        binary_vector: Sequence[int],
    ) -> float:
        """
        Evaluate Ising energy using a binary solution.

        Binary values are converted using:

            z_i = 1 - 2*x_i
        """

        vector = np.asarray(
            binary_vector,
            dtype=float,
        )

        if vector.ndim != 1:
            raise QAOASolverError(
                "binary_vector must be one-dimensional."
            )

        if vector.shape[0] != len(
            model.linear
        ):
            raise QAOASolverError(
                "binary_vector length must match "
                "the Ising model."
            )

        if not np.all(
            np.logical_or(
                vector == 0.0,
                vector == 1.0,
            )
        ):
            raise QAOASolverError(
                "binary_vector must contain "
                "only 0 or 1."
            )

        spins = (
            1.0
            - 2.0
            * vector
        )

        energy = float(
            model.constant
        )

        for index, coefficient in enumerate(
            model.linear
        ):
            energy += (
                coefficient
                * spins[index]
            )

        for (
            first,
            second,
        ), coefficient in (
            model.quadratic.items()
        ):
            energy += (
                coefficient
                * spins[first]
                * spins[second]
            )

        return float(
            energy
        )

    # ==========================================================
    # Circuit construction
    # ==========================================================

    @staticmethod
    def _apply_cost_layer(
        *,
        circuit: Any,
        ising: IsingQUBOModel,
        gamma: Any,
    ) -> None:
        """
        Apply exp(-i gamma H_cost).

        Constant Ising energy does not affect measurement
        probabilities and is therefore omitted from the
        circuit.
        """

        for index, coefficient in enumerate(
            ising.linear
        ):
            if abs(
                coefficient
            ) <= 1e-15:
                continue

            circuit.rz(
                2.0
                * coefficient
                * gamma,
                index,
            )

        for (
            first,
            second,
        ), coefficient in (
            ising.quadratic.items()
        ):
            if abs(
                coefficient
            ) <= 1e-15:
                continue

            circuit.cx(
                first,
                second,
            )

            circuit.rz(
                2.0
                * coefficient
                * gamma,
                second,
            )

            circuit.cx(
                first,
                second,
            )

    @staticmethod
    def _apply_mixer_layer(
        *,
        circuit: Any,
        beta: Any,
        qubit_count: int,
    ) -> None:
        """
        Standard X mixer:

            exp(-i beta sum X_i)
        """

        for qubit in range(
            qubit_count
        ):
            circuit.rx(
                2.0 * beta,
                qubit,
            )

    # ==========================================================
    # State evaluation
    # ==========================================================

    @staticmethod
    def _expected_qubo_energy(
        *,
        problem: PortfolioQUBOProblem,
        statevector: np.ndarray,
    ) -> float:
        """
        Calculate expected QUBO energy directly from a
        statevector probability distribution.
        """

        amplitudes = np.asarray(
            statevector,
            dtype=complex,
        )

        expected_size = (
            2 ** problem.variable_count
        )

        if amplitudes.ndim != 1:
            raise QAOASolverError(
                "statevector must be one-dimensional."
            )

        if amplitudes.shape[0] != expected_size:
            raise QAOASolverError(
                "statevector size does not match "
                "the QUBO variable count."
            )

        probabilities = (
            np.abs(
                amplitudes
            )
            ** 2
        )

        expected_energy = 0.0

        for basis_index, probability in enumerate(
            probabilities
        ):
            if probability <= 1e-18:
                continue

            binary_solution = (
                QAOAPortfolioSolver
                .basis_index_to_binary_solution(
                    basis_index=(
                        basis_index
                    ),
                    variable_count=(
                        problem.variable_count
                    ),
                )
            )

            energy = problem.energy(
                binary_solution
            )

            expected_energy += (
                float(
                    probability
                )
                * energy
            )

        return float(
            expected_energy
        )

    @staticmethod
    def _best_measured_state(
        *,
        problem: PortfolioQUBOProblem,
        counts: Mapping[str, int],
    ) -> tuple[int, ...]:
        """
        Select the lowest-energy state actually observed.

        Frequency is used as a deterministic tie-breaker.
        """

        if not counts:
            raise QAOASolverError(
                "counts cannot be empty."
            )

        best_state: Optional[
            tuple[int, ...]
        ] = None

        best_energy = float("inf")
        best_count = -1

        for bitstring, count in (
            counts.items()
        ):
            if count < 0:
                raise QAOASolverError(
                    "measurement counts cannot "
                    "be negative."
                )

            state = (
                QAOAPortfolioSolver
                .bitstring_to_binary_solution(
                    bitstring=bitstring,
                    variable_count=(
                        problem.variable_count
                    ),
                )
            )

            energy = float(
                problem.energy(
                    state
                )
            )

            if (
                energy
                < best_energy
                - 1e-12
            ):
                best_state = state
                best_energy = energy
                best_count = int(
                    count
                )

            elif (
                best_state is not None
                and abs(
                    energy
                    - best_energy
                )
                <= 1e-12
                and count > best_count
            ):
                best_state = state
                best_count = int(
                    count
                )

        if best_state is None:
            raise QAOASolverError(
                "Unable to decode a measured state."
            )

        return best_state

    # ==========================================================
    # Bit ordering
    # ==========================================================

    @staticmethod
    def basis_index_to_binary_solution(
        *,
        basis_index: int,
        variable_count: int,
    ) -> tuple[int, ...]:
        """
        Convert Qiskit's little-endian computational basis
        index into variable order:

            (x_0, x_1, ..., x_n)
        """

        if basis_index < 0:
            raise QAOASolverError(
                "basis_index cannot be negative."
            )

        if variable_count <= 0:
            raise QAOASolverError(
                "variable_count must be positive."
            )

        if basis_index >= (
            2 ** variable_count
        ):
            raise QAOASolverError(
                "basis_index exceeds the state space."
            )

        return tuple(
            int(
                (
                    basis_index
                    >> index
                )
                & 1
            )
            for index in range(
                variable_count
            )
        )

    @staticmethod
    def bitstring_to_binary_solution(
        *,
        bitstring: str,
        variable_count: int,
    ) -> tuple[int, ...]:
        """
        Convert a Qiskit measurement bitstring to QUBO
        variable order.

        Qiskit prints classical bits high-to-low, so the
        displayed bitstring is reversed.
        """

        if variable_count <= 0:
            raise QAOASolverError(
                "variable_count must be positive."
            )

        cleaned = (
            bitstring
            .replace(
                " ",
                "",
            )
        )

        if len(
            cleaned
        ) != variable_count:
            raise QAOASolverError(
                "bitstring length must match "
                "variable_count."
            )

        if any(
            character not in (
                "0",
                "1",
            )
            for character in cleaned
        ):
            raise QAOASolverError(
                "bitstring must contain "
                "only 0 or 1."
            )

        return tuple(
            int(character)
            for character
            in reversed(
                cleaned
            )
        )

    @staticmethod
    def binary_solution_to_bitstring(
        binary_solution: Sequence[int],
    ) -> str:
        """
        Convert QUBO variable order to Qiskit's displayed
        measurement bitstring.
        """

        values = tuple(
            int(value)
            for value
            in binary_solution
        )

        if not values:
            raise QAOASolverError(
                "binary_solution cannot be empty."
            )

        if any(
            value not in (
                0,
                1,
            )
            for value
            in values
        ):
            raise QAOASolverError(
                "binary_solution must contain "
                "only 0 or 1."
            )

        return "".join(
            str(value)
            for value
            in reversed(
                values
            )
        )

    # ==========================================================
    # Portfolio decoding
    # ==========================================================

    @staticmethod
    def _selected_assets(
        *,
        problem: PortfolioQUBOProblem,
        binary_solution: Sequence[int],
    ) -> tuple[str, ...]:
        selected: list[str] = []

        for variable, value in zip(
            problem.variables,
            binary_solution,
        ):
            if value != 1:
                continue

            if variable.asset_symbol is None:
                continue

            selected.append(
                variable.asset_symbol
            )

        return tuple(
            selected
        )

    @staticmethod
    def _derive_equal_weights(
        *,
        problem: PortfolioQUBOProblem,
        binary_solution: Sequence[int],
    ) -> dict[str, float]:
        selected_assets = (
            QAOAPortfolioSolver
            ._selected_assets(
                problem=problem,
                binary_solution=(
                    binary_solution
                ),
            )
        )

        weights = {
            symbol: 0.0
            for symbol
            in problem.asset_symbols
        }

        if not selected_assets:
            return weights

        equal_weight = (
            1.0
            / len(
                selected_assets
            )
        )

        for symbol in selected_assets:
            weights[symbol] = float(
                equal_weight
            )

        return weights

    @staticmethod
    def _constraint_diagnostics(
        *,
        problem: PortfolioQUBOProblem,
        binary_solution: Sequence[int],
    ) -> tuple[
        QUBOConstraintDiagnostic,
        ...,
    ]:
        diagnostics: list[
            QUBOConstraintDiagnostic
        ] = []

        if (
            problem.target_cardinality
            is not None
        ):
            selected_count = int(
                sum(
                    int(value)
                    for value
                    in binary_solution
                )
            )

            difference = (
                selected_count
                - problem.target_cardinality
            )

            violation = float(
                abs(
                    difference
                )
            )

            penalty = float(
                problem.budget_penalty
                * difference
                * difference
            )

            diagnostics.append(
                QUBOConstraintDiagnostic(
                    name="cardinality",
                    satisfied=(
                        violation == 0.0
                    ),
                    violation=violation,
                    penalty=penalty,
                    metadata={
                        "selected_count": (
                            selected_count
                        ),
                        "target_cardinality": (
                            problem.target_cardinality
                        ),
                    },
                )
            )

        return tuple(
            diagnostics
        )

    # ==========================================================
    # Validation
    # ==========================================================

    @staticmethod
    def _validate_problem(
        problem: PortfolioQUBOProblem,
    ) -> None:
        if not isinstance(
            problem,
            PortfolioQUBOProblem,
        ):
            raise QAOASolverError(
                "problem must be a PortfolioQUBOProblem."
            )

        if problem.variable_count <= 0:
            raise QAOASolverError(
                "QUBO problem must contain "
                "at least one variable."
            )


__all__ = [
    "IsingQUBOModel",
    "QAOAPortfolioSolver",
    "QAOASolverConfig",
    "QAOASolverError",
]