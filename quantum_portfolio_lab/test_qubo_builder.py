"""
Tests for the v5.0.8 Quantum Portfolio Lab QUBO builder.

Coverage:

- request validation
- QUBO variable construction
- risk-return objective formulation
- exact-cardinality penalty
- QUBO matrix symmetry
- QUBO offset
- objective evaluation
- constraint violation evaluation
- exhaustive mathematical equivalence between:
      original objective + penalty
  and:
      x^T Q x + offset
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from quantum_portfolio_lab.qubo_builder import (
    PortfolioQUBOBuilder,
    PortfolioSelectionQUBORequest,
    QUBOBuilderError,
)
from quantum_portfolio_lab.qubo_models import (
    PortfolioQUBOProblem,
)


def build_request(
    *,
    target_cardinality: int = 2,
    risk_aversion: float = 1.5,
    return_weight: float = 2.0,
    budget_penalty: float = 10.0,
) -> PortfolioSelectionQUBORequest:
    return PortfolioSelectionQUBORequest(
        asset_symbols=(
            "SPY",
            "GLD",
            "TLT",
        ),
        expected_returns=np.array(
            [
                0.08,
                0.04,
                0.03,
            ],
            dtype=float,
        ),
        covariance_matrix=np.array(
            [
                [0.040, 0.010, -0.005],
                [0.010, 0.020, 0.002],
                [-0.005, 0.002, 0.015],
            ],
            dtype=float,
        ),
        target_cardinality=target_cardinality,
        risk_aversion=risk_aversion,
        return_weight=return_weight,
        budget_penalty=budget_penalty,
        constraint_penalty=5.0,
        metadata={
            "dataset": "unit_test",
        },
    )


def test_request_is_created() -> None:
    request = build_request()

    assert request.asset_symbols == (
        "SPY",
        "GLD",
        "TLT",
    )

    assert request.target_cardinality == 2

    assert request.risk_aversion == pytest.approx(
        1.5
    )

    assert request.return_weight == pytest.approx(
        2.0
    )


def test_request_strips_asset_symbols() -> None:
    request = PortfolioSelectionQUBORequest(
        asset_symbols=(
            " SPY ",
            " GLD ",
        ),
        expected_returns=np.array(
            [0.08, 0.04]
        ),
        covariance_matrix=np.array(
            [
                [0.04, 0.01],
                [0.01, 0.02],
            ]
        ),
        target_cardinality=1,
    )

    assert request.asset_symbols == (
        "SPY",
        "GLD",
    )


def test_empty_asset_symbols_raise_error() -> None:
    with pytest.raises(
        QUBOBuilderError,
        match="asset_symbols cannot be empty",
    ):
        PortfolioSelectionQUBORequest(
            asset_symbols=(),
            expected_returns=np.array([]),
            covariance_matrix=np.empty(
                (0, 0)
            ),
            target_cardinality=1,
        )


def test_empty_asset_symbol_raises_error() -> None:
    with pytest.raises(
        QUBOBuilderError,
        match="non-empty string",
    ):
        PortfolioSelectionQUBORequest(
            asset_symbols=(
                "SPY",
                "",
            ),
            expected_returns=np.array(
                [0.08, 0.04]
            ),
            covariance_matrix=np.eye(2),
            target_cardinality=1,
        )


def test_duplicate_asset_symbols_raise_error() -> None:
    with pytest.raises(
        QUBOBuilderError,
        match="must be unique",
    ):
        PortfolioSelectionQUBORequest(
            asset_symbols=(
                "SPY",
                "SPY",
            ),
            expected_returns=np.array(
                [0.08, 0.04]
            ),
            covariance_matrix=np.eye(2),
            target_cardinality=1,
        )


def test_expected_returns_must_be_one_dimensional() -> None:
    with pytest.raises(
        QUBOBuilderError,
        match="one-dimensional",
    ):
        PortfolioSelectionQUBORequest(
            asset_symbols=(
                "SPY",
                "GLD",
            ),
            expected_returns=np.array(
                [
                    [0.08, 0.04],
                ]
            ),
            covariance_matrix=np.eye(2),
            target_cardinality=1,
        )


def test_expected_returns_length_must_match_assets() -> None:
    with pytest.raises(
        QUBOBuilderError,
        match="length must match",
    ):
        PortfolioSelectionQUBORequest(
            asset_symbols=(
                "SPY",
                "GLD",
            ),
            expected_returns=np.array(
                [0.08]
            ),
            covariance_matrix=np.eye(2),
            target_cardinality=1,
        )


def test_expected_returns_must_be_finite() -> None:
    with pytest.raises(
        QUBOBuilderError,
        match="finite",
    ):
        PortfolioSelectionQUBORequest(
            asset_symbols=(
                "SPY",
                "GLD",
            ),
            expected_returns=np.array(
                [
                    0.08,
                    np.nan,
                ]
            ),
            covariance_matrix=np.eye(2),
            target_cardinality=1,
        )


def test_covariance_must_be_two_dimensional() -> None:
    with pytest.raises(
        QUBOBuilderError,
        match="two-dimensional",
    ):
        PortfolioSelectionQUBORequest(
            asset_symbols=(
                "SPY",
                "GLD",
            ),
            expected_returns=np.array(
                [0.08, 0.04]
            ),
            covariance_matrix=np.array(
                [0.04, 0.02]
            ),
            target_cardinality=1,
        )


def test_covariance_dimension_must_match_assets() -> None:
    with pytest.raises(
        QUBOBuilderError,
        match="dimension must match",
    ):
        PortfolioSelectionQUBORequest(
            asset_symbols=(
                "SPY",
                "GLD",
            ),
            expected_returns=np.array(
                [0.08, 0.04]
            ),
            covariance_matrix=np.eye(3),
            target_cardinality=1,
        )


def test_covariance_must_be_finite() -> None:
    covariance = np.array(
        [
            [0.04, np.nan],
            [np.nan, 0.02],
        ]
    )

    with pytest.raises(
        QUBOBuilderError,
        match="finite",
    ):
        PortfolioSelectionQUBORequest(
            asset_symbols=(
                "SPY",
                "GLD",
            ),
            expected_returns=np.array(
                [0.08, 0.04]
            ),
            covariance_matrix=covariance,
            target_cardinality=1,
        )


def test_covariance_must_be_symmetric() -> None:
    covariance = np.array(
        [
            [0.04, 0.02],
            [0.01, 0.02],
        ]
    )

    with pytest.raises(
        QUBOBuilderError,
        match="symmetric",
    ):
        PortfolioSelectionQUBORequest(
            asset_symbols=(
                "SPY",
                "GLD",
            ),
            expected_returns=np.array(
                [0.08, 0.04]
            ),
            covariance_matrix=covariance,
            target_cardinality=1,
        )


def test_target_cardinality_must_be_positive() -> None:
    with pytest.raises(
        QUBOBuilderError,
        match="must be positive",
    ):
        build_request(
            target_cardinality=0
        )


def test_target_cardinality_cannot_exceed_assets() -> None:
    with pytest.raises(
        QUBOBuilderError,
        match="cannot exceed",
    ):
        build_request(
            target_cardinality=4
        )


@pytest.mark.parametrize(
    "field_name,value",
    [
        (
            "risk_aversion",
            -1.0,
        ),
        (
            "return_weight",
            -1.0,
        ),
        (
            "budget_penalty",
            -1.0,
        ),
        (
            "constraint_penalty",
            -1.0,
        ),
    ],
)
def test_negative_parameters_raise_error(
    field_name: str,
    value: float,
) -> None:
    kwargs = {
        "asset_symbols": (
            "SPY",
            "GLD",
        ),
        "expected_returns": np.array(
            [0.08, 0.04]
        ),
        "covariance_matrix": np.eye(2),
        "target_cardinality": 1,
        field_name: value,
    }

    with pytest.raises(
        QUBOBuilderError,
        match=field_name,
    ):
        PortfolioSelectionQUBORequest(
            **kwargs
        )


def test_builder_returns_qubo_problem() -> None:
    builder = PortfolioQUBOBuilder()

    problem = builder.build(
        build_request()
    )

    assert isinstance(
        problem,
        PortfolioQUBOProblem,
    )


def test_builder_rejects_invalid_request_type() -> None:
    builder = PortfolioQUBOBuilder()

    with pytest.raises(
        QUBOBuilderError,
        match="PortfolioSelectionQUBORequest",
    ):
        builder.build(
            "invalid"
        )  # type: ignore[arg-type]


def test_builder_creates_one_variable_per_asset() -> None:
    builder = PortfolioQUBOBuilder()

    problem = builder.build(
        build_request()
    )

    assert problem.variable_count == 3

    assert tuple(
        variable.name
        for variable in problem.variables
    ) == (
        "x_SPY",
        "x_GLD",
        "x_TLT",
    )


def test_variable_indices_are_contiguous() -> None:
    builder = PortfolioQUBOBuilder()

    problem = builder.build(
        build_request()
    )

    assert tuple(
        variable.index
        for variable in problem.variables
    ) == (
        0,
        1,
        2,
    )


def test_variable_asset_symbols_are_preserved() -> None:
    builder = PortfolioQUBOBuilder()

    problem = builder.build(
        build_request()
    )

    assert tuple(
        variable.asset_symbol
        for variable in problem.variables
    ) == (
        "SPY",
        "GLD",
        "TLT",
    )


def test_qubo_matrix_is_symmetric() -> None:
    builder = PortfolioQUBOBuilder()

    problem = builder.build(
        build_request()
    )

    assert np.allclose(
        problem.qubo_matrix,
        problem.qubo_matrix.T,
    )


def test_problem_preserves_expected_returns() -> None:
    request = build_request()

    problem = PortfolioQUBOBuilder().build(
        request
    )

    assert np.allclose(
        problem.expected_returns,
        request.expected_returns,
    )


def test_problem_preserves_covariance_matrix() -> None:
    request = build_request()

    problem = PortfolioQUBOBuilder().build(
        request
    )

    assert np.allclose(
        problem.covariance_matrix,
        request.covariance_matrix,
    )


def test_problem_preserves_parameters() -> None:
    request = build_request()

    problem = PortfolioQUBOBuilder().build(
        request
    )

    assert problem.risk_aversion == pytest.approx(
        request.risk_aversion
    )

    assert problem.return_weight == pytest.approx(
        request.return_weight
    )

    assert problem.budget_penalty == pytest.approx(
        request.budget_penalty
    )

    assert (
        problem.target_cardinality
        == request.target_cardinality
    )


def test_problem_metadata_contains_formulation() -> None:
    problem = PortfolioQUBOBuilder().build(
        build_request()
    )

    assert (
        problem.metadata["formulation"]
        == "binary_portfolio_selection"
    )

    assert (
        problem.metadata["objective"]
        == "risk_minus_return"
    )

    assert (
        problem.metadata[
            "cardinality_constraint"
        ]
        == "exact"
    )


def test_request_metadata_is_preserved() -> None:
    problem = PortfolioQUBOBuilder().build(
        build_request()
    )

    assert (
        problem.metadata["dataset"]
        == "unit_test"
    )


def test_base_objective_matches_manual_calculation() -> None:
    request = build_request()

    builder = PortfolioQUBOBuilder()

    vector = np.array(
        [1.0, 0.0, 1.0]
    )

    risk = float(
        vector
        @ request.covariance_matrix
        @ vector
    )

    expected_return = float(
        request.expected_returns
        @ vector
    )

    expected = (
        request.risk_aversion
        * risk
        - request.return_weight
        * expected_return
    )

    actual = builder.evaluate_base_objective(
        request=request,
        binary_vector=vector,
    )

    assert actual == pytest.approx(
        expected
    )


def test_cardinality_violation_zero_when_feasible() -> None:
    request = build_request(
        target_cardinality=2
    )

    builder = PortfolioQUBOBuilder()

    violation = (
        builder.evaluate_cardinality_violation(
            request=request,
            binary_vector=[
                1,
                0,
                1,
            ],
        )
    )

    assert violation == pytest.approx(
        0.0
    )


def test_cardinality_violation_when_too_few_assets() -> None:
    request = build_request(
        target_cardinality=2
    )

    builder = PortfolioQUBOBuilder()

    violation = (
        builder.evaluate_cardinality_violation(
            request=request,
            binary_vector=[
                1,
                0,
                0,
            ],
        )
    )

    assert violation == pytest.approx(
        1.0
    )


def test_cardinality_violation_when_too_many_assets() -> None:
    request = build_request(
        target_cardinality=1
    )

    builder = PortfolioQUBOBuilder()

    violation = (
        builder.evaluate_cardinality_violation(
            request=request,
            binary_vector=[
                1,
                1,
                1,
            ],
        )
    )

    assert violation == pytest.approx(
        2.0
    )


def test_cardinality_penalty_zero_when_feasible() -> None:
    request = build_request(
        target_cardinality=2,
        budget_penalty=10.0,
    )

    builder = PortfolioQUBOBuilder()

    penalty = (
        builder.evaluate_cardinality_penalty(
            request=request,
            binary_vector=[
                1,
                0,
                1,
            ],
        )
    )

    assert penalty == pytest.approx(
        0.0
    )


def test_cardinality_penalty_is_squared_violation() -> None:
    request = build_request(
        target_cardinality=1,
        budget_penalty=10.0,
    )

    builder = PortfolioQUBOBuilder()

    penalty = (
        builder.evaluate_cardinality_penalty(
            request=request,
            binary_vector=[
                1,
                1,
                1,
            ],
        )
    )

    assert penalty == pytest.approx(
        40.0
    )


def test_zero_vector_cardinality_penalty() -> None:
    request = build_request(
        target_cardinality=2,
        budget_penalty=10.0,
    )

    builder = PortfolioQUBOBuilder()

    penalty = (
        builder.evaluate_cardinality_penalty(
            request=request,
            binary_vector=[
                0,
                0,
                0,
            ],
        )
    )

    assert penalty == pytest.approx(
        40.0
    )


def test_qubo_offset_matches_cardinality_constant() -> None:
    request = build_request(
        target_cardinality=2,
        budget_penalty=10.0,
    )

    problem = PortfolioQUBOBuilder().build(
        request
    )

    expected_offset = (
        request.budget_penalty
        * request.target_cardinality**2
    )

    assert problem.offset == pytest.approx(
        expected_offset
    )


def test_zero_vector_energy_equals_offset() -> None:
    problem = PortfolioQUBOBuilder().build(
        build_request()
    )

    assert problem.energy(
        [0, 0, 0]
    ) == pytest.approx(
        problem.offset
    )


def test_total_objective_matches_problem_energy() -> None:
    request = build_request()

    builder = PortfolioQUBOBuilder()

    problem = builder.build(
        request
    )

    vector = [
        1,
        0,
        1,
    ]

    expected = (
        builder.evaluate_total_objective(
            request=request,
            binary_vector=vector,
        )
    )

    actual = problem.energy(
        vector
    )

    assert actual == pytest.approx(
        expected
    )


def test_qubo_energy_matches_base_objective_for_feasible_solution() -> None:
    request = build_request(
        target_cardinality=2
    )

    builder = PortfolioQUBOBuilder()

    problem = builder.build(
        request
    )

    vector = [
        1,
        1,
        0,
    ]

    qubo_energy = problem.energy(
        vector
    )

    base_objective = (
        builder.evaluate_base_objective(
            request=request,
            binary_vector=vector,
        )
    )

    assert qubo_energy == pytest.approx(
        base_objective
    )


def test_qubo_energy_includes_penalty_for_infeasible_solution() -> None:
    request = build_request(
        target_cardinality=2,
        budget_penalty=10.0,
    )

    builder = PortfolioQUBOBuilder()

    problem = builder.build(
        request
    )

    vector = [
        1,
        0,
        0,
    ]

    base = builder.evaluate_base_objective(
        request=request,
        binary_vector=vector,
    )

    penalty = (
        builder.evaluate_cardinality_penalty(
            request=request,
            binary_vector=vector,
        )
    )

    assert problem.energy(
        vector
    ) == pytest.approx(
        base + penalty
    )


def test_all_binary_states_match_original_formulation() -> None:
    """
    Exhaustively verify every binary state.

    This is the key mathematical equivalence test:

        QUBO energy
            ==
        original risk-return objective
            +
        exact-cardinality penalty
    """

    request = build_request(
        target_cardinality=2,
        risk_aversion=1.7,
        return_weight=2.3,
        budget_penalty=13.0,
    )

    builder = PortfolioQUBOBuilder()

    problem = builder.build(
        request
    )

    for state in itertools.product(
        (0, 1),
        repeat=len(
            request.asset_symbols
        ),
    ):
        qubo_energy = problem.energy(
            state
        )

        original_objective = (
            builder.evaluate_total_objective(
                request=request,
                binary_vector=state,
            )
        )

        assert qubo_energy == pytest.approx(
            original_objective,
            abs=1e-10,
        )


def test_all_feasible_states_have_zero_cardinality_penalty() -> None:
    request = build_request(
        target_cardinality=2
    )

    builder = PortfolioQUBOBuilder()

    for state in itertools.product(
        (0, 1),
        repeat=3,
    ):
        if sum(state) == 2:
            penalty = (
                builder.evaluate_cardinality_penalty(
                    request=request,
                    binary_vector=state,
                )
            )

            assert penalty == pytest.approx(
                0.0
            )


def test_binary_vector_wrong_length_raises_error() -> None:
    request = build_request()

    builder = PortfolioQUBOBuilder()

    with pytest.raises(
        QUBOBuilderError,
        match="length must match",
    ):
        builder.evaluate_total_objective(
            request=request,
            binary_vector=[
                1,
                0,
            ],
        )


def test_nonbinary_vector_raises_error() -> None:
    request = build_request()

    builder = PortfolioQUBOBuilder()

    with pytest.raises(
        QUBOBuilderError,
        match="only 0 or 1",
    ):
        builder.evaluate_total_objective(
            request=request,
            binary_vector=[
                1,
                2,
                0,
            ],
        )


def test_multidimensional_binary_vector_raises_error() -> None:
    request = build_request()

    builder = PortfolioQUBOBuilder()

    with pytest.raises(
        QUBOBuilderError,
        match="one-dimensional",
    ):
        builder.evaluate_total_objective(
            request=request,
            binary_vector=[
                [1, 0, 1],
            ],
        )


def test_zero_risk_aversion_produces_return_only_objective() -> None:
    request = build_request(
        target_cardinality=1,
        risk_aversion=0.0,
        return_weight=1.0,
    )

    builder = PortfolioQUBOBuilder()

    value = builder.evaluate_base_objective(
        request=request,
        binary_vector=[
            1,
            0,
            0,
        ],
    )

    assert value == pytest.approx(
        -0.08
    )


def test_zero_return_weight_produces_risk_only_objective() -> None:
    request = build_request(
        target_cardinality=1,
        risk_aversion=1.0,
        return_weight=0.0,
    )

    builder = PortfolioQUBOBuilder()

    value = builder.evaluate_base_objective(
        request=request,
        binary_vector=[
            1,
            0,
            0,
        ],
    )

    assert value == pytest.approx(
        0.040
    )


def test_zero_budget_penalty_removes_cardinality_penalty() -> None:
    request = build_request(
        target_cardinality=2,
        budget_penalty=0.0,
    )

    builder = PortfolioQUBOBuilder()

    problem = builder.build(
        request
    )

    vector = [
        1,
        0,
        0,
    ]

    assert problem.energy(
        vector
    ) == pytest.approx(
        builder.evaluate_base_objective(
            request=request,
            binary_vector=vector,
        )
    )


def test_cardinality_matrix_diagonal_is_correct() -> None:
    matrix, offset = (
        PortfolioQUBOBuilder
        ._build_cardinality_penalty(
            variable_count=3,
            target_cardinality=2,
            penalty=10.0,
        )
    )

    expected_diagonal = (
        10.0
        * (
            1.0
            - 2.0 * 2
        )
    )

    assert np.allclose(
        np.diag(matrix),
        np.array(
            [
                expected_diagonal,
                expected_diagonal,
                expected_diagonal,
            ]
        ),
    )

    assert offset == pytest.approx(
        40.0
    )


def test_cardinality_matrix_off_diagonal_is_correct() -> None:
    matrix, _ = (
        PortfolioQUBOBuilder
        ._build_cardinality_penalty(
            variable_count=3,
            target_cardinality=2,
            penalty=10.0,
        )
    )

    assert matrix[0, 1] == pytest.approx(
        10.0
    )

    assert matrix[1, 0] == pytest.approx(
        10.0
    )

    assert matrix[0, 2] == pytest.approx(
        10.0
    )

    assert matrix[2, 0] == pytest.approx(
        10.0
    )

    assert matrix[1, 2] == pytest.approx(
        10.0
    )

    assert matrix[2, 1] == pytest.approx(
        10.0
    )