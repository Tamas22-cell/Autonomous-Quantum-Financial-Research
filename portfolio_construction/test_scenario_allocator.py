"""
Tests for the v5.0.7 Scenario-Aware Portfolio Allocation Engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from portfolio_construction.models import (
    AssetDefinition,
    ConstraintType,
    CovarianceMatrix,
    ExpectedReturnEstimate,
    MarketScenario,
    OptimizationMethod,
    OptimizationRequest,
    OptimizationStatus,
    PortfolioAllocation,
    PortfolioConstraint,
)

from portfolio_construction.scenario_allocator import (
    ScenarioAllocatorError,
    ScenarioAwareAllocator,
)


# ==============================================================
# Helpers
# ==============================================================


def build_assets() -> list[AssetDefinition]:
    return [
        AssetDefinition(
            symbol="SPY",
            asset_class="equity",
            sector="broad_market",
        ),
        AssetDefinition(
            symbol="NVDA",
            asset_class="equity",
            sector="technology",
        ),
        AssetDefinition(
            symbol="GLD",
            asset_class="commodity",
            sector="gold",
        ),
    ]


def build_expected_returns() -> list[ExpectedReturnEstimate]:
    return [
        ExpectedReturnEstimate(
            symbol="SPY",
            expected_return=0.08,
        ),
        ExpectedReturnEstimate(
            symbol="NVDA",
            expected_return=0.15,
        ),
        ExpectedReturnEstimate(
            symbol="GLD",
            expected_return=0.05,
        ),
    ]


def build_covariance() -> CovarianceMatrix:
    return CovarianceMatrix(
        symbols=[
            "SPY",
            "NVDA",
            "GLD",
        ],
        matrix=[
            [
                0.0400,
                0.0300,
                0.0050,
            ],
            [
                0.0300,
                0.0900,
                0.0020,
            ],
            [
                0.0050,
                0.0020,
                0.0225,
            ],
        ],
        annualized=True,
    )


def build_scenarios() -> list[MarketScenario]:
    return [
        MarketScenario(
            scenario_id="bull",
            name="Bull Market",
            probability=0.30,
            regime="risk_on",
            asset_returns={
                "SPY": 0.15,
                "NVDA": 0.30,
                "GLD": -0.03,
            },
        ),
        MarketScenario(
            scenario_id="base",
            name="Base Case",
            probability=0.50,
            regime="neutral",
            asset_returns={
                "SPY": 0.07,
                "NVDA": 0.12,
                "GLD": 0.04,
            },
        ),
        MarketScenario(
            scenario_id="bear",
            name="Bear Market",
            probability=0.20,
            regime="risk_off",
            asset_returns={
                "SPY": -0.20,
                "NVDA": -0.35,
                "GLD": 0.15,
            },
        ),
    ]


def build_equal_probability_scenarios() -> list[MarketScenario]:
    return [
        MarketScenario(
            scenario_id="scenario_1",
            name="Scenario 1",
            probability=None,
            regime="risk_on",
            asset_returns={
                "SPY": 0.10,
                "NVDA": 0.20,
                "GLD": -0.02,
            },
        ),
        MarketScenario(
            scenario_id="scenario_2",
            name="Scenario 2",
            probability=None,
            regime="neutral",
            asset_returns={
                "SPY": 0.05,
                "NVDA": 0.08,
                "GLD": 0.04,
            },
        ),
        MarketScenario(
            scenario_id="scenario_3",
            name="Scenario 3",
            probability=None,
            regime="risk_off",
            asset_returns={
                "SPY": -0.10,
                "NVDA": -0.20,
                "GLD": 0.12,
            },
        ),
    ]


def build_request(
    *,
    scenarios: list[MarketScenario] | None = None,
    constraints: list[PortfolioConstraint] | None = None,
    current_allocation: PortfolioAllocation | None = None,
    method: OptimizationMethod = (
        OptimizationMethod.SCENARIO_AWARE
    ),
    target_return: float | None = None,
) -> OptimizationRequest:
    if scenarios is None:
        scenarios = build_scenarios()

    return OptimizationRequest(
        request_id="scenario_allocator_test",
        method=method,
        assets=build_assets(),
        expected_returns=build_expected_returns(),
        covariance=build_covariance(),
        constraints=constraints or [],
        scenarios=scenarios,
        current_allocation=current_allocation,
        risk_free_rate=0.02,
        target_return=target_return,
    )


# ==============================================================
# Scenario probability handling
# ==============================================================


def test_scenario_probabilities_are_normalized() -> None:
    scenarios = [
        MarketScenario(
            scenario_id="one",
            name="One",
            probability=3.0 / 10.0,
            asset_returns={
                "SPY": 0.10,
                "NVDA": 0.20,
                "GLD": 0.00,
            },
        ),
        MarketScenario(
            scenario_id="two",
            name="Two",
            probability=7.0 / 10.0,
            asset_returns={
                "SPY": -0.05,
                "NVDA": -0.10,
                "GLD": 0.10,
            },
        ),
    ]

    probabilities = (
        ScenarioAwareAllocator._scenario_probabilities(
            scenarios
        )
    )

    assert np.sum(
        probabilities
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )

    assert probabilities[0] == pytest.approx(
        0.30
    )

    assert probabilities[1] == pytest.approx(
        0.70
    )


def test_missing_probabilities_use_equal_weights() -> None:
    scenarios = (
        build_equal_probability_scenarios()
    )

    probabilities = (
        ScenarioAwareAllocator._scenario_probabilities(
            scenarios
        )
    )

    assert probabilities[0] == pytest.approx(
        1.0 / 3.0
    )

    assert probabilities[1] == pytest.approx(
        1.0 / 3.0
    )

    assert probabilities[2] == pytest.approx(
        1.0 / 3.0
    )


# ==============================================================
# Scenario matrix
# ==============================================================


def test_scenario_return_matrix_has_correct_shape() -> None:
    scenarios = build_scenarios()

    matrix = (
        ScenarioAwareAllocator._scenario_return_matrix(
            scenarios=scenarios,
            symbols=[
                "SPY",
                "NVDA",
                "GLD",
            ],
        )
    )

    assert matrix.shape == (
        3,
        3,
    )


def test_scenario_return_matrix_preserves_asset_order() -> None:
    scenarios = build_scenarios()

    matrix = (
        ScenarioAwareAllocator._scenario_return_matrix(
            scenarios=scenarios,
            symbols=[
                "SPY",
                "NVDA",
                "GLD",
            ],
        )
    )

    assert matrix[0, 0] == pytest.approx(
        0.15
    )

    assert matrix[0, 1] == pytest.approx(
        0.30
    )

    assert matrix[0, 2] == pytest.approx(
        -0.03
    )


# ==============================================================
# Scenario expected returns
# ==============================================================


def test_probability_weighted_expected_returns() -> None:
    allocator = ScenarioAwareAllocator()

    scenarios = build_scenarios()

    probabilities = (
        allocator._scenario_probabilities(
            scenarios
        )
    )

    matrix = (
        allocator._scenario_return_matrix(
            scenarios=scenarios,
            symbols=[
                "SPY",
                "NVDA",
                "GLD",
            ],
        )
    )

    expected = (
        allocator._scenario_expected_returns(
            scenario_matrix=matrix,
            probabilities=probabilities,
        )
    )

    expected_spy = (
        0.30 * 0.15
        + 0.50 * 0.07
        + 0.20 * -0.20
    )

    expected_nvda = (
        0.30 * 0.30
        + 0.50 * 0.12
        + 0.20 * -0.35
    )

    expected_gld = (
        0.30 * -0.03
        + 0.50 * 0.04
        + 0.20 * 0.15
    )

    assert expected[0] == pytest.approx(
        expected_spy
    )

    assert expected[1] == pytest.approx(
        expected_nvda
    )

    assert expected[2] == pytest.approx(
        expected_gld
    )


# ==============================================================
# Blended returns
# ==============================================================


def test_scenario_weight_zero_returns_nominal_estimates() -> None:
    allocator = ScenarioAwareAllocator(
        scenario_weight=0.0
    )

    nominal = np.asarray(
        [
            0.08,
            0.15,
            0.05,
        ],
        dtype=float,
    )

    scenario = np.asarray(
        [
            0.04,
            0.08,
            0.10,
        ],
        dtype=float,
    )

    blended = (
        allocator._blended_expected_returns(
            nominal_returns=nominal,
            scenario_returns=scenario,
        )
    )

    assert np.allclose(
        blended,
        nominal,
        atol=1e-12,
    )


def test_scenario_weight_one_returns_scenario_estimates() -> None:
    allocator = ScenarioAwareAllocator(
        scenario_weight=1.0
    )

    nominal = np.asarray(
        [
            0.08,
            0.15,
            0.05,
        ],
        dtype=float,
    )

    scenario = np.asarray(
        [
            0.04,
            0.08,
            0.10,
        ],
        dtype=float,
    )

    blended = (
        allocator._blended_expected_returns(
            nominal_returns=nominal,
            scenario_returns=scenario,
        )
    )

    assert np.allclose(
        blended,
        scenario,
        atol=1e-12,
    )


def test_half_scenario_weight_blends_returns_equally() -> None:
    allocator = ScenarioAwareAllocator(
        scenario_weight=0.50
    )

    nominal = np.asarray(
        [
            0.10,
            0.20,
            0.04,
        ],
        dtype=float,
    )

    scenario = np.asarray(
        [
            0.04,
            0.10,
            0.08,
        ],
        dtype=float,
    )

    blended = (
        allocator._blended_expected_returns(
            nominal_returns=nominal,
            scenario_returns=scenario,
        )
    )

    expected = (
        0.50 * nominal
        + 0.50 * scenario
    )

    assert np.allclose(
        blended,
        expected,
        atol=1e-12,
    )


# ==============================================================
# Basic optimization
# ==============================================================


def test_scenario_optimization_completes_successfully() -> None:
    allocator = ScenarioAwareAllocator()

    result = allocator.optimize(
        build_request()
    )

    assert (
        result.status
        == OptimizationStatus.COMPLETED
    )

    assert result.allocation is not None
    assert result.risk_metrics is not None
    assert result.objective_value is not None
    assert result.iterations is not None
    assert result.iterations >= 1


def test_scenario_weights_sum_to_one() -> None:
    allocator = ScenarioAwareAllocator()

    result = allocator.optimize(
        build_request()
    )

    assert result.allocation is not None

    assert sum(
        result.allocation.weights.values()
    ) == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_scenario_portfolio_is_long_only() -> None:
    allocator = ScenarioAwareAllocator()

    result = allocator.optimize(
        build_request()
    )

    assert result.allocation is not None

    for weight in (
        result.allocation.weights.values()
    ):
        assert weight >= 0.0


# ==============================================================
# Scenario diagnostics
# ==============================================================


def test_scenario_diagnostics_are_generated() -> None:
    allocator = ScenarioAwareAllocator()

    result = allocator.optimize(
        build_request()
    )

    assert (
        result.diagnostics[
            "scenario_count"
        ]
        == 3
    )

    probabilities = (
        result.diagnostics[
            "scenario_probabilities"
        ]
    )

    assert sum(
        probabilities.values()
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )

    assert (
        "scenario_expected_returns"
        in result.diagnostics
    )

    assert (
        "blended_expected_returns"
        in result.diagnostics
    )


def test_worst_and_best_scenarios_are_reported() -> None:
    allocator = ScenarioAwareAllocator()

    result = allocator.optimize(
        build_request()
    )

    assert result.risk_metrics is not None

    metadata = (
        result.risk_metrics.metadata
    )

    assert (
        "worst_scenario_return"
        in metadata
    )

    assert (
        "best_scenario_return"
        in metadata
    )

    assert (
        "worst_scenario_id"
        in metadata
    )

    assert (
        "best_scenario_id"
        in metadata
    )

    assert (
        metadata["worst_scenario_return"]
        <= metadata["best_scenario_return"]
    )


def test_scenario_dispersion_is_reported() -> None:
    allocator = ScenarioAwareAllocator()

    result = allocator.optimize(
        build_request()
    )

    assert result.risk_metrics is not None

    dispersion = (
        result.risk_metrics.metadata[
            "scenario_dispersion"
        ]
    )

    assert set(
        dispersion.keys()
    ) == {
        "SPY",
        "NVDA",
        "GLD",
    }

    for value in dispersion.values():
        assert value >= 0.0


# ==============================================================
# Scenario-awareness behavior
# ==============================================================


def test_scenario_weight_changes_allocation() -> None:
    request = build_request()

    nominal_allocator = (
        ScenarioAwareAllocator(
            scenario_weight=0.0,
        )
    )

    scenario_allocator = (
        ScenarioAwareAllocator(
            scenario_weight=1.0,
        )
    )

    nominal_result = (
        nominal_allocator.optimize(
            request
        )
    )

    scenario_result = (
        scenario_allocator.optimize(
            request
        )
    )

    assert (
        nominal_result.allocation
        is not None
    )

    assert (
        scenario_result.allocation
        is not None
    )

    nominal_weights = np.asarray(
        [
            nominal_result.allocation.weights[
                "SPY"
            ],
            nominal_result.allocation.weights[
                "NVDA"
            ],
            nominal_result.allocation.weights[
                "GLD"
            ],
        ],
        dtype=float,
    )

    scenario_weights = np.asarray(
        [
            scenario_result.allocation.weights[
                "SPY"
            ],
            scenario_result.allocation.weights[
                "NVDA"
            ],
            scenario_result.allocation.weights[
                "GLD"
            ],
        ],
        dtype=float,
    )

    assert not np.allclose(
        nominal_weights,
        scenario_weights,
        atol=1e-4,
    )


def test_downside_aversion_changes_allocation() -> None:
    request = build_request()

    low_downside = (
        ScenarioAwareAllocator(
            downside_aversion=0.0
        )
    )

    high_downside = (
        ScenarioAwareAllocator(
            downside_aversion=20.0
        )
    )

    first = low_downside.optimize(
        request
    )

    second = high_downside.optimize(
        request
    )

    assert first.allocation is not None
    assert second.allocation is not None

    first_weights = np.asarray(
        [
            first.allocation.weights["SPY"],
            first.allocation.weights["NVDA"],
            first.allocation.weights["GLD"],
        ],
        dtype=float,
    )

    second_weights = np.asarray(
        [
            second.allocation.weights["SPY"],
            second.allocation.weights["NVDA"],
            second.allocation.weights["GLD"],
        ],
        dtype=float,
    )

    assert not np.allclose(
        first_weights,
        second_weights,
        atol=1e-4,
    )


# ==============================================================
# Position constraints
# ==============================================================


def test_maximum_position_constraint_is_respected() -> None:
    constraints = [
        PortfolioConstraint(
            constraint_type=(
                ConstraintType.POSITION
            ),
            target="GLD",
            maximum=0.40,
        ),
    ]

    allocator = ScenarioAwareAllocator()

    result = allocator.optimize(
        build_request(
            constraints=constraints
        )
    )

    assert result.allocation is not None

    assert (
        result.allocation.weights["GLD"]
        <= 0.40 + 1e-6
    )


def test_minimum_position_constraint_is_respected() -> None:
    constraints = [
        PortfolioConstraint(
            constraint_type=(
                ConstraintType.POSITION
            ),
            target="SPY",
            minimum=0.30,
        ),
    ]

    allocator = ScenarioAwareAllocator()

    result = allocator.optimize(
        build_request(
            constraints=constraints
        )
    )

    assert result.allocation is not None

    assert (
        result.allocation.weights["SPY"]
        >= 0.30 - 1e-6
    )


# ==============================================================
# Target return
# ==============================================================


def test_target_return_influences_allocation() -> None:
    allocator = ScenarioAwareAllocator()

    no_target = allocator.optimize(
        build_request()
    )

    target = allocator.optimize(
        build_request(
            target_return=0.10
        )
    )

    assert no_target.risk_metrics is not None
    assert target.risk_metrics is not None

    assert (
        no_target.risk_metrics.expected_return
        is not None
    )

    assert (
        target.risk_metrics.expected_return
        is not None
    )

    assert (
        target.risk_metrics.expected_return
        >= no_target.risk_metrics.expected_return
        - 1e-6
    )


# ==============================================================
# Turnover
# ==============================================================


def test_turnover_is_calculated() -> None:
    current = PortfolioAllocation(
        weights={
            "SPY": 0.50,
            "NVDA": 0.30,
            "GLD": 0.20,
        }
    )

    current.validate()

    allocator = ScenarioAwareAllocator()

    result = allocator.optimize(
        build_request(
            current_allocation=current
        )
    )

    assert result.risk_metrics is not None

    assert (
        result.risk_metrics.turnover
        is not None
    )

    assert (
        result.risk_metrics.turnover
        >= 0.0
    )


# ==============================================================
# Risk metrics
# ==============================================================


def test_risk_metrics_are_generated() -> None:
    allocator = ScenarioAwareAllocator()

    result = allocator.optimize(
        build_request()
    )

    metrics = result.risk_metrics

    assert metrics is not None
    assert metrics.expected_return is not None
    assert metrics.volatility is not None
    assert metrics.sharpe_ratio is not None

    assert metrics.volatility > 0.0

    assert metrics.gross_exposure == pytest.approx(
        1.0,
        abs=1e-6,
    )

    assert metrics.net_exposure == pytest.approx(
        1.0,
        abs=1e-6,
    )

    assert (
        "scenario_expected_return"
        in metrics.metadata
    )

    assert (
        "blended_expected_return"
        in metrics.metadata
    )

    assert (
        "scenario_volatility"
        in metrics.metadata
    )


# ==============================================================
# Error handling
# ==============================================================


def test_wrong_method_raises_error() -> None:
    allocator = ScenarioAwareAllocator()

    request = build_request(
        method=(
            OptimizationMethod.MEAN_VARIANCE
        )
    )

    with pytest.raises(
        ScenarioAllocatorError
    ):
        allocator.optimize(
            request
        )


def test_no_scenarios_raises_error() -> None:
    allocator = ScenarioAwareAllocator()

    request = build_request(
        scenarios=[]
    )

    with pytest.raises(
        ScenarioAllocatorError
    ):
        allocator.optimize(
            request
        )


def test_partial_probabilities_raise_error() -> None:
    scenarios = [
        MarketScenario(
            scenario_id="one",
            name="One",
            probability=0.50,
            asset_returns={
                "SPY": 0.10,
                "NVDA": 0.20,
                "GLD": 0.00,
            },
        ),
        MarketScenario(
            scenario_id="two",
            name="Two",
            probability=None,
            asset_returns={
                "SPY": -0.05,
                "NVDA": -0.10,
                "GLD": 0.10,
            },
        ),
    ]

    allocator = ScenarioAwareAllocator()

    with pytest.raises(
        ScenarioAllocatorError
    ):
        allocator.optimize(
            build_request(
                scenarios=scenarios
            )
        )


def test_missing_asset_return_raises_error() -> None:
    scenarios = [
        MarketScenario(
            scenario_id="broken",
            name="Broken Scenario",
            probability=1.0,
            asset_returns={
                "SPY": 0.10,
                "NVDA": 0.20,
            },
        ),
    ]

    allocator = ScenarioAwareAllocator()

    with pytest.raises(
        ScenarioAllocatorError
    ):
        allocator.optimize(
            build_request(
                scenarios=scenarios
            )
        )


def test_infeasible_minimum_constraints_raise_error() -> None:
    constraints = [
        PortfolioConstraint(
            constraint_type=(
                ConstraintType.POSITION
            ),
            target="SPY",
            minimum=0.50,
        ),
        PortfolioConstraint(
            constraint_type=(
                ConstraintType.POSITION
            ),
            target="NVDA",
            minimum=0.50,
        ),
        PortfolioConstraint(
            constraint_type=(
                ConstraintType.POSITION
            ),
            target="GLD",
            minimum=0.50,
        ),
    ]

    allocator = ScenarioAwareAllocator()

    with pytest.raises(
        ScenarioAllocatorError
    ):
        allocator.optimize(
            build_request(
                constraints=constraints
            )
        )


def test_infeasible_maximum_constraints_raise_error() -> None:
    constraints = [
        PortfolioConstraint(
            constraint_type=(
                ConstraintType.POSITION
            ),
            target="SPY",
            maximum=0.20,
        ),
        PortfolioConstraint(
            constraint_type=(
                ConstraintType.POSITION
            ),
            target="NVDA",
            maximum=0.20,
        ),
        PortfolioConstraint(
            constraint_type=(
                ConstraintType.POSITION
            ),
            target="GLD",
            maximum=0.20,
        ),
    ]

    allocator = ScenarioAwareAllocator()

    with pytest.raises(
        ScenarioAllocatorError
    ):
        allocator.optimize(
            build_request(
                constraints=constraints
            )
        )


def test_invalid_optimizer_configuration_raises_error() -> None:
    with pytest.raises(
        ScenarioAllocatorError
    ):
        ScenarioAwareAllocator(
            scenario_weight=-0.1
        )

    with pytest.raises(
        ScenarioAllocatorError
    ):
        ScenarioAwareAllocator(
            scenario_weight=1.1
        )

    with pytest.raises(
        ScenarioAllocatorError
    ):
        ScenarioAwareAllocator(
            risk_aversion=-1.0
        )

    with pytest.raises(
        ScenarioAllocatorError
    ):
        ScenarioAwareAllocator(
            scenario_risk_aversion=-1.0
        )

    with pytest.raises(
        ScenarioAllocatorError
    ):
        ScenarioAwareAllocator(
            downside_aversion=-1.0
        )

    with pytest.raises(
        ScenarioAllocatorError
    ):
        ScenarioAwareAllocator(
            learning_rate=0.0
        )

    with pytest.raises(
        ScenarioAllocatorError
    ):
        ScenarioAwareAllocator(
            max_iterations=0
        )

    with pytest.raises(
        ScenarioAllocatorError
    ):
        ScenarioAwareAllocator(
            tolerance=0.0
        )

    with pytest.raises(
        ScenarioAllocatorError
    ):
        ScenarioAwareAllocator(
            target_return_penalty=-1.0
        )