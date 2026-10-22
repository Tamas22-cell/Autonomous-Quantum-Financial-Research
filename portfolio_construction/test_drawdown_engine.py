"""
Tests for the v5.0.7 Drawdown-Aware De-Risking Engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from portfolio_construction.drawdown_engine import (
    DeRiskingResult,
    DrawdownAwareDeRiskingEngine,
    DrawdownEngineError,
    DrawdownMetrics,
)
from portfolio_construction.models import (
    DrawdownPolicy,
    PortfolioAllocation,
)


# ==============================================================
# Helpers
# ==============================================================


def build_allocation() -> PortfolioAllocation:
    allocation = PortfolioAllocation(
        weights={
            "SPY": 0.50,
            "NVDA": 0.30,
            "GLD": 0.20,
        },
        cash_weight=0.0,
    )

    allocation.validate()

    return allocation


def build_policy() -> DrawdownPolicy:
    return DrawdownPolicy(
        trigger_drawdown=0.10,
        target_risk_multiplier=0.70,
        minimum_risk_multiplier=0.30,
        recovery_threshold=0.05,
    )


# ==============================================================
# Drawdown series
# ==============================================================


def test_drawdown_series_is_calculated_correctly() -> None:
    values = [
        100.0,
        110.0,
        99.0,
        88.0,
    ]

    drawdowns = (
        DrawdownAwareDeRiskingEngine
        .calculate_drawdown_series(
            values
        )
    )

    assert drawdowns[0] == pytest.approx(
        0.0
    )

    assert drawdowns[1] == pytest.approx(
        0.0
    )

    assert drawdowns[2] == pytest.approx(
        0.10
    )

    assert drawdowns[3] == pytest.approx(
        0.20
    )


def test_monotonically_rising_series_has_zero_drawdown() -> None:
    values = [
        100.0,
        105.0,
        110.0,
        120.0,
    ]

    drawdowns = (
        DrawdownAwareDeRiskingEngine
        .calculate_drawdown_series(
            values
        )
    )

    assert np.allclose(
        drawdowns,
        0.0,
        atol=1e-12,
    )


# ==============================================================
# Current drawdown
# ==============================================================


def test_current_drawdown_is_calculated() -> None:
    values = [
        100.0,
        120.0,
        108.0,
    ]

    drawdown = (
        DrawdownAwareDeRiskingEngine
        .calculate_current_drawdown(
            values
        )
    )

    assert drawdown == pytest.approx(
        0.10
    )


def test_current_drawdown_is_zero_at_new_high() -> None:
    values = [
        100.0,
        90.0,
        110.0,
    ]

    drawdown = (
        DrawdownAwareDeRiskingEngine
        .calculate_current_drawdown(
            values
        )
    )

    assert drawdown == pytest.approx(
        0.0
    )


# ==============================================================
# Maximum drawdown
# ==============================================================


def test_maximum_drawdown_is_calculated() -> None:
    values = [
        100.0,
        120.0,
        90.0,
        100.0,
        130.0,
    ]

    maximum_drawdown = (
        DrawdownAwareDeRiskingEngine
        .calculate_maximum_drawdown(
            values
        )
    )

    assert maximum_drawdown == pytest.approx(
        0.25
    )


def test_maximum_drawdown_remembers_historical_loss() -> None:
    values = [
        100.0,
        120.0,
        90.0,
        110.0,
        130.0,
        125.0,
    ]

    maximum_drawdown = (
        DrawdownAwareDeRiskingEngine
        .calculate_maximum_drawdown(
            values
        )
    )

    assert maximum_drawdown == pytest.approx(
        0.25
    )


# ==============================================================
# Drawdown analytics
# ==============================================================


def test_analyze_returns_drawdown_metrics() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    metrics = engine.analyze(
        [
            100.0,
            120.0,
            90.0,
            100.0,
        ]
    )

    assert isinstance(
        metrics,
        DrawdownMetrics,
    )

    assert metrics.current_value == pytest.approx(
        100.0
    )

    assert metrics.peak_value == pytest.approx(
        120.0
    )

    assert metrics.current_drawdown == pytest.approx(
        1.0 - 100.0 / 120.0
    )

    assert metrics.maximum_drawdown == pytest.approx(
        0.25
    )

    assert metrics.observations == 4


def test_analyze_identifies_peak_and_trough() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    metrics = engine.analyze(
        [
            100.0,
            120.0,
            90.0,
            100.0,
        ]
    )

    assert metrics.peak_index == 1
    assert metrics.trough_index == 2


# ==============================================================
# Risk multiplier before trigger
# ==============================================================


def test_drawdown_below_trigger_keeps_full_risk() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    multiplier = (
        engine.calculate_risk_multiplier(
            current_drawdown=0.05,
            policy=build_policy(),
        )
    )

    assert multiplier == pytest.approx(
        1.0
    )


def test_zero_drawdown_keeps_full_risk() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    multiplier = (
        engine.calculate_risk_multiplier(
            current_drawdown=0.0,
            policy=build_policy(),
        )
    )

    assert multiplier == pytest.approx(
        1.0
    )


# ==============================================================
# Trigger behavior
# ==============================================================


def test_drawdown_at_trigger_uses_target_multiplier() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    multiplier = (
        engine.calculate_risk_multiplier(
            current_drawdown=0.10,
            policy=build_policy(),
        )
    )

    assert multiplier == pytest.approx(
        0.70
    )


def test_drawdown_above_trigger_reduces_risk_further() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    multiplier = (
        engine.calculate_risk_multiplier(
            current_drawdown=0.30,
            policy=build_policy(),
        )
    )

    assert multiplier < 0.70
    assert multiplier >= 0.30


def test_extreme_drawdown_never_goes_below_minimum_multiplier() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    multiplier = (
        engine.calculate_risk_multiplier(
            current_drawdown=1.0,
            policy=build_policy(),
        )
    )

    assert multiplier == pytest.approx(
        0.30
    )


def test_progressive_derisking_is_monotonic() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    policy = build_policy()

    multiplier_10 = (
        engine.calculate_risk_multiplier(
            current_drawdown=0.10,
            policy=policy,
        )
    )

    multiplier_20 = (
        engine.calculate_risk_multiplier(
            current_drawdown=0.20,
            policy=policy,
        )
    )

    multiplier_40 = (
        engine.calculate_risk_multiplier(
            current_drawdown=0.40,
            policy=policy,
        )
    )

    assert multiplier_10 >= multiplier_20
    assert multiplier_20 >= multiplier_40


# ==============================================================
# Portfolio scaling
# ==============================================================


def test_apply_risk_multiplier_scales_asset_weights() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    adjusted = engine.apply_risk_multiplier(
        allocation=build_allocation(),
        risk_multiplier=0.50,
    )

    assert adjusted.weights["SPY"] == pytest.approx(
        0.25
    )

    assert adjusted.weights["NVDA"] == pytest.approx(
        0.15
    )

    assert adjusted.weights["GLD"] == pytest.approx(
        0.10
    )


def test_derisked_capital_moves_to_cash() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    adjusted = engine.apply_risk_multiplier(
        allocation=build_allocation(),
        risk_multiplier=0.50,
    )

    assert adjusted.cash_weight == pytest.approx(
        0.50
    )


def test_adjusted_portfolio_still_sums_to_one() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    adjusted = engine.apply_risk_multiplier(
        allocation=build_allocation(),
        risk_multiplier=0.40,
    )

    total = (
        sum(
            adjusted.weights.values()
        )
        + adjusted.cash_weight
    )

    assert total == pytest.approx(
        1.0,
        abs=1e-8,
    )


def test_full_risk_multiplier_preserves_portfolio() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    original = build_allocation()

    adjusted = engine.apply_risk_multiplier(
        allocation=original,
        risk_multiplier=1.0,
    )

    assert adjusted.weights == original.weights

    assert adjusted.cash_weight == pytest.approx(
        original.cash_weight
    )


# ==============================================================
# Existing cash
# ==============================================================


def test_derisking_portfolio_with_existing_cash() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    allocation = PortfolioAllocation(
        weights={
            "SPY": 0.40,
            "NVDA": 0.30,
            "GLD": 0.20,
        },
        cash_weight=0.10,
    )

    allocation.validate()

    adjusted = engine.apply_risk_multiplier(
        allocation=allocation,
        risk_multiplier=0.50,
    )

    assert sum(
        adjusted.weights.values()
    ) == pytest.approx(
        0.45
    )

    assert adjusted.cash_weight == pytest.approx(
        0.55
    )

    assert adjusted.total_weight == pytest.approx(
        1.0
    )


# ==============================================================
# Full de-risking result
# ==============================================================


def test_apply_returns_derisking_result() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    result = engine.apply(
        allocation=build_allocation(),
        values=[
            100.0,
            120.0,
            105.0,
        ],
        policy=build_policy(),
    )

    assert isinstance(
        result,
        DeRiskingResult,
    )

    assert isinstance(
        result.metrics,
        DrawdownMetrics,
    )


def test_apply_triggers_derisking_above_threshold() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    result = engine.apply(
        allocation=build_allocation(),
        values=[
            100.0,
            120.0,
            100.0,
        ],
        policy=build_policy(),
    )

    assert result.triggered is True

    assert result.risk_multiplier < 1.0

    assert (
        result.adjusted_allocation.cash_weight
        > result.original_allocation.cash_weight
    )


def test_apply_does_not_trigger_below_threshold() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    result = engine.apply(
        allocation=build_allocation(),
        values=[
            100.0,
            110.0,
            105.0,
        ],
        policy=build_policy(),
    )

    assert result.triggered is False

    assert result.risk_multiplier == pytest.approx(
        1.0
    )

    assert result.cash_increase == pytest.approx(
        0.0
    )


def test_risk_reduction_matches_released_exposure() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    result = engine.apply(
        allocation=build_allocation(),
        values=[
            100.0,
            120.0,
            100.0,
        ],
        policy=build_policy(),
    )

    expected_reduction = (
        1.0
        - result.risk_multiplier
    )

    assert result.risk_reduction == pytest.approx(
        expected_reduction
    )

    assert result.cash_increase == pytest.approx(
        expected_reduction
    )


# ==============================================================
# Recovery behavior
# ==============================================================


def test_recovery_is_detected_after_trough() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    result = engine.apply(
        allocation=build_allocation(),
        values=[
            100.0,
            120.0,
            90.0,
            100.0,
        ],
        policy=build_policy(),
    )

    assert result.recovering is True


def test_recovery_below_threshold_restores_full_risk() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    result = engine.apply(
        allocation=build_allocation(),
        values=[
            100.0,
            120.0,
            90.0,
            115.0,
        ],
        policy=build_policy(),
    )

    assert result.recovering is True

    assert (
        result.metrics.current_drawdown
        <= 0.05
    )

    assert result.risk_multiplier == pytest.approx(
        1.0
    )

    assert result.triggered is False


def test_recovery_above_recovery_threshold_does_not_force_full_risk() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    result = engine.apply(
        allocation=build_allocation(),
        values=[
            100.0,
            120.0,
            90.0,
            105.0,
        ],
        policy=build_policy(),
    )

    assert result.recovering is True

    assert (
        result.metrics.current_drawdown
        > 0.05
    )

    assert result.risk_multiplier < 1.0


def test_no_recovery_detection_without_recovery_threshold() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    policy = DrawdownPolicy(
        trigger_drawdown=0.10,
        target_risk_multiplier=0.70,
        minimum_risk_multiplier=0.30,
        recovery_threshold=None,
    )

    result = engine.apply(
        allocation=build_allocation(),
        values=[
            100.0,
            120.0,
            90.0,
            100.0,
        ],
        policy=policy,
    )

    assert result.recovering is False


# ==============================================================
# Metadata
# ==============================================================


def test_result_contains_policy_metadata() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    result = engine.apply(
        allocation=build_allocation(),
        values=[
            100.0,
            120.0,
            100.0,
        ],
        policy=build_policy(),
    )

    assert result.metadata[
        "trigger_drawdown"
    ] == pytest.approx(
        0.10
    )

    assert result.metadata[
        "target_risk_multiplier"
    ] == pytest.approx(
        0.70
    )

    assert result.metadata[
        "minimum_risk_multiplier"
    ] == pytest.approx(
        0.30
    )

    assert result.metadata[
        "recovery_threshold"
    ] == pytest.approx(
        0.05
    )


def test_adjusted_allocation_contains_drawdown_metadata() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    adjusted = engine.apply_risk_multiplier(
        allocation=build_allocation(),
        risk_multiplier=0.50,
    )

    assert (
        adjusted.metadata[
            "drawdown_adjusted"
        ]
        is True
    )

    assert adjusted.metadata[
        "risk_multiplier"
    ] == pytest.approx(
        0.50
    )


# ==============================================================
# Validation errors
# ==============================================================


def test_empty_value_series_raises_error() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    with pytest.raises(
        DrawdownEngineError
    ):
        engine.analyze(
            []
        )


def test_nonfinite_value_series_raises_error() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    with pytest.raises(
        DrawdownEngineError
    ):
        engine.analyze(
            [
                100.0,
                np.nan,
                90.0,
            ]
        )


def test_infinite_value_series_raises_error() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    with pytest.raises(
        DrawdownEngineError
    ):
        engine.analyze(
            [
                100.0,
                np.inf,
                90.0,
            ]
        )


def test_nonpositive_portfolio_value_raises_error() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    with pytest.raises(
        DrawdownEngineError
    ):
        engine.analyze(
            [
                100.0,
                0.0,
                90.0,
            ]
        )


def test_multidimensional_value_series_raises_error() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    with pytest.raises(
        DrawdownEngineError
    ):
        engine.analyze(
            [
                [
                    100.0,
                    90.0,
                ],
                [
                    110.0,
                    95.0,
                ],
            ]
        )


def test_invalid_current_drawdown_below_zero_raises_error() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    with pytest.raises(
        DrawdownEngineError
    ):
        engine.calculate_risk_multiplier(
            current_drawdown=-0.01,
            policy=build_policy(),
        )


def test_invalid_current_drawdown_above_one_raises_error() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    with pytest.raises(
        DrawdownEngineError
    ):
        engine.calculate_risk_multiplier(
            current_drawdown=1.01,
            policy=build_policy(),
        )


def test_invalid_risk_multiplier_below_zero_raises_error() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    with pytest.raises(
        DrawdownEngineError
    ):
        engine.apply_risk_multiplier(
            allocation=build_allocation(),
            risk_multiplier=-0.10,
        )


def test_invalid_risk_multiplier_above_one_raises_error() -> None:
    engine = DrawdownAwareDeRiskingEngine()

    with pytest.raises(
        DrawdownEngineError
    ):
        engine.apply_risk_multiplier(
            allocation=build_allocation(),
            risk_multiplier=1.10,
        )


def test_invalid_engine_tolerance_raises_error() -> None:
    with pytest.raises(
        DrawdownEngineError
    ):
        DrawdownAwareDeRiskingEngine(
            tolerance=-0.01
        )