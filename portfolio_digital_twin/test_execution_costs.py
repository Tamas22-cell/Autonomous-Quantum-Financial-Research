
import pytest

from portfolio_digital_twin.execution_costs import (
    ExecutionCostConfig,
    ExecutionCostEngine,
    ExecutionCostError,
)


@pytest.fixture
def engine():
    return ExecutionCostEngine()


def test_default_configuration(engine):
    config = engine.config

    assert config.commission_bps == 2.0
    assert config.spread_bps == 5.0
    assert config.base_slippage_bps == 3.0
    assert config.impact_coefficient == 0.10
    assert config.max_participation_rate == 0.10


def test_transaction_cost_calculation(engine):
    report = engine.estimate(
        symbol="SPY",
        order_notional=10_000.0,
        daily_volume_notional=1_000_000.0,
        volatility=0.02,
    )

    assert report.commission == pytest.approx(2.0)
    assert report.spread_cost == pytest.approx(2.5)
    assert report.slippage_cost == pytest.approx(3.0)
    assert report.market_impact_cost == pytest.approx(2.0)

    assert report.total_cost == pytest.approx(9.5)
    assert report.total_cost_bps == pytest.approx(9.5)


def test_liquidity_warning(engine):
    report = engine.estimate(
        symbol="BTC",
        order_notional=200_000.0,
        daily_volume_notional=1_000_000.0,
        volatility=0.04,
    )

    assert report.participation_rate == pytest.approx(0.20)
    assert report.liquidity_warning is True


def test_no_liquidity_warning(engine):
    report = engine.estimate(
        symbol="SPY",
        order_notional=10_000.0,
        daily_volume_notional=1_000_000.0,
    )

    assert report.liquidity_warning is False


def test_zero_order(engine):
    report = engine.estimate(
        symbol="SPY",
        order_notional=0.0,
        daily_volume_notional=1_000_000.0,
    )

    assert report.total_cost == pytest.approx(0.0)
    assert report.total_cost_bps == pytest.approx(0.0)
    assert report.participation_rate == pytest.approx(0.0)


def test_invalid_order_notional(engine):
    with pytest.raises(ExecutionCostError):
        engine.estimate(
            symbol="SPY",
            order_notional=-100.0,
            daily_volume_notional=1_000_000.0,
        )


def test_invalid_daily_volume(engine):
    with pytest.raises(ExecutionCostError):
        engine.estimate(
            symbol="SPY",
            order_notional=10_000.0,
            daily_volume_notional=0.0,
        )


def test_invalid_volatility(engine):
    with pytest.raises(ExecutionCostError):
        engine.estimate(
            symbol="SPY",
            order_notional=10_000.0,
            daily_volume_notional=1_000_000.0,
            volatility=float("nan"),
        )


def test_invalid_symbol(engine):
    with pytest.raises(ExecutionCostError):
        engine.estimate(
            symbol="",
            order_notional=10_000.0,
            daily_volume_notional=1_000_000.0,
        )


def test_invalid_configuration():
    with pytest.raises(ExecutionCostError):
        ExecutionCostConfig(
            commission_bps=-1.0,
        )


def test_custom_configuration():
    config = ExecutionCostConfig(
        commission_bps=1.0,
        spread_bps=2.0,
        base_slippage_bps=1.0,
        impact_coefficient=0.0,
        max_participation_rate=0.05,
    )

    engine = ExecutionCostEngine(config)

    report = engine.estimate(
        symbol="NVDA",
        order_notional=10_000.0,
        daily_volume_notional=1_000_000.0,
    )

    assert report.commission == pytest.approx(1.0)
    assert report.spread_cost == pytest.approx(1.0)
    assert report.slippage_cost == pytest.approx(1.0)
    assert report.market_impact_cost == pytest.approx(0.0)
    assert report.total_cost == pytest.approx(3.0)


def test_market_impact_increases_with_participation(engine):
    small_order = engine.estimate(
        symbol="SPY",
        order_notional=10_000.0,
        daily_volume_notional=1_000_000.0,
    )

    large_order = engine.estimate(
        symbol="SPY",
        order_notional=100_000.0,
        daily_volume_notional=1_000_000.0,
    )

    assert (
        large_order.market_impact_cost
        > small_order.market_impact_cost
    )


def test_higher_volatility_increases_market_impact(engine):
    low_volatility = engine.estimate(
        symbol="BTC",
        order_notional=10_000.0,
        daily_volume_notional=1_000_000.0,
        volatility=0.01,
    )

    high_volatility = engine.estimate(
        symbol="BTC",
        order_notional=10_000.0,
        daily_volume_notional=1_000_000.0,
        volatility=0.05,
    )

    assert (
        high_volatility.market_impact_cost
        > low_volatility.market_impact_cost
    )


def test_cost_components_sum_to_total(engine):
    report = engine.estimate(
        symbol="NVDA",
        order_notional=25_000.0,
        daily_volume_notional=2_000_000.0,
        volatility=0.03,
    )

    expected_total = (
        report.commission
        + report.spread_cost
        + report.slippage_cost
        + report.market_impact_cost
    )

    assert report.total_cost == pytest.approx(expected_total)