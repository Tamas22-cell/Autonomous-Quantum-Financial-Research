
import pytest

from portfolio_construction.models import PortfolioAllocation
from portfolio_digital_twin.execution_costs import (
    ExecutionCostConfig,
    ExecutionCostEngine,
)
from portfolio_digital_twin.shadow_simulation import (
    ShadowSimulationEngine,
)
from portfolio_digital_twin.trade_execution import (
    TradeExecutionEngine,
    TradeExecutionError,
    VirtualTrade,
)


@pytest.fixture
def engine():
    allocation = PortfolioAllocation(
        weights={
            "SPY": 0.50,
            "BTC": 0.30,
        },
        cash_weight=0.20,
    )

    shadow = ShadowSimulationEngine(
        initial_capital=100_000.0,
        allocation=allocation,
    )

    return TradeExecutionEngine(
        shadow=shadow,
    )


def make_trade(
    symbol="SPY",
    side="BUY",
    notional=1_000.0,
    daily_volume=1_000_000.0,
    volatility=0.02,
):
    return VirtualTrade(
        symbol=symbol,
        side=side,
        notional=notional,
        daily_volume_notional=daily_volume,
        volatility=volatility,
    )


def test_initial_portfolio(engine):
    assert engine.shadow.nav == pytest.approx(100_000.0)
    assert engine.shadow.cash == pytest.approx(20_000.0)
    assert engine.shadow.positions["SPY"] == pytest.approx(
        50_000.0
    )
    assert len(engine.trade_history) == 0


def test_virtual_buy(engine):
    result = engine.execute(make_trade())

    cost = result.execution_cost.total_cost

    assert engine.shadow.positions["SPY"] == pytest.approx(
        51_000.0
    )

    assert engine.shadow.cash == pytest.approx(
        20_000.0 - 1_000.0 - cost
    )

    assert engine.shadow.nav == pytest.approx(
        100_000.0 - cost
    )

    assert result.side == "BUY"
    assert result.notional == pytest.approx(1_000.0)


def test_virtual_sell(engine):
    result = engine.execute(
        make_trade(
            side="SELL",
            notional=5_000.0,
        )
    )

    cost = result.execution_cost.total_cost

    assert engine.shadow.positions["SPY"] == pytest.approx(
        45_000.0
    )

    assert engine.shadow.cash == pytest.approx(
        25_000.0 - cost
    )

    assert engine.shadow.nav == pytest.approx(
        100_000.0 - cost
    )


def test_transaction_cost_reduces_nav(engine):
    nav_before = engine.shadow.nav

    result = engine.execute(make_trade())

    assert result.nav_before == pytest.approx(nav_before)

    assert result.nav_after == pytest.approx(
        nav_before - result.execution_cost.total_cost
    )


def test_trade_history(engine):
    engine.execute(make_trade())

    engine.execute(
        make_trade(
            symbol="BTC",
            side="SELL",
            notional=2_000.0,
        )
    )

    assert len(engine.trade_history) == 2
    assert engine.trade_history[0].trade_id == 1
    assert engine.trade_history[1].trade_id == 2

    assert len(engine.shadow.history) == 3


def test_cumulative_execution_cost(engine):
    first = engine.execute(make_trade())

    second = engine.execute(
        make_trade(
            symbol="BTC",
            side="SELL",
            notional=2_000.0,
        )
    )

    expected = (
        first.execution_cost.total_cost
        + second.execution_cost.total_cost
    )

    assert engine.cumulative_execution_cost == pytest.approx(
        expected
    )


def test_cumulative_turnover(engine):
    engine.execute(make_trade(notional=1_000.0))

    engine.execute(
        make_trade(
            side="SELL",
            notional=2_000.0,
        )
    )

    assert engine.cumulative_turnover == pytest.approx(
        0.01 + 2_000.0 / engine.trade_history[1].nav_before
    )


def test_insufficient_cash(engine):
    with pytest.raises(
        TradeExecutionError,
        match="Insufficient virtual cash",
    ):
        engine.execute(
            make_trade(notional=30_000.0)
        )


def test_insufficient_position(engine):
    with pytest.raises(
        TradeExecutionError,
        match="Insufficient virtual position",
    ):
        engine.execute(
            make_trade(
                side="SELL",
                notional=60_000.0,
            )
        )


def test_liquidity_limit(engine):
    with pytest.raises(
        TradeExecutionError,
        match="liquidity participation",
    ):
        engine.execute(
            make_trade(
                notional=15_000.0,
                daily_volume=100_000.0,
            )
        )


def test_unknown_symbol(engine):
    with pytest.raises(
        TradeExecutionError,
        match="Unknown portfolio symbol",
    ):
        engine.execute(
            make_trade(symbol="NVDA")
        )


def test_invalid_trade_side(engine):
    with pytest.raises(
        TradeExecutionError,
        match="BUY or SELL",
    ):
        engine.execute(
            make_trade(side="HOLD")
        )


def test_invalid_notional(engine):
    with pytest.raises(TradeExecutionError):
        engine.execute(
            make_trade(notional=-100.0)
        )


def test_failed_trade_does_not_change_portfolio(engine):
    nav_before = engine.shadow.nav
    cash_before = engine.shadow.cash
    positions_before = dict(engine.shadow.positions)
    history_before = len(engine.shadow.history)

    with pytest.raises(TradeExecutionError):
        engine.execute(
            make_trade(notional=50_000.0)
        )

    assert engine.shadow.nav == pytest.approx(nav_before)
    assert engine.shadow.cash == pytest.approx(cash_before)
    assert engine.shadow.positions == positions_before
    assert len(engine.shadow.history) == history_before
    assert len(engine.trade_history) == 0


def test_zero_execution_cost():
    allocation = PortfolioAllocation(
        weights={"SPY": 0.50},
        cash_weight=0.50,
    )

    shadow = ShadowSimulationEngine(
        initial_capital=100_000.0,
        allocation=allocation,
    )

    cost_engine = ExecutionCostEngine(
        ExecutionCostConfig(
            commission_bps=0.0,
            spread_bps=0.0,
            base_slippage_bps=0.0,
            impact_coefficient=0.0,
        )
    )

    engine = TradeExecutionEngine(
        shadow=shadow,
        cost_engine=cost_engine,
    )

    result = engine.execute(
        make_trade(notional=10_000.0)
    )

    assert result.execution_cost.total_cost == pytest.approx(
        0.0
    )

    assert engine.shadow.nav == pytest.approx(100_000.0)
    assert engine.shadow.cash == pytest.approx(40_000.0)
    assert engine.shadow.positions["SPY"] == pytest.approx(
        60_000.0
    )