
import pytest

from portfolio_construction.models import PortfolioAllocation
from portfolio_digital_twin.derisking_execution import (
    AssetExecutionData,
    DeRiskingExecutionEngine,
    DeRiskingExecutionError,
)
from portfolio_digital_twin.regime_derisking import (
    MarketRegime,
    RegimeDeRiskingEngine,
)
from portfolio_digital_twin.shadow_simulation import (
    ShadowSimulationEngine,
)
from portfolio_digital_twin.trade_execution import (
    TradeExecutionEngine,
)


@pytest.fixture
def setup():
    allocation = PortfolioAllocation(
        weights={
            "SPY": 0.40,
            "BTC": 0.30,
            "NVDA": 0.20,
        },
        cash_weight=0.10,
    )

    shadow = ShadowSimulationEngine(
        initial_capital=100_000.0,
        allocation=allocation,
    )

    trade_engine = TradeExecutionEngine(shadow)
    execution_engine = DeRiskingExecutionEngine(trade_engine)

    market_data = {
        "SPY": AssetExecutionData(
            daily_volume_notional=100_000_000.0
        ),
        "BTC": AssetExecutionData(
            daily_volume_notional=100_000_000.0
        ),
        "NVDA": AssetExecutionData(
            daily_volume_notional=100_000_000.0
        ),
    }

    return shadow, trade_engine, execution_engine, market_data


def make_plan(shadow, regime=MarketRegime.RISK_OFF):
    return RegimeDeRiskingEngine().plan(shadow, regime)


def test_executes_all_planned_sales(setup):
    shadow, trade_engine, engine, market_data = setup
    plan = make_plan(shadow)

    result = engine.execute(plan, market_data)

    assert len(result.trades) == 3
    assert result.executed_notional == pytest.approx(30_000.0)
    assert len(trade_engine.trade_history) == 3
    assert result.nav_after < result.nav_before


def test_exposure_decreases_after_execution(setup):
    shadow, _, engine, market_data = setup
    plan = make_plan(shadow)

    result = engine.execute(plan, market_data)

    assert result.exposure_before == pytest.approx(0.90)
    assert result.exposure_after < result.exposure_before
    assert result.exposure_after == pytest.approx(
        sum(shadow.positions.values()) / shadow.nav
    )


def test_execution_cost_is_recorded(setup):
    _, trade_engine, engine, market_data = setup
    plan = make_plan(trade_engine.shadow)

    result = engine.execute(plan, market_data)

    assert result.estimated_execution_cost > 0.0
    assert result.actual_execution_cost == pytest.approx(
        result.estimated_execution_cost
    )
    assert trade_engine.cumulative_execution_cost == pytest.approx(
        result.actual_execution_cost
    )


def test_cash_increases_by_net_sale_proceeds(setup):
    shadow, _, engine, market_data = setup
    cash_before = shadow.cash
    plan = make_plan(shadow)

    result = engine.execute(plan, market_data)

    assert shadow.cash == pytest.approx(
        cash_before
        + result.executed_notional
        - result.actual_execution_cost
    )


def test_positions_decrease_by_sale_notionals(setup):
    shadow, _, engine, market_data = setup
    positions_before = dict(shadow.positions)
    plan = make_plan(shadow)

    engine.execute(plan, market_data)

    for sale in plan.planned_sales:
        assert shadow.positions[sale.symbol] == pytest.approx(
            positions_before[sale.symbol] - sale.notional
        )


def test_risk_report_matches_final_portfolio(setup):
    shadow, _, engine, market_data = setup
    plan = make_plan(shadow)

    result = engine.execute(plan, market_data)

    assert result.risk_report.nav == pytest.approx(shadow.nav)
    assert result.risk_report.gross_exposure == pytest.approx(
        result.exposure_after
    )


def test_no_action_plan_does_not_trade(setup):
    shadow, trade_engine, engine, market_data = setup
    plan = make_plan(shadow, MarketRegime.RISK_ON)
    nav_before = shadow.nav

    result = engine.execute(plan, market_data)

    assert result.trades == ()
    assert result.executed_notional == pytest.approx(0.0)
    assert result.actual_execution_cost == pytest.approx(0.0)
    assert shadow.nav == pytest.approx(nav_before)
    assert len(trade_engine.trade_history) == 0
    assert result.target_reached is True


def test_rejects_stale_plan_after_market_move(setup):
    shadow, trade_engine, engine, market_data = setup
    plan = make_plan(shadow)

    shadow.update(
        {
            "SPY": 0.01,
            "BTC": 0.01,
            "NVDA": 0.01,
        }
    )

    with pytest.raises(
        DeRiskingExecutionError,
        match="Stale",
    ):
        engine.execute(plan, market_data)

    assert len(trade_engine.trade_history) == 0


def test_rejects_stale_plan_after_cash_position_change(setup):
    shadow, trade_engine, engine, market_data = setup
    plan = make_plan(shadow)

    shadow.cash += 1_000.0

    with pytest.raises(
        DeRiskingExecutionError,
        match="Stale",
    ):
        engine.execute(plan, market_data)

    assert len(trade_engine.trade_history) == 0


def test_missing_market_data_rejects_entire_plan(setup):
    shadow, trade_engine, engine, market_data = setup
    plan = make_plan(shadow)
    positions_before = dict(shadow.positions)
    cash_before = shadow.cash

    del market_data["NVDA"]

    with pytest.raises(
        DeRiskingExecutionError,
        match="Missing market data",
    ):
        engine.execute(plan, market_data)

    assert shadow.positions == positions_before
    assert shadow.cash == pytest.approx(cash_before)
    assert len(trade_engine.trade_history) == 0


def test_liquidity_failure_rejects_entire_plan(setup):
    shadow, trade_engine, engine, market_data = setup
    plan = make_plan(shadow)
    positions_before = dict(shadow.positions)
    cash_before = shadow.cash

    market_data["NVDA"] = AssetExecutionData(
        daily_volume_notional=1.0
    )

    with pytest.raises(
        DeRiskingExecutionError,
        match="Liquidity",
    ):
        engine.execute(plan, market_data)

    assert shadow.positions == positions_before
    assert shadow.cash == pytest.approx(cash_before)
    assert len(trade_engine.trade_history) == 0


def test_invalid_market_data_rejects_entire_plan(setup):
    shadow, trade_engine, engine, market_data = setup
    plan = make_plan(shadow)

    market_data["BTC"] = AssetExecutionData(
        daily_volume_notional=-1.0
    )

    with pytest.raises(
        DeRiskingExecutionError,
        match="Invalid market data",
    ):
        engine.execute(plan, market_data)

    assert len(trade_engine.trade_history) == 0


def test_no_action_plan_preserves_history(setup):
    shadow, trade_engine, engine, market_data = setup
    plan = make_plan(shadow, MarketRegime.RISK_ON)
    history_before = list(shadow.history)

    engine.execute(plan, market_data)

    assert shadow.history == history_before
    assert trade_engine.trade_history == []


def test_execution_updates_virtual_trade_history(setup):
    shadow, trade_engine, engine, market_data = setup
    plan = make_plan(shadow)
    history_length_before = len(shadow.history)

    result = engine.execute(plan, market_data)

    assert len(shadow.history) == (
        history_length_before + len(result.trades)
    )
    assert len(trade_engine.trade_history) == len(result.trades)