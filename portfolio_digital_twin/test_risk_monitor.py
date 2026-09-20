
import pytest

from portfolio_construction.models import PortfolioAllocation
from portfolio_digital_twin.risk_monitor import (
    PortfolioRiskMonitor,
    RiskLimits,
    RiskMonitorError,
)
from portfolio_digital_twin.shadow_simulation import (
    ShadowSimulationEngine,
)
from portfolio_digital_twin.trade_execution import (
    TradeExecutionEngine,
    VirtualTrade,
)


def make_shadow(
    weights=None,
    cash_weight=0.20,
    initial_capital=100_000.0,
):
    if weights is None:
        weights = {
            "SPY": 0.30,
            "BTC": 0.25,
            "NVDA": 0.25,
        }

    allocation = PortfolioAllocation(
        weights=weights,
        cash_weight=cash_weight,
    )

    return ShadowSimulationEngine(
        initial_capital=initial_capital,
        allocation=allocation,
    )


def test_portfolio_within_limits():
    shadow = make_shadow()

    report = PortfolioRiskMonitor().evaluate(shadow)

    assert report.nav == pytest.approx(100_000.0)
    assert report.cash_weight == pytest.approx(0.20)
    assert report.gross_exposure == pytest.approx(0.80)
    assert report.net_exposure == pytest.approx(0.80)
    assert report.drawdown == pytest.approx(0.0)
    assert report.violations == ()
    assert report.within_limits is True


def test_position_weights():
    shadow = make_shadow()

    report = PortfolioRiskMonitor().evaluate(shadow)

    assert report.position_weights["SPY"] == pytest.approx(0.30)
    assert report.position_weights["BTC"] == pytest.approx(0.25)
    assert report.position_weights["NVDA"] == pytest.approx(0.25)


def test_largest_position():
    shadow = make_shadow()

    report = PortfolioRiskMonitor().evaluate(shadow)

    assert report.largest_position_symbol == "SPY"
    assert report.largest_position_weight == pytest.approx(0.30)


def test_single_position_concentration_violation():
    shadow = make_shadow(
        weights={
            "SPY": 0.60,
            "BTC": 0.20,
        },
        cash_weight=0.20,
    )

    report = PortfolioRiskMonitor().evaluate(shadow)

    assert "SINGLE_POSITION_CONCENTRATION" in report.violations
    assert report.within_limits is False


def test_minimum_cash_violation():
    shadow = make_shadow(
        weights={
            "SPY": 0.35,
            "BTC": 0.35,
            "NVDA": 0.28,
        },
        cash_weight=0.02,
    )

    report = PortfolioRiskMonitor().evaluate(shadow)

    assert "MIN_CASH_WEIGHT" in report.violations
    assert report.within_limits is False


def test_drawdown_violation():
    shadow = make_shadow()

    shadow.update(
        {
            "SPY": -0.40,
            "BTC": -0.40,
            "NVDA": -0.40,
        }
    )

    report = PortfolioRiskMonitor().evaluate(shadow)

    assert report.drawdown == pytest.approx(0.32)
    assert "MAX_DRAWDOWN" in report.violations
    assert report.within_limits is False


def test_custom_limits():
    shadow = make_shadow()

    monitor = PortfolioRiskMonitor(
        limits=RiskLimits(
            max_single_position_weight=0.25,
            max_gross_exposure=0.75,
            max_drawdown=0.10,
            min_cash_weight=0.25,
        )
    )

    report = monitor.evaluate(shadow)

    assert "SINGLE_POSITION_CONCENTRATION" in report.violations
    assert "GROSS_EXPOSURE_LIMIT" in report.violations
    assert "MIN_CASH_WEIGHT" in report.violations
    assert "MAX_DRAWDOWN" not in report.violations
    assert report.within_limits is False


def test_all_cash_portfolio():
    shadow = make_shadow(
        weights={"SPY": 0.50},
        cash_weight=0.50,
    )

    execution = TradeExecutionEngine(shadow=shadow)

    execution.execute(
        VirtualTrade(
            symbol="SPY",
            side="SELL",
            notional=50_000.0,
            daily_volume_notional=1_000_000.0,
        )
    )

    report = PortfolioRiskMonitor().evaluate(shadow)

    assert report.largest_position_weight == pytest.approx(0.0)
    assert report.gross_exposure == pytest.approx(0.0)
    assert report.net_exposure == pytest.approx(0.0)
    assert report.cash_weight == pytest.approx(1.0)
    assert report.within_limits is True


def test_monitor_does_not_modify_portfolio():
    shadow = make_shadow()

    cash_before = shadow.cash
    positions_before = dict(shadow.positions)
    history_before = len(shadow.history)

    PortfolioRiskMonitor().evaluate(shadow)

    assert shadow.cash == cash_before
    assert shadow.positions == positions_before
    assert len(shadow.history) == history_before


def test_invalid_risk_limit():
    with pytest.raises(RiskMonitorError):
        RiskLimits(max_drawdown=-0.10)

    with pytest.raises(RiskMonitorError):
        RiskLimits(min_cash_weight=float("nan"))


def test_invalid_portfolio_nav():
    shadow = make_shadow()
    shadow.cash = -100_000.0

    with pytest.raises(RiskMonitorError, match="NAV"):
        PortfolioRiskMonitor().evaluate(shadow)


def test_invalid_position_value():
    shadow = make_shadow()
    shadow.positions["SPY"] = float("nan")

    with pytest.raises(RiskMonitorError):
        PortfolioRiskMonitor().evaluate(shadow)


def test_risk_report_after_virtual_trade():
    shadow = make_shadow()
    execution = TradeExecutionEngine(shadow=shadow)

    execution.execute(
        VirtualTrade(
            symbol="SPY",
            side="BUY",
            notional=1_000.0,
            daily_volume_notional=1_000_000.0,
        )
    )

    report = PortfolioRiskMonitor().evaluate(shadow)

    assert report.nav < 100_000.0
    assert report.position_weights["SPY"] > 0.30
    assert report.cash_weight < 0.20


def test_drawdown_at_limit_is_allowed():
    shadow = make_shadow(
        weights={"SPY": 1.0},
        cash_weight=0.0,
    )

    shadow.update({"SPY": -0.20})

    monitor = PortfolioRiskMonitor(
        limits=RiskLimits(
            max_single_position_weight=1.0,
            max_gross_exposure=1.0,
            max_drawdown=0.20,
            min_cash_weight=0.0,
        )
    )

    report = monitor.evaluate(shadow)

    assert report.drawdown == pytest.approx(0.20)
    assert "MAX_DRAWDOWN" not in report.violations
    assert report.within_limits is True