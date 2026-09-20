
import pytest

from portfolio_construction.models import PortfolioAllocation
from portfolio_digital_twin.regime_derisking import (
    DeRiskingConfig,
    DeRiskingError,
    MarketRegime,
    RegimeDeRiskingEngine,
)
from portfolio_digital_twin.shadow_simulation import (
    ShadowSimulationEngine,
)


@pytest.fixture
def shadow():
    allocation = PortfolioAllocation(
        weights={
            "SPY": 0.40,
            "BTC": 0.30,
            "NVDA": 0.20,
        },
        cash_weight=0.10,
    )

    return ShadowSimulationEngine(
        initial_capital=100_000.0,
        allocation=allocation,
    )


@pytest.fixture
def engine():
    return RegimeDeRiskingEngine()


def test_risk_on_requires_no_action(shadow, engine):
    plan = engine.plan(shadow, MarketRegime.RISK_ON)

    assert plan.regime == MarketRegime.RISK_ON
    assert plan.current_exposure == pytest.approx(0.90)
    assert plan.target_max_exposure == pytest.approx(1.00)
    assert plan.required_reduction == pytest.approx(0.0)
    assert plan.planned_sales == ()
    assert plan.action_required is False


def test_neutral_regime_reduces_exposure(shadow, engine):
    plan = engine.plan(shadow, MarketRegime.NEUTRAL)

    assert plan.target_max_exposure == pytest.approx(0.85)
    assert plan.required_reduction == pytest.approx(5_000.0)
    assert plan.action_required is True
    assert "REGIME_EXPOSURE_LIMIT" in plan.reasons


def test_risk_off_regime_reduces_exposure(shadow, engine):
    plan = engine.plan(shadow, MarketRegime.RISK_OFF)

    assert plan.target_max_exposure == pytest.approx(0.60)
    assert plan.required_reduction == pytest.approx(30_000.0)
    assert plan.action_required is True
    assert "REGIME_EXPOSURE_LIMIT" in plan.reasons


def test_crisis_regime_reduces_exposure(shadow, engine):
    plan = engine.plan(shadow, MarketRegime.CRISIS)

    assert plan.target_max_exposure == pytest.approx(0.30)
    assert plan.required_reduction == pytest.approx(60_000.0)
    assert plan.action_required is True
    assert "REGIME_EXPOSURE_LIMIT" in plan.reasons


def test_planned_sales_are_proportional(shadow, engine):
    plan = engine.plan(shadow, MarketRegime.RISK_OFF)

    sales = {
        sale.symbol: sale.notional
        for sale in plan.planned_sales
    }

    assert sales["SPY"] == pytest.approx(
        30_000.0 * 40_000.0 / 90_000.0
    )
    assert sales["BTC"] == pytest.approx(
        30_000.0 * 30_000.0 / 90_000.0
    )
    assert sales["NVDA"] == pytest.approx(
        30_000.0 * 20_000.0 / 90_000.0
    )

    assert sum(sales.values()) == pytest.approx(30_000.0)


def test_drawdown_triggers_additional_derisking(shadow, engine):
    shadow.update(
        {
            "SPY": -0.20,
            "BTC": -0.20,
            "NVDA": -0.20,
        }
    )

    plan = engine.plan(shadow, MarketRegime.RISK_ON)

    assert plan.nav == pytest.approx(82_000.0)
    assert plan.current_exposure == pytest.approx(
        72_000.0 / 82_000.0
    )
    assert plan.target_max_exposure == pytest.approx(0.50)
    assert plan.required_reduction == pytest.approx(31_000.0)
    assert "DRAWDOWN_DERISKING" in plan.reasons
    assert plan.action_required is True


def test_crisis_limit_remains_stricter_than_drawdown_limit(
    shadow,
    engine,
):
    shadow.update(
        {
            "SPY": -0.20,
            "BTC": -0.20,
            "NVDA": -0.20,
        }
    )

    plan = engine.plan(shadow, MarketRegime.CRISIS)

    assert plan.target_max_exposure == pytest.approx(0.30)
    assert plan.required_reduction == pytest.approx(47_400.0)
    assert plan.action_required is True


def test_minimum_trade_threshold(shadow):
    engine = RegimeDeRiskingEngine(
        config=DeRiskingConfig(
            neutral_max_exposure=0.89995,
            minimum_trade_notional=10.0,
        )
    )

    plan = engine.plan(shadow, MarketRegime.NEUTRAL)

    assert plan.required_reduction == pytest.approx(0.0)
    assert plan.planned_sales == ()
    assert plan.action_required is False


def test_custom_exposure_limits(shadow):
    engine = RegimeDeRiskingEngine(
        config=DeRiskingConfig(
            risk_on_max_exposure=0.95,
            neutral_max_exposure=0.80,
            risk_off_max_exposure=0.50,
            crisis_max_exposure=0.20,
        )
    )

    plan = engine.plan(shadow, MarketRegime.RISK_OFF)

    assert plan.target_max_exposure == pytest.approx(0.50)
    assert plan.required_reduction == pytest.approx(40_000.0)


def test_invalid_regime(shadow, engine):
    with pytest.raises(DeRiskingError):
        engine.plan(shadow, "RISK_OFF")


def test_invalid_exposure_configuration():
    with pytest.raises(DeRiskingError):
        DeRiskingConfig(
            risk_on_max_exposure=0.50,
            neutral_max_exposure=0.85,
        )


def test_invalid_minimum_trade_notional():
    with pytest.raises(DeRiskingError):
        DeRiskingConfig(
            minimum_trade_notional=-1.0,
        )


def test_planning_does_not_modify_portfolio(shadow, engine):
    nav_before = shadow.nav
    cash_before = shadow.cash
    positions_before = dict(shadow.positions)
    history_before = list(shadow.history)

    engine.plan(shadow, MarketRegime.CRISIS)

    assert shadow.nav == pytest.approx(nav_before)
    assert shadow.cash == pytest.approx(cash_before)
    assert shadow.positions == positions_before
    assert shadow.history == history_before


def test_no_action_after_exposure_reduction(shadow, engine):
    shadow.positions["SPY"] = 10_000.0
    shadow.positions["BTC"] = 10_000.0
    shadow.positions["NVDA"] = 10_000.0
    shadow.cash = 70_000.0

    plan = engine.plan(shadow, MarketRegime.CRISIS)

    assert plan.current_exposure == pytest.approx(0.30)
    assert plan.required_reduction == pytest.approx(0.0)
    assert plan.planned_sales == ()
    assert plan.action_required is False