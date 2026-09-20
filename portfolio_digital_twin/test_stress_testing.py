
import pytest

from portfolio_construction.models import PortfolioAllocation
from portfolio_digital_twin.risk_monitor import RiskLimits
from portfolio_digital_twin.shadow_simulation import (
    ShadowSimulationEngine,
)
from portfolio_digital_twin.stress_testing import (
    PortfolioStressTestingEngine,
    StressScenario,
    StressTestingError,
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
    return PortfolioStressTestingEngine()


def test_no_market_change(shadow, engine):
    result = engine.run(
        shadow,
        StressScenario(
            name="No market change",
            asset_returns={},
        ),
    )

    assert result.nav_before == pytest.approx(100_000.0)
    assert result.nav_after == pytest.approx(100_000.0)
    assert result.pnl == pytest.approx(0.0)
    assert result.pnl_pct == pytest.approx(0.0)
    assert result.drawdown_after == pytest.approx(0.0)


def test_market_crash(shadow, engine):
    result = engine.run(
        shadow,
        StressScenario(
            name="Market crash",
            asset_returns={
                "SPY": -0.30,
                "BTC": -0.50,
                "NVDA": -0.40,
            },
        ),
    )

    # Losses: 12,000 + 15,000 + 8,000 = 35,000.
    assert result.nav_after == pytest.approx(65_000.0)
    assert result.pnl == pytest.approx(-35_000.0)
    assert result.pnl_pct == pytest.approx(-0.35)
    assert result.drawdown_after == pytest.approx(0.35)

    assert "MAX_DRAWDOWN" in result.risk_violations
    assert result.within_limits is False


def test_single_asset_shock(shadow, engine):
    result = engine.run(
        shadow,
        StressScenario(
            name="BTC crash",
            asset_returns={"BTC": -0.50},
        ),
    )

    assert result.positions_after["SPY"] == pytest.approx(
        40_000.0
    )
    assert result.positions_after["BTC"] == pytest.approx(
        15_000.0
    )
    assert result.positions_after["NVDA"] == pytest.approx(
        20_000.0
    )

    assert result.nav_after == pytest.approx(85_000.0)
    assert result.pnl_pct == pytest.approx(-0.15)


def test_positive_market_scenario(shadow, engine):
    result = engine.run(
        shadow,
        StressScenario(
            name="Market rally",
            asset_returns={
                "SPY": 0.10,
                "BTC": 0.20,
                "NVDA": 0.15,
            },
        ),
    )

    # Gains: 4,000 + 6,000 + 3,000 = 13,000.
    assert result.nav_after == pytest.approx(113_000.0)
    assert result.pnl == pytest.approx(13_000.0)
    assert result.pnl_pct == pytest.approx(0.13)
    assert result.drawdown_after == pytest.approx(0.0)


def test_cash_remains_unchanged(shadow, engine):
    result = engine.run(
        shadow,
        StressScenario(
            name="Equity decline",
            asset_returns={"SPY": -0.20},
        ),
    )

    assert result.cash_after == pytest.approx(10_000.0)


def test_unspecified_assets_receive_zero_return(shadow, engine):
    result = engine.run(
        shadow,
        StressScenario(
            name="SPY shock only",
            asset_returns={"SPY": -0.10},
        ),
    )

    assert result.positions_after["BTC"] == pytest.approx(
        30_000.0
    )
    assert result.positions_after["NVDA"] == pytest.approx(
        20_000.0
    )
    assert result.nav_after == pytest.approx(96_000.0)


def test_stress_test_does_not_modify_original_portfolio(
    shadow,
    engine,
):
    nav_before = shadow.nav
    cash_before = shadow.cash
    positions_before = dict(shadow.positions)
    history_before = list(shadow.history)

    engine.run(
        shadow,
        StressScenario(
            name="Severe crash",
            asset_returns={
                "SPY": -0.60,
                "BTC": -0.70,
                "NVDA": -0.50,
            },
        ),
    )

    assert shadow.nav == pytest.approx(nav_before)
    assert shadow.cash == pytest.approx(cash_before)
    assert shadow.positions == positions_before
    assert shadow.history == history_before


def test_custom_drawdown_limit(shadow):
    engine = PortfolioStressTestingEngine(
        risk_limits=RiskLimits(
            max_single_position_weight=1.0,
            max_gross_exposure=1.0,
            max_drawdown=0.10,
            min_cash_weight=0.0,
        )
    )

    result = engine.run(
        shadow,
        StressScenario(
            name="Moderate decline",
            asset_returns={"SPY": -0.30},
        ),
    )

    assert result.drawdown_after == pytest.approx(0.12)
    assert "MAX_DRAWDOWN" in result.risk_violations


def test_concentration_after_asset_rally(shadow, engine):
    result = engine.run(
        shadow,
        StressScenario(
            name="SPY concentration",
            asset_returns={"SPY": 1.0},
        ),
    )

    assert result.nav_after == pytest.approx(140_000.0)
    assert result.positions_after["SPY"] == pytest.approx(
        80_000.0
    )
    assert (
        "SINGLE_POSITION_CONCENTRATION"
        in result.risk_violations
    )


def test_invalid_empty_scenario_name():
    with pytest.raises(StressTestingError):
        StressScenario(
            name="",
            asset_returns={"SPY": -0.10},
        )


def test_invalid_return_below_minus_one():
    with pytest.raises(StressTestingError):
        StressScenario(
            name="Invalid loss",
            asset_returns={"SPY": -1.10},
        )


def test_invalid_nonfinite_return():
    with pytest.raises(StressTestingError):
        StressScenario(
            name="Invalid return",
            asset_returns={"SPY": float("nan")},
        )


def test_total_loss_of_all_positions(shadow, engine):
    result = engine.run(
        shadow,
        StressScenario(
            name="Total asset loss",
            asset_returns={
                "SPY": -1.0,
                "BTC": -1.0,
                "NVDA": -1.0,
            },
        ),
    )

    # The remaining 10,000 cash keeps NAV positive.
    assert result.nav_after == pytest.approx(10_000.0)
    assert result.cash_after == pytest.approx(10_000.0)
    assert result.drawdown_after == pytest.approx(0.90)
    assert "MAX_DRAWDOWN" in result.risk_violations


def test_nonpositive_stressed_nav_is_rejected(engine):
    allocation = PortfolioAllocation(
        weights={"SPY": 1.0},
        cash_weight=0.0,
    )

    fully_invested_shadow = ShadowSimulationEngine(
        initial_capital=100_000.0,
        allocation=allocation,
    )

    with pytest.raises(
        StressTestingError,
        match="non-positive NAV",
    ):
        engine.run(
            fully_invested_shadow,
            StressScenario(
                name="Total portfolio loss",
                asset_returns={"SPY": -1.0},
            ),
        )