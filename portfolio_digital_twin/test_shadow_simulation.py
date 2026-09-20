
import pytest

from portfolio_construction.models import PortfolioAllocation
from portfolio_digital_twin.shadow_simulation import (
    ShadowSimulationEngine,
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

    return ShadowSimulationEngine(
        initial_capital=100_000.0,
        allocation=allocation,
    )


def test_initial_portfolio(engine):
    assert engine.nav == pytest.approx(100_000.0)
    assert engine.cash == pytest.approx(20_000.0)
    assert engine.positions["SPY"] == pytest.approx(50_000.0)
    assert engine.positions["BTC"] == pytest.approx(30_000.0)
    assert engine.pnl == pytest.approx(0.0)
    assert len(engine.history) == 1


def test_positive_returns(engine):
    snapshot = engine.update({
        "SPY": 0.10,
        "BTC": 0.20,
    })

    assert snapshot.nav == pytest.approx(111_000.0)
    assert snapshot.pnl == pytest.approx(11_000.0)
    assert snapshot.pnl_pct == pytest.approx(0.11)
    assert snapshot.cash == pytest.approx(20_000.0)
    assert snapshot.drawdown == pytest.approx(0.0)


def test_negative_returns_and_drawdown(engine):
    engine.update({
        "SPY": 0.10,
        "BTC": 0.20,
    })

    snapshot = engine.update({
        "SPY": -0.10,
        "BTC": -0.20,
    })

    assert snapshot.nav == pytest.approx(98_300.0)
    assert snapshot.pnl == pytest.approx(-1_700.0)
    assert snapshot.drawdown == pytest.approx(
        98_300.0 / 111_000.0 - 1.0
    )


def test_portfolio_exposure(engine):
    snapshot = engine.history[0]

    assert snapshot.gross_exposure == pytest.approx(0.80)
    assert snapshot.net_exposure == pytest.approx(0.80)

    assert sum(
        snapshot.weights.values()
    ) == pytest.approx(0.80)


def test_invalid_initial_capital():
    allocation = PortfolioAllocation(
        weights={"SPY": 0.80},
        cash_weight=0.20,
    )

    with pytest.raises(ValueError):
        ShadowSimulationEngine(
            initial_capital=-100.0,
            allocation=allocation,
        )


def test_missing_asset_returns(engine):
    with pytest.raises(ValueError):
        engine.update({
            "SPY": 0.05,
        })


def test_invalid_returns(engine):
    with pytest.raises(ValueError):
        engine.update({
            "SPY": float("nan"),
            "BTC": 0.05,
        })


def test_failed_update_does_not_change_portfolio(engine):
    original_nav = engine.nav
    original_history_length = len(engine.history)

    with pytest.raises(ValueError):
        engine.update({
            "SPY": 0.05,
            "BTC": -1.50,
        })

    assert engine.nav == pytest.approx(original_nav)

    assert len(engine.history) == (
        original_history_length
    )