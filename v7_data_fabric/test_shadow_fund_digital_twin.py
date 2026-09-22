from __future__ import annotations

from datetime import datetime, timezone

import pytest

from v7_data_fabric.shadow_fund_digital_twin import (
    ScenarioSeverity,
    ShadowFundDigitalTwin,
    ShadowFundDigitalTwinError,
    StressScenario,
    VirtualFill,
    VirtualOrderSide,
)


def _buy_fill(
    fill_id: str = "fill-001",
    asset_id: str = "BTC",
    quantity: float = 10.0,
    price: float = 1_000.0,
) -> VirtualFill:
    return VirtualFill(
        fill_id=fill_id,
        asset_id=asset_id,
        side=VirtualOrderSide.BUY,
        quantity=quantity,
        price=price,
        commission=5.0,
        spread_cost=2.0,
        slippage_cost=1.0,
        impact_cost=1.0,
        executed_at=datetime.now(timezone.utc),
    )


def test_version_and_safety_flags():
    twin = ShadowFundDigitalTwin()

    assert twin.VERSION == "v7.0.9"
    assert twin.RESEARCH_ONLY is True
    assert twin.REAL_MONEY_EXECUTION is False
    assert twin.BROKER_CONNECTION is False


def test_virtual_fill_updates_cash_and_position():
    twin = ShadowFundDigitalTwin(
        initial_cash=100_000.0
    )

    fill = twin.record_virtual_fill(
        _buy_fill()
    )

    position = twin.get_position("BTC")

    assert fill.fill_id == "fill-001"
    assert twin.fill_count == 1
    assert position.quantity == 10.0
    assert position.market_price == 1_000.0
    assert twin.cash_balance == 89_991.0
    assert twin.cumulative_costs == 9.0


def test_mark_to_market_updates_position_value():
    twin = ShadowFundDigitalTwin(
        initial_cash=100_000.0
    )

    twin.record_virtual_fill(
        _buy_fill()
    )

    positions = twin.mark_to_market(
        {"BTC": 1_200.0}
    )

    assert len(positions) == 1
    assert positions[0].market_price == 1_200.0
    assert positions[0].market_value == 12_000.0
    assert round(positions[0].unrealized_pnl, 6) == 1_991.0


def test_snapshot_reconstructs_portfolio_state():
    twin = ShadowFundDigitalTwin(
        initial_cash=100_000.0
    )

    twin.record_virtual_fill(
        _buy_fill()
    )
    twin.mark_to_market(
        {"BTC": 1_100.0}
    )

    snapshot = twin.snapshot("snapshot-001")

    assert snapshot.snapshot_id == "snapshot-001"
    assert snapshot.cash_balance == 89_991.0
    assert snapshot.total_market_value == 11_000.0
    assert snapshot.total_equity == 100_991.0
    assert snapshot.cumulative_costs == 9.0
    assert round(snapshot.cumulative_unrealized_pnl, 6) == 991.0
    assert twin.snapshot_count == 1


def test_virtual_sell_records_realized_pnl():
    twin = ShadowFundDigitalTwin(
        initial_cash=100_000.0
    )

    twin.record_virtual_fill(
        _buy_fill()
    )

    sell = VirtualFill(
        fill_id="fill-002",
        asset_id="BTC",
        side=VirtualOrderSide.SELL,
        quantity=4.0,
        price=1_200.0,
        commission=3.0,
        executed_at=datetime.now(timezone.utc),
    )

    twin.record_virtual_fill(sell)

    position = twin.get_position("BTC")

    assert position.quantity == 6.0
    assert round(position.realized_pnl, 6) == 793.4
    assert twin.fill_count == 2


def test_stress_scenario_detects_critical_loss():
    twin = ShadowFundDigitalTwin(
        initial_cash=100_000.0
    )

    twin.record_virtual_fill(
        _buy_fill()
    )

    scenario = StressScenario(
        scenario_id="stress-001",
        name="Severe BTC shock",
        price_shocks={"BTC": -0.50},
        liquidity_factor=0.90,
        description="Severe market stress.",
    )

    result = twin.run_scenario(scenario)

    assert result.starting_value == 10_000.0
    assert result.stressed_value == 4_500.0
    assert result.pnl_change == -5_500.0
    assert result.pnl_change_ratio == -0.55
    assert result.severity == ScenarioSeverity.CRITICAL
    assert result.affected_assets == ("BTC",)
    assert twin.scenario_count == 1


def test_zero_shock_scenario_is_clear():
    twin = ShadowFundDigitalTwin(
        initial_cash=100_000.0
    )

    twin.record_virtual_fill(
        _buy_fill()
    )

    scenario = StressScenario(
        scenario_id="stress-002",
        name="Neutral scenario",
        price_shocks={},
    )

    result = twin.run_scenario(scenario)

    assert result.pnl_change == 0.0
    assert result.pnl_change_ratio == 0.0
    assert result.severity == ScenarioSeverity.NONE


def test_overselling_virtual_position_is_rejected():
    twin = ShadowFundDigitalTwin(
        initial_cash=100_000.0
    )

    twin.record_virtual_fill(
        _buy_fill(quantity=2.0)
    )

    sell = VirtualFill(
        fill_id="fill-invalid",
        asset_id="BTC",
        side=VirtualOrderSide.SELL,
        quantity=3.0,
        price=1_100.0,
        executed_at=datetime.now(timezone.utc),
    )

    with pytest.raises(
        ShadowFundDigitalTwinError
    ):
        twin.record_virtual_fill(sell)


def test_duplicate_snapshot_is_rejected():
    twin = ShadowFundDigitalTwin()

    twin.snapshot("snapshot-001")

    with pytest.raises(
        ShadowFundDigitalTwinError
    ):
        twin.snapshot("snapshot-001")


def test_summary_contains_digital_twin_state():
    twin = ShadowFundDigitalTwin(
        initial_cash=100_000.0
    )

    twin.record_virtual_fill(
        _buy_fill()
    )
    twin.snapshot("snapshot-001")

    summary = twin.summary()

    assert summary["version"] == "v7.0.9"
    assert summary["initial_cash"] == 100_000.0
    assert summary["fill_count"] == 1
    assert summary["snapshot_count"] == 1
    assert summary["scenario_count"] == 0
    assert summary["research_only"] is True
    assert summary["real_money_execution"] is False
    assert summary["broker_connection"] is False