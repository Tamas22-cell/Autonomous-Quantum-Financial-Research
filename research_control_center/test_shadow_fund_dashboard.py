import pytest

from research_control_center.models import AllocationMode
from research_control_center.shadow_fund_dashboard import (
    ShadowFundDashboard,
    ShadowFundDashboardConfig,
)


def make_dashboard() -> ShadowFundDashboard:
    return ShadowFundDashboard()


# ============================================================
# CONFIG
# ============================================================


def test_default_config():
    config = ShadowFundDashboardConfig()

    assert config.initial_capital == 100_000.0
    assert config.max_gross_exposure == 1.50
    assert config.max_net_exposure == 1.00
    assert config.max_single_position_weight == 0.25
    assert config.max_drawdown_limit == 0.20


def test_custom_config():
    config = ShadowFundDashboardConfig(
        initial_capital=200_000.0,
        max_gross_exposure=2.0,
        max_net_exposure=0.80,
        max_single_position_weight=0.20,
        max_drawdown_limit=0.15,
    )

    assert config.initial_capital == 200_000.0
    assert config.max_drawdown_limit == 0.15


@pytest.mark.parametrize(
    "field",
    [
        "initial_capital",
        "max_gross_exposure",
        "max_net_exposure",
        "max_single_position_weight",
        "max_drawdown_limit",
    ],
)
def test_boolean_config_rejected(field):
    kwargs = {field: True}

    with pytest.raises(TypeError):
        ShadowFundDashboardConfig(**kwargs)


def test_zero_initial_capital_rejected():
    with pytest.raises(ValueError):
        ShadowFundDashboardConfig(
            initial_capital=0.0
        )


def test_single_position_limit_above_one_rejected():
    with pytest.raises(ValueError):
        ShadowFundDashboardConfig(
            max_single_position_weight=1.1
        )


def test_drawdown_limit_above_one_rejected():
    with pytest.raises(ValueError):
        ShadowFundDashboardConfig(
            max_drawdown_limit=1.1
        )


# ============================================================
# POSITIONS
# ============================================================


def test_set_long_position():
    dashboard = make_dashboard()

    dashboard.set_position(
        "BTC",
        20_000.0,
    )

    assert dashboard.positions()["BTC"] == 20_000.0


def test_set_short_position():
    dashboard = make_dashboard()

    dashboard.set_position(
        "SPY",
        -15_000.0,
    )

    assert dashboard.positions()["SPY"] == -15_000.0


def test_zero_position_removes_asset():
    dashboard = make_dashboard()

    dashboard.set_position(
        "BTC",
        20_000.0,
    )

    dashboard.set_position(
        "BTC",
        0.0,
    )

    assert "BTC" not in dashboard.positions()


def test_remove_position():
    dashboard = make_dashboard()

    dashboard.set_position(
        "BTC",
        20_000.0,
    )

    dashboard.remove_position(
        "BTC"
    )

    assert "BTC" not in dashboard.positions()


def test_positions_returns_copy():
    dashboard = make_dashboard()

    dashboard.set_position(
        "BTC",
        20_000.0,
    )

    positions = dashboard.positions()
    positions["BTC"] = 999_999.0

    assert (
        dashboard.positions()["BTC"]
        == 20_000.0
    )


def test_empty_asset_rejected():
    dashboard = make_dashboard()

    with pytest.raises(ValueError):
        dashboard.set_position(
            "",
            10_000.0,
        )


def test_non_numeric_position_rejected():
    dashboard = make_dashboard()

    with pytest.raises(TypeError):
        dashboard.set_position(
            "BTC",
            "invalid",
        )


def test_non_finite_position_rejected():
    dashboard = make_dashboard()

    with pytest.raises(ValueError):
        dashboard.set_position(
            "BTC",
            float("inf"),
        )


# ============================================================
# CAPITAL
# ============================================================


def test_allocated_capital_uses_absolute_notional():
    dashboard = make_dashboard()

    dashboard.set_position(
        "BTC",
        20_000.0,
    )

    dashboard.set_position(
        "SPY",
        -10_000.0,
    )

    assert (
        dashboard.allocated_capital()
        == 30_000.0
    )


def test_cash():
    dashboard = make_dashboard()

    dashboard.set_position(
        "BTC",
        25_000.0,
    )

    assert dashboard.cash() == 75_000.0


def test_gross_exposure():
    dashboard = make_dashboard()

    dashboard.set_position(
        "BTC",
        20_000.0,
    )

    dashboard.set_position(
        "SPY",
        -10_000.0,
    )

    assert (
        dashboard.gross_exposure()
        == pytest.approx(0.30)
    )


def test_net_notional():
    dashboard = make_dashboard()

    dashboard.set_position(
        "BTC",
        20_000.0,
    )

    dashboard.set_position(
        "SPY",
        -10_000.0,
    )

    assert (
        dashboard.net_notional()
        == 10_000.0
    )


def test_net_exposure():
    dashboard = make_dashboard()

    dashboard.set_position(
        "BTC",
        20_000.0,
    )

    dashboard.set_position(
        "SPY",
        -10_000.0,
    )

    assert (
        dashboard.net_exposure()
        == pytest.approx(0.10)
    )


def test_position_weights():
    dashboard = make_dashboard()

    dashboard.set_position(
        "BTC",
        20_000.0,
    )

    dashboard.set_position(
        "SPY",
        -10_000.0,
    )

    weights = dashboard.position_weights()

    assert weights["BTC"] == pytest.approx(0.20)
    assert weights["SPY"] == pytest.approx(-0.10)


# ============================================================
# P&L
# ============================================================


def test_set_pnl():
    dashboard = make_dashboard()

    dashboard.set_pnl(
        5_000.0
    )

    assert dashboard.pnl() == 5_000.0


def test_add_pnl():
    dashboard = make_dashboard()

    dashboard.set_pnl(
        1_000.0
    )

    dashboard.add_pnl(
        2_500.0
    )

    assert dashboard.pnl() == 3_500.0


def test_negative_pnl():
    dashboard = make_dashboard()

    dashboard.set_pnl(
        -5_000.0
    )

    assert dashboard.pnl() == -5_000.0


def test_return_pct():
    dashboard = make_dashboard()

    dashboard.set_pnl(
        10_000.0
    )

    assert (
        dashboard.return_pct()
        == pytest.approx(0.10)
    )


# ============================================================
# DRAWDOWN
# ============================================================


def test_set_positive_drawdown():
    dashboard = make_dashboard()

    dashboard.set_max_drawdown(
        0.15
    )

    assert (
        dashboard.max_drawdown()
        == pytest.approx(0.15)
    )


def test_negative_drawdown_is_normalized():
    dashboard = make_dashboard()

    dashboard.set_max_drawdown(
        -0.12
    )

    assert (
        dashboard.max_drawdown()
        == pytest.approx(0.12)
    )


def test_drawdown_above_one_rejected():
    dashboard = make_dashboard()

    with pytest.raises(ValueError):
        dashboard.set_max_drawdown(
            1.1
        )


# ============================================================
# EXPOSURE BREAKDOWNS
# ============================================================


def test_set_asset_exposure():
    dashboard = make_dashboard()

    dashboard.set_asset_exposure(
        {
            "BTC": 0.30,
            "SPY": 0.20,
        }
    )

    snapshot = dashboard.snapshot()

    assert (
        snapshot.asset_exposure["BTC"]
        == 0.30
    )


def test_set_sector_exposure():
    dashboard = make_dashboard()

    dashboard.set_sector_exposure(
        {
            "Technology": 0.40,
            "Energy": 0.10,
        }
    )

    snapshot = dashboard.snapshot()

    assert (
        snapshot.sector_exposure[
            "Technology"
        ]
        == 0.40
    )


def test_set_factor_exposure():
    dashboard = make_dashboard()

    dashboard.set_factor_exposure(
        {
            "Momentum": 0.50,
            "Value": -0.10,
        }
    )

    snapshot = dashboard.snapshot()

    assert (
        snapshot.factor_exposure[
            "Momentum"
        ]
        == 0.50
    )


def test_invalid_exposure_mapping_rejected():
    dashboard = make_dashboard()

    with pytest.raises(TypeError):
        dashboard.set_asset_exposure(
            "invalid"
        )


def test_invalid_exposure_value_rejected():
    dashboard = make_dashboard()

    with pytest.raises(TypeError):
        dashboard.set_asset_exposure(
            {
                "BTC": "invalid",
            }
        )


# ============================================================
# ATTRIBUTION
# ============================================================


def test_set_attribution():
    dashboard = make_dashboard()

    dashboard.set_attribution(
        {
            "MacroAgent": 1500.0,
            "TechnicalAgent": 500.0,
        }
    )

    snapshot = dashboard.snapshot()

    assert (
        snapshot.attribution[
            "MacroAgent"
        ]
        == 1500.0
    )


def test_add_attribution():
    dashboard = make_dashboard()

    dashboard.add_attribution(
        "MacroAgent",
        1000.0,
    )

    dashboard.add_attribution(
        "MacroAgent",
        500.0,
    )

    snapshot = dashboard.snapshot()

    assert (
        snapshot.attribution[
            "MacroAgent"
        ]
        == 1500.0
    )


# ============================================================
# RISK
# ============================================================


def test_portfolio_within_limits():
    dashboard = make_dashboard()

    dashboard.set_position(
        "BTC",
        20_000.0,
    )

    dashboard.set_position(
        "SPY",
        15_000.0,
    )

    report = dashboard.risk_report()

    assert report["within_limits"] is True
    assert report["breach_count"] == 0


def test_single_position_breach():
    dashboard = make_dashboard()

    dashboard.set_position(
        "BTC",
        30_000.0,
    )

    report = dashboard.risk_report()

    assert (
        report[
            "position_breaches"
        ]["BTC"]
        == pytest.approx(0.30)
    )

    assert (
        "MAX_SINGLE_POSITION"
        in report["breaches"]
    )


def test_gross_exposure_breach():
    config = ShadowFundDashboardConfig(
        max_gross_exposure=0.50,
    )

    dashboard = ShadowFundDashboard(
        config
    )

    dashboard.set_position(
        "BTC",
        30_000.0,
    )

    dashboard.set_position(
        "SPY",
        -30_000.0,
    )

    report = dashboard.risk_report()

    assert (
        report[
            "gross_exposure_breach"
        ]
        is True
    )


def test_net_exposure_breach():
    config = ShadowFundDashboardConfig(
        max_net_exposure=0.30,
    )

    dashboard = ShadowFundDashboard(
        config
    )

    dashboard.set_position(
        "BTC",
        40_000.0,
    )

    report = dashboard.risk_report()

    assert (
        report[
            "net_exposure_breach"
        ]
        is True
    )


def test_drawdown_breach():
    dashboard = make_dashboard()

    dashboard.set_max_drawdown(
        0.25
    )

    report = dashboard.risk_report()

    assert (
        report[
            "drawdown_breach"
        ]
        is True
    )

    assert (
        "MAX_DRAWDOWN"
        in report["breaches"]
    )


def test_risk_report_is_research_only():
    dashboard = make_dashboard()

    report = dashboard.risk_report()

    assert report["research_only"] is True
    assert (
        report["real_money_execution"]
        is False
    )


# ============================================================
# SNAPSHOT
# ============================================================


def test_snapshot():
    dashboard = make_dashboard()

    dashboard.set_position(
        "BTC",
        20_000.0,
    )

    dashboard.set_pnl(
        5_000.0
    )

    snapshot = dashboard.snapshot()

    assert snapshot.total_capital == 100_000.0
    assert snapshot.allocated_capital == 20_000.0
    assert snapshot.cash == 80_000.0
    assert snapshot.pnl == 5_000.0
    assert snapshot.return_pct == pytest.approx(0.05)
    assert snapshot.mode == AllocationMode.SHADOW


def test_snapshot_safety_metadata():
    dashboard = make_dashboard()

    snapshot = dashboard.snapshot()

    assert snapshot.metadata["research_only"] is True
    assert snapshot.metadata["paper_allocation"] is True
    assert (
        snapshot.metadata[
            "real_money_execution"
        ]
        is False
    )
    assert (
        snapshot.metadata[
            "broker_order_allowed"
        ]
        is False
    )


# ============================================================
# DASHBOARD
# ============================================================


def test_dashboard_view():
    dashboard = make_dashboard()

    dashboard.set_position(
        "BTC",
        20_000.0,
    )

    dashboard.set_pnl(
        2_500.0
    )

    view = dashboard.dashboard()

    assert (
        view["capital"][
            "initial_capital"
        ]
        == 100_000.0
    )

    assert (
        view["performance"]["pnl"]
        == 2_500.0
    )

    assert (
        view["exposure"]["positions"][
            "BTC"
        ]
        == 20_000.0
    )


def test_dashboard_is_shadow_only():
    dashboard = make_dashboard()

    view = dashboard.dashboard()

    assert view["mode"] == "SHADOW"
    assert view["research_only"] is True
    assert view["paper_allocation"] is True
    assert view["real_money_execution"] is False
    assert view["broker_order_allowed"] is False


# ============================================================
# SUMMARY
# ============================================================


def test_summary():
    dashboard = make_dashboard()

    dashboard.set_position(
        "BTC",
        20_000.0,
    )

    dashboard.set_pnl(
        5_000.0
    )

    summary = dashboard.summary()

    assert summary["positions"] == 1
    assert summary["allocated_capital"] == 20_000.0
    assert summary["cash"] == 80_000.0
    assert summary["pnl"] == 5_000.0
    assert summary["within_risk_limits"] is True


# ============================================================
# AUDIT
# ============================================================


def test_position_update_creates_audit_event():
    dashboard = make_dashboard()

    dashboard.set_position(
        "BTC",
        20_000.0,
    )

    events = dashboard.audit_log()

    assert len(events) == 1
    assert (
        events[0]["event"]
        == "SHADOW_POSITION_UPDATED"
    )


def test_multiple_operations_are_audited():
    dashboard = make_dashboard()

    dashboard.set_position(
        "BTC",
        20_000.0,
    )

    dashboard.set_pnl(
        1_000.0
    )

    dashboard.set_max_drawdown(
        0.10
    )

    dashboard.add_attribution(
        "MacroAgent",
        500.0,
    )

    events = dashboard.audit_log()

    assert len(events) == 4


def test_audit_is_research_only():
    dashboard = make_dashboard()

    dashboard.set_position(
        "BTC",
        10_000.0,
    )

    event = dashboard.audit_log()[0]

    assert event["research_only"] is True
    assert (
        event["real_money_execution"]
        is False
    )