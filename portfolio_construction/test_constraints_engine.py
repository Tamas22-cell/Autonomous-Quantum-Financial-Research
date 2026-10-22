"""
Tests for the v5.0.7 Institutional Portfolio Constraints Engine.
"""

from __future__ import annotations

import pytest

from portfolio_construction.constraints_engine import (
    ConstraintsEngineError,
    PortfolioConstraintsEngine,
)
from portfolio_construction.models import (
    AssetDefinition,
    ConstraintType,
    LeverageConstraint,
    PortfolioAllocation,
    PortfolioConstraint,
    TurnoverConstraint,
)


# ==============================================================
# Helpers
# ==============================================================


def build_assets() -> list[AssetDefinition]:
    return [
        AssetDefinition(
            symbol="SPY",
            asset_class="equity",
            sector="broad_market",
            liquidity_score=0.95,
            factor_exposures={
                "market": 1.00,
                "growth": 0.20,
                "value": 0.10,
            },
        ),
        AssetDefinition(
            symbol="NVDA",
            asset_class="equity",
            sector="technology",
            liquidity_score=0.90,
            factor_exposures={
                "market": 1.30,
                "growth": 1.20,
                "value": -0.20,
            },
        ),
        AssetDefinition(
            symbol="GLD",
            asset_class="commodity",
            sector="gold",
            liquidity_score=0.85,
            factor_exposures={
                "market": 0.10,
                "growth": -0.10,
                "value": 0.30,
            },
        ),
    ]


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


def build_current_allocation() -> PortfolioAllocation:
    allocation = PortfolioAllocation(
        weights={
            "SPY": 0.40,
            "NVDA": 0.20,
            "GLD": 0.30,
        },
        cash_weight=0.10,
    )
    allocation.validate()
    return allocation


# ==============================================================
# Basic compliance
# ==============================================================


def test_empty_constraints_are_compliant() -> None:
    engine = PortfolioConstraintsEngine()

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
    )

    assert report.compliant is True
    assert report.violation_count == 0
    assert report.violations == []


def test_report_contains_basic_exposures() -> None:
    engine = PortfolioConstraintsEngine()

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
    )

    assert "position" in report.exposures
    assert "sector" in report.exposures
    assert "asset_class" in report.exposures
    assert "factor" in report.exposures
    assert "liquidity" in report.exposures

    assert report.gross_exposure == pytest.approx(1.0)
    assert report.net_exposure == pytest.approx(1.0)
    assert report.cash_weight == pytest.approx(0.0)


# ==============================================================
# Position exposures
# ==============================================================


def test_position_exposures_are_calculated() -> None:
    exposures = PortfolioConstraintsEngine.position_exposures(
        allocation=build_allocation(),
        assets=build_assets(),
    )

    assert exposures["SPY"] == pytest.approx(0.50)
    assert exposures["NVDA"] == pytest.approx(0.30)
    assert exposures["GLD"] == pytest.approx(0.20)


def test_position_maximum_constraint_passes() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.POSITION,
        target="NVDA",
        maximum=0.35,
    )

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        constraints=[constraint],
    )

    assert report.compliant is True


def test_position_maximum_constraint_detects_violation() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.POSITION,
        target="NVDA",
        maximum=0.25,
    )

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        constraints=[constraint],
    )

    assert report.compliant is False
    assert report.violation_count == 1

    violation = report.violations[0]

    assert violation.constraint_type == ConstraintType.POSITION
    assert violation.target == "NVDA"
    assert violation.actual == pytest.approx(0.30)
    assert violation.maximum == pytest.approx(0.25)
    assert violation.amount == pytest.approx(0.05)


def test_position_minimum_constraint_detects_violation() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.POSITION,
        target="GLD",
        minimum=0.25,
    )

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        constraints=[constraint],
    )

    assert report.compliant is False

    violation = report.violations[0]

    assert violation.target == "GLD"
    assert violation.actual == pytest.approx(0.20)
    assert violation.minimum == pytest.approx(0.25)
    assert violation.amount == pytest.approx(0.05)


# ==============================================================
# MIN_WEIGHT / MAX_WEIGHT
# ==============================================================


def test_min_weight_constraint() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.MIN_WEIGHT,
        target="SPY",
        minimum=0.60,
    )

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        constraints=[constraint],
    )

    assert report.compliant is False
    assert report.violations[0].actual == pytest.approx(0.50)


def test_max_weight_constraint() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.MAX_WEIGHT,
        target="NVDA",
        maximum=0.20,
    )

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        constraints=[constraint],
    )

    assert report.compliant is False
    assert report.violations[0].actual == pytest.approx(0.30)


# ==============================================================
# Sector constraints
# ==============================================================


def test_sector_exposures_are_calculated() -> None:
    exposures = PortfolioConstraintsEngine.sector_exposures(
        allocation=build_allocation(),
        assets=build_assets(),
    )

    assert exposures["broad_market"] == pytest.approx(0.50)
    assert exposures["technology"] == pytest.approx(0.30)
    assert exposures["gold"] == pytest.approx(0.20)


def test_sector_constraint_detects_violation() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.SECTOR,
        target="technology",
        maximum=0.25,
    )

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        constraints=[constraint],
    )

    assert report.compliant is False

    violation = report.violations[0]

    assert violation.constraint_type == ConstraintType.SECTOR
    assert violation.target == "technology"
    assert violation.actual == pytest.approx(0.30)


def test_sector_constraint_passes() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.SECTOR,
        target="technology",
        maximum=0.35,
    )

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        constraints=[constraint],
    )

    assert report.compliant is True


# ==============================================================
# Asset-class constraints
# ==============================================================


def test_asset_class_exposures_are_calculated() -> None:
    exposures = PortfolioConstraintsEngine.asset_class_exposures(
        allocation=build_allocation(),
        assets=build_assets(),
    )

    assert exposures["equity"] == pytest.approx(0.80)
    assert exposures["commodity"] == pytest.approx(0.20)


def test_asset_class_constraint_detects_violation() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.ASSET_CLASS,
        target="equity",
        maximum=0.75,
    )

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        constraints=[constraint],
    )

    assert report.compliant is False

    violation = report.violations[0]

    assert violation.constraint_type == ConstraintType.ASSET_CLASS
    assert violation.actual == pytest.approx(0.80)


def test_asset_class_minimum_constraint_passes() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.ASSET_CLASS,
        target="commodity",
        minimum=0.15,
    )

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        constraints=[constraint],
    )

    assert report.compliant is True


# ==============================================================
# Factor constraints
# ==============================================================


def test_factor_exposures_are_calculated() -> None:
    exposures = PortfolioConstraintsEngine.factor_exposures(
        allocation=build_allocation(),
        assets=build_assets(),
    )

    expected_market = (
        0.50 * 1.00
        + 0.30 * 1.30
        + 0.20 * 0.10
    )

    expected_growth = (
        0.50 * 0.20
        + 0.30 * 1.20
        + 0.20 * -0.10
    )

    expected_value = (
        0.50 * 0.10
        + 0.30 * -0.20
        + 0.20 * 0.30
    )

    assert exposures["market"] == pytest.approx(expected_market)
    assert exposures["growth"] == pytest.approx(expected_growth)
    assert exposures["value"] == pytest.approx(expected_value)


def test_factor_maximum_constraint_detects_violation() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.FACTOR,
        target="growth",
        maximum=0.40,
    )

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        constraints=[constraint],
    )

    assert report.compliant is False

    violation = report.violations[0]

    assert violation.constraint_type == ConstraintType.FACTOR
    assert violation.target == "growth"
    assert violation.actual == pytest.approx(0.44)


def test_factor_minimum_constraint_detects_violation() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.FACTOR,
        target="value",
        minimum=0.10,
    )

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        constraints=[constraint],
    )

    assert report.compliant is False

    violation = report.violations[0]

    assert violation.target == "value"
    assert violation.actual == pytest.approx(0.05)


# ==============================================================
# Liquidity constraints
# ==============================================================


def test_weighted_liquidity_score_is_calculated() -> None:
    score = PortfolioConstraintsEngine.weighted_liquidity_score(
        allocation=build_allocation(),
        assets=build_assets(),
    )

    expected = (
        0.50 * 0.95
        + 0.30 * 0.90
        + 0.20 * 0.85
    )

    assert score == pytest.approx(expected)


def test_liquidity_constraint_passes() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.LIQUIDITY,
        target="portfolio",
        minimum=0.90,
    )

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        constraints=[constraint],
    )

    assert report.compliant is True


def test_liquidity_constraint_detects_violation() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.LIQUIDITY,
        target="portfolio",
        minimum=0.93,
    )

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        constraints=[constraint],
    )

    assert report.compliant is False
    assert report.violations[0].constraint_type == ConstraintType.LIQUIDITY


# ==============================================================
# Turnover
# ==============================================================


def test_turnover_is_calculated() -> None:
    turnover = PortfolioConstraintsEngine.calculate_turnover(
        allocation=build_allocation(),
        current_allocation=build_current_allocation(),
        assets=build_assets(),
    )

    assert turnover == pytest.approx(0.20)


def test_turnover_constraint_passes() -> None:
    engine = PortfolioConstraintsEngine()

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        current_allocation=build_current_allocation(),
        turnover_constraint=TurnoverConstraint(
            maximum_turnover=0.25
        ),
    )

    assert report.compliant is True
    assert report.turnover == pytest.approx(0.20)


def test_turnover_constraint_detects_violation() -> None:
    engine = PortfolioConstraintsEngine()

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        current_allocation=build_current_allocation(),
        turnover_constraint=TurnoverConstraint(
            maximum_turnover=0.15
        ),
    )

    assert report.compliant is False

    violation = report.violations[0]

    assert violation.constraint_type == ConstraintType.TURNOVER
    assert violation.actual == pytest.approx(0.20)
    assert violation.maximum == pytest.approx(0.15)
    assert violation.amount == pytest.approx(0.05)


def test_generic_turnover_constraint() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.TURNOVER,
        target="portfolio",
        maximum=0.15,
    )

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        constraints=[constraint],
        current_allocation=build_current_allocation(),
    )

    assert report.compliant is False
    assert report.violations[0].constraint_type == ConstraintType.TURNOVER


# ==============================================================
# Leverage
# ==============================================================


def test_gross_and_net_exposure_are_calculated() -> None:
    allocation = build_allocation()

    gross = PortfolioConstraintsEngine.gross_exposure(
        allocation=allocation
    )

    net = PortfolioConstraintsEngine.net_exposure(
        allocation=allocation
    )

    assert gross == pytest.approx(1.0)
    assert net == pytest.approx(1.0)


def test_leverage_constraint_passes() -> None:
    engine = PortfolioConstraintsEngine()

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        leverage_constraint=LeverageConstraint(
            maximum_gross_exposure=1.0
        ),
    )

    assert report.compliant is True


def test_leverage_constraint_detects_violation() -> None:
    engine = PortfolioConstraintsEngine()

    allocation = PortfolioAllocation(
        weights={
            "SPY": 0.70,
            "NVDA": 0.50,
            "GLD": 0.20,
        },
        cash_weight=0.0,
    )

    report = engine.validate(
        allocation=allocation,
        assets=build_assets(),
        leverage_constraint=LeverageConstraint(
            maximum_gross_exposure=1.20
        ),
    )

    assert report.compliant is False

    violation = report.violations[0]

    assert violation.constraint_type == ConstraintType.LEVERAGE
    assert violation.actual == pytest.approx(1.40)
    assert violation.maximum == pytest.approx(1.20)


def test_generic_leverage_constraint() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.LEVERAGE,
        target="portfolio",
        maximum=0.90,
    )

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        constraints=[constraint],
    )

    assert report.compliant is False
    assert report.violations[0].constraint_type == ConstraintType.LEVERAGE


# ==============================================================
# Cash constraints
# ==============================================================


def test_cash_constraint_passes() -> None:
    engine = PortfolioConstraintsEngine()

    allocation = build_current_allocation()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.CASH,
        target="portfolio",
        minimum=0.05,
        maximum=0.15,
    )

    report = engine.validate(
        allocation=allocation,
        assets=build_assets(),
        constraints=[constraint],
    )

    assert report.compliant is True


def test_cash_minimum_constraint_detects_violation() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.CASH,
        target="portfolio",
        minimum=0.10,
    )

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        constraints=[constraint],
    )

    assert report.compliant is False
    assert report.violations[0].actual == pytest.approx(0.0)


# ==============================================================
# Multiple constraints
# ==============================================================


def test_multiple_violations_are_reported() -> None:
    engine = PortfolioConstraintsEngine()

    constraints = [
        PortfolioConstraint(
            constraint_type=ConstraintType.POSITION,
            target="NVDA",
            maximum=0.20,
        ),
        PortfolioConstraint(
            constraint_type=ConstraintType.ASSET_CLASS,
            target="equity",
            maximum=0.70,
        ),
        PortfolioConstraint(
            constraint_type=ConstraintType.FACTOR,
            target="growth",
            maximum=0.40,
        ),
    ]

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        constraints=constraints,
    )

    assert report.compliant is False
    assert report.violation_count == 3


def test_disabled_constraint_is_ignored() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.POSITION,
        target="NVDA",
        maximum=0.10,
        enabled=False,
    )

    report = engine.validate(
        allocation=build_allocation(),
        assets=build_assets(),
        constraints=[constraint],
    )

    assert report.compliant is True
    assert report.violation_count == 0


# ==============================================================
# Error handling
# ==============================================================


def test_unknown_position_target_raises_error() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.POSITION,
        target="UNKNOWN",
        maximum=0.20,
    )

    with pytest.raises(ConstraintsEngineError):
        engine.validate(
            allocation=build_allocation(),
            assets=build_assets(),
            constraints=[constraint],
        )


def test_turnover_constraint_without_current_allocation_raises_error() -> None:
    engine = PortfolioConstraintsEngine()

    with pytest.raises(ConstraintsEngineError):
        engine.validate(
            allocation=build_allocation(),
            assets=build_assets(),
            turnover_constraint=TurnoverConstraint(
                maximum_turnover=0.20
            ),
        )


def test_generic_turnover_without_current_allocation_raises_error() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.TURNOVER,
        target="portfolio",
        maximum=0.20,
    )

    with pytest.raises(ConstraintsEngineError):
        engine.validate(
            allocation=build_allocation(),
            assets=build_assets(),
            constraints=[constraint],
        )


def test_liquidity_constraint_without_liquidity_data_raises_error() -> None:
    assets = [
        AssetDefinition(
            symbol="SPY",
            asset_class="equity",
        ),
        AssetDefinition(
            symbol="NVDA",
            asset_class="equity",
        ),
        AssetDefinition(
            symbol="GLD",
            asset_class="commodity",
        ),
    ]

    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.LIQUIDITY,
        target="portfolio",
        minimum=0.80,
    )

    with pytest.raises(ConstraintsEngineError):
        engine.validate(
            allocation=build_allocation(),
            assets=assets,
            constraints=[constraint],
        )


def test_unknown_allocation_symbol_raises_error() -> None:
    engine = PortfolioConstraintsEngine()

    allocation = PortfolioAllocation(
        weights={
            "SPY": 0.40,
            "NVDA": 0.30,
            "GLD": 0.20,
            "UNKNOWN": 0.10,
        },
        cash_weight=0.0,
    )

    with pytest.raises(ConstraintsEngineError):
        engine.validate(
            allocation=allocation,
            assets=build_assets(),
        )


def test_empty_asset_list_raises_error() -> None:
    engine = PortfolioConstraintsEngine()

    with pytest.raises(ConstraintsEngineError):
        engine.validate(
            allocation=build_allocation(),
            assets=[],
        )


def test_invalid_engine_tolerance_raises_error() -> None:
    with pytest.raises(ConstraintsEngineError):
        PortfolioConstraintsEngine(
            tolerance=-0.01
        )


def test_min_weight_without_minimum_raises_error() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.MIN_WEIGHT,
        target="SPY",
        maximum=0.80,
    )

    with pytest.raises(ConstraintsEngineError):
        engine.validate(
            allocation=build_allocation(),
            assets=build_assets(),
            constraints=[constraint],
        )


def test_max_weight_without_maximum_raises_error() -> None:
    engine = PortfolioConstraintsEngine()

    constraint = PortfolioConstraint(
        constraint_type=ConstraintType.MAX_WEIGHT,
        target="SPY",
        minimum=0.20,
    )

    with pytest.raises(ConstraintsEngineError):
        engine.validate(
            allocation=build_allocation(),
            assets=build_assets(),
            constraints=[constraint],
        )