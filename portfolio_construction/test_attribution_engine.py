"""
Tests for the v5.0.7 Portfolio Attribution Engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from portfolio_construction.attribution_engine import (
    AttributionEngineError,
    AttributionReport,
    PortfolioAttributionEngine,
)
from portfolio_construction.models import (
    AssetDefinition,
    AttributionRecord,
    AttributionSource,
    CovarianceMatrix,
    PortfolioAllocation,
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


def build_asset_returns() -> dict[str, float]:
    return {
        "SPY": 0.05,
        "NVDA": 0.10,
        "GLD": -0.02,
    }


def build_covariance() -> CovarianceMatrix:
    return CovarianceMatrix(
        symbols=[
            "SPY",
            "NVDA",
            "GLD",
        ],
        matrix=[
            [
                0.0400,
                0.0200,
                0.0050,
            ],
            [
                0.0200,
                0.0900,
                0.0020,
            ],
            [
                0.0050,
                0.0020,
                0.0225,
            ],
        ],
    )


def build_alpha_sources() -> dict[str, str]:
    return {
        "SPY": "market_alpha",
        "NVDA": "growth_alpha",
        "GLD": "defensive_alpha",
    }


def build_agent_sources() -> dict[str, str]:
    return {
        "SPY": "macro_agent",
        "NVDA": "stock_agent",
        "GLD": "macro_agent",
    }


def build_regime_sources() -> dict[str, str]:
    return {
        "SPY": "risk_on",
        "NVDA": "risk_on",
        "GLD": "defensive",
    }


def build_risk_sources() -> dict[str, str]:
    return {
        "SPY": "market_risk",
        "NVDA": "growth_risk",
        "GLD": "commodity_risk",
    }


def build_report() -> AttributionReport:
    engine = PortfolioAttributionEngine()

    return engine.analyze(
        allocation=build_allocation(),
        assets=build_assets(),
        asset_returns=build_asset_returns(),
        covariance=build_covariance(),
        alpha_sources=build_alpha_sources(),
        agent_sources=build_agent_sources(),
        regime_sources=build_regime_sources(),
        risk_sources=build_risk_sources(),
    )


# ==============================================================
# Basic report
# ==============================================================


def test_analyze_returns_attribution_report() -> None:
    report = build_report()

    assert isinstance(
        report,
        AttributionReport,
    )


def test_report_contains_records() -> None:
    report = build_report()

    assert report.record_count > 0


def test_return_reconciliation_passes() -> None:
    report = build_report()

    assert report.return_reconciled is True


def test_risk_reconciliation_passes() -> None:
    report = build_report()

    assert report.risk_reconciled is True


# ==============================================================
# Portfolio return
# ==============================================================


def test_portfolio_return_is_calculated() -> None:
    report = build_report()

    expected = (
        0.50 * 0.05
        + 0.30 * 0.10
        + 0.20 * -0.02
    )

    assert report.portfolio_return == pytest.approx(
        expected
    )


def test_total_return_contribution_matches_portfolio_return() -> None:
    report = build_report()

    assert (
        report.total_return_contribution
        == pytest.approx(
            report.portfolio_return
        )
    )


# ==============================================================
# Portfolio risk
# ==============================================================


def test_portfolio_variance_is_calculated() -> None:
    report = build_report()

    weights = np.asarray(
        [
            0.50,
            0.30,
            0.20,
        ],
        dtype=float,
    )

    covariance = np.asarray(
        build_covariance().matrix,
        dtype=float,
    )

    expected_variance = float(
        weights.T.dot(
            covariance
        ).dot(
            weights
        )
    )

    assert report.portfolio_variance == pytest.approx(
        expected_variance
    )


def test_portfolio_volatility_is_calculated() -> None:
    report = build_report()

    expected = np.sqrt(
        report.portfolio_variance
    )

    assert report.portfolio_volatility == pytest.approx(
        expected
    )


def test_total_risk_contribution_matches_variance() -> None:
    report = build_report()

    assert (
        report.total_risk_contribution
        == pytest.approx(
            report.portfolio_variance
        )
    )


# ==============================================================
# Asset return attribution
# ==============================================================


def test_asset_return_contributions_are_calculated() -> None:
    contributions = (
        PortfolioAttributionEngine
        .asset_return_contributions(
            allocation=build_allocation(),
            assets=build_assets(),
            asset_returns=build_asset_returns(),
        )
    )

    assert contributions["SPY"] == pytest.approx(
        0.025
    )

    assert contributions["NVDA"] == pytest.approx(
        0.030
    )

    assert contributions["GLD"] == pytest.approx(
        -0.004
    )


def test_asset_records_are_generated() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.ASSET
    )

    assert len(records) == 3

    source_ids = {
        record.source_id
        for record in records
    }

    assert source_ids == {
        "SPY",
        "NVDA",
        "GLD",
    }


def test_asset_records_reconcile_return() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.ASSET
    )

    total = sum(
        record.return_contribution
        for record in records
    )

    assert total == pytest.approx(
        report.portfolio_return
    )


def test_asset_weights_reconcile_invested_weight() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.ASSET
    )

    total = sum(
        record.weight_contribution
        for record in records
    )

    assert total == pytest.approx(
        1.0
    )


# ==============================================================
# Asset risk attribution
# ==============================================================


def test_asset_risk_contributions_are_calculated() -> None:
    engine = PortfolioAttributionEngine()

    contributions = (
        engine.asset_risk_contributions(
            allocation=build_allocation(),
            assets=build_assets(),
            covariance=build_covariance(),
        )
    )

    assert set(
        contributions
    ) == {
        "SPY",
        "NVDA",
        "GLD",
    }


def test_asset_risk_contributions_sum_to_variance() -> None:
    engine = PortfolioAttributionEngine()

    contributions = (
        engine.asset_risk_contributions(
            allocation=build_allocation(),
            assets=build_assets(),
            covariance=build_covariance(),
        )
    )

    report = build_report()

    assert sum(
        contributions.values()
    ) == pytest.approx(
        report.portfolio_variance
    )


def test_percentage_risk_contributions_sum_to_one() -> None:
    report = build_report()

    percentages = (
        report.metadata[
            "percentage_risk_contribution"
        ]
    )

    assert sum(
        percentages.values()
    ) == pytest.approx(
        1.0
    )


# ==============================================================
# Sector attribution
# ==============================================================


def test_sector_records_are_generated() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.SECTOR
    )

    source_ids = {
        record.source_id
        for record in records
    }

    assert source_ids == {
        "broad_market",
        "technology",
        "gold",
    }


def test_sector_return_attribution_reconciles() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.SECTOR
    )

    total = sum(
        record.return_contribution
        for record in records
    )

    assert total == pytest.approx(
        report.portfolio_return
    )


def test_sector_risk_attribution_reconciles() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.SECTOR
    )

    total = sum(
        record.risk_contribution
        for record in records
    )

    assert total == pytest.approx(
        report.portfolio_variance
    )


# ==============================================================
# Asset-class attribution
# ==============================================================


def test_asset_class_records_are_generated() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.ASSET_CLASS
    )

    source_ids = {
        record.source_id
        for record in records
    }

    assert source_ids == {
        "equity",
        "commodity",
    }


def test_asset_class_weight_attribution_reconciles() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.ASSET_CLASS
    )

    total = sum(
        record.weight_contribution
        for record in records
    )

    assert total == pytest.approx(
        1.0
    )


# ==============================================================
# Alpha attribution
# ==============================================================


def test_alpha_records_are_generated() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.ALPHA
    )

    source_ids = {
        record.source_id
        for record in records
    }

    assert source_ids == {
        "market_alpha",
        "growth_alpha",
        "defensive_alpha",
    }


def test_alpha_return_attribution_reconciles() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.ALPHA
    )

    total = sum(
        record.return_contribution
        for record in records
    )

    assert total == pytest.approx(
        report.portfolio_return
    )


# ==============================================================
# Agent attribution
# ==============================================================


def test_agent_records_are_generated() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.AGENT
    )

    source_ids = {
        record.source_id
        for record in records
    }

    assert source_ids == {
        "macro_agent",
        "stock_agent",
    }


def test_agent_group_aggregation_is_correct() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.AGENT
    )

    macro_record = next(
        record
        for record in records
        if record.source_id
        == "macro_agent"
    )

    expected_weight = (
        0.50
        + 0.20
    )

    expected_return = (
        0.50 * 0.05
        + 0.20 * -0.02
    )

    assert macro_record.weight_contribution == pytest.approx(
        expected_weight
    )

    assert macro_record.return_contribution == pytest.approx(
        expected_return
    )


def test_agent_return_attribution_reconciles() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.AGENT
    )

    total = sum(
        record.return_contribution
        for record in records
    )

    assert total == pytest.approx(
        report.portfolio_return
    )


# ==============================================================
# Regime attribution
# ==============================================================


def test_regime_records_are_generated() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.REGIME
    )

    source_ids = {
        record.source_id
        for record in records
    }

    assert source_ids == {
        "risk_on",
        "defensive",
    }


def test_regime_return_attribution_reconciles() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.REGIME
    )

    total = sum(
        record.return_contribution
        for record in records
    )

    assert total == pytest.approx(
        report.portfolio_return
    )


# ==============================================================
# Risk-source attribution
# ==============================================================


def test_risk_source_records_are_generated() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.RISK
    )

    source_ids = {
        record.source_id
        for record in records
    }

    assert source_ids == {
        "market_risk",
        "growth_risk",
        "commodity_risk",
    }


def test_risk_source_attribution_reconciles_variance() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.RISK
    )

    total = sum(
        record.risk_contribution
        for record in records
    )

    assert total == pytest.approx(
        report.portfolio_variance
    )


# ==============================================================
# Factor attribution
# ==============================================================


def test_factor_records_are_generated() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.FACTOR
    )

    source_ids = {
        record.source_id
        for record in records
    }

    assert source_ids == {
        "market",
        "growth",
        "value",
    }


def test_factor_market_exposure_is_correct() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.FACTOR
    )

    market_record = next(
        record
        for record in records
        if record.source_id
        == "market"
    )

    expected = (
        0.50 * 1.00
        + 0.30 * 1.30
        + 0.20 * 0.10
    )

    assert market_record.weight_contribution == pytest.approx(
        expected
    )


def test_factor_growth_exposure_is_correct() -> None:
    report = build_report()

    records = report.records_by_source(
        AttributionSource.FACTOR
    )

    growth_record = next(
        record
        for record in records
        if record.source_id
        == "growth"
    )

    expected = (
        0.50 * 0.20
        + 0.30 * 1.20
        + 0.20 * -0.10
    )

    assert growth_record.weight_contribution == pytest.approx(
        expected
    )


# ==============================================================
# Aggregation helper
# ==============================================================


def test_aggregate_records_groups_source_ids() -> None:
    records = [
        AttributionRecord(
            source_type=AttributionSource.AGENT,
            source_id="macro_agent",
            return_contribution=0.01,
            risk_contribution=0.02,
            weight_contribution=0.30,
        ),
        AttributionRecord(
            source_type=AttributionSource.AGENT,
            source_id="macro_agent",
            return_contribution=0.02,
            risk_contribution=0.03,
            weight_contribution=0.20,
        ),
        AttributionRecord(
            source_type=AttributionSource.AGENT,
            source_id="stock_agent",
            return_contribution=0.04,
            risk_contribution=0.05,
            weight_contribution=0.50,
        ),
    ]

    aggregated = (
        PortfolioAttributionEngine
        .aggregate_records(
            records=records,
            source_type=AttributionSource.AGENT,
        )
    )

    assert aggregated[
        "macro_agent"
    ][
        "return_contribution"
    ] == pytest.approx(
        0.03
    )

    assert aggregated[
        "macro_agent"
    ][
        "risk_contribution"
    ] == pytest.approx(
        0.05
    )

    assert aggregated[
        "macro_agent"
    ][
        "weight_contribution"
    ] == pytest.approx(
        0.50
    )


def test_aggregate_records_ignores_other_source_types() -> None:
    records = [
        AttributionRecord(
            source_type=AttributionSource.AGENT,
            source_id="macro_agent",
            return_contribution=0.01,
        ),
        AttributionRecord(
            source_type=AttributionSource.ASSET,
            source_id="SPY",
            return_contribution=0.02,
        ),
    ]

    aggregated = (
        PortfolioAttributionEngine
        .aggregate_records(
            records=records,
            source_type=AttributionSource.AGENT,
        )
    )

    assert set(
        aggregated
    ) == {
        "macro_agent"
    }


# ==============================================================
# Unclassified groups
# ==============================================================


def test_missing_agent_mapping_becomes_unclassified() -> None:
    engine = PortfolioAttributionEngine()

    report = engine.analyze(
        allocation=build_allocation(),
        assets=build_assets(),
        asset_returns=build_asset_returns(),
        covariance=build_covariance(),
        agent_sources={
            "SPY": "macro_agent",
            "NVDA": "stock_agent",
        },
    )

    records = report.records_by_source(
        AttributionSource.AGENT
    )

    source_ids = {
        record.source_id
        for record in records
    }

    assert "unclassified" in source_ids


def test_missing_sector_becomes_unclassified() -> None:
    assets = [
        AssetDefinition(
            symbol="SPY",
            asset_class="equity",
            sector=None,
        ),
        AssetDefinition(
            symbol="NVDA",
            asset_class="equity",
            sector="technology",
        ),
        AssetDefinition(
            symbol="GLD",
            asset_class="commodity",
            sector="gold",
        ),
    ]

    engine = PortfolioAttributionEngine()

    report = engine.analyze(
        allocation=build_allocation(),
        assets=assets,
        asset_returns=build_asset_returns(),
        covariance=build_covariance(),
    )

    records = report.records_by_source(
        AttributionSource.SECTOR
    )

    source_ids = {
        record.source_id
        for record in records
    }

    assert "unclassified" in source_ids


# ==============================================================
# Metadata
# ==============================================================


def test_report_metadata_contains_asset_count() -> None:
    report = build_report()

    assert report.metadata[
        "asset_count"
    ] == 3


def test_report_metadata_contains_component_variance() -> None:
    report = build_report()

    components = report.metadata[
        "component_variance"
    ]

    assert set(
        components
    ) == {
        "SPY",
        "NVDA",
        "GLD",
    }


def test_report_metadata_contains_marginal_variance() -> None:
    report = build_report()

    marginal = report.metadata[
        "marginal_variance"
    ]

    assert set(
        marginal
    ) == {
        "SPY",
        "NVDA",
        "GLD",
    }


# ==============================================================
# Validation
# ==============================================================


def test_missing_asset_return_raises_error() -> None:
    engine = PortfolioAttributionEngine()

    asset_returns = {
        "SPY": 0.05,
        "NVDA": 0.10,
    }

    with pytest.raises(
        AttributionEngineError
    ):
        engine.analyze(
            allocation=build_allocation(),
            assets=build_assets(),
            asset_returns=asset_returns,
            covariance=build_covariance(),
        )


def test_nonfinite_asset_return_raises_error() -> None:
    engine = PortfolioAttributionEngine()

    asset_returns = build_asset_returns()

    asset_returns[
        "NVDA"
    ] = np.nan

    with pytest.raises(
        AttributionEngineError
    ):
        engine.analyze(
            allocation=build_allocation(),
            assets=build_assets(),
            asset_returns=asset_returns,
            covariance=build_covariance(),
        )


def test_covariance_symbol_order_mismatch_raises_error() -> None:
    engine = PortfolioAttributionEngine()

    covariance = CovarianceMatrix(
        symbols=[
            "NVDA",
            "SPY",
            "GLD",
        ],
        matrix=[
            [
                0.09,
                0.02,
                0.002,
            ],
            [
                0.02,
                0.04,
                0.005,
            ],
            [
                0.002,
                0.005,
                0.0225,
            ],
        ],
    )

    with pytest.raises(
        AttributionEngineError
    ):
        engine.analyze(
            allocation=build_allocation(),
            assets=build_assets(),
            asset_returns=build_asset_returns(),
            covariance=covariance,
        )


def test_non_psd_covariance_raises_error() -> None:
    engine = PortfolioAttributionEngine()

    covariance = CovarianceMatrix(
        symbols=[
            "SPY",
            "NVDA",
            "GLD",
        ],
        matrix=[
            [
                1.0,
                2.0,
                0.0,
            ],
            [
                2.0,
                1.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
    )

    with pytest.raises(
        AttributionEngineError
    ):
        engine.analyze(
            allocation=build_allocation(),
            assets=build_assets(),
            asset_returns=build_asset_returns(),
            covariance=covariance,
        )


def test_unknown_allocation_symbol_raises_error() -> None:
    engine = PortfolioAttributionEngine()

    allocation = PortfolioAllocation(
        weights={
            "SPY": 0.40,
            "NVDA": 0.30,
            "GLD": 0.20,
            "UNKNOWN": 0.10,
        },
        cash_weight=0.0,
    )

    with pytest.raises(
        AttributionEngineError
    ):
        engine.analyze(
            allocation=allocation,
            assets=build_assets(),
            asset_returns=build_asset_returns(),
            covariance=build_covariance(),
        )


def test_empty_asset_list_raises_error() -> None:
    engine = PortfolioAttributionEngine()

    with pytest.raises(
        AttributionEngineError
    ):
        engine.analyze(
            allocation=build_allocation(),
            assets=[],
            asset_returns=build_asset_returns(),
            covariance=build_covariance(),
        )


def test_invalid_engine_tolerance_raises_error() -> None:
    with pytest.raises(
        AttributionEngineError
    ):
        PortfolioAttributionEngine(
            tolerance=-0.01
        )