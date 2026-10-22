"""
Portfolio Attribution Engine for the
v5.0.7 Portfolio Construction Engine.

Capabilities:

- Asset return attribution
- Asset risk attribution
- Alpha attribution
- Agent attribution
- Regime attribution
- Risk-source attribution
- Sector attribution
- Asset-class attribution
- Factor attribution
- Portfolio return reconciliation
- Portfolio variance / volatility decomposition
- Standard AttributionRecord output
- Audit-friendly attribution report

The implementation uses NumPy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .models import (
    AssetDefinition,
    AttributionRecord,
    AttributionSource,
    CovarianceMatrix,
    PortfolioAllocation,
)


class AttributionEngineError(RuntimeError):
    """
    Raised when portfolio attribution cannot be
    calculated safely.
    """


@dataclass
class AttributionReport:
    """
    Complete portfolio attribution result.
    """

    records: List[AttributionRecord] = field(
        default_factory=list
    )

    portfolio_return: Optional[float] = None
    portfolio_variance: Optional[float] = None
    portfolio_volatility: Optional[float] = None

    total_return_contribution: float = 0.0
    total_risk_contribution: float = 0.0

    return_reconciled: bool = True
    risk_reconciled: bool = True

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def record_count(self) -> int:
        return len(self.records)

    def records_by_source(
        self,
        source_type: AttributionSource,
    ) -> List[AttributionRecord]:
        """
        Return attribution records for one source type.
        """

        return [
            record
            for record in self.records
            if record.source_type == source_type
        ]


class PortfolioAttributionEngine:
    """
    Institutional portfolio attribution engine.

    Return attribution
    ------------------

    Asset contribution:

        weight_i * return_i

    Risk attribution
    ----------------

    Portfolio variance:

        variance = w' Σ w

    Marginal variance contribution:

        MRC_i = (Σw)_i

    Component variance contribution:

        CRC_i = w_i * MRC_i

    The component variance contributions sum to
    portfolio variance.

    Percentage risk contribution:

        CRC_i / portfolio_variance

    The engine can additionally aggregate asset-level
    contributions by:

    - sector
    - asset class
    - alpha source
    - agent
    - regime
    - risk source

    Factor attribution uses weighted asset factor
    exposures.
    """

    def __init__(
        self,
        *,
        tolerance: float = 1e-8,
    ) -> None:
        if tolerance < 0.0:
            raise AttributionEngineError(
                "tolerance cannot be negative."
            )

        self.tolerance = float(
            tolerance
        )

    # ==========================================================
    # Main API
    # ==========================================================

    def analyze(
        self,
        *,
        allocation: PortfolioAllocation,
        assets: List[AssetDefinition],
        asset_returns: Mapping[str, float],
        covariance: CovarianceMatrix,
        alpha_sources: Optional[
            Mapping[str, str]
        ] = None,
        agent_sources: Optional[
            Mapping[str, str]
        ] = None,
        regime_sources: Optional[
            Mapping[str, str]
        ] = None,
        risk_sources: Optional[
            Mapping[str, str]
        ] = None,
    ) -> AttributionReport:
        """
        Produce a complete portfolio attribution report.
        """

        self._validate_inputs(
            allocation=allocation,
            assets=assets,
            asset_returns=asset_returns,
            covariance=covariance,
        )

        symbols = [
            asset.symbol
            for asset in assets
        ]

        weights = self._weight_vector(
            allocation=allocation,
            symbols=symbols,
        )

        returns = self._return_vector(
            asset_returns=asset_returns,
            symbols=symbols,
        )

        covariance_matrix = np.asarray(
            covariance.matrix,
            dtype=float,
        )

        asset_return_contributions = (
            weights * returns
        )

        portfolio_return = float(
            np.sum(
                asset_return_contributions
            )
        )

        marginal_variance = (
            covariance_matrix.dot(
                weights
            )
        )

        component_variance = (
            weights
            * marginal_variance
        )

        portfolio_variance = float(
            weights.T.dot(
                covariance_matrix
            ).dot(
                weights
            )
        )

        portfolio_variance = max(
            portfolio_variance,
            0.0,
        )

        portfolio_volatility = float(
            np.sqrt(
                portfolio_variance
            )
        )

        records: List[
            AttributionRecord
        ] = []

        records.extend(
            self._asset_records(
                assets=assets,
                weights=weights,
                return_contributions=(
                    asset_return_contributions
                ),
                risk_contributions=(
                    component_variance
                ),
                portfolio_variance=(
                    portfolio_variance
                ),
            )
        )

        records.extend(
            self._group_records(
                source_type=(
                    AttributionSource.SECTOR
                ),
                groups={
                    asset.symbol: (
                        asset.sector
                        if asset.sector is not None
                        else "unclassified"
                    )
                    for asset in assets
                },
                symbols=symbols,
                weights=weights,
                return_contributions=(
                    asset_return_contributions
                ),
                risk_contributions=(
                    component_variance
                ),
                portfolio_variance=(
                    portfolio_variance
                ),
            )
        )

        records.extend(
            self._group_records(
                source_type=(
                    AttributionSource.ASSET_CLASS
                ),
                groups={
                    asset.symbol: (
                        asset.asset_class
                    )
                    for asset in assets
                },
                symbols=symbols,
                weights=weights,
                return_contributions=(
                    asset_return_contributions
                ),
                risk_contributions=(
                    component_variance
                ),
                portfolio_variance=(
                    portfolio_variance
                ),
            )
        )

        if alpha_sources is not None:
            records.extend(
                self._group_records(
                    source_type=(
                        AttributionSource.ALPHA
                    ),
                    groups=alpha_sources,
                    symbols=symbols,
                    weights=weights,
                    return_contributions=(
                        asset_return_contributions
                    ),
                    risk_contributions=(
                        component_variance
                    ),
                    portfolio_variance=(
                        portfolio_variance
                    ),
                )
            )

        if agent_sources is not None:
            records.extend(
                self._group_records(
                    source_type=(
                        AttributionSource.AGENT
                    ),
                    groups=agent_sources,
                    symbols=symbols,
                    weights=weights,
                    return_contributions=(
                        asset_return_contributions
                    ),
                    risk_contributions=(
                        component_variance
                    ),
                    portfolio_variance=(
                        portfolio_variance
                    ),
                )
            )

        if regime_sources is not None:
            records.extend(
                self._group_records(
                    source_type=(
                        AttributionSource.REGIME
                    ),
                    groups=regime_sources,
                    symbols=symbols,
                    weights=weights,
                    return_contributions=(
                        asset_return_contributions
                    ),
                    risk_contributions=(
                        component_variance
                    ),
                    portfolio_variance=(
                        portfolio_variance
                    ),
                )
            )

        if risk_sources is not None:
            records.extend(
                self._group_records(
                    source_type=(
                        AttributionSource.RISK
                    ),
                    groups=risk_sources,
                    symbols=symbols,
                    weights=weights,
                    return_contributions=(
                        asset_return_contributions
                    ),
                    risk_contributions=(
                        component_variance
                    ),
                    portfolio_variance=(
                        portfolio_variance
                    ),
                )
            )

        records.extend(
            self._factor_records(
                allocation=allocation,
                assets=assets,
            )
        )

        total_return_contribution = float(
            np.sum(
                asset_return_contributions
            )
        )

        total_risk_contribution = float(
            np.sum(
                component_variance
            )
        )

        return_reconciled = bool(
            abs(
                total_return_contribution
                - portfolio_return
            )
            <= self.tolerance
        )

        risk_reconciled = bool(
            abs(
                total_risk_contribution
                - portfolio_variance
            )
            <= self.tolerance
        )

        return AttributionReport(
            records=records,
            portfolio_return=(
                portfolio_return
            ),
            portfolio_variance=(
                portfolio_variance
            ),
            portfolio_volatility=(
                portfolio_volatility
            ),
            total_return_contribution=(
                total_return_contribution
            ),
            total_risk_contribution=(
                total_risk_contribution
            ),
            return_reconciled=(
                return_reconciled
            ),
            risk_reconciled=(
                risk_reconciled
            ),
            metadata={
                "asset_count": len(
                    assets
                ),
                "invested_weight": float(
                    np.sum(
                        weights
                    )
                ),
                "cash_weight": float(
                    allocation.cash_weight
                ),
                "marginal_variance": {
                    symbol: float(
                        value
                    )
                    for symbol, value
                    in zip(
                        symbols,
                        marginal_variance,
                    )
                },
                "component_variance": {
                    symbol: float(
                        value
                    )
                    for symbol, value
                    in zip(
                        symbols,
                        component_variance,
                    )
                },
                "percentage_risk_contribution": {
                    symbol: (
                        float(
                            component
                            / portfolio_variance
                        )
                        if portfolio_variance
                        > self.tolerance
                        else 0.0
                    )
                    for symbol, component
                    in zip(
                        symbols,
                        component_variance,
                    )
                },
            },
        )

    # ==========================================================
    # Asset attribution
    # ==========================================================

    @staticmethod
    def asset_return_contributions(
        *,
        allocation: PortfolioAllocation,
        assets: List[AssetDefinition],
        asset_returns: Mapping[str, float],
    ) -> Dict[str, float]:
        """
        Calculate return contribution by asset.
        """

        result: Dict[
            str,
            float
        ] = {}

        for asset in assets:
            if (
                asset.symbol
                not in asset_returns
            ):
                raise AttributionEngineError(
                    "Missing asset return for "
                    f"{asset.symbol}."
                )

            weight = float(
                allocation.weights.get(
                    asset.symbol,
                    0.0,
                )
            )

            asset_return = float(
                asset_returns[
                    asset.symbol
                ]
            )

            result[
                asset.symbol
            ] = (
                weight
                * asset_return
            )

        return result

    def asset_risk_contributions(
        self,
        *,
        allocation: PortfolioAllocation,
        assets: List[AssetDefinition],
        covariance: CovarianceMatrix,
    ) -> Dict[str, float]:
        """
        Calculate component variance contribution
        by asset.
        """

        symbols = [
            asset.symbol
            for asset in assets
        ]

        self._validate_covariance(
            covariance=covariance,
            symbols=symbols,
        )

        weights = self._weight_vector(
            allocation=allocation,
            symbols=symbols,
        )

        matrix = np.asarray(
            covariance.matrix,
            dtype=float,
        )

        marginal = matrix.dot(
            weights
        )

        component = (
            weights
            * marginal
        )

        return {
            symbol: float(
                contribution
            )
            for symbol, contribution
            in zip(
                symbols,
                component,
            )
        }

    # ==========================================================
    # Record generation
    # ==========================================================

    @staticmethod
    def _asset_records(
        *,
        assets: List[AssetDefinition],
        weights: np.ndarray,
        return_contributions: np.ndarray,
        risk_contributions: np.ndarray,
        portfolio_variance: float,
    ) -> List[AttributionRecord]:
        records: List[
            AttributionRecord
        ] = []

        for (
            asset,
            weight,
            return_contribution,
            risk_contribution,
        ) in zip(
            assets,
            weights,
            return_contributions,
            risk_contributions,
        ):
            percentage_risk = (
                float(
                    risk_contribution
                    / portfolio_variance
                )
                if portfolio_variance > 0.0
                else 0.0
            )

            records.append(
                AttributionRecord(
                    source_type=(
                        AttributionSource.ASSET
                    ),
                    source_id=(
                        asset.symbol
                    ),
                    return_contribution=float(
                        return_contribution
                    ),
                    risk_contribution=float(
                        risk_contribution
                    ),
                    weight_contribution=float(
                        weight
                    ),
                    metadata={
                        "asset_class": (
                            asset.asset_class
                        ),
                        "sector": (
                            asset.sector
                        ),
                        "percentage_risk_contribution": (
                            percentage_risk
                        ),
                    },
                )
            )

        return records

    @staticmethod
    def _group_records(
        *,
        source_type: AttributionSource,
        groups: Mapping[str, str],
        symbols: Sequence[str],
        weights: np.ndarray,
        return_contributions: np.ndarray,
        risk_contributions: np.ndarray,
        portfolio_variance: float,
    ) -> List[AttributionRecord]:
        grouped: Dict[
            str,
            Dict[str, float]
        ] = {}

        for (
            symbol,
            weight,
            return_contribution,
            risk_contribution,
        ) in zip(
            symbols,
            weights,
            return_contributions,
            risk_contributions,
        ):
            group = groups.get(
                symbol,
                "unclassified",
            )

            if not group:
                group = "unclassified"

            if group not in grouped:
                grouped[group] = {
                    "weight": 0.0,
                    "return": 0.0,
                    "risk": 0.0,
                }

            grouped[group][
                "weight"
            ] += float(
                weight
            )

            grouped[group][
                "return"
            ] += float(
                return_contribution
            )

            grouped[group][
                "risk"
            ] += float(
                risk_contribution
            )

        records: List[
            AttributionRecord
        ] = []

        for (
            group,
            values,
        ) in sorted(
            grouped.items()
        ):
            percentage_risk = (
                values["risk"]
                / portfolio_variance
                if portfolio_variance > 0.0
                else 0.0
            )

            records.append(
                AttributionRecord(
                    source_type=(
                        source_type
                    ),
                    source_id=(
                        group
                    ),
                    return_contribution=float(
                        values[
                            "return"
                        ]
                    ),
                    risk_contribution=float(
                        values[
                            "risk"
                        ]
                    ),
                    weight_contribution=float(
                        values[
                            "weight"
                        ]
                    ),
                    metadata={
                        "percentage_risk_contribution": float(
                            percentage_risk
                        ),
                    },
                )
            )

        return records

    @staticmethod
    def _factor_records(
        *,
        allocation: PortfolioAllocation,
        assets: List[AssetDefinition],
    ) -> List[AttributionRecord]:
        """
        Generate weighted factor-exposure records.

        Factor records represent exposure rather than
        realized return attribution because factor-return
        series are not part of OptimizationRequest.
        """

        factor_exposures: Dict[
            str,
            float
        ] = {}

        for asset in assets:
            weight = float(
                allocation.weights.get(
                    asset.symbol,
                    0.0,
                )
            )

            for (
                factor,
                exposure,
            ) in (
                asset.factor_exposures.items()
            ):
                factor_exposures[
                    factor
                ] = (
                    factor_exposures.get(
                        factor,
                        0.0,
                    )
                    + weight
                    * float(
                        exposure
                    )
                )

        return [
            AttributionRecord(
                source_type=(
                    AttributionSource.FACTOR
                ),
                source_id=factor,
                return_contribution=0.0,
                risk_contribution=0.0,
                weight_contribution=float(
                    exposure
                ),
                metadata={
                    "record_type": (
                        "factor_exposure"
                    ),
                },
            )
            for factor, exposure
            in sorted(
                factor_exposures.items()
            )
        ]

    # ==========================================================
    # Aggregation helpers
    # ==========================================================

    @staticmethod
    def aggregate_records(
        *,
        records: Sequence[
            AttributionRecord
        ],
        source_type: AttributionSource,
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate attribution records by source ID.
        """

        result: Dict[
            str,
            Dict[str, float]
        ] = {}

        for record in records:
            if (
                record.source_type
                != source_type
            ):
                continue

            if (
                record.source_id
                not in result
            ):
                result[
                    record.source_id
                ] = {
                    "return_contribution": 0.0,
                    "risk_contribution": 0.0,
                    "weight_contribution": 0.0,
                }

            result[
                record.source_id
            ][
                "return_contribution"
            ] += float(
                record.return_contribution
            )

            result[
                record.source_id
            ][
                "risk_contribution"
            ] += float(
                record.risk_contribution
            )

            result[
                record.source_id
            ][
                "weight_contribution"
            ] += float(
                record.weight_contribution
            )

        return result

    # ==========================================================
    # Validation
    # ==========================================================

    def _validate_inputs(
        self,
        *,
        allocation: PortfolioAllocation,
        assets: List[AssetDefinition],
        asset_returns: Mapping[str, float],
        covariance: CovarianceMatrix,
    ) -> None:
        if not assets:
            raise AttributionEngineError(
                "At least one asset is required."
            )

        symbols = [
            asset.symbol
            for asset in assets
        ]

        if len(
            set(
                symbols
            )
        ) != len(
            symbols
        ):
            raise AttributionEngineError(
                "Asset symbols must be unique."
            )

        unknown_symbols = (
            set(
                allocation.weights
            )
            - set(
                symbols
            )
        )

        if unknown_symbols:
            raise AttributionEngineError(
                "Allocation contains unknown "
                "symbols: "
                + ", ".join(
                    sorted(
                        unknown_symbols
                    )
                )
            )

        for symbol in symbols:
            if (
                symbol
                not in asset_returns
            ):
                raise AttributionEngineError(
                    "Missing asset return for "
                    f"{symbol}."
                )

            value = float(
                asset_returns[
                    symbol
                ]
            )

            if not np.isfinite(
                value
            ):
                raise AttributionEngineError(
                    "Asset return for "
                    f"{symbol} must be finite."
                )

        for (
            symbol,
            weight,
        ) in (
            allocation.weights.items()
        ):
            if not np.isfinite(
                float(
                    weight
                )
            ):
                raise AttributionEngineError(
                    "Allocation weight for "
                    f"{symbol} must be finite."
                )

            if weight < 0.0:
                raise AttributionEngineError(
                    "Attribution engine currently "
                    "supports long-only portfolios."
                )

        if not np.isfinite(
            float(
                allocation.cash_weight
            )
        ):
            raise AttributionEngineError(
                "cash_weight must be finite."
            )

        self._validate_covariance(
            covariance=covariance,
            symbols=symbols,
        )

    @staticmethod
    def _validate_covariance(
        *,
        covariance: CovarianceMatrix,
        symbols: Sequence[str],
    ) -> None:
        if list(
            covariance.symbols
        ) != list(
            symbols
        ):
            raise AttributionEngineError(
                "Covariance symbol ordering must "
                "exactly match asset ordering."
            )

        matrix = np.asarray(
            covariance.matrix,
            dtype=float,
        )

        size = len(
            symbols
        )

        if matrix.shape != (
            size,
            size,
        ):
            raise AttributionEngineError(
                "Covariance matrix has "
                "incorrect dimensions."
            )

        if not np.all(
            np.isfinite(
                matrix
            )
        ):
            raise AttributionEngineError(
                "Covariance matrix contains "
                "non-finite values."
            )

        if not np.allclose(
            matrix,
            matrix.T,
            atol=1e-10,
        ):
            raise AttributionEngineError(
                "Covariance matrix must "
                "be symmetric."
            )

        eigenvalues = (
            np.linalg.eigvalsh(
                matrix
            )
        )

        if np.min(
            eigenvalues
        ) < -1e-8:
            raise AttributionEngineError(
                "Covariance matrix must be "
                "positive semi-definite."
            )

    @staticmethod
    def _weight_vector(
        *,
        allocation: PortfolioAllocation,
        symbols: Sequence[str],
    ) -> np.ndarray:
        return np.asarray(
            [
                allocation.weights.get(
                    symbol,
                    0.0,
                )
                for symbol
                in symbols
            ],
            dtype=float,
        )

    @staticmethod
    def _return_vector(
        *,
        asset_returns: Mapping[str, float],
        symbols: Sequence[str],
    ) -> np.ndarray:
        return np.asarray(
            [
                asset_returns[
                    symbol
                ]
                for symbol
                in symbols
            ],
            dtype=float,
        )


__all__ = [
    "AttributionEngineError",
    "AttributionReport",
    "PortfolioAttributionEngine",
]