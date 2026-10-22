"""
Central Portfolio Construction Engine for v5.0.7.

This module provides the unified orchestration layer for the
Portfolio Construction Engine.

Integrated components:

- Mean-Variance Optimization
- Minimum-Variance Optimization
- Risk Parity
- CVaR Optimization
- Robust Portfolio Optimization
- Scenario-Aware Allocation
- Institutional Portfolio Constraints
- Drawdown-Aware De-Risking
- Portfolio Attribution

The goal is to expose one stable entry point for portfolio
construction while keeping individual engines modular and
independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from .attribution_engine import (
    AttributionReport,
    PortfolioAttributionEngine,
)
from .constraints_engine import (
    ConstraintReport,
    PortfolioConstraintsEngine,
)
from .cvar_optimizer import CVaROptimizer
from .drawdown_engine import (
    DeRiskingResult,
    DrawdownAwareDeRiskingEngine,
)
from .mean_variance import MeanVarianceOptimizer
from .models import (
    AttributionRecord,
    DrawdownPolicy,
    OptimizationMethod,
    OptimizationRequest,
    OptimizationResult,
    OptimizationStatus,
    PortfolioAllocation,
)
from .risk_parity import RiskParityOptimizer
from .robust_optimizer import RobustPortfolioOptimizer
from .scenario_allocator import ScenarioAwareAllocator


class PortfolioConstructionEngineError(RuntimeError):
    """
    Raised when unified portfolio construction fails.
    """


@dataclass
class PortfolioConstructionResult:
    """
    Complete result produced by the unified portfolio
    construction pipeline.
    """

    optimization_result: OptimizationResult

    pre_derisk_allocation: PortfolioAllocation
    final_allocation: PortfolioAllocation

    constraint_report: Optional[
        ConstraintReport
    ] = None

    derisking_result: Optional[
        DeRiskingResult
    ] = None

    attribution_report: Optional[
        AttributionReport
    ] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def status(
        self,
    ) -> OptimizationStatus:
        return self.optimization_result.status

    @property
    def method(
        self,
    ) -> OptimizationMethod:
        return self.optimization_result.method

    @property
    def compliant(
        self,
    ) -> Optional[bool]:
        if self.constraint_report is None:
            return None

        return self.constraint_report.compliant

    @property
    def derisked(
        self,
    ) -> bool:
        return (
            self.derisking_result is not None
            and self.derisking_result.triggered
        )


class PortfolioConstructionEngine:
    """
    Unified v5.0.7 Portfolio Construction Engine.

    Pipeline:

        Optimization
            |
            v
        Optimized Allocation
            |
            v
        Optional Drawdown De-Risking
            |
            v
        Final Allocation
            |
            +--> Constraint Validation
            |
            +--> Attribution
            |
            v
        PortfolioConstructionResult

    Optimization dispatch is selected automatically from
    OptimizationRequest.method.
    """

    def __init__(
        self,
        *,
        mean_variance_optimizer: Optional[
            MeanVarianceOptimizer
        ] = None,
        risk_parity_optimizer: Optional[
            RiskParityOptimizer
        ] = None,
        cvar_optimizer: Optional[
            CVaROptimizer
        ] = None,
        robust_optimizer: Optional[
            RobustPortfolioOptimizer
        ] = None,
        scenario_allocator: Optional[
            ScenarioAwareAllocator
        ] = None,
        constraints_engine: Optional[
            PortfolioConstraintsEngine
        ] = None,
        drawdown_engine: Optional[
            DrawdownAwareDeRiskingEngine
        ] = None,
        attribution_engine: Optional[
            PortfolioAttributionEngine
        ] = None,
    ) -> None:
        self.mean_variance_optimizer = (
            mean_variance_optimizer
            or MeanVarianceOptimizer()
        )

        self.risk_parity_optimizer = (
            risk_parity_optimizer
            or RiskParityOptimizer()
        )

        self.cvar_optimizer = (
            cvar_optimizer
            or CVaROptimizer()
        )

        self.robust_optimizer = (
            robust_optimizer
            or RobustPortfolioOptimizer()
        )

        self.scenario_allocator = (
            scenario_allocator
            or ScenarioAwareAllocator()
        )

        self.constraints_engine = (
            constraints_engine
            or PortfolioConstraintsEngine()
        )

        self.drawdown_engine = (
            drawdown_engine
            or DrawdownAwareDeRiskingEngine()
        )

        self.attribution_engine = (
            attribution_engine
            or PortfolioAttributionEngine()
        )

    # ==========================================================
    # Main pipeline
    # ==========================================================

    def construct(
        self,
        request: OptimizationRequest,
        *,
        portfolio_values: Optional[
            Sequence[float]
        ] = None,
        drawdown_policy: Optional[
            DrawdownPolicy
        ] = None,
        asset_returns: Optional[
            Mapping[str, float]
        ] = None,
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
        validate_constraints: bool = True,
        run_attribution: bool = True,
    ) -> PortfolioConstructionResult:
        """
        Execute the complete portfolio construction pipeline.
        """

        optimization_result = (
            self.optimize(
                request
            )
        )

        if (
            optimization_result.status
            != OptimizationStatus.COMPLETED
        ):
            raise PortfolioConstructionEngineError(
                "Optimization did not complete successfully. "
                f"Status: {optimization_result.status.value}"
            )

        if (
            optimization_result.allocation
            is None
        ):
            raise PortfolioConstructionEngineError(
                "Optimization completed without "
                "a portfolio allocation."
            )

        pre_derisk_allocation = (
            optimization_result.allocation
        )

        final_allocation = (
            pre_derisk_allocation
        )

        derisking_result: Optional[
            DeRiskingResult
        ] = None

        effective_drawdown_policy = (
            drawdown_policy
            if drawdown_policy is not None
            else request.drawdown_policy
        )

        if (
            portfolio_values is not None
            and effective_drawdown_policy
            is not None
        ):
            derisking_result = (
                self.drawdown_engine.apply(
                    allocation=(
                        pre_derisk_allocation
                    ),
                    values=portfolio_values,
                    policy=(
                        effective_drawdown_policy
                    ),
                )
            )

            final_allocation = (
                derisking_result
                .adjusted_allocation
            )

        constraint_report: Optional[
            ConstraintReport
        ] = None

        if validate_constraints:
            constraint_report = (
                self.constraints_engine.validate(
                    allocation=(
                        final_allocation
                    ),
                    assets=request.assets,
                    constraints=(
                        request.constraints
                    ),
                    current_allocation=(
                        request.current_allocation
                    ),
                    turnover_constraint=(
                        request.turnover_constraint
                    ),
                    leverage_constraint=(
                        request.leverage_constraint
                    ),
                )
            )

        attribution_report: Optional[
            AttributionReport
        ] = None

        if run_attribution:
            if asset_returns is not None:
                attribution_report = (
                    self.attribution_engine.analyze(
                        allocation=(
                            final_allocation
                        ),
                        assets=request.assets,
                        asset_returns=(
                            asset_returns
                        ),
                        covariance=(
                            request.covariance
                        ),
                        alpha_sources=(
                            alpha_sources
                        ),
                        agent_sources=(
                            agent_sources
                        ),
                        regime_sources=(
                            regime_sources
                        ),
                        risk_sources=(
                            risk_sources
                        ),
                    )
                )

        if (
            attribution_report
            is not None
        ):
            optimization_result.attribution = (
                list(
                    attribution_report.records
                )
            )

        metadata = {
            "optimization_method": (
                request.method.value
            ),
            "optimization_completed": True,
            "drawdown_policy_available": (
                effective_drawdown_policy
                is not None
            ),
            "drawdown_analysis_run": (
                derisking_result
                is not None
            ),
            "derisking_triggered": (
                derisking_result.triggered
                if derisking_result
                is not None
                else False
            ),
            "constraint_validation_run": (
                constraint_report
                is not None
            ),
            "constraint_compliant": (
                constraint_report.compliant
                if constraint_report
                is not None
                else None
            ),
            "attribution_run": (
                attribution_report
                is not None
            ),
            "pre_derisk_cash_weight": float(
                pre_derisk_allocation.cash_weight
            ),
            "final_cash_weight": float(
                final_allocation.cash_weight
            ),
        }

        return PortfolioConstructionResult(
            optimization_result=(
                optimization_result
            ),
            pre_derisk_allocation=(
                pre_derisk_allocation
            ),
            final_allocation=(
                final_allocation
            ),
            constraint_report=(
                constraint_report
            ),
            derisking_result=(
                derisking_result
            ),
            attribution_report=(
                attribution_report
            ),
            metadata=metadata,
        )

    # ==========================================================
    # Optimization dispatch
    # ==========================================================

    def optimize(
        self,
        request: OptimizationRequest,
    ) -> OptimizationResult:
        """
        Dispatch an OptimizationRequest to the correct
        portfolio optimizer.
        """

        method = request.method

        try:
            if (
                method == OptimizationMethod.MEAN_VARIANCE
                or method == OptimizationMethod.MINIMUM_VARIANCE
            ):
                return (
                    self.mean_variance_optimizer
                    .optimize(
                        request
                    )
                )

            if (
                method
                == OptimizationMethod.RISK_PARITY
            ):
                return (
                    self.risk_parity_optimizer
                    .optimize(
                        request
                    )
                )

            if (
                method
                == OptimizationMethod.CVAR
            ):
                return (
                    self.cvar_optimizer
                    .optimize(
                        request
                    )
                )

            if (
                method
                == OptimizationMethod.ROBUST
            ):
                return (
                    self.robust_optimizer
                    .optimize(
                        request
                    )
                )

            if (
                method
                == OptimizationMethod.SCENARIO_AWARE
            ):
                return (
                    self.scenario_allocator
                    .optimize(
                        request
                    )
                )

            raise PortfolioConstructionEngineError(
                "Unsupported optimization method: "
                f"{getattr(method, 'value', method)}"
            )

        except PortfolioConstructionEngineError:
            raise

        except Exception as exc:
            method_name = getattr(
                method,
                "value",
                str(method),
            )

            raise PortfolioConstructionEngineError(
                "Portfolio optimization failed "
                f"for method '{method_name}': "
                f"{exc}"
            ) from exc

    # ==========================================================
    # Individual pipeline stages
    # ==========================================================

    def validate_allocation(
        self,
        *,
        request: OptimizationRequest,
        allocation: PortfolioAllocation,
    ) -> ConstraintReport:
        """
        Run institutional constraint validation
        independently.
        """

        return (
            self.constraints_engine.validate(
                allocation=allocation,
                assets=request.assets,
                constraints=request.constraints,
                current_allocation=(
                    request.current_allocation
                ),
                turnover_constraint=(
                    request.turnover_constraint
                ),
                leverage_constraint=(
                    request.leverage_constraint
                ),
            )
        )

    def apply_drawdown_policy(
        self,
        *,
        allocation: PortfolioAllocation,
        portfolio_values: Sequence[float],
        policy: DrawdownPolicy,
    ) -> DeRiskingResult:
        """
        Apply drawdown-aware de-risking independently.
        """

        return (
            self.drawdown_engine.apply(
                allocation=allocation,
                values=portfolio_values,
                policy=policy,
            )
        )

    def attribute(
        self,
        *,
        request: OptimizationRequest,
        allocation: PortfolioAllocation,
        asset_returns: Mapping[str, float],
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
        Run portfolio attribution independently.
        """

        return (
            self.attribution_engine.analyze(
                allocation=allocation,
                assets=request.assets,
                asset_returns=asset_returns,
                covariance=request.covariance,
                alpha_sources=alpha_sources,
                agent_sources=agent_sources,
                regime_sources=regime_sources,
                risk_sources=risk_sources,
            )
        )

    # ==========================================================
    # Convenience helpers
    # ==========================================================

    @staticmethod
    def allocation_changed(
        *,
        before: PortfolioAllocation,
        after: PortfolioAllocation,
        tolerance: float = 1e-10,
    ) -> bool:
        """
        Determine whether two allocations differ.
        """

        if tolerance < 0.0:
            raise PortfolioConstructionEngineError(
                "tolerance cannot be negative."
            )

        symbols = (
            set(
                before.weights
            )
            | set(
                after.weights
            )
        )

        for symbol in symbols:
            before_weight = float(
                before.weights.get(
                    symbol,
                    0.0,
                )
            )

            after_weight = float(
                after.weights.get(
                    symbol,
                    0.0,
                )
            )

            if (
                abs(
                    before_weight
                    - after_weight
                )
                > tolerance
            ):
                return True

        if (
            abs(
                float(
                    before.cash_weight
                )
                - float(
                    after.cash_weight
                )
            )
            > tolerance
        ):
            return True

        return False

    @staticmethod
    def attribution_records(
        result: PortfolioConstructionResult,
    ) -> list[AttributionRecord]:
        """
        Return attribution records from a complete
        construction result.
        """

        if (
            result.attribution_report
            is None
        ):
            return []

        return list(
            result.attribution_report.records
        )


__all__ = [
    "PortfolioConstructionEngine",
    "PortfolioConstructionEngineError",
    "PortfolioConstructionResult",
]