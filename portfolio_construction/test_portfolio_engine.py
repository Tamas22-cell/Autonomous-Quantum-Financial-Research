"""
Integration tests for the v5.0.7
Central Portfolio Construction Engine.

The tests focus on orchestration:

- optimizer dispatch
- unified construction pipeline
- drawdown integration
- constraint integration
- attribution integration
- result metadata
- convenience helpers
- failure handling

Individual mathematical engines are tested in their
own dedicated test modules.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from portfolio_construction.models import (
    AttributionRecord,
    AttributionSource,
    DrawdownPolicy,
    OptimizationMethod,
    OptimizationStatus,
    PortfolioAllocation,
)
from portfolio_construction.portfolio_engine import (
    PortfolioConstructionEngine,
    PortfolioConstructionEngineError,
    PortfolioConstructionResult,
)


# ==============================================================
# Helpers
# ==============================================================


def build_allocation() -> PortfolioAllocation:
    return PortfolioAllocation(
        weights={
            "SPY": 0.60,
            "GLD": 0.40,
        },
        cash_weight=0.0,
    )


def build_derisked_allocation() -> PortfolioAllocation:
    return PortfolioAllocation(
        weights={
            "SPY": 0.30,
            "GLD": 0.20,
        },
        cash_weight=0.50,
    )


def build_request(
    method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE,
) -> SimpleNamespace:
    return SimpleNamespace(
        method=method,
        assets=[],
        covariance=None,
        constraints=[],
        current_allocation=None,
        turnover_constraint=None,
        leverage_constraint=None,
        drawdown_policy=None,
    )


def build_optimization_result(
    method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE,
    allocation: PortfolioAllocation | None = None,
    status: OptimizationStatus = OptimizationStatus.COMPLETED,
) -> SimpleNamespace:
    return SimpleNamespace(
        method=method,
        status=status,
        allocation=(
            allocation
            if allocation is not None
            else build_allocation()
        ),
        attribution=[],
    )


# ==============================================================
# Fake components
# ==============================================================


class FakeOptimizer:
    def __init__(
        self,
        result: Any | None = None,
    ) -> None:
        self.result = result
        self.calls = 0
        self.last_request = None

    def optimize(
        self,
        request: Any,
    ) -> Any:
        self.calls += 1
        self.last_request = request

        if self.result is not None:
            return self.result

        return build_optimization_result(
            method=request.method
        )


class RaisingOptimizer:
    def optimize(
        self,
        request: Any,
    ) -> Any:
        raise RuntimeError(
            "optimizer exploded"
        )


class FakeConstraintsEngine:
    def __init__(
        self,
        compliant: bool = True,
    ) -> None:
        self.compliant = compliant
        self.calls = 0
        self.last_kwargs = None

    def validate(
        self,
        **kwargs: Any,
    ) -> SimpleNamespace:
        self.calls += 1
        self.last_kwargs = kwargs

        return SimpleNamespace(
            compliant=self.compliant,
            violations=[],
        )


class FakeDrawdownEngine:
    def __init__(
        self,
        triggered: bool = True,
    ) -> None:
        self.triggered = triggered
        self.calls = 0
        self.last_kwargs = None

    def apply(
        self,
        **kwargs: Any,
    ) -> SimpleNamespace:
        self.calls += 1
        self.last_kwargs = kwargs

        allocation = kwargs[
            "allocation"
        ]

        adjusted = (
            build_derisked_allocation()
            if self.triggered
            else allocation
        )

        return SimpleNamespace(
            original_allocation=allocation,
            adjusted_allocation=adjusted,
            triggered=self.triggered,
        )


class FakeAttributionEngine:
    def __init__(self) -> None:
        self.calls = 0
        self.last_kwargs = None

        self.records = [
            AttributionRecord(
                source_type=AttributionSource.ASSET,
                source_id="SPY",
                return_contribution=0.01,
                risk_contribution=0.02,
                weight_contribution=0.60,
            )
        ]

    def analyze(
        self,
        **kwargs: Any,
    ) -> SimpleNamespace:
        self.calls += 1
        self.last_kwargs = kwargs

        return SimpleNamespace(
            records=list(
                self.records
            )
        )


# ==============================================================
# Engine builder
# ==============================================================


def build_engine(
    *,
    mean_variance_optimizer: Any | None = None,
    risk_parity_optimizer: Any | None = None,
    cvar_optimizer: Any | None = None,
    robust_optimizer: Any | None = None,
    scenario_allocator: Any | None = None,
    constraints_engine: Any | None = None,
    drawdown_engine: Any | None = None,
    attribution_engine: Any | None = None,
) -> PortfolioConstructionEngine:
    return PortfolioConstructionEngine(
        mean_variance_optimizer=(
            mean_variance_optimizer
            or FakeOptimizer()
        ),
        risk_parity_optimizer=(
            risk_parity_optimizer
            or FakeOptimizer()
        ),
        cvar_optimizer=(
            cvar_optimizer
            or FakeOptimizer()
        ),
        robust_optimizer=(
            robust_optimizer
            or FakeOptimizer()
        ),
        scenario_allocator=(
            scenario_allocator
            or FakeOptimizer()
        ),
        constraints_engine=(
            constraints_engine
            or FakeConstraintsEngine()
        ),
        drawdown_engine=(
            drawdown_engine
            or FakeDrawdownEngine()
        ),
        attribution_engine=(
            attribution_engine
            or FakeAttributionEngine()
        ),
    )


# ==============================================================
# Optimizer dispatch
# ==============================================================


def test_mean_variance_dispatch() -> None:
    optimizer = FakeOptimizer()

    engine = build_engine(
        mean_variance_optimizer=optimizer
    )

    request = build_request(
        OptimizationMethod.MEAN_VARIANCE
    )

    result = engine.optimize(
        request
    )

    assert optimizer.calls == 1
    assert optimizer.last_request is request
    assert result.method == OptimizationMethod.MEAN_VARIANCE


def test_minimum_variance_dispatch() -> None:
    optimizer = FakeOptimizer()

    engine = build_engine(
        mean_variance_optimizer=optimizer
    )

    request = build_request(
        OptimizationMethod.MINIMUM_VARIANCE
    )

    result = engine.optimize(
        request
    )

    assert optimizer.calls == 1
    assert result.method == OptimizationMethod.MINIMUM_VARIANCE


def test_risk_parity_dispatch() -> None:
    optimizer = FakeOptimizer()

    engine = build_engine(
        risk_parity_optimizer=optimizer
    )

    request = build_request(
        OptimizationMethod.RISK_PARITY
    )

    result = engine.optimize(
        request
    )

    assert optimizer.calls == 1
    assert result.method == OptimizationMethod.RISK_PARITY


def test_cvar_dispatch() -> None:
    optimizer = FakeOptimizer()

    engine = build_engine(
        cvar_optimizer=optimizer
    )

    request = build_request(
        OptimizationMethod.CVAR
    )

    result = engine.optimize(
        request
    )

    assert optimizer.calls == 1
    assert result.method == OptimizationMethod.CVAR


def test_robust_dispatch() -> None:
    optimizer = FakeOptimizer()

    engine = build_engine(
        robust_optimizer=optimizer
    )

    request = build_request(
        OptimizationMethod.ROBUST
    )

    result = engine.optimize(
        request
    )

    assert optimizer.calls == 1
    assert result.method == OptimizationMethod.ROBUST


def test_scenario_aware_dispatch() -> None:
    optimizer = FakeOptimizer()

    engine = build_engine(
        scenario_allocator=optimizer
    )

    request = build_request(
        OptimizationMethod.SCENARIO_AWARE
    )

    result = engine.optimize(
        request
    )

    assert optimizer.calls == 1
    assert result.method == OptimizationMethod.SCENARIO_AWARE


# ==============================================================
# Optimization error handling
# ==============================================================


def test_optimizer_exception_is_wrapped() -> None:
    engine = build_engine(
        mean_variance_optimizer=(
            RaisingOptimizer()
        )
    )

    request = build_request()

    with pytest.raises(
        PortfolioConstructionEngineError,
        match="Portfolio optimization failed",
    ):
        engine.optimize(
            request
        )


def test_unsupported_method_raises_error() -> None:
    engine = build_engine()

    request = build_request()

    request.method = SimpleNamespace(
        value="unsupported"
    )

    with pytest.raises(
        PortfolioConstructionEngineError,
        match="Unsupported optimization method",
    ):
        engine.optimize(
            request
        )


# ==============================================================
# Basic construction pipeline
# ==============================================================


def test_construct_returns_portfolio_construction_result() -> None:
    engine = build_engine()

    result = engine.construct(
        build_request(),
        validate_constraints=False,
        run_attribution=False,
    )

    assert isinstance(
        result,
        PortfolioConstructionResult,
    )


def test_construct_preserves_optimizer_allocation_without_derisking() -> None:
    engine = build_engine()

    result = engine.construct(
        build_request(),
        validate_constraints=False,
        run_attribution=False,
    )

    assert (
        result.final_allocation
        is result.pre_derisk_allocation
    )


def test_construct_exposes_method() -> None:
    engine = build_engine()

    result = engine.construct(
        build_request(),
        validate_constraints=False,
        run_attribution=False,
    )

    assert (
        result.method
        == OptimizationMethod.MEAN_VARIANCE
    )


def test_construct_exposes_status() -> None:
    engine = build_engine()

    result = engine.construct(
        build_request(),
        validate_constraints=False,
        run_attribution=False,
    )

    assert (
        result.status
        == OptimizationStatus.COMPLETED
    )


# ==============================================================
# Failed optimization
# ==============================================================


def test_construct_rejects_noncompleted_optimization() -> None:
    optimizer = FakeOptimizer(
        result=build_optimization_result(
            status=OptimizationStatus.FAILED
        )
    )

    engine = build_engine(
        mean_variance_optimizer=optimizer
    )

    with pytest.raises(
        PortfolioConstructionEngineError,
        match="did not complete successfully",
    ):
        engine.construct(
            build_request(),
            validate_constraints=False,
            run_attribution=False,
        )


def test_construct_rejects_missing_allocation() -> None:
    result = build_optimization_result()

    result.allocation = None

    optimizer = FakeOptimizer(
        result=result
    )

    engine = build_engine(
        mean_variance_optimizer=optimizer
    )

    with pytest.raises(
        PortfolioConstructionEngineError,
        match="without a portfolio allocation",
    ):
        engine.construct(
            build_request(),
            validate_constraints=False,
            run_attribution=False,
        )


# ==============================================================
# Constraint integration
# ==============================================================


def test_constraint_validation_runs() -> None:
    constraints = FakeConstraintsEngine(
        compliant=True
    )

    engine = build_engine(
        constraints_engine=constraints
    )

    result = engine.construct(
        build_request(),
        run_attribution=False,
    )

    assert constraints.calls == 1
    assert result.constraint_report is not None
    assert result.compliant is True


def test_noncompliant_constraint_report_is_exposed() -> None:
    constraints = FakeConstraintsEngine(
        compliant=False
    )

    engine = build_engine(
        constraints_engine=constraints
    )

    result = engine.construct(
        build_request(),
        run_attribution=False,
    )

    assert result.compliant is False


def test_constraint_validation_can_be_disabled() -> None:
    constraints = FakeConstraintsEngine()

    engine = build_engine(
        constraints_engine=constraints
    )

    result = engine.construct(
        build_request(),
        validate_constraints=False,
        run_attribution=False,
    )

    assert constraints.calls == 0
    assert result.constraint_report is None
    assert result.compliant is None


def test_validate_allocation_helper_calls_constraints_engine() -> None:
    constraints = FakeConstraintsEngine()

    engine = build_engine(
        constraints_engine=constraints
    )

    request = build_request()

    report = engine.validate_allocation(
        request=request,
        allocation=build_allocation(),
    )

    assert constraints.calls == 1
    assert report.compliant is True


# ==============================================================
# Drawdown integration
# ==============================================================


def test_drawdown_engine_runs_when_values_and_policy_exist() -> None:
    drawdown = FakeDrawdownEngine(
        triggered=True
    )

    engine = build_engine(
        drawdown_engine=drawdown
    )

    policy = DrawdownPolicy(
        trigger_drawdown=0.10,
        target_risk_multiplier=0.50,
        minimum_risk_multiplier=0.25,
    )

    result = engine.construct(
        build_request(),
        portfolio_values=[
            100.0,
            120.0,
            90.0,
        ],
        drawdown_policy=policy,
        validate_constraints=False,
        run_attribution=False,
    )

    assert drawdown.calls == 1
    assert result.derisking_result is not None
    assert result.derisked is True


def test_drawdown_changes_final_allocation() -> None:
    drawdown = FakeDrawdownEngine(
        triggered=True
    )

    engine = build_engine(
        drawdown_engine=drawdown
    )

    policy = DrawdownPolicy(
        trigger_drawdown=0.10,
        target_risk_multiplier=0.50,
        minimum_risk_multiplier=0.25,
    )

    result = engine.construct(
        build_request(),
        portfolio_values=[
            100.0,
            120.0,
            90.0,
        ],
        drawdown_policy=policy,
        validate_constraints=False,
        run_attribution=False,
    )

    assert (
        result.final_allocation.cash_weight
        == pytest.approx(
            0.50
        )
    )


def test_drawdown_does_not_run_without_values() -> None:
    drawdown = FakeDrawdownEngine()

    engine = build_engine(
        drawdown_engine=drawdown
    )

    policy = DrawdownPolicy(
        trigger_drawdown=0.10,
        target_risk_multiplier=0.50,
        minimum_risk_multiplier=0.25,
    )

    engine.construct(
        build_request(),
        drawdown_policy=policy,
        validate_constraints=False,
        run_attribution=False,
    )

    assert drawdown.calls == 0


def test_drawdown_does_not_run_without_policy() -> None:
    drawdown = FakeDrawdownEngine()

    engine = build_engine(
        drawdown_engine=drawdown
    )

    engine.construct(
        build_request(),
        portfolio_values=[
            100.0,
            90.0,
        ],
        validate_constraints=False,
        run_attribution=False,
    )

    assert drawdown.calls == 0


def test_request_drawdown_policy_is_used() -> None:
    drawdown = FakeDrawdownEngine()

    engine = build_engine(
        drawdown_engine=drawdown
    )

    request = build_request()

    request.drawdown_policy = DrawdownPolicy(
        trigger_drawdown=0.10,
        target_risk_multiplier=0.50,
        minimum_risk_multiplier=0.25,
    )

    engine.construct(
        request,
        portfolio_values=[
            100.0,
            80.0,
        ],
        validate_constraints=False,
        run_attribution=False,
    )

    assert drawdown.calls == 1


def test_explicit_drawdown_policy_overrides_request_policy() -> None:
    drawdown = FakeDrawdownEngine()

    engine = build_engine(
        drawdown_engine=drawdown
    )

    request = build_request()

    request.drawdown_policy = DrawdownPolicy(
        trigger_drawdown=0.20,
        target_risk_multiplier=0.80,
        minimum_risk_multiplier=0.50,
    )

    explicit_policy = DrawdownPolicy(
        trigger_drawdown=0.10,
        target_risk_multiplier=0.50,
        minimum_risk_multiplier=0.25,
    )

    engine.construct(
        request,
        portfolio_values=[
            100.0,
            80.0,
        ],
        drawdown_policy=explicit_policy,
        validate_constraints=False,
        run_attribution=False,
    )

    assert (
        drawdown.last_kwargs[
            "policy"
        ]
        is explicit_policy
    )


def test_apply_drawdown_policy_helper() -> None:
    drawdown = FakeDrawdownEngine(
        triggered=True
    )

    engine = build_engine(
        drawdown_engine=drawdown
    )

    policy = DrawdownPolicy(
        trigger_drawdown=0.10,
        target_risk_multiplier=0.50,
        minimum_risk_multiplier=0.25,
    )

    result = engine.apply_drawdown_policy(
        allocation=build_allocation(),
        portfolio_values=[
            100.0,
            80.0,
        ],
        policy=policy,
    )

    assert drawdown.calls == 1
    assert result.triggered is True


# ==============================================================
# Attribution integration
# ==============================================================


def test_attribution_runs_when_returns_are_supplied() -> None:
    attribution = FakeAttributionEngine()

    engine = build_engine(
        attribution_engine=attribution
    )

    result = engine.construct(
        build_request(),
        asset_returns={
            "SPY": 0.05,
            "GLD": 0.01,
        },
        validate_constraints=False,
    )

    assert attribution.calls == 1
    assert result.attribution_report is not None


def test_attribution_does_not_run_without_returns() -> None:
    attribution = FakeAttributionEngine()

    engine = build_engine(
        attribution_engine=attribution
    )

    result = engine.construct(
        build_request(),
        validate_constraints=False,
    )

    assert attribution.calls == 0
    assert result.attribution_report is None


def test_attribution_can_be_disabled() -> None:
    attribution = FakeAttributionEngine()

    engine = build_engine(
        attribution_engine=attribution
    )

    result = engine.construct(
        build_request(),
        asset_returns={
            "SPY": 0.05,
            "GLD": 0.01,
        },
        validate_constraints=False,
        run_attribution=False,
    )

    assert attribution.calls == 0
    assert result.attribution_report is None


def test_attribution_records_are_attached_to_optimization_result() -> None:
    attribution = FakeAttributionEngine()

    engine = build_engine(
        attribution_engine=attribution
    )

    result = engine.construct(
        build_request(),
        asset_returns={
            "SPY": 0.05,
            "GLD": 0.01,
        },
        validate_constraints=False,
    )

    assert (
        result.optimization_result.attribution
        == attribution.records
    )


def test_attribute_helper_calls_attribution_engine() -> None:
    attribution = FakeAttributionEngine()

    engine = build_engine(
        attribution_engine=attribution
    )

    report = engine.attribute(
        request=build_request(),
        allocation=build_allocation(),
        asset_returns={
            "SPY": 0.05,
            "GLD": 0.01,
        },
    )

    assert attribution.calls == 1
    assert len(
        report.records
    ) == 1


def test_attribution_receives_agent_sources() -> None:
    attribution = FakeAttributionEngine()

    engine = build_engine(
        attribution_engine=attribution
    )

    agent_sources = {
        "SPY": "macro_agent",
        "GLD": "risk_agent",
    }

    engine.construct(
        build_request(),
        asset_returns={
            "SPY": 0.05,
            "GLD": 0.01,
        },
        agent_sources=agent_sources,
        validate_constraints=False,
    )

    assert (
        attribution.last_kwargs[
            "agent_sources"
        ]
        == agent_sources
    )


# ==============================================================
# Pipeline ordering
# ==============================================================


def test_constraints_receive_final_derisked_allocation() -> None:
    drawdown = FakeDrawdownEngine(
        triggered=True
    )

    constraints = FakeConstraintsEngine()

    engine = build_engine(
        drawdown_engine=drawdown,
        constraints_engine=constraints,
    )

    policy = DrawdownPolicy(
        trigger_drawdown=0.10,
        target_risk_multiplier=0.50,
        minimum_risk_multiplier=0.25,
    )

    engine.construct(
        build_request(),
        portfolio_values=[
            100.0,
            80.0,
        ],
        drawdown_policy=policy,
        run_attribution=False,
    )

    validated_allocation = (
        constraints.last_kwargs[
            "allocation"
        ]
    )

    assert (
        validated_allocation.cash_weight
        == pytest.approx(
            0.50
        )
    )


def test_attribution_receives_final_derisked_allocation() -> None:
    drawdown = FakeDrawdownEngine(
        triggered=True
    )

    attribution = FakeAttributionEngine()

    engine = build_engine(
        drawdown_engine=drawdown,
        attribution_engine=attribution,
    )

    policy = DrawdownPolicy(
        trigger_drawdown=0.10,
        target_risk_multiplier=0.50,
        minimum_risk_multiplier=0.25,
    )

    engine.construct(
        build_request(),
        portfolio_values=[
            100.0,
            80.0,
        ],
        drawdown_policy=policy,
        asset_returns={
            "SPY": 0.05,
            "GLD": 0.01,
        },
        validate_constraints=False,
    )

    attributed_allocation = (
        attribution.last_kwargs[
            "allocation"
        ]
    )

    assert (
        attributed_allocation.cash_weight
        == pytest.approx(
            0.50
        )
    )


# ==============================================================
# Result metadata
# ==============================================================


def test_metadata_contains_optimization_method() -> None:
    engine = build_engine()

    result = engine.construct(
        build_request(),
        validate_constraints=False,
        run_attribution=False,
    )

    assert (
        result.metadata[
            "optimization_method"
        ]
        == OptimizationMethod.MEAN_VARIANCE.value
    )


def test_metadata_reports_constraint_validation() -> None:
    engine = build_engine()

    result = engine.construct(
        build_request(),
        run_attribution=False,
    )

    assert (
        result.metadata[
            "constraint_validation_run"
        ]
        is True
    )


def test_metadata_reports_attribution_run() -> None:
    engine = build_engine()

    result = engine.construct(
        build_request(),
        asset_returns={
            "SPY": 0.05,
            "GLD": 0.01,
        },
        validate_constraints=False,
    )

    assert (
        result.metadata[
            "attribution_run"
        ]
        is True
    )


def test_metadata_reports_derisking_trigger() -> None:
    drawdown = FakeDrawdownEngine(
        triggered=True
    )

    engine = build_engine(
        drawdown_engine=drawdown
    )

    policy = DrawdownPolicy(
        trigger_drawdown=0.10,
        target_risk_multiplier=0.50,
        minimum_risk_multiplier=0.25,
    )

    result = engine.construct(
        build_request(),
        portfolio_values=[
            100.0,
            80.0,
        ],
        drawdown_policy=policy,
        validate_constraints=False,
        run_attribution=False,
    )

    assert (
        result.metadata[
            "derisking_triggered"
        ]
        is True
    )


def test_metadata_contains_final_cash_weight() -> None:
    drawdown = FakeDrawdownEngine(
        triggered=True
    )

    engine = build_engine(
        drawdown_engine=drawdown
    )

    policy = DrawdownPolicy(
        trigger_drawdown=0.10,
        target_risk_multiplier=0.50,
        minimum_risk_multiplier=0.25,
    )

    result = engine.construct(
        build_request(),
        portfolio_values=[
            100.0,
            80.0,
        ],
        drawdown_policy=policy,
        validate_constraints=False,
        run_attribution=False,
    )

    assert (
        result.metadata[
            "final_cash_weight"
        ]
        == pytest.approx(
            0.50
        )
    )


# ==============================================================
# Allocation comparison
# ==============================================================


def test_allocation_changed_detects_weight_change() -> None:
    before = build_allocation()

    after = PortfolioAllocation(
        weights={
            "SPY": 0.50,
            "GLD": 0.50,
        },
        cash_weight=0.0,
    )

    assert (
        PortfolioConstructionEngine
        .allocation_changed(
            before=before,
            after=after,
        )
        is True
    )


def test_allocation_changed_detects_cash_change() -> None:
    before = build_allocation()

    after = PortfolioAllocation(
        weights={
            "SPY": 0.30,
            "GLD": 0.20,
        },
        cash_weight=0.50,
    )

    assert (
        PortfolioConstructionEngine
        .allocation_changed(
            before=before,
            after=after,
        )
        is True
    )


def test_allocation_changed_returns_false_for_equal_allocations() -> None:
    before = build_allocation()

    after = build_allocation()

    assert (
        PortfolioConstructionEngine
        .allocation_changed(
            before=before,
            after=after,
        )
        is False
    )


def test_allocation_changed_respects_tolerance() -> None:
    before = build_allocation()

    after = PortfolioAllocation(
        weights={
            "SPY": 0.60000000001,
            "GLD": 0.39999999999,
        },
        cash_weight=0.0,
    )

    assert (
        PortfolioConstructionEngine
        .allocation_changed(
            before=before,
            after=after,
            tolerance=1e-8,
        )
        is False
    )


def test_negative_allocation_comparison_tolerance_raises_error() -> None:
    with pytest.raises(
        PortfolioConstructionEngineError
    ):
        PortfolioConstructionEngine.allocation_changed(
            before=build_allocation(),
            after=build_allocation(),
            tolerance=-0.01,
        )


# ==============================================================
# Attribution record convenience helper
# ==============================================================


def test_attribution_records_returns_records() -> None:
    attribution = FakeAttributionEngine()

    engine = build_engine(
        attribution_engine=attribution
    )

    result = engine.construct(
        build_request(),
        asset_returns={
            "SPY": 0.05,
            "GLD": 0.01,
        },
        validate_constraints=False,
    )

    records = (
        PortfolioConstructionEngine
        .attribution_records(
            result
        )
    )

    assert len(
        records
    ) == 1

    assert (
        records[0].source_id
        == "SPY"
    )


def test_attribution_records_returns_empty_list_when_not_run() -> None:
    engine = build_engine()

    result = engine.construct(
        build_request(),
        validate_constraints=False,
        run_attribution=False,
    )

    records = (
        PortfolioConstructionEngine
        .attribution_records(
            result
        )
    )

    assert records == []