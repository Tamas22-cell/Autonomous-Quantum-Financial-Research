"""
Tests for v5.0.9 Shadow Outcome Observer.

Covered:
- LONG return
- SHORT return
- NEUTRAL return
- benchmark return
- alpha calculation
- drawdown normalization
- successful observation
- negative-return failure
- negative-alpha failure
- excessive-drawdown failure
- strategy/allocation identity protection
- missing entry price
- invalid prices
- benchmark validation
- safety metadata
- callable interface
"""

import pytest

from autonomous_research_loop.models import (
    ShadowAllocation,
    StrategyCandidate,
)
from autonomous_research_loop.outcome_observer import (
    OutcomeObserver,
    OutcomeObserverConfig,
)


def make_candidate(
    direction: str = "LONG",
) -> StrategyCandidate:
    return StrategyCandidate(
        hypothesis_id="hypothesis-outcome-test",
        name="BTC forward observation candidate",
        entry_rule="Enter research exposure.",
        exit_rule="Exit after observation horizon.",
        asset="BTC",
        horizon="7D",
        position_size=0.08,
        stop_condition="Stop on risk invalidation.",
        parameters={
            "direction": direction,
            "research_only": True,
            "real_money_execution": False,
        },
    )


def make_allocation(
    candidate: StrategyCandidate,
    entry_price: float | None = 100.0,
) -> ShadowAllocation:
    return ShadowAllocation(
        strategy_id=candidate.strategy_id,
        asset=candidate.asset,
        weight=0.08,
        notional_value=8_000.0,
        entry_price=entry_price,
        metadata={
            "mode": "SHADOW",
            "research_only": True,
            "real_money_execution": False,
        },
    )


def test_long_return() -> None:
    observer = OutcomeObserver()

    result = observer.calculate_return(
        100.0,
        110.0,
        "LONG",
    )

    assert result == pytest.approx(0.10)


def test_long_loss() -> None:
    observer = OutcomeObserver()

    result = observer.calculate_return(
        100.0,
        90.0,
        "LONG",
    )

    assert result == pytest.approx(-0.10)


def test_short_return() -> None:
    observer = OutcomeObserver()

    result = observer.calculate_return(
        100.0,
        90.0,
        "SHORT",
    )

    assert result == pytest.approx(0.10)


def test_short_loss() -> None:
    observer = OutcomeObserver()

    result = observer.calculate_return(
        100.0,
        110.0,
        "SHORT",
    )

    assert result == pytest.approx(-0.10)


def test_neutral_return_is_zero() -> None:
    observer = OutcomeObserver()

    result = observer.calculate_return(
        100.0,
        110.0,
        "NEUTRAL",
    )

    assert result == pytest.approx(0.0)


def test_invalid_direction_is_rejected() -> None:
    observer = OutcomeObserver()

    with pytest.raises(ValueError):
        observer.calculate_return(
            100.0,
            110.0,
            "INVALID",
        )


def test_benchmark_return() -> None:
    observer = OutcomeObserver()

    result = observer.calculate_benchmark_return(
        100.0,
        105.0,
    )

    assert result == pytest.approx(0.05)


def test_missing_benchmark_returns_zero() -> None:
    observer = OutcomeObserver()

    result = observer.calculate_benchmark_return(
        None,
        None,
    )

    assert result == pytest.approx(0.0)


def test_partial_benchmark_is_rejected() -> None:
    observer = OutcomeObserver()

    with pytest.raises(ValueError):
        observer.calculate_benchmark_return(
            100.0,
            None,
        )


def test_alpha_calculation() -> None:
    observer = OutcomeObserver()

    result = observer.calculate_alpha(
        0.10,
        0.04,
    )

    assert result == pytest.approx(0.06)


def test_negative_drawdown_is_normalized() -> None:
    observer = OutcomeObserver()

    result = observer.normalize_drawdown(
        -0.12
    )

    assert result == pytest.approx(0.12)


def test_positive_drawdown_is_preserved() -> None:
    observer = OutcomeObserver()

    result = observer.normalize_drawdown(
        0.12
    )

    assert result == pytest.approx(0.12)


def test_excessive_numeric_drawdown_is_rejected() -> None:
    observer = OutcomeObserver()

    with pytest.raises(ValueError):
        observer.normalize_drawdown(
            1.20
        )


def test_successful_shadow_observation() -> None:
    observer = OutcomeObserver()

    candidate = make_candidate(
        "LONG"
    )
    allocation = make_allocation(
        candidate,
        100.0,
    )

    result = observer.observe(
        candidate=candidate,
        allocation=allocation,
        exit_price=110.0,
        max_drawdown=0.05,
        benchmark_entry_price=100.0,
        benchmark_exit_price=104.0,
    )

    assert result.strategy_id == candidate.strategy_id
    assert result.metadata["shadow_allocation_id"] == allocation.allocation_id

    assert result.return_pct == pytest.approx(0.10)
    assert result.benchmark_return_pct == pytest.approx(0.04)
    assert result.alpha == pytest.approx(0.06)
    assert result.metadata["max_drawdown"] == pytest.approx(0.05)
    assert result.hit is True


def test_negative_return_fails_observation() -> None:
    observer = OutcomeObserver()

    candidate = make_candidate(
        "LONG"
    )
    allocation = make_allocation(
        candidate,
        100.0,
    )

    result = observer.observe(
        candidate=candidate,
        allocation=allocation,
        exit_price=95.0,
        max_drawdown=0.08,
    )

    assert result.return_pct == pytest.approx(-0.05)
    assert result.hit is False


def test_negative_alpha_fails_observation() -> None:
    observer = OutcomeObserver()

    candidate = make_candidate(
        "LONG"
    )
    allocation = make_allocation(
        candidate,
        100.0,
    )

    result = observer.observe(
        candidate=candidate,
        allocation=allocation,
        exit_price=105.0,
        max_drawdown=0.05,
        benchmark_entry_price=100.0,
        benchmark_exit_price=110.0,
    )

    assert result.return_pct == pytest.approx(0.05)
    assert result.benchmark_return_pct == pytest.approx(0.10)
    assert result.alpha == pytest.approx(-0.05)
    assert result.hit is False


def test_excessive_drawdown_fails_observation() -> None:
    observer = OutcomeObserver(
        OutcomeObserverConfig(
            maximum_acceptable_drawdown=0.20,
        )
    )

    candidate = make_candidate(
        "LONG"
    )
    allocation = make_allocation(
        candidate,
        100.0,
    )

    result = observer.observe(
        candidate=candidate,
        allocation=allocation,
        exit_price=110.0,
        max_drawdown=0.25,
    )

    assert result.return_pct == pytest.approx(0.10)
    assert result.metadata["max_drawdown"] == pytest.approx(0.25)
    assert result.hit is False


def test_short_candidate_observation() -> None:
    observer = OutcomeObserver()

    candidate = make_candidate(
        "SHORT"
    )
    allocation = make_allocation(
        candidate,
        100.0,
    )

    result = observer.observe(
        candidate=candidate,
        allocation=allocation,
        exit_price=90.0,
        max_drawdown=0.05,
    )

    assert result.return_pct == pytest.approx(0.10)
    assert result.hit is True


def test_allocation_must_belong_to_candidate() -> None:
    observer = OutcomeObserver()

    candidate = make_candidate()

    allocation = ShadowAllocation(
        strategy_id="wrong-strategy-id",
        asset="BTC",
        weight=0.08,
        notional_value=8_000.0,
        entry_price=100.0,
        metadata={},
    )

    with pytest.raises(ValueError):
        observer.observe(
            candidate=candidate,
            allocation=allocation,
            exit_price=110.0,
            max_drawdown=0.05,
        )


def test_missing_entry_price_is_rejected() -> None:
    observer = OutcomeObserver()

    candidate = make_candidate()
    allocation = make_allocation(
        candidate,
        None,
    )

    with pytest.raises(ValueError):
        observer.observe(
            candidate=candidate,
            allocation=allocation,
            exit_price=110.0,
            max_drawdown=0.05,
        )


def test_invalid_candidate_is_rejected() -> None:
    observer = OutcomeObserver()

    candidate = make_candidate()
    allocation = make_allocation(candidate)

    with pytest.raises(TypeError):
        observer.observe(
            candidate="invalid-candidate",
            allocation=allocation,
            exit_price=110.0,
            max_drawdown=0.05,
        )


def test_invalid_allocation_is_rejected() -> None:
    observer = OutcomeObserver()

    candidate = make_candidate()

    with pytest.raises(TypeError):
        observer.observe(
            candidate=candidate,
            allocation="invalid-allocation",
            exit_price=110.0,
            max_drawdown=0.05,
        )


def test_zero_entry_price_is_rejected() -> None:
    observer = OutcomeObserver()

    with pytest.raises(ValueError):
        observer.calculate_return(
            0.0,
            110.0,
            "LONG",
        )


def test_negative_exit_price_is_rejected() -> None:
    observer = OutcomeObserver()

    with pytest.raises(ValueError):
        observer.calculate_return(
            100.0,
            -10.0,
            "LONG",
        )


def test_non_numeric_price_is_rejected() -> None:
    observer = OutcomeObserver()

    with pytest.raises(ValueError):
        observer.calculate_return(
            "invalid",
            110.0,
            "LONG",
        )


def test_custom_success_threshold() -> None:
    observer = OutcomeObserver(
        OutcomeObserverConfig(
            minimum_success_return=0.05,
            minimum_success_alpha=0.02,
        )
    )

    assert (
        observer.determine_success(
            strategy_return=0.04,
            alpha=0.03,
            max_drawdown=0.05,
        )
        is False
    )

    assert (
        observer.determine_success(
            strategy_return=0.06,
            alpha=0.03,
            max_drawdown=0.05,
        )
        is True
    )


def test_metadata_is_research_only() -> None:
    observer = OutcomeObserver()

    candidate = make_candidate()
    allocation = make_allocation(
        candidate,
        100.0,
    )

    result = observer.observe(
        candidate=candidate,
        allocation=allocation,
        exit_price=110.0,
        max_drawdown=0.05,
        benchmark_entry_price=100.0,
        benchmark_exit_price=104.0,
    )

    assert (
        result.metadata["observer"]
        == "v5.0.9-outcome-observer"
    )
    assert result.metadata["mode"] == "SHADOW_OBSERVATION"
    assert result.metadata["research_only"] is True
    assert result.metadata["paper_observation"] is True
    assert result.metadata["real_money_execution"] is False

    assert (
        result.metadata["eligible_for_automatic_live_promotion"]
        is False
    )


def test_callable_interface() -> None:
    observer = OutcomeObserver()

    candidate = make_candidate()
    allocation = make_allocation(
        candidate,
        100.0,
    )

    result = observer(
        candidate,
        allocation,
        110.0,
        0.05,
        100.0,
        104.0,
    )

    assert result.return_pct == pytest.approx(0.10)
    assert result.alpha == pytest.approx(0.06)
    assert result.hit is True


def test_config_rejects_negative_drawdown_limit() -> None:
    with pytest.raises(ValueError):
        OutcomeObserverConfig(
            maximum_acceptable_drawdown=-0.01,
        )


def test_config_rejects_drawdown_limit_above_one() -> None:
    with pytest.raises(ValueError):
        OutcomeObserverConfig(
            maximum_acceptable_drawdown=1.01,
        )
