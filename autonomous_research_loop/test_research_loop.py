"""
Tests for v5.0.9 Autonomous Research Loop.

Covered paths:
- research prioritization
- successful promotion
- shadow allocation
- validation rejection
- risk veto
- outcome observation
- learning feedback
- full cycle completion
"""

from autonomous_research_loop.models import (
    Hypothesis,
    LearningFeedback,
    ResearchOutcome,
    ResearchPriority,
    ResearchStatus,
    ResearchTask,
    ResearchTrigger,
    RiskAssessment,
    RiskDecision,
    ShadowAllocation,
    StrategyCandidate,
    ValidationDecision,
    ValidationResult,
)
from autonomous_research_loop.research_loop import (
    AutonomousResearchLoop,
    ResearchLoopDependencies,
)


def make_task() -> ResearchTask:
    return ResearchTask(
        title="BTC regime research",
        description="Investigate BTC momentum after regime transition.",
        trigger=ResearchTrigger.REGIME_CHANGE,
        asset="BTC",
        regime="risk_on",
        horizon="7D",
        uncertainty_score=0.80,
        performance_gap_score=0.75,
    )


def hypothesis_generator(task: ResearchTask) -> Hypothesis:
    return Hypothesis(
        task_id=task.task_id,
        statement=(
            "BTC positive momentum persists after "
            "a confirmed risk-on regime transition."
        ),
        asset=task.asset,
        regime=task.regime,
        horizon=task.horizon,
        confidence=0.78,
        evidence=[
            "positive momentum",
            "regime transition",
        ],
    )


def strategy_generator(
    task: ResearchTask,
    hypothesis: Hypothesis,
) -> StrategyCandidate:
    return StrategyCandidate(
        hypothesis_id=hypothesis.hypothesis_id,
        name="BTC Risk-On Momentum",
        entry_rule="Enter after confirmed positive momentum.",
        exit_rule="Exit when momentum or regime invalidates.",
        asset=task.asset,
        horizon=task.horizon,
        position_size=0.05,
        stop_condition="Risk-off regime detected.",
    )


def promoted_validator(
    task: ResearchTask,
    hypothesis: Hypothesis,
    strategy: StrategyCandidate,
) -> ValidationResult:
    return ValidationResult(
        strategy_id=strategy.strategy_id,
        decision=ValidationDecision.PROMOTE,
        score=0.84,
        out_of_sample_score=0.81,
        robustness_score=0.79,
        causal_score=0.73,
        sharpe_ratio=1.42,
        sortino_ratio=1.71,
        max_drawdown=0.09,
        reasons=[
            "Passed validation thresholds.",
        ],
    )


def rejected_validator(
    task: ResearchTask,
    hypothesis: Hypothesis,
    strategy: StrategyCandidate,
) -> ValidationResult:
    return ValidationResult(
        strategy_id=strategy.strategy_id,
        decision=ValidationDecision.REJECT,
        score=0.31,
        robustness_score=0.28,
        reasons=[
            "Insufficient robustness.",
        ],
    )


def approving_risk_governor(
    task: ResearchTask,
    strategy: StrategyCandidate,
    validation: ValidationResult,
) -> RiskAssessment:
    return RiskAssessment(
        strategy_id=strategy.strategy_id,
        decision=RiskDecision.APPROVE,
        risk_score=0.22,
        approved_position_size=strategy.position_size,
        reasons=[
            "Risk within shadow-allocation limits.",
        ],
    )


def vetoing_risk_governor(
    task: ResearchTask,
    strategy: StrategyCandidate,
    validation: ValidationResult,
) -> RiskAssessment:
    return RiskAssessment(
        strategy_id=strategy.strategy_id,
        decision=RiskDecision.VETO,
        risk_score=0.96,
        approved_position_size=0.0,
        reasons=[
            "Risk threshold exceeded.",
        ],
    )


def shadow_allocator(
    task: ResearchTask,
    strategy: StrategyCandidate,
    risk: RiskAssessment,
) -> ShadowAllocation:
    return ShadowAllocation(
        strategy_id=strategy.strategy_id,
        asset=task.asset or "UNKNOWN",
        weight=risk.approved_position_size or 0.0,
        notional_value=10_000.0,
        entry_price=60_000.0,
    )


def outcome_observer(
    task: ResearchTask,
    strategy: StrategyCandidate,
    allocation: ShadowAllocation,
) -> ResearchOutcome:
    return ResearchOutcome(
        strategy_id=strategy.strategy_id,
        pnl=320.0,
        return_pct=0.032,
        benchmark_return_pct=0.018,
        alpha=0.014,
        hit=True,
    )


def feedback_generator(
    task: ResearchTask,
    strategy: StrategyCandidate,
    validation: ValidationResult,
    risk: RiskAssessment,
    outcome: ResearchOutcome,
) -> LearningFeedback:
    return LearningFeedback(
        strategy_id=strategy.strategy_id,
        outcome_id=outcome.outcome_id,
        performance_score=0.82,
        lessons=[
            "Regime-aware momentum hypothesis remained valid.",
        ],
        recommended_actions=[
            "Continue shadow observation.",
        ],
        create_new_research_task=False,
    )


def make_dependencies(
    validator=promoted_validator,
    risk_governor=approving_risk_governor,
) -> ResearchLoopDependencies:
    return ResearchLoopDependencies(
        hypothesis_generator=hypothesis_generator,
        strategy_generator=strategy_generator,
        strategy_validator=validator,
        risk_governor=risk_governor,
        shadow_allocator=shadow_allocator,
        outcome_observer=outcome_observer,
        feedback_generator=feedback_generator,
    )


def test_successful_research_cycle_reaches_shadow() -> None:
    loop = AutonomousResearchLoop(
        dependencies=make_dependencies()
    )

    task = make_task()

    cycle = loop.run_until_shadow(task)

    assert cycle.task.status == ResearchStatus.SHADOW

    assert cycle.hypothesis is not None
    assert cycle.strategy is not None
    assert cycle.validation is not None
    assert cycle.risk_assessment is not None
    assert cycle.shadow_allocation is not None

    assert cycle.validation.decision == ValidationDecision.PROMOTE
    assert cycle.risk_assessment.decision == RiskDecision.APPROVE

    assert cycle.task.priority in {
        ResearchPriority.HIGH,
        ResearchPriority.CRITICAL,
    }

    events = [
        event["event"]
        for event in cycle.audit_log
    ]

    assert "RESEARCH_CYCLE_CREATED" in events
    assert "RESEARCH_TASK_PRIORITIZED" in events
    assert "HYPOTHESIS_GENERATED" in events
    assert "STRATEGY_GENERATED" in events
    assert "STRATEGY_VALIDATED" in events
    assert "STRATEGY_PROMOTED" in events
    assert "RISK_GATE_COMPLETED" in events
    assert "SHADOW_ALLOCATION_CREATED" in events


def test_validation_rejection_blocks_shadow_allocation() -> None:
    loop = AutonomousResearchLoop(
        dependencies=make_dependencies(
            validator=rejected_validator
        )
    )

    task = make_task()

    cycle = loop.run_until_shadow(task)

    assert cycle.task.status == ResearchStatus.REJECTED
    assert cycle.validation is not None
    assert cycle.validation.decision == ValidationDecision.REJECT

    assert cycle.risk_assessment is None
    assert cycle.shadow_allocation is None

    events = [
        event["event"]
        for event in cycle.audit_log
    ]

    assert "RESEARCH_CYCLE_REJECTED" in events
    assert "RISK_GATE_COMPLETED" not in events
    assert "SHADOW_ALLOCATION_CREATED" not in events


def test_risk_veto_blocks_shadow_allocation() -> None:
    loop = AutonomousResearchLoop(
        dependencies=make_dependencies(
            risk_governor=vetoing_risk_governor
        )
    )

    task = make_task()

    cycle = loop.run_until_shadow(task)

    assert cycle.task.status == ResearchStatus.REJECTED

    assert cycle.validation is not None
    assert cycle.validation.decision == ValidationDecision.PROMOTE

    assert cycle.risk_assessment is not None
    assert cycle.risk_assessment.decision == RiskDecision.VETO

    assert cycle.shadow_allocation is None

    events = [
        event["event"]
        for event in cycle.audit_log
    ]

    assert "STRATEGY_PROMOTED" in events
    assert "RISK_GATE_COMPLETED" in events
    assert "RESEARCH_CYCLE_REJECTED" in events
    assert "SHADOW_ALLOCATION_CREATED" not in events


def test_complete_shadow_cycle_generates_feedback() -> None:
    loop = AutonomousResearchLoop(
        dependencies=make_dependencies()
    )

    task = make_task()

    cycle = loop.run_until_shadow(task)

    assert cycle.task.status == ResearchStatus.SHADOW

    completed = loop.complete_shadow_cycle(cycle)

    assert completed.task.status == ResearchStatus.COMPLETED
    assert completed.outcome is not None
    assert completed.feedback is not None
    assert completed.completed_at is not None

    assert completed.outcome.alpha == 0.014
    assert completed.feedback.performance_score == 0.82

    events = [
        event["event"]
        for event in completed.audit_log
    ]

    assert "RESEARCH_OUTCOME_OBSERVED" in events
    assert "LEARNING_FEEDBACK_GENERATED" in events
    assert "RESEARCH_CYCLE_COMPLETED" in events


def test_rank_tasks_returns_highest_priority_first() -> None:
    loop = AutonomousResearchLoop(
        dependencies=make_dependencies()
    )

    low_task = ResearchTask(
        title="Low-priority research",
        description="Routine research.",
        trigger=ResearchTrigger.MANUAL,
        uncertainty_score=0.10,
        performance_gap_score=0.10,
    )

    high_task = ResearchTask(
        title="High-priority research",
        description="Major risk event research.",
        trigger=ResearchTrigger.RISK_EVENT,
        uncertainty_score=0.95,
        performance_gap_score=0.90,
    )

    ranked = loop.rank_tasks(
        [
            low_task,
            high_task,
        ]
    )

    assert ranked[0].task_id == high_task.task_id
    assert ranked[1].task_id == low_task.task_id