"""
v5.0.9 - Autonomous Research Loop
Core Research Loop Engine.

Pipeline:

Research Task
    -> Prioritization
    -> Hypothesis
    -> Strategy Candidate
    -> Validation
    -> Independent Risk Gate
    -> Shadow Allocation
    -> Outcome
    -> Learning Feedback

Important:
- No automatic real-money execution.
- No uncontrolled source-code modification.
- Risk remains an independent hard gate.
- Every state transition is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

from autonomous_research_loop.models import (
    Hypothesis,
    LearningFeedback,
    ResearchCycle,
    ResearchOutcome,
    ResearchStatus,
    ResearchTask,
    RiskAssessment,
    RiskDecision,
    ShadowAllocation,
    StrategyCandidate,
    ValidationDecision,
    ValidationResult,
)
from autonomous_research_loop.research_prioritizer import ResearchPrioritizer


HypothesisGenerator = Callable[[ResearchTask], Hypothesis]

StrategyGenerator = Callable[
    [ResearchTask, Hypothesis],
    StrategyCandidate,
]

StrategyValidator = Callable[
    [ResearchTask, Hypothesis, StrategyCandidate],
    ValidationResult,
]

RiskGovernor = Callable[
    [ResearchTask, StrategyCandidate, ValidationResult],
    RiskAssessment,
]

ShadowAllocator = Callable[
    [ResearchTask, StrategyCandidate, RiskAssessment],
    ShadowAllocation,
]

OutcomeObserver = Callable[
    [ResearchTask, StrategyCandidate, ShadowAllocation],
    ResearchOutcome,
]

FeedbackGenerator = Callable[
    [
        ResearchTask,
        StrategyCandidate,
        ValidationResult,
        RiskAssessment,
        ResearchOutcome,
    ],
    LearningFeedback,
]


@dataclass
class ResearchLoopDependencies:
    """
    External components used by the autonomous research loop.

    These callables form adapter boundaries between v5.0.9 and
    the existing platform modules.

    Later they can connect to:
    - existing agents
    - institutional backtesting
    - causal research
    - market regime intelligence
    - portfolio construction
    - alpha lifecycle
    - risk governor
    - meta-learning
    """

    hypothesis_generator: HypothesisGenerator
    strategy_generator: StrategyGenerator
    strategy_validator: StrategyValidator
    risk_governor: RiskGovernor
    shadow_allocator: ShadowAllocator
    outcome_observer: OutcomeObserver
    feedback_generator: FeedbackGenerator


class AutonomousResearchLoop:
    """
    Central orchestration engine for v5.0.9.

    The engine deliberately depends on injected adapters instead of
    importing existing platform modules directly.

    This keeps the research loop:
    - testable
    - modular
    - auditable
    - reversible
    - independent from implementation details
    """

    def __init__(
        self,
        dependencies: ResearchLoopDependencies,
        prioritizer: Optional[ResearchPrioritizer] = None,
    ) -> None:
        self.dependencies = dependencies
        self.prioritizer = prioritizer or ResearchPrioritizer()

    def create_cycle(
        self,
        task: ResearchTask,
    ) -> ResearchCycle:
        """
        Create a new auditable research cycle.
        """

        cycle = ResearchCycle(task=task)

        cycle.add_audit_event(
            "RESEARCH_CYCLE_CREATED",
            {
                "task_id": task.task_id,
                "trigger": task.trigger.value,
            },
        )

        return cycle

    def prioritize_cycle(
        self,
        cycle: ResearchCycle,
    ) -> float:
        """
        Prioritize the research task associated with the cycle.
        """

        score = self.prioritizer.prioritize(cycle.task)

        cycle.add_audit_event(
            "RESEARCH_TASK_PRIORITIZED",
            {
                "priority": cycle.task.priority.value,
                "priority_score": score,
            },
        )

        return score

    def generate_hypothesis(
        self,
        cycle: ResearchCycle,
    ) -> Hypothesis:
        """
        Generate a testable hypothesis.
        """

        hypothesis = self.dependencies.hypothesis_generator(
            cycle.task
        )

        if hypothesis.task_id != cycle.task.task_id:
            raise ValueError(
                "Generated hypothesis does not belong "
                "to the active research task."
            )

        cycle.hypothesis = hypothesis

        cycle.task.update_status(
            ResearchStatus.HYPOTHESIS_GENERATED
        )

        cycle.add_audit_event(
            "HYPOTHESIS_GENERATED",
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "confidence": hypothesis.confidence,
            },
        )

        return hypothesis

    def generate_strategy(
        self,
        cycle: ResearchCycle,
    ) -> StrategyCandidate:
        """
        Convert the hypothesis into a strategy candidate.
        """

        if cycle.hypothesis is None:
            raise RuntimeError(
                "Cannot generate strategy without hypothesis."
            )

        strategy = self.dependencies.strategy_generator(
            cycle.task,
            cycle.hypothesis,
        )

        if (
            strategy.hypothesis_id
            != cycle.hypothesis.hypothesis_id
        ):
            raise ValueError(
                "Generated strategy does not belong "
                "to the active hypothesis."
            )

        cycle.strategy = strategy

        cycle.task.update_status(
            ResearchStatus.STRATEGY_GENERATED
        )

        cycle.add_audit_event(
            "STRATEGY_GENERATED",
            {
                "strategy_id": strategy.strategy_id,
                "strategy_name": strategy.name,
            },
        )

        return strategy

    def validate_strategy(
        self,
        cycle: ResearchCycle,
    ) -> ValidationResult:
        """
        Validate the candidate using the injected validation layer.
        """

        if cycle.hypothesis is None:
            raise RuntimeError(
                "Cannot validate without hypothesis."
            )

        if cycle.strategy is None:
            raise RuntimeError(
                "Cannot validate without strategy."
            )

        cycle.task.update_status(
            ResearchStatus.VALIDATING
        )

        validation = self.dependencies.strategy_validator(
            cycle.task,
            cycle.hypothesis,
            cycle.strategy,
        )

        if validation.strategy_id != cycle.strategy.strategy_id:
            raise ValueError(
                "Validation result does not belong "
                "to the active strategy."
            )

        cycle.validation = validation

        cycle.add_audit_event(
            "STRATEGY_VALIDATED",
            {
                "validation_id": validation.validation_id,
                "decision": validation.decision.value,
                "score": validation.score,
            },
        )

        return validation

    def apply_risk_gate(
        self,
        cycle: ResearchCycle,
    ) -> RiskAssessment:
        """
        Apply the independent Risk Governor.

        The Risk Governor can:
        - approve
        - reduce
        - veto
        - shutdown

        A research score can never bypass the risk gate.
        """

        if cycle.strategy is None:
            raise RuntimeError(
                "Cannot apply risk gate without strategy."
            )

        if cycle.validation is None:
            raise RuntimeError(
                "Cannot apply risk gate without validation."
            )

        assessment = self.dependencies.risk_governor(
            cycle.task,
            cycle.strategy,
            cycle.validation,
        )

        if assessment.strategy_id != cycle.strategy.strategy_id:
            raise ValueError(
                "Risk assessment does not belong "
                "to the active strategy."
            )

        cycle.risk_assessment = assessment

        cycle.add_audit_event(
            "RISK_GATE_COMPLETED",
            {
                "assessment_id": assessment.assessment_id,
                "decision": assessment.decision.value,
                "risk_score": assessment.risk_score,
            },
        )

        return assessment

    def create_shadow_allocation(
        self,
        cycle: ResearchCycle,
    ) -> ShadowAllocation:
        """
        Create a virtual/shadow allocation.

        This function must never be interpreted as permission
        for automatic real-money execution.
        """

        if cycle.strategy is None:
            raise RuntimeError(
                "Cannot create shadow allocation without strategy."
            )

        if cycle.validation is None:
            raise RuntimeError(
                "Cannot create shadow allocation without validation."
            )

        if cycle.risk_assessment is None:
            raise RuntimeError(
                "Cannot create shadow allocation without risk assessment."
            )

        if (
            cycle.validation.decision
            != ValidationDecision.PROMOTE
        ):
            raise RuntimeError(
                "Only promoted strategies may enter shadow allocation."
            )

        if cycle.risk_assessment.decision in {
            RiskDecision.VETO,
            RiskDecision.SHUTDOWN,
        }:
            raise RuntimeError(
                "Risk Governor blocked shadow allocation."
            )

        allocation = self.dependencies.shadow_allocator(
            cycle.task,
            cycle.strategy,
            cycle.risk_assessment,
        )

        if allocation.strategy_id != cycle.strategy.strategy_id:
            raise ValueError(
                "Shadow allocation does not belong "
                "to the active strategy."
            )

        cycle.shadow_allocation = allocation

        cycle.task.update_status(
            ResearchStatus.SHADOW
        )

        cycle.add_audit_event(
            "SHADOW_ALLOCATION_CREATED",
            {
                "allocation_id": allocation.allocation_id,
                "weight": allocation.weight,
                "notional_value": allocation.notional_value,
            },
        )

        return allocation

    def observe_outcome(
        self,
        cycle: ResearchCycle,
    ) -> ResearchOutcome:
        """
        Observe the performance of the shadow candidate.
        """

        if cycle.strategy is None:
            raise RuntimeError(
                "Cannot observe outcome without strategy."
            )

        if cycle.shadow_allocation is None:
            raise RuntimeError(
                "Cannot observe outcome without shadow allocation."
            )

        cycle.task.update_status(
            ResearchStatus.OBSERVING
        )

        outcome = self.dependencies.outcome_observer(
            cycle.task,
            cycle.strategy,
            cycle.shadow_allocation,
        )

        if outcome.strategy_id != cycle.strategy.strategy_id:
            raise ValueError(
                "Outcome does not belong to the active strategy."
            )

        cycle.outcome = outcome

        cycle.add_audit_event(
            "RESEARCH_OUTCOME_OBSERVED",
            {
                "outcome_id": outcome.outcome_id,
                "return_pct": outcome.return_pct,
                "alpha": outcome.alpha,
            },
        )

        return outcome

    def generate_feedback(
        self,
        cycle: ResearchCycle,
    ) -> LearningFeedback:
        """
        Convert observed performance into learning feedback.
        """

        if cycle.strategy is None:
            raise RuntimeError(
                "Cannot generate feedback without strategy."
            )

        if cycle.validation is None:
            raise RuntimeError(
                "Cannot generate feedback without validation."
            )

        if cycle.risk_assessment is None:
            raise RuntimeError(
                "Cannot generate feedback without risk assessment."
            )

        if cycle.outcome is None:
            raise RuntimeError(
                "Cannot generate feedback without outcome."
            )

        feedback = self.dependencies.feedback_generator(
            cycle.task,
            cycle.strategy,
            cycle.validation,
            cycle.risk_assessment,
            cycle.outcome,
        )

        if feedback.strategy_id != cycle.strategy.strategy_id:
            raise ValueError(
                "Feedback does not belong "
                "to the active strategy."
            )

        if feedback.outcome_id != cycle.outcome.outcome_id:
            raise ValueError(
                "Feedback does not belong "
                "to the active outcome."
            )

        cycle.feedback = feedback

        cycle.add_audit_event(
            "LEARNING_FEEDBACK_GENERATED",
            {
                "feedback_id": feedback.feedback_id,
                "performance_score": feedback.performance_score,
                "create_new_research_task":
                    feedback.create_new_research_task,
            },
        )

        return feedback

    def reject_cycle(
        self,
        cycle: ResearchCycle,
        reason: str,
    ) -> ResearchCycle:
        """
        Reject and close a research cycle.
        """

        cycle.task.update_status(
            ResearchStatus.REJECTED
        )

        cycle.add_audit_event(
            "RESEARCH_CYCLE_REJECTED",
            {
                "reason": reason,
            },
        )

        return cycle

    def run_until_shadow(
        self,
        task: ResearchTask,
    ) -> ResearchCycle:
        """
        Run the autonomous research process until shadow allocation.

        The process intentionally stops at shadow allocation.
        No real-money execution exists in this engine.
        """

        cycle = self.create_cycle(task)

        try:
            self.prioritize_cycle(cycle)

            self.generate_hypothesis(cycle)
            self.generate_strategy(cycle)

            validation = self.validate_strategy(cycle)

            if validation.decision != ValidationDecision.PROMOTE:
                return self.reject_cycle(
                    cycle,
                    reason=(
                        "Strategy was not promoted "
                        "by validation layer."
                    ),
                )

            cycle.task.update_status(
                ResearchStatus.PROMOTED
            )

            cycle.add_audit_event(
                "STRATEGY_PROMOTED",
                {
                    "strategy_id": cycle.strategy.strategy_id,
                },
            )

            risk = self.apply_risk_gate(cycle)

            if risk.decision in {
                RiskDecision.VETO,
                RiskDecision.SHUTDOWN,
            }:
                return self.reject_cycle(
                    cycle,
                    reason=(
                        f"Risk Governor decision: "
                        f"{risk.decision.value}"
                    ),
                )

            self.create_shadow_allocation(cycle)

            return cycle

        except Exception as exc:
            cycle.task.update_status(
                ResearchStatus.FAILED
            )

            cycle.add_audit_event(
                "RESEARCH_CYCLE_FAILED",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

            raise

    def complete_shadow_cycle(
        self,
        cycle: ResearchCycle,
    ) -> ResearchCycle:
        """
        Observe a shadow candidate, generate learning feedback,
        and complete the research cycle.
        """

        if cycle.task.status != ResearchStatus.SHADOW:
            raise RuntimeError(
                "Only a SHADOW research cycle can be completed."
            )

        self.observe_outcome(cycle)
        self.generate_feedback(cycle)

        cycle.complete()

        return cycle

    def rank_tasks(
        self,
        tasks: Iterable[ResearchTask],
    ) -> List[ResearchTask]:
        """
        Return research tasks ordered by autonomous priority.
        """

        ranked = self.prioritizer.rank_tasks(tasks)

        return [
            task
            for task, _score in ranked
        ]