from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class ResearchLoopStage(str, Enum):
    MARKET_DATA = "MARKET_DATA"
    UNCERTAINTY_DETECTION = "UNCERTAINTY_DETECTION"
    HYPOTHESIS = "HYPOTHESIS"
    EXPERIMENT_DESIGN = "EXPERIMENT_DESIGN"
    AGENT_RESEARCH = "AGENT_RESEARCH"
    VALIDATION = "VALIDATION"
    ROBUSTNESS = "ROBUSTNESS"
    SHADOW_PORTFOLIO = "SHADOW_PORTFOLIO"
    ATTRIBUTION = "ATTRIBUTION"
    META_LEARNING = "META_LEARNING"
    NEW_RESEARCH_TASK = "NEW_RESEARCH_TASK"


class ResearchLoopStatus(str, Enum):
    INITIALIZED = "INITIALIZED"
    RUNNING = "RUNNING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class ValidationStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    PASSED = "PASSED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ResearchLoopEvent:
    sequence: int
    stage: ResearchLoopStage
    action: str
    status: str
    metadata: Mapping[str, object] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ResearchLoopResult:
    cycle_id: str
    status: ResearchLoopStatus
    current_stage: ResearchLoopStage
    hypothesis: str | None
    research_question: str | None
    experiment_id: str | None

    validation_status: ValidationStatus
    robustness_status: ValidationStatus
    shadow_portfolio_status: ValidationStatus

    outcome_attribution: Mapping[str, object]
    meta_learning_output: Mapping[str, object]
    next_research_task: str | None

    audit_trail: tuple[ResearchLoopEvent, ...]
    reproducibility_metadata: Mapping[str, object]

    human_approval_required: bool
    human_approved: bool

    error: str | None = None


class ContinuousResearchLoop:
    """
    v6.0.8 Continuous Research Loop.

    Full research cycle:

    Market Data / Research Memory
        ->
    Uncertainty / Regime Detection
        ->
    Hypothesis / Research Question
        ->
    Experiment Design
        ->
    Agent Research
        ->
    Causal / Statistical Validation
        ->
    Out-of-Sample / Robustness
        ->
    Shadow Portfolio / Digital Twin
        ->
    Outcome Attribution
        ->
    Meta-Learning
        ->
    New Research Task

    The loop is:
    - measurable
    - auditable
    - reproducible
    - human-governed
    """

    def __init__(
        self,
        *,
        require_human_approval: bool = True,
    ) -> None:

        if not isinstance(
            require_human_approval,
            bool,
        ):
            raise ValueError(
                "require_human_approval must be a boolean."
            )

        self.require_human_approval = (
            require_human_approval
        )

    # =====================================================
    # VALIDATION
    # =====================================================

    @staticmethod
    def _validate_string(
        value: object,
        field_name: str,
    ) -> str:

        if not isinstance(value, str):
            raise ValueError(
                f"{field_name} must be a string."
            )

        value = value.strip()

        if not value:
            raise ValueError(
                f"{field_name} must not be empty."
            )

        return value

    @staticmethod
    def _validate_market_data(
        value: object,
    ) -> tuple[object, ...]:
        """
        Strict market_data validation.

        Invalid:
            None
            strings
            bytes
            mappings/dicts
            numbers
            arbitrary non-iterables

        Valid:
            list
            tuple
            set
            other iterable sequences
        """

        # CRITICAL:
        # None is NOT an empty dataset.
        if value is None:
            raise ValueError(
                "market_data must be a sequence."
            )

        # Strings are iterable but are not market datasets.
        if isinstance(
            value,
            (str, bytes, bytearray),
        ):
            raise ValueError(
                "market_data must be a sequence."
            )

        # Mapping / dict is explicitly forbidden.
        if isinstance(
            value,
            Mapping,
        ):
            raise ValueError(
                "market_data must be a sequence, "
                "not a mapping."
            )

        # Numbers and similar scalar values.
        if isinstance(
            value,
            (int, float, complex, bool),
        ):
            raise ValueError(
                "market_data must be a sequence."
            )

        try:
            normalized = tuple(value)
        except TypeError as exc:
            raise ValueError(
                "market_data must be a sequence."
            ) from exc

        return normalized

    @staticmethod
    def _validate_mapping(
        value: object,
        field_name: str,
    ) -> dict[str, object]:

        if value is None:
            return {}

        if not isinstance(
            value,
            Mapping,
        ):
            raise ValueError(
                f"{field_name} must be a mapping."
            )

        return dict(value)

    @staticmethod
    def _validate_callable(
        value: object,
        field_name: str,
    ) -> None:

        if value is not None and not callable(value):
            raise ValueError(
                f"{field_name} must be callable or None."
            )

    # =====================================================
    # AUDIT
    # =====================================================

    @staticmethod
    def _create_event(
        sequence: int,
        stage: ResearchLoopStage,
        action: str,
        status: str,
        metadata: Mapping[str, object] | None = None,
    ) -> ResearchLoopEvent:

        return ResearchLoopEvent(
            sequence=sequence,
            stage=stage,
            action=action,
            status=status,
            metadata=dict(metadata or {}),
        )

    # =====================================================
    # BLOCKED RESULT
    # =====================================================

    def _blocked_result(
        self,
        *,
        cycle_id: str,
        human_approved: bool,
        metadata: Mapping[str, object],
    ) -> ResearchLoopResult:

        event = self._create_event(
            sequence=1,
            stage=ResearchLoopStage.HYPOTHESIS,
            action="human_approval_gate",
            status="BLOCKED",
        )

        reproducibility = {
            **dict(metadata),
            "cycle_id": cycle_id,
            "market_data_size": 0,
            "execution_started": False,
            "human_approval_required": (
                self.require_human_approval
            ),
            "human_approved": human_approved,
            "continuous_loop": True,
            "measurable": True,
            "auditable": True,
            "reproducible": True,
            "human_governed": True,
            "audit_event_count": 1,
        }

        return ResearchLoopResult(
            cycle_id=cycle_id,
            status=ResearchLoopStatus.BLOCKED,
            current_stage=ResearchLoopStage.HYPOTHESIS,
            hypothesis=None,
            research_question=None,
            experiment_id=None,
            validation_status=ValidationStatus.NOT_RUN,
            robustness_status=ValidationStatus.NOT_RUN,
            shadow_portfolio_status=ValidationStatus.NOT_RUN,
            outcome_attribution={},
            meta_learning_output={},
            next_research_task=None,
            audit_trail=(event,),
            reproducibility_metadata=reproducibility,
            human_approval_required=(
                self.require_human_approval
            ),
            human_approved=human_approved,
            error=(
                "Research loop blocked because "
                "human approval was not granted."
            ),
        )

    # =====================================================
    # MAIN LOOP
    # =====================================================

    def run(
        self,
        *,
        cycle_id: str,
        market_data: Sequence[object],
        uncertainty_signal: Mapping[str, object] | None = None,
        hypothesis: str | None = None,
        research_question: str | None = None,
        experiment_id: str | None = None,
        research_runner: Any = None,
        validation_runner: Any = None,
        robustness_runner: Any = None,
        shadow_portfolio_runner: Any = None,
        attribution_runner: Any = None,
        meta_learning_runner: Any = None,
        next_task_generator: Any = None,
        human_approval: bool = False,
        metadata: Mapping[str, object] | None = None,
    ) -> ResearchLoopResult:

        # -------------------------------------------------
        # INPUT VALIDATION
        # -------------------------------------------------

        cycle_id = self._validate_string(
            cycle_id,
            "cycle_id",
        )

        # IMPORTANT:
        # This is deliberately before the human gate.
        # None and {} MUST raise ValueError.
        market_data = self._validate_market_data(
            market_data
        )

        uncertainty_signal = self._validate_mapping(
            uncertainty_signal,
            "uncertainty_signal",
        )

        metadata = self._validate_mapping(
            metadata,
            "metadata",
        )

        runners = {
            "research_runner": research_runner,
            "validation_runner": validation_runner,
            "robustness_runner": robustness_runner,
            "shadow_portfolio_runner": (
                shadow_portfolio_runner
            ),
            "attribution_runner": attribution_runner,
            "meta_learning_runner": (
                meta_learning_runner
            ),
            "next_task_generator": (
                next_task_generator
            ),
        }

        for name, runner in runners.items():
            self._validate_callable(
                runner,
                name,
            )

        if not isinstance(
            human_approval,
            bool,
        ):
            raise ValueError(
                "human_approval must be a boolean."
            )

        # -------------------------------------------------
        # HUMAN GOVERNANCE GATE
        # -------------------------------------------------

        if (
            self.require_human_approval
            and not human_approval
        ):
            return self._blocked_result(
                cycle_id=cycle_id,
                human_approved=human_approval,
                metadata=metadata,
            )

        # -------------------------------------------------
        # STATE
        # -------------------------------------------------

        audit: list[ResearchLoopEvent] = []
        sequence = 0

        validation_status = ValidationStatus.NOT_RUN
        robustness_status = ValidationStatus.NOT_RUN
        shadow_portfolio_status = ValidationStatus.NOT_RUN

        normalized_hypothesis = None
        normalized_question = None
        normalized_experiment_id = None

        attribution_output: dict[str, object] = {}
        meta_learning_output: dict[str, object] = {}

        next_research_task = None

        def add_event(
            stage: ResearchLoopStage,
            action: str,
            status: str,
            event_metadata: Mapping[str, object] | None = None,
        ) -> None:

            nonlocal sequence

            sequence += 1

            audit.append(
                self._create_event(
                    sequence=sequence,
                    stage=stage,
                    action=action,
                    status=status,
                    metadata=event_metadata,
                )
            )

        try:
            # =============================================
            # 1. MARKET DATA
            # =============================================

            add_event(
                ResearchLoopStage.MARKET_DATA,
                "market_data_loaded",
                "COMPLETED",
                {
                    "data_points": len(market_data),
                },
            )

            # =============================================
            # 2. UNCERTAINTY / REGIME
            # =============================================

            add_event(
                ResearchLoopStage.UNCERTAINTY_DETECTION,
                "uncertainty_signal_evaluated",
                "COMPLETED",
                {
                    "signal_count": len(
                        uncertainty_signal
                    ),
                },
            )

            # =============================================
            # 3. HYPOTHESIS
            # =============================================

            if hypothesis is not None:
                normalized_hypothesis = (
                    self._validate_string(
                        hypothesis,
                        "hypothesis",
                    )
                )

            if research_question is not None:
                normalized_question = (
                    self._validate_string(
                        research_question,
                        "research_question",
                    )
                )

            add_event(
                ResearchLoopStage.HYPOTHESIS,
                "research_hypothesis_defined",
                "COMPLETED",
                {
                    "has_hypothesis": (
                        normalized_hypothesis
                        is not None
                    ),
                    "has_research_question": (
                        normalized_question
                        is not None
                    ),
                },
            )

            # =============================================
            # 4. EXPERIMENT DESIGN
            # =============================================

            if experiment_id is not None:
                normalized_experiment_id = (
                    self._validate_string(
                        experiment_id,
                        "experiment_id",
                    )
                )

            add_event(
                ResearchLoopStage.EXPERIMENT_DESIGN,
                "experiment_defined",
                "COMPLETED",
                {
                    "experiment_id": (
                        normalized_experiment_id
                    ),
                },
            )

            # =============================================
            # 5. AGENT RESEARCH
            # =============================================

            research_output = None

            if research_runner is not None:
                research_output = research_runner(
                    market_data,
                    normalized_hypothesis,
                    normalized_question,
                    normalized_experiment_id,
                )

            add_event(
                ResearchLoopStage.AGENT_RESEARCH,
                "agent_research_completed",
                "COMPLETED",
                {
                    "runner_executed": (
                        research_runner is not None
                    ),
                },
            )

            # =============================================
            # 6. VALIDATION
            # =============================================

            validation_output = None

            if validation_runner is not None:
                validation_output = validation_runner(
                    research_output,
                    market_data,
                )

                if isinstance(
                    validation_output,
                    Mapping,
                ):
                    passed = validation_output.get(
                        "passed"
                    )

                    if passed is True:
                        validation_status = (
                            ValidationStatus.PASSED
                        )
                    elif passed is False:
                        validation_status = (
                            ValidationStatus.FAILED
                        )

            add_event(
                ResearchLoopStage.VALIDATION,
                "validation_completed",
                validation_status.value,
            )

            # =============================================
            # 7. ROBUSTNESS / OOS
            # =============================================

            robustness_output = None

            if robustness_runner is not None:
                robustness_output = robustness_runner(
                    research_output,
                    market_data,
                )

                if isinstance(
                    robustness_output,
                    Mapping,
                ):
                    passed = robustness_output.get(
                        "passed"
                    )

                    if passed is True:
                        robustness_status = (
                            ValidationStatus.PASSED
                        )
                    elif passed is False:
                        robustness_status = (
                            ValidationStatus.FAILED
                        )

            add_event(
                ResearchLoopStage.ROBUSTNESS,
                "robustness_test_completed",
                robustness_status.value,
            )

            # =============================================
            # 8. SHADOW PORTFOLIO
            # =============================================

            shadow_output = None

            if shadow_portfolio_runner is not None:
                shadow_output = shadow_portfolio_runner(
                    research_output,
                    market_data,
                )

                if isinstance(
                    shadow_output,
                    Mapping,
                ):
                    passed = shadow_output.get(
                        "passed"
                    )

                    if passed is True:
                        shadow_portfolio_status = (
                            ValidationStatus.PASSED
                        )
                    elif passed is False:
                        shadow_portfolio_status = (
                            ValidationStatus.FAILED
                        )

            add_event(
                ResearchLoopStage.SHADOW_PORTFOLIO,
                "shadow_portfolio_evaluated",
                shadow_portfolio_status.value,
            )

            # =============================================
            # 9. ATTRIBUTION
            # =============================================

            if attribution_runner is not None:
                raw_attribution = attribution_runner(
                    research_output,
                    validation_output,
                    robustness_output,
                    shadow_output,
                )

                attribution_output = (
                    self._validate_mapping(
                        raw_attribution,
                        "outcome_attribution",
                    )
                )

            add_event(
                ResearchLoopStage.ATTRIBUTION,
                "outcome_attribution_completed",
                "COMPLETED",
                {
                    "field_count": len(
                        attribution_output
                    ),
                },
            )

            # =============================================
            # 10. META LEARNING
            # =============================================

            if meta_learning_runner is not None:
                raw_meta = meta_learning_runner(
                    research_output,
                    validation_output,
                    robustness_output,
                    shadow_output,
                    attribution_output,
                )

                meta_learning_output = (
                    self._validate_mapping(
                        raw_meta,
                        "meta_learning_output",
                    )
                )

            add_event(
                ResearchLoopStage.META_LEARNING,
                "meta_learning_completed",
                "COMPLETED",
                {
                    "field_count": len(
                        meta_learning_output
                    ),
                },
            )

            # =============================================
            # 11. NEW RESEARCH TASK
            # =============================================

            if next_task_generator is not None:
                generated_task = next_task_generator(
                    meta_learning_output,
                    attribution_output,
                )

                if generated_task is not None:
                    next_research_task = (
                        self._validate_string(
                            generated_task,
                            "next_research_task",
                        )
                    )

            add_event(
                ResearchLoopStage.NEW_RESEARCH_TASK,
                "next_research_task_generated",
                "COMPLETED",
                {
                    "task_generated": (
                        next_research_task is not None
                    ),
                },
            )

            # =============================================
            # FINAL RESULT
            # =============================================

            reproducibility_metadata = {
                **metadata,
                "cycle_id": cycle_id,
                "market_data_size": len(market_data),
                "human_approval_required": (
                    self.require_human_approval
                ),
                "human_approved": human_approval,
                "execution_started": True,
                "audit_event_count": len(audit),
                "validation_status": (
                    validation_status.value
                ),
                "robustness_status": (
                    robustness_status.value
                ),
                "shadow_portfolio_status": (
                    shadow_portfolio_status.value
                ),
                "continuous_loop": True,
                "measurable": True,
                "auditable": True,
                "reproducible": True,
                "human_governed": True,
            }

            return ResearchLoopResult(
                cycle_id=cycle_id,
                status=ResearchLoopStatus.COMPLETED,
                current_stage=(
                    ResearchLoopStage.NEW_RESEARCH_TASK
                ),
                hypothesis=normalized_hypothesis,
                research_question=normalized_question,
                experiment_id=normalized_experiment_id,
                validation_status=validation_status,
                robustness_status=robustness_status,
                shadow_portfolio_status=(
                    shadow_portfolio_status
                ),
                outcome_attribution=attribution_output,
                meta_learning_output=(
                    meta_learning_output
                ),
                next_research_task=next_research_task,
                audit_trail=tuple(audit),
                reproducibility_metadata=(
                    reproducibility_metadata
                ),
                human_approval_required=(
                    self.require_human_approval
                ),
                human_approved=human_approval,
                error=None,
            )

        except Exception as exc:

            current_stage = (
                audit[-1].stage
                if audit
                else ResearchLoopStage.MARKET_DATA
            )

            add_event(
                current_stage,
                "research_loop_failed",
                "FAILED",
                {
                    "error": str(exc),
                },
            )

            return ResearchLoopResult(
                cycle_id=cycle_id,
                status=ResearchLoopStatus.FAILED,
                current_stage=current_stage,
                hypothesis=normalized_hypothesis,
                research_question=normalized_question,
                experiment_id=normalized_experiment_id,
                validation_status=validation_status,
                robustness_status=robustness_status,
                shadow_portfolio_status=(
                    shadow_portfolio_status
                ),
                outcome_attribution=attribution_output,
                meta_learning_output=(
                    meta_learning_output
                ),
                next_research_task=next_research_task,
                audit_trail=tuple(audit),
                reproducibility_metadata={
                    **metadata,
                    "cycle_id": cycle_id,
                    "market_data_size": len(market_data),
                    "human_approval_required": (
                        self.require_human_approval
                    ),
                    "human_approved": human_approval,
                    "execution_started": True,
                    "continuous_loop": True,
                    "measurable": True,
                    "auditable": True,
                    "reproducible": True,
                    "human_governed": True,
                    "audit_event_count": len(audit),
                },
                human_approval_required=(
                    self.require_human_approval
                ),
                human_approved=human_approval,
                error=str(exc),
            )


__all__ = [
    "ResearchLoopStage",
    "ResearchLoopStatus",
    "ValidationStatus",
    "ResearchLoopEvent",
    "ResearchLoopResult",
    "ContinuousResearchLoop",
]