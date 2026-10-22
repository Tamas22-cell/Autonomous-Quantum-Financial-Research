import pytest

from autonomous_experiment_designer.continuous_research_loop import (
    ContinuousResearchLoop,
    ResearchLoopEvent,
    ResearchLoopResult,
    ResearchLoopStage,
    ResearchLoopStatus,
    ValidationStatus,
)


def test_default_configuration():
    loop = ContinuousResearchLoop()
    assert loop.require_human_approval is True


def test_disabled_human_approval():
    loop = ContinuousResearchLoop(
        require_human_approval=False
    )
    assert loop.require_human_approval is False


def test_invalid_configuration():
    with pytest.raises(ValueError):
        ContinuousResearchLoop(
            require_human_approval="invalid"
        )


def test_invalid_cycle_id():
    with pytest.raises(ValueError):
        ContinuousResearchLoop(
            require_human_approval=False
        ).run(
            cycle_id=None,
            market_data=(1, 2),
            human_approval=True,
        )


def test_invalid_market_data_none():
    with pytest.raises(ValueError):
        ContinuousResearchLoop(
            require_human_approval=False
        ).run(
            cycle_id="CYCLE-001",
            market_data=None,
            human_approval=True,
        )


def test_invalid_market_data_mapping():
    with pytest.raises(ValueError):
        ContinuousResearchLoop(
            require_human_approval=False
        ).run(
            cycle_id="CYCLE-002",
            market_data={},
            human_approval=True,
        )


def test_invalid_market_data_string():
    with pytest.raises(ValueError):
        ContinuousResearchLoop(
            require_human_approval=False
        ).run(
            cycle_id="CYCLE-003",
            market_data="BTC",
            human_approval=True,
        )


def test_valid_market_data():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-004",
        market_data=(1, 2, 3),
        human_approval=True,
    )

    assert result.status == ResearchLoopStatus.COMPLETED


def test_human_gate_blocks():
    result = ContinuousResearchLoop(
        require_human_approval=True
    ).run(
        cycle_id="CYCLE-005",
        market_data=(1, 2),
        human_approval=False,
    )

    assert result.status == ResearchLoopStatus.BLOCKED
    assert result.human_approval_required is True
    assert result.human_approved is False


def test_human_gate_allows():
    result = ContinuousResearchLoop(
        require_human_approval=True
    ).run(
        cycle_id="CYCLE-006",
        market_data=(1, 2),
        human_approval=True,
    )

    assert result.status == ResearchLoopStatus.COMPLETED


def test_cycle_id_preserved():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="MY-CYCLE",
        market_data=(1, 2),
        human_approval=True,
    )

    assert result.cycle_id == "MY-CYCLE"


def test_hypothesis_preserved():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-007",
        market_data=(1, 2),
        hypothesis="Regime changed.",
        human_approval=True,
    )

    assert result.hypothesis == "Regime changed."


def test_research_question_preserved():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-008",
        market_data=(1, 2),
        research_question="Is strategy robust?",
        human_approval=True,
    )

    assert (
        result.research_question
        == "Is strategy robust?"
    )


def test_experiment_id_preserved():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-009",
        market_data=(1, 2),
        experiment_id="EXP-001",
        human_approval=True,
    )

    assert result.experiment_id == "EXP-001"


def test_research_runner():
    calls = []

    def research(
        market_data,
        hypothesis,
        question,
        experiment_id,
    ):
        calls.append(True)
        return {"strategy": "QAOA"}

    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-010",
        market_data=(1, 2),
        research_runner=research,
        human_approval=True,
    )

    assert result.status == ResearchLoopStatus.COMPLETED
    assert calls == [True]


def test_validation_passed():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-011",
        market_data=(1, 2),
        validation_runner=lambda *args: {
            "passed": True
        },
        human_approval=True,
    )

    assert (
        result.validation_status
        == ValidationStatus.PASSED
    )


def test_validation_failed():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-012",
        market_data=(1, 2),
        validation_runner=lambda *args: {
            "passed": False
        },
        human_approval=True,
    )

    assert (
        result.validation_status
        == ValidationStatus.FAILED
    )


def test_robustness_passed():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-013",
        market_data=(1, 2),
        robustness_runner=lambda *args: {
            "passed": True
        },
        human_approval=True,
    )

    assert (
        result.robustness_status
        == ValidationStatus.PASSED
    )


def test_shadow_portfolio_passed():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-014",
        market_data=(1, 2),
        shadow_portfolio_runner=lambda *args: {
            "passed": True
        },
        human_approval=True,
    )

    assert (
        result.shadow_portfolio_status
        == ValidationStatus.PASSED
    )


def test_attribution():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-015",
        market_data=(1, 2),
        attribution_runner=lambda *args: {
            "alpha": 0.05
        },
        human_approval=True,
    )

    assert result.outcome_attribution == {
        "alpha": 0.05
    }


def test_meta_learning():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-016",
        market_data=(1, 2),
        meta_learning_runner=lambda *args: {
            "gain": 0.1
        },
        human_approval=True,
    )

    assert result.meta_learning_output == {
        "gain": 0.1
    }


def test_next_research_task():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-017",
        market_data=(1, 2),
        next_task_generator=lambda *args: (
            "Next research task"
        ),
        human_approval=True,
    )

    assert (
        result.next_research_task
        == "Next research task"
    )


def test_audit_trail():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-018",
        market_data=(1, 2),
        human_approval=True,
    )

    assert isinstance(
        result.audit_trail,
        tuple,
    )

    assert len(result.audit_trail) == 11


def test_audit_sequence():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-019",
        market_data=(1, 2),
        human_approval=True,
    )

    assert [
        event.sequence
        for event in result.audit_trail
    ] == list(range(1, 12))


def test_audit_stages():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-020",
        market_data=(1, 2),
        human_approval=True,
    )

    stages = [
        event.stage
        for event in result.audit_trail
    ]

    assert stages == [
        ResearchLoopStage.MARKET_DATA,
        ResearchLoopStage.UNCERTAINTY_DETECTION,
        ResearchLoopStage.HYPOTHESIS,
        ResearchLoopStage.EXPERIMENT_DESIGN,
        ResearchLoopStage.AGENT_RESEARCH,
        ResearchLoopStage.VALIDATION,
        ResearchLoopStage.ROBUSTNESS,
        ResearchLoopStage.SHADOW_PORTFOLIO,
        ResearchLoopStage.ATTRIBUTION,
        ResearchLoopStage.META_LEARNING,
        ResearchLoopStage.NEW_RESEARCH_TASK,
    ]


def test_audit_event_type():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-021",
        market_data=(1, 2),
        human_approval=True,
    )

    assert all(
        isinstance(
            event,
            ResearchLoopEvent,
        )
        for event in result.audit_trail
    )


def test_result_type():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-022",
        market_data=(1, 2),
        human_approval=True,
    )

    assert isinstance(
        result,
        ResearchLoopResult,
    )


def test_reproducibility_metadata():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-023",
        market_data=(1, 2, 3),
        metadata={
            "version": "v6.0.8"
        },
        human_approval=True,
    )

    metadata = result.reproducibility_metadata

    assert metadata["cycle_id"] == "CYCLE-023"
    assert metadata["market_data_size"] == 3
    assert metadata["version"] == "v6.0.8"
    assert metadata["continuous_loop"] is True
    assert metadata["measurable"] is True
    assert metadata["auditable"] is True
    assert metadata["reproducible"] is True
    assert metadata["human_governed"] is True


def test_final_stage():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-024",
        market_data=(1, 2),
        human_approval=True,
    )

    assert (
        result.current_stage
        == ResearchLoopStage.NEW_RESEARCH_TASK
    )


def test_success_has_no_error():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-025",
        market_data=(1, 2),
        human_approval=True,
    )

    assert result.error is None


def test_research_exception_is_captured():
    def failing(*args):
        raise RuntimeError("research failure")

    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-026",
        market_data=(1, 2),
        research_runner=failing,
        human_approval=True,
    )

    assert result.status == ResearchLoopStatus.FAILED
    assert "research failure" in result.error


def test_validation_exception_is_captured():
    def failing(*args):
        raise RuntimeError("validation failure")

    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-027",
        market_data=(1, 2),
        validation_runner=failing,
        human_approval=True,
    )

    assert result.status == ResearchLoopStatus.FAILED
    assert "validation failure" in result.error


def test_uncertainty_signal():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-028",
        market_data=(1, 2),
        uncertainty_signal={
            "regime_distance": 0.8,
            "performance_gap": 0.2,
        },
        human_approval=True,
    )

    event = next(
        event
        for event in result.audit_trail
        if event.stage
        == ResearchLoopStage.UNCERTAINTY_DETECTION
    )

    assert event.metadata["signal_count"] == 2


def test_full_pipeline():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="V6.0.8-FULL",
        market_data=(
            "BTC",
            "ETH",
            "NVDA",
        ),
        hypothesis="Regime changed.",
        research_question="Is the strategy robust?",
        experiment_id="EXP-001",
        research_runner=lambda *args: {
            "strategy": "QAOA"
        },
        validation_runner=lambda *args: {
            "passed": True
        },
        robustness_runner=lambda *args: {
            "passed": True
        },
        shadow_portfolio_runner=lambda *args: {
            "passed": True
        },
        attribution_runner=lambda *args: {
            "alpha": 0.04
        },
        meta_learning_runner=lambda *args: {
            "gain": 0.12
        },
        next_task_generator=lambda *args: (
            "Investigate next regime."
        ),
        human_approval=True,
    )

    assert result.status == ResearchLoopStatus.COMPLETED
    assert (
        result.validation_status
        == ValidationStatus.PASSED
    )
    assert (
        result.robustness_status
        == ValidationStatus.PASSED
    )
    assert (
        result.shadow_portfolio_status
        == ValidationStatus.PASSED
    )
    assert result.outcome_attribution == {
        "alpha": 0.04
    }
    assert result.meta_learning_output == {
        "gain": 0.12
    }
    assert result.next_research_task == (
        "Investigate next regime."
    )


def test_enum_values():
    assert ValidationStatus.NOT_RUN.value == "NOT_RUN"
    assert ValidationStatus.PASSED.value == "PASSED"
    assert ValidationStatus.FAILED.value == "FAILED"

    assert ResearchLoopStatus.COMPLETED.value == (
        "COMPLETED"
    )
    assert ResearchLoopStatus.FAILED.value == "FAILED"
    assert ResearchLoopStatus.BLOCKED.value == "BLOCKED"


def test_empty_market_data():
    result = ContinuousResearchLoop(
        require_human_approval=False
    ).run(
        cycle_id="CYCLE-EMPTY",
        market_data=(),
        human_approval=True,
    )

    assert result.status == ResearchLoopStatus.COMPLETED