import pytest

from autonomous_experiment_designer.system_validation_engine import (
    SystemValidationEngine,
    ValidationCriterion,
    ValidationStatus,
)


def _all_success_criteria() -> dict:
    return {
        criterion: True
        for criterion
        in SystemValidationEngine.REQUIRED_CRITERIA
    }


def _full_evidence() -> dict:
    return {
        criterion: (
            f"validated:{criterion.value}",
        )
        for criterion
        in SystemValidationEngine.REQUIRED_CRITERIA
    }


def test_v6_0_9_final_acceptance_passes():
    engine = SystemValidationEngine()

    result = engine.validate(
        version="v6.0.9",
        criteria=_all_success_criteria(),
        evidence=_full_evidence(),
        human_approval=True,
        metadata={
            "release": "v6.0.9",
            "acceptance_stage": "final",
        },
    )

    assert result.status == ValidationStatus.PASSED
    assert result.version == "v6.0.9"
    assert result.total_count == 10
    assert result.passed_count == 10
    assert result.failed_count == 0
    assert result.blocked_count == 0
    assert result.all_criteria_passed is True


def test_provenance_requirement_is_accepted():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    evidence = _full_evidence()
    evidence[
        ValidationCriterion.REPRODUCIBLE_PROVENANCE
    ] = (
        "raw-data-hash",
        "experiment-id",
        "source-timestamp",
    )

    result = engine.validate(
        criteria=_all_success_criteria(),
        evidence=evidence,
        human_approval=False,
    )

    item = next(
        item
        for item in result.criteria_results
        if item.criterion
        == ValidationCriterion.REPRODUCIBLE_PROVENANCE
    )

    assert item.status == ValidationStatus.PASSED
    assert item.evidence == (
        "raw-data-hash",
        "experiment-id",
        "source-timestamp",
    )


def test_predictive_causal_separation_is_required():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = _all_success_criteria()
    criteria[
        ValidationCriterion.PREDICTIVE_CAUSAL_SEPARATION
    ] = False

    result = engine.validate(
        criteria=criteria,
        human_approval=False,
    )

    assert result.status == ValidationStatus.FAILED
    assert result.passed_count == 9
    assert result.failed_count == 1


def test_oos_separation_is_required():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = _all_success_criteria()
    criteria[
        ValidationCriterion.OOS_IS_SEPARATED
    ] = False

    result = engine.validate(
        criteria=criteria,
        human_approval=False,
    )

    assert result.status == ValidationStatus.FAILED


def test_change_measurement_is_required():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = _all_success_criteria()
    criteria[
        ValidationCriterion.CHANGES_MEASURABLE
    ] = False

    result = engine.validate(
        criteria=criteria,
        human_approval=False,
    )

    assert result.status == ValidationStatus.FAILED


def test_change_reversibility_is_required():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = _all_success_criteria()
    criteria[
        ValidationCriterion.CHANGES_REVERSIBLE
    ] = False

    result = engine.validate(
        criteria=criteria,
        human_approval=False,
    )

    assert result.status == ValidationStatus.FAILED


def test_portfolio_reconstruction_is_required():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = _all_success_criteria()
    criteria[
        ValidationCriterion.PORTFOLIO_RECONSTRUCTABLE
    ] = False

    result = engine.validate(
        criteria=criteria,
        human_approval=False,
    )

    assert result.status == ValidationStatus.FAILED


def test_quantum_classical_equivalence_requirement():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = _all_success_criteria()
    criteria[
        ValidationCriterion.QUANTUM_CLASSICAL_BENCHMARK
    ] = False

    result = engine.validate(
        criteria=criteria,
        human_approval=False,
    )

    assert result.status == ValidationStatus.FAILED


def test_autonomous_research_safety_requirement():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = _all_success_criteria()
    criteria[
        ValidationCriterion.AUTONOMOUS_RESEARCH_CONTROLLED
    ] = False

    result = engine.validate(
        criteria=criteria,
        human_approval=False,
    )

    assert result.status == ValidationStatus.FAILED


def test_end_to_end_acceptance_is_required():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = _all_success_criteria()
    criteria[
        ValidationCriterion.END_TO_END_ACCEPTANCE
    ] = False

    result = engine.validate(
        criteria=criteria,
        human_approval=False,
    )

    assert result.status == ValidationStatus.FAILED


def test_full_regression_is_required():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = _all_success_criteria()
    criteria[
        ValidationCriterion.FULL_REGRESSION
    ] = False

    result = engine.validate(
        criteria=criteria,
        human_approval=False,
    )

    assert result.status == ValidationStatus.FAILED


def test_human_approval_blocks_final_acceptance():
    engine = SystemValidationEngine(
        require_human_approval=True
    )

    result = engine.validate(
        version="v6.0.9",
        criteria=_all_success_criteria(),
        evidence=_full_evidence(),
        human_approval=False,
    )

    assert result.status == ValidationStatus.BLOCKED
    assert result.criteria_results == ()
    assert result.passed_count == 0
    assert result.failed_count == 0
    assert result.blocked_count == 1
    assert result.metadata["execution_started"] is False


def test_approved_final_acceptance_is_executable():
    engine = SystemValidationEngine(
        require_human_approval=True
    )

    result = engine.validate(
        version="v6.0.9",
        criteria=_all_success_criteria(),
        evidence=_full_evidence(),
        human_approval=True,
    )

    assert result.status == ValidationStatus.PASSED
    assert result.metadata["human_approved"] is True
    assert result.metadata["execution_started"] is True


def test_final_acceptance_contains_all_criteria():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate(
        criteria=_all_success_criteria(),
        evidence=_full_evidence(),
        human_approval=False,
    )

    criteria = {
        item.criterion
        for item in result.criteria_results
    }

    assert criteria == set(
        SystemValidationEngine.REQUIRED_CRITERIA
    )


def test_final_acceptance_preserves_evidence():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    evidence = _full_evidence()

    evidence[
        ValidationCriterion.FULL_REGRESSION
    ] = (
        "4996 passed",
        "0 failed",
        "77.27 seconds",
    )

    result = engine.validate(
        criteria=_all_success_criteria(),
        evidence=evidence,
        human_approval=False,
    )

    regression = next(
        item
        for item in result.criteria_results
        if item.criterion
        == ValidationCriterion.FULL_REGRESSION
    )

    assert regression.evidence == (
        "4996 passed",
        "0 failed",
        "77.27 seconds",
    )


def test_final_acceptance_metadata():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate(
        version="v6.0.9",
        criteria=_all_success_criteria(),
        evidence=_full_evidence(),
        human_approval=False,
        metadata={
            "project": "MultiAgentFinancialResearch",
            "release": "v6.0.9",
        },
    )

    assert (
        result.metadata["project"]
        == "MultiAgentFinancialResearch"
    )
    assert result.metadata["release"] == "v6.0.9"
    assert (
        result.metadata["validation_version"]
        == "v6.0.9"
    )
    assert result.metadata["required_criteria"] == 10
    assert result.metadata["passed_count"] == 10
    assert result.metadata["failed_count"] == 0


def test_multiple_acceptance_failures_are_detected():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = _all_success_criteria()

    criteria[
        ValidationCriterion.FULL_REGRESSION
    ] = False

    criteria[
        ValidationCriterion.END_TO_END_ACCEPTANCE
    ] = False

    criteria[
        ValidationCriterion.CHANGES_REVERSIBLE
    ] = False

    criteria[
        ValidationCriterion.QUANTUM_CLASSICAL_BENCHMARK
    ] = False

    result = engine.validate(
        criteria=criteria,
        evidence=_full_evidence(),
        human_approval=False,
    )

    assert result.status == ValidationStatus.FAILED
    assert result.passed_count == 6
    assert result.failed_count == 4
    assert result.blocked_count == 0


def test_final_acceptance_is_repeatable():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    first = engine.validate(
        version="v6.0.9",
        criteria=_all_success_criteria(),
        evidence=_full_evidence(),
        human_approval=False,
    )

    second = engine.validate(
        version="v6.0.9",
        criteria=_all_success_criteria(),
        evidence=_full_evidence(),
        human_approval=False,
    )

    assert first == second


def test_final_acceptance_result_is_immutable():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate(
        criteria=_all_success_criteria(),
        human_approval=False,
    )

    with pytest.raises(AttributeError):
        result.status = ValidationStatus.FAILED