import pytest

from autonomous_experiment_designer.system_validation_engine import (
    SystemValidationEngine,
    ValidationCriterion,
    ValidationStatus,
)


def _criteria(**overrides):
    criteria = {
        criterion: True
        for criterion
        in SystemValidationEngine.REQUIRED_CRITERIA
    }
    criteria.update(overrides)
    return criteria


def _evidence():
    return {
        criterion: (
            f"validated:{criterion.value}",
        )
        for criterion
        in SystemValidationEngine.REQUIRED_CRITERIA
    }


def test_v6_0_9_system_integration_full_acceptance():
    engine = SystemValidationEngine()

    result = engine.validate(
        version="v6.0.9",
        criteria=_criteria(),
        evidence=_evidence(),
        human_approval=True,
        metadata={
            "project": "MultiAgentFinancialResearch",
            "milestone": "v6.0.9",
            "stage": "system-integration",
        },
    )

    assert result.status == ValidationStatus.PASSED
    assert result.all_criteria_passed is True
    assert result.total_count == 10
    assert result.passed_count == 10
    assert result.failed_count == 0
    assert result.blocked_count == 0


def test_provenance_is_part_of_system_acceptance():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate(
        criteria=_criteria(),
        evidence={
            **_evidence(),
            ValidationCriterion.REPRODUCIBLE_PROVENANCE: (
                "raw-data-hash",
                "experiment-id",
                "source-timestamp",
                "model-version",
            ),
        },
        human_approval=False,
    )

    item = next(
        item
        for item in result.criteria_results
        if item.criterion
        == ValidationCriterion.REPRODUCIBLE_PROVENANCE
    )

    assert item.status == ValidationStatus.PASSED
    assert len(item.evidence) == 4


def test_predictive_and_causal_evidence_must_be_separated():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = _criteria(
        **{
            ValidationCriterion.PREDICTIVE_CAUSAL_SEPARATION:
                False
        }
    )

    result = engine.validate(
        criteria=criteria,
        evidence=_evidence(),
        human_approval=False,
    )

    assert result.status == ValidationStatus.FAILED
    assert result.failed_count == 1


def test_in_sample_and_oos_evidence_must_be_separated():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = _criteria(
        **{
            ValidationCriterion.OOS_IS_SEPARATED: False
        }
    )

    result = engine.validate(
        criteria=criteria,
        evidence=_evidence(),
        human_approval=False,
    )

    assert result.status == ValidationStatus.FAILED


def test_agent_model_prompt_policy_changes_are_measurable():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = _criteria(
        **{
            ValidationCriterion.CHANGES_MEASURABLE:
                True
        }
    )

    result = engine.validate(
        criteria=criteria,
        evidence={
            **_evidence(),
            ValidationCriterion.CHANGES_MEASURABLE: (
                "agent-version",
                "model-version",
                "prompt-version",
                "policy-version",
                "evaluation-metrics",
            ),
        },
        human_approval=False,
    )

    item = next(
        item
        for item in result.criteria_results
        if item.criterion
        == ValidationCriterion.CHANGES_MEASURABLE
    )

    assert item.status == ValidationStatus.PASSED
    assert len(item.evidence) == 5


def test_changes_must_be_reversible():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = _criteria(
        **{
            ValidationCriterion.CHANGES_REVERSIBLE:
                False
        }
    )

    result = engine.validate(
        criteria=criteria,
        human_approval=False,
    )

    assert result.status == ValidationStatus.FAILED


def test_portfolio_decision_reconstruction_is_required():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate(
        criteria=_criteria(),
        evidence={
            **_evidence(),
            ValidationCriterion.PORTFOLIO_RECONSTRUCTABLE: (
                "raw-market-data",
                "research-output",
                "decision-record",
                "execution-record",
                "final-outcome",
            ),
        },
        human_approval=False,
    )

    item = next(
        item
        for item in result.criteria_results
        if item.criterion
        == ValidationCriterion.PORTFOLIO_RECONSTRUCTABLE
    )

    assert item.status == ValidationStatus.PASSED
    assert len(item.evidence) == 5


def test_quantum_experiment_requires_classical_baseline():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = _criteria(
        **{
            ValidationCriterion.QUANTUM_CLASSICAL_BENCHMARK:
                True
        }
    )

    result = engine.validate(
        criteria=criteria,
        evidence={
            **_evidence(),
            ValidationCriterion.QUANTUM_CLASSICAL_BENCHMARK: (
                "quantum-result",
                "classical-result",
                "same-problem-definition",
                "same-dataset",
                "benchmark-metrics",
            ),
        },
        human_approval=False,
    )

    item = next(
        item
        for item in result.criteria_results
        if item.criterion
        == ValidationCriterion.QUANTUM_CLASSICAL_BENCHMARK
    )

    assert item.status == ValidationStatus.PASSED


def test_uncontrolled_autonomous_execution_fails_acceptance():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = _criteria(
        **{
            ValidationCriterion.AUTONOMOUS_RESEARCH_CONTROLLED:
                False
        }
    )

    result = engine.validate(
        criteria=criteria,
        human_approval=False,
    )

    assert result.status == ValidationStatus.FAILED


def test_end_to_end_acceptance_is_mandatory():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = _criteria(
        **{
            ValidationCriterion.END_TO_END_ACCEPTANCE:
                False
        }
    )

    result = engine.validate(
        criteria=criteria,
        human_approval=False,
    )

    assert result.status == ValidationStatus.FAILED


def test_complete_regression_is_mandatory():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = _criteria(
        **{
            ValidationCriterion.FULL_REGRESSION:
                False
        }
    )

    result = engine.validate(
        criteria=criteria,
        human_approval=False,
    )

    assert result.status == ValidationStatus.FAILED


def test_human_gate_blocks_unapproved_system_acceptance():
    engine = SystemValidationEngine(
        require_human_approval=True
    )

    result = engine.validate(
        version="v6.0.9",
        criteria=_criteria(),
        evidence=_evidence(),
        human_approval=False,
    )

    assert result.status == ValidationStatus.BLOCKED
    assert result.criteria_results == ()
    assert result.metadata["execution_started"] is False


def test_human_approval_allows_system_acceptance():
    engine = SystemValidationEngine(
        require_human_approval=True
    )

    result = engine.validate(
        version="v6.0.9",
        criteria=_criteria(),
        evidence=_evidence(),
        human_approval=True,
    )

    assert result.status == ValidationStatus.PASSED
    assert result.metadata["human_approved"] is True
    assert result.metadata["execution_started"] is True


def test_multiple_system_failures_are_detected():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = _criteria(
        **{
            ValidationCriterion.FULL_REGRESSION:
                False,
            ValidationCriterion.END_TO_END_ACCEPTANCE:
                False,
            ValidationCriterion.CHANGES_REVERSIBLE:
                False,
            ValidationCriterion.QUANTUM_CLASSICAL_BENCHMARK:
                False,
        }
    )

    result = engine.validate(
        criteria=criteria,
        evidence=_evidence(),
        human_approval=False,
    )

    assert result.status == ValidationStatus.FAILED
    assert result.passed_count == 6
    assert result.failed_count == 4


def test_system_acceptance_preserves_metadata():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate(
        version="v6.0.9",
        criteria=_criteria(),
        evidence=_evidence(),
        human_approval=False,
        metadata={
            "release": "v6.0.9",
            "test_count": 4996,
            "failures": 0,
        },
    )

    assert result.metadata["release"] == "v6.0.9"
    assert result.metadata["test_count"] == 4996
    assert result.metadata["failures"] == 0


def test_system_acceptance_is_repeatable():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    first = engine.validate(
        version="v6.0.9",
        criteria=_criteria(),
        evidence=_evidence(),
        human_approval=False,
    )

    second = engine.validate(
        version="v6.0.9",
        criteria=_criteria(),
        evidence=_evidence(),
        human_approval=False,
    )

    assert first == second


def test_system_acceptance_result_is_immutable():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate(
        criteria=_criteria(),
        human_approval=False,
    )

    with pytest.raises(AttributeError):
        result.status = ValidationStatus.FAILED


def test_all_v6_0_9_criteria_are_evaluated():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate(
        criteria=_criteria(),
        evidence=_evidence(),
        human_approval=False,
    )

    returned = {
        item.criterion
        for item in result.criteria_results
    }

    assert returned == set(
        SystemValidationEngine.REQUIRED_CRITERIA
    )

    assert (
        result.metadata[
            "all_criteria_evaluated"
        ]
        is True
    )


def test_final_system_acceptance_has_no_failures():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate_full_system(
        provenance=True,
        predictive_causal_separation=True,
        oos_separation=True,
        changes_measurable=True,
        changes_reversible=True,
        portfolio_reconstructable=True,
        quantum_classical_benchmark=True,
        autonomous_research_controlled=True,
        end_to_end_acceptance=True,
        full_regression=True,
        human_approval=False,
        metadata={
            "project": "MultiAgentFinancialResearch",
            "release": "v6.0.9",
        },
    )

    assert result.status == ValidationStatus.PASSED
    assert result.passed_count == 10
    assert result.failed_count == 0
    assert result.blocked_count == 0
    assert result.all_criteria_passed is True
