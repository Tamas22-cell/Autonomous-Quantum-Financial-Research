import pytest

from autonomous_experiment_designer.system_validation_engine import (
    CriterionResult,
    SystemValidationEngine,
    SystemValidationResult,
    ValidationCriterion,
    ValidationStatus,
)


def all_true_criteria():
    return {
        criterion: True
        for criterion in SystemValidationEngine.REQUIRED_CRITERIA
    }


def all_evidence():
    return {
        criterion: (
            f"Validated evidence for {criterion.value}.",
        )
        for criterion in SystemValidationEngine.REQUIRED_CRITERIA
    }


def test_default_configuration():
    engine = SystemValidationEngine()

    assert engine.require_human_approval is True
    assert engine.require_all_criteria is True
    assert engine.VERSION == "v6.0.9"


def test_custom_configuration():
    engine = SystemValidationEngine(
        require_human_approval=False,
        require_all_criteria=False,
    )

    assert engine.require_human_approval is False
    assert engine.require_all_criteria is False


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        "true",
        [],
    ],
)
def test_invalid_human_approval_configuration(value):
    with pytest.raises(ValueError):
        SystemValidationEngine(
            require_human_approval=value
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        "true",
        [],
    ],
)
def test_invalid_require_all_criteria_configuration(value):
    with pytest.raises(ValueError):
        SystemValidationEngine(
            require_all_criteria=value
        )


def test_required_criteria_count():
    assert len(
        SystemValidationEngine.REQUIRED_CRITERIA
    ) == 10


def test_required_criteria_are_unique():
    criteria = SystemValidationEngine.REQUIRED_CRITERIA

    assert len(criteria) == len(set(criteria))


def test_all_required_criteria_are_enum_values():
    for criterion in (
        SystemValidationEngine.REQUIRED_CRITERIA
    ):
        assert isinstance(
            criterion,
            ValidationCriterion,
        )


def test_validate_criterion_passed():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate_criterion(
        ValidationCriterion.REPRODUCIBLE_PROVENANCE,
        passed=True,
        evidence=(
            "Event ID recorded.",
            "Source data hash recorded.",
        ),
        details="Provenance is reproducible.",
    )

    assert isinstance(result, CriterionResult)
    assert (
        result.criterion
        == ValidationCriterion.REPRODUCIBLE_PROVENANCE
    )
    assert (
        result.status
        == ValidationStatus.PASSED
    )
    assert len(result.evidence) == 2
    assert (
        result.details
        == "Provenance is reproducible."
    )


def test_validate_criterion_failed():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate_criterion(
        ValidationCriterion.FULL_REGRESSION,
        passed=False,
        evidence=("Regression incomplete.",),
        details="Acceptance failed.",
    )

    assert (
        result.status
        == ValidationStatus.FAILED
    )


@pytest.mark.parametrize(
    "criterion",
    [
        None,
        "REPRODUCIBLE_PROVENANCE",
        123,
    ],
)
def test_invalid_criterion_is_rejected(criterion):
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    with pytest.raises(ValueError):
        engine.validate_criterion(
            criterion,
            passed=True,
            evidence=("valid evidence",),
        )


@pytest.mark.parametrize(
    "passed",
    [
        None,
        1,
        "true",
        [],
    ],
)
def test_invalid_passed_value_is_rejected(passed):
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    with pytest.raises(ValueError):
        engine.validate_criterion(
            ValidationCriterion.FULL_REGRESSION,
            passed=passed,
            evidence=("valid evidence",),
        )


@pytest.mark.parametrize(
    "evidence",
    [
        None,
        123,
        "single string",
        b"bytes",
    ],
)
def test_invalid_criterion_evidence_is_rejected(
    evidence,
):
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    with pytest.raises(ValueError):
        engine.validate_criterion(
            ValidationCriterion.FULL_REGRESSION,
            passed=True,
            evidence=evidence,
        )


def test_duplicate_evidence_is_removed():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate_criterion(
        ValidationCriterion.FULL_REGRESSION,
        passed=True,
        evidence=(
            "same evidence",
            "same evidence",
            "another evidence",
        ),
    )

    assert result.evidence == (
        "same evidence",
        "another evidence",
    )


def test_evidence_is_trimmed():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate_criterion(
        ValidationCriterion.FULL_REGRESSION,
        passed=True,
        evidence=(
            "  evidence one  ",
            "\tevidence two\t",
        ),
    )

    assert result.evidence == (
        "evidence one",
        "evidence two",
    )


@pytest.mark.parametrize(
    "evidence",
    [
        ("",),
        (" ",),
        ("\t",),
        (None,),
        (123,),
    ],
)
def test_invalid_evidence_item_is_rejected(
    evidence,
):
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    with pytest.raises(ValueError):
        engine.validate_criterion(
            ValidationCriterion.FULL_REGRESSION,
            passed=True,
            evidence=evidence,
        )


def test_successful_full_validation():
    engine = SystemValidationEngine()

    result = engine.validate(
        version="v6.0.9",
        criteria=all_true_criteria(),
        evidence=all_evidence(),
        human_approval=True,
    )

    assert isinstance(
        result,
        SystemValidationResult,
    )

    assert (
        result.status
        == ValidationStatus.PASSED
    )

    assert result.version == "v6.0.9"
    assert result.passed_count == 10
    assert result.failed_count == 0
    assert result.blocked_count == 0
    assert result.total_count == 10
    assert result.all_criteria_passed is True


def test_successful_validation_contains_all_results():
    engine = SystemValidationEngine()

    result = engine.validate(
        criteria=all_true_criteria(),
        evidence=all_evidence(),
        human_approval=True,
    )

    assert len(result.criteria_results) == 10

    returned = {
        item.criterion
        for item in result.criteria_results
    }

    assert returned == set(
        SystemValidationEngine.REQUIRED_CRITERIA
    )


def test_human_gate_blocks_validation():
    engine = SystemValidationEngine(
        require_human_approval=True
    )

    result = engine.validate(
        criteria=all_true_criteria(),
        evidence=all_evidence(),
        human_approval=False,
    )

    assert (
        result.status
        == ValidationStatus.BLOCKED
    )

    assert result.passed_count == 0
    assert result.failed_count == 0
    assert result.blocked_count == 1
    assert result.criteria_results == ()

    assert (
        result.metadata[
            "execution_started"
        ]
        is False
    )

    assert result.error is not None


def test_validation_can_run_without_human_gate():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate(
        criteria=all_true_criteria(),
        evidence=all_evidence(),
        human_approval=False,
    )

    assert (
        result.status
        == ValidationStatus.PASSED
    )

    assert result.passed_count == 10


def test_one_failed_criterion_fails_system():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = all_true_criteria()

    criteria[
        ValidationCriterion.FULL_REGRESSION
    ] = False

    result = engine.validate(
        criteria=criteria,
        evidence=all_evidence(),
        human_approval=False,
    )

    assert (
        result.status
        == ValidationStatus.FAILED
    )

    assert result.passed_count == 9
    assert result.failed_count == 1
    assert result.blocked_count == 0
    assert result.all_criteria_passed is False


def test_multiple_failed_criteria_are_counted():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = all_true_criteria()

    criteria[
        ValidationCriterion.FULL_REGRESSION
    ] = False

    criteria[
        ValidationCriterion.END_TO_END_ACCEPTANCE
    ] = False

    criteria[
        ValidationCriterion.CHANGES_REVERSIBLE
    ] = False

    result = engine.validate(
        criteria=criteria,
        evidence=all_evidence(),
        human_approval=False,
    )

    assert result.passed_count == 7
    assert result.failed_count == 3
    assert result.blocked_count == 0
    assert (
        result.status
        == ValidationStatus.FAILED
    )


def test_missing_criterion_is_failed():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = all_true_criteria()

    del criteria[
        ValidationCriterion.FULL_REGRESSION
    ]

    result = engine.validate(
        criteria=criteria,
        evidence=all_evidence(),
        human_approval=False,
    )

    assert (
        result.status
        == ValidationStatus.FAILED
    )

    assert result.failed_count == 1

    failed = [
        item
        for item in result.criteria_results
        if item.criterion
        == ValidationCriterion.FULL_REGRESSION
    ]

    assert len(failed) == 1
    assert failed[0].evidence == ()


def test_custom_version_is_preserved():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate(
        version="v6.0.9-final",
        criteria=all_true_criteria(),
        human_approval=False,
    )

    assert result.version == "v6.0.9-final"
    assert (
        result.metadata[
            "validation_version"
        ]
        == "v6.0.9-final"
    )


@pytest.mark.parametrize(
    "version",
    [
        None,
        "",
        " ",
        123,
    ],
)
def test_invalid_version_is_rejected(version):
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    with pytest.raises(ValueError):
        engine.validate(
            version=version,
            criteria=all_true_criteria(),
            human_approval=False,
        )


@pytest.mark.parametrize(
    "criteria",
    [
        None,
        [],
        (),
        "invalid",
    ],
)
def test_invalid_criteria_mapping_is_rejected(
    criteria,
):
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    with pytest.raises(ValueError):
        engine.validate(
            criteria=criteria,
            human_approval=False,
        )


@pytest.mark.parametrize(
    "evidence",
    [
        [],
        (),
        "invalid",
        123,
    ],
)
def test_invalid_evidence_mapping_is_rejected(
    evidence,
):
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    with pytest.raises(ValueError):
        engine.validate(
            criteria=all_true_criteria(),
            evidence=evidence,
            human_approval=False,
        )


@pytest.mark.parametrize(
    "details",
    [
        [],
        (),
        "invalid",
        123,
    ],
)
def test_invalid_details_mapping_is_rejected(
    details,
):
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    with pytest.raises(ValueError):
        engine.validate(
            criteria=all_true_criteria(),
            details=details,
            human_approval=False,
        )


def test_metadata_is_preserved():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate(
        criteria=all_true_criteria(),
        human_approval=False,
        metadata={
            "project": "MultiAgentFinancialResearch",
            "milestone": "v6.0.9",
        },
    )

    assert (
        result.metadata["project"]
        == "MultiAgentFinancialResearch"
    )

    assert (
        result.metadata["milestone"]
        == "v6.0.9"
    )


def test_metadata_is_copied():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    metadata = {
        "source": "acceptance",
    }

    result = engine.validate(
        criteria=all_true_criteria(),
        human_approval=False,
        metadata=metadata,
    )

    metadata["source"] = "changed"

    assert (
        result.metadata["source"]
        == "acceptance"
    )


def test_success_metadata():
    engine = SystemValidationEngine(
        require_human_approval=True
    )

    result = engine.validate(
        criteria=all_true_criteria(),
        evidence=all_evidence(),
        human_approval=True,
    )

    assert (
        result.metadata[
            "human_approval_required"
        ]
        is True
    )

    assert (
        result.metadata[
            "human_approved"
        ]
        is True
    )

    assert (
        result.metadata[
            "execution_started"
        ]
        is True
    )

    assert (
        result.metadata[
            "required_criteria"
        ]
        == 10
    )

    assert (
        result.metadata[
            "passed_count"
        ]
        == 10
    )


def test_failed_criterion_preserves_evidence():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = all_true_criteria()

    criteria[
        ValidationCriterion.QUANTUM_CLASSICAL_BENCHMARK
    ] = False

    evidence = all_evidence()

    result = engine.validate(
        criteria=criteria,
        evidence=evidence,
        human_approval=False,
    )

    failed = next(
        item
        for item in result.criteria_results
        if item.criterion
        == ValidationCriterion.QUANTUM_CLASSICAL_BENCHMARK
    )

    assert (
        failed.status
        == ValidationStatus.FAILED
    )

    assert failed.evidence == (
        "Validated evidence for "
        "quantum_classical_benchmark.",
    )


def test_predictive_causal_separation_is_supported():
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
    )

    item = next(
        item
        for item in result.criteria_results
        if item.criterion
        == ValidationCriterion.PREDICTIVE_CAUSAL_SEPARATION
    )

    assert item.status == ValidationStatus.PASSED


def test_oos_separation_is_supported():
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
    )

    item = next(
        item
        for item in result.criteria_results
        if item.criterion
        == ValidationCriterion.OOS_IS_SEPARATED
    )

    assert item.status == ValidationStatus.PASSED


def test_reversibility_is_supported():
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
    )

    item = next(
        item
        for item in result.criteria_results
        if item.criterion
        == ValidationCriterion.CHANGES_REVERSIBLE
    )

    assert item.status == ValidationStatus.PASSED


def test_portfolio_reconstruction_is_supported():
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
    )

    item = next(
        item
        for item in result.criteria_results
        if item.criterion
        == ValidationCriterion.PORTFOLIO_RECONSTRUCTABLE
    )

    assert item.status == ValidationStatus.PASSED


def test_quantum_classical_benchmark_is_supported():
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
    )

    item = next(
        item
        for item in result.criteria_results
        if item.criterion
        == ValidationCriterion.QUANTUM_CLASSICAL_BENCHMARK
    )

    assert item.status == ValidationStatus.PASSED


def test_autonomous_research_control_is_supported():
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
    )

    item = next(
        item
        for item in result.criteria_results
        if item.criterion
        == ValidationCriterion.AUTONOMOUS_RESEARCH_CONTROLLED
    )

    assert item.status == ValidationStatus.PASSED


def test_end_to_end_acceptance_is_supported():
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
    )

    item = next(
        item
        for item in result.criteria_results
        if item.criterion
        == ValidationCriterion.END_TO_END_ACCEPTANCE
    )

    assert item.status == ValidationStatus.PASSED


def test_full_regression_is_supported():
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
    )

    item = next(
        item
        for item in result.criteria_results
        if item.criterion
        == ValidationCriterion.FULL_REGRESSION
    )

    assert item.status == ValidationStatus.PASSED


def test_full_system_failure_is_propagated():
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
        quantum_classical_benchmark=False,
        autonomous_research_controlled=True,
        end_to_end_acceptance=True,
        full_regression=True,
    )

    assert (
        result.status
        == ValidationStatus.FAILED
    )

    assert result.passed_count == 9
    assert result.failed_count == 1


def test_full_system_human_gate():
    engine = SystemValidationEngine(
        require_human_approval=True
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
    )

    assert (
        result.status
        == ValidationStatus.BLOCKED
    )

    assert result.criteria_results == ()


def test_full_system_approval():
    engine = SystemValidationEngine(
        require_human_approval=True
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
        human_approval=True,
    )

    assert (
        result.status
        == ValidationStatus.PASSED
    )

    assert result.all_criteria_passed is True


def test_full_system_evidence_is_preserved():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    evidence = {
        ValidationCriterion.REPRODUCIBLE_PROVENANCE: (
            "raw data hash",
            "experiment ID",
        ),
        ValidationCriterion.FULL_REGRESSION: (
            "4996 tests passed",
        ),
    }

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
        evidence=evidence,
    )

    provenance_result = next(
        item
        for item in result.criteria_results
        if item.criterion
        == ValidationCriterion.REPRODUCIBLE_PROVENANCE
    )

    regression_result = next(
        item
        for item in result.criteria_results
        if item.criterion
        == ValidationCriterion.FULL_REGRESSION
    )

    assert provenance_result.evidence == (
        "raw data hash",
        "experiment ID",
    )

    assert regression_result.evidence == (
        "4996 tests passed",
    )


def test_result_is_immutable():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate(
        criteria=all_true_criteria(),
        human_approval=False,
    )

    with pytest.raises(AttributeError):
        result.status = ValidationStatus.FAILED


def test_criterion_result_is_immutable():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate_criterion(
        ValidationCriterion.FULL_REGRESSION,
        passed=True,
        evidence=("tests passed",),
    )

    with pytest.raises(AttributeError):
        result.status = ValidationStatus.FAILED


def test_all_criteria_passed_property():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    passed = engine.validate(
        criteria=all_true_criteria(),
        human_approval=False,
    )

    assert passed.all_criteria_passed is True

    criteria = all_true_criteria()

    criteria[
        ValidationCriterion.FULL_REGRESSION
    ] = False

    failed = engine.validate(
        criteria=criteria,
        human_approval=False,
    )

    assert failed.all_criteria_passed is False


def test_validation_is_repeatable():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    first = engine.validate(
        criteria=all_true_criteria(),
        evidence=all_evidence(),
        human_approval=False,
    )

    second = engine.validate(
        criteria=all_true_criteria(),
        evidence=all_evidence(),
        human_approval=False,
    )

    assert first == second


def test_validation_does_not_modify_criteria():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    criteria = all_true_criteria()
    original = dict(criteria)

    engine.validate(
        criteria=criteria,
        human_approval=False,
    )

    assert criteria == original


def test_validation_does_not_modify_evidence():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    evidence = all_evidence()

    original = {
        key: tuple(value)
        for key, value in evidence.items()
    }

    engine.validate(
        criteria=all_true_criteria(),
        evidence=evidence,
        human_approval=False,
    )

    assert evidence == original


def test_validation_does_not_modify_metadata():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    metadata = {
        "release": "v6.0.9",
    }

    original = dict(metadata)

    engine.validate(
        criteria=all_true_criteria(),
        human_approval=False,
        metadata=metadata,
    )

    assert metadata == original


def test_empty_evidence_is_allowed_for_criterion():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate_criterion(
        ValidationCriterion.FULL_REGRESSION,
        passed=False,
        evidence=(),
    )

    assert result.evidence == ()
    assert (
        result.status
        == ValidationStatus.FAILED
    )


def test_missing_evidence_defaults_to_empty():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate(
        criteria=all_true_criteria(),
        human_approval=False,
    )

    for criterion_result in result.criteria_results:
        assert criterion_result.evidence == ()


def test_details_are_optional():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate_criterion(
        ValidationCriterion.FULL_REGRESSION,
        passed=True,
        evidence=("tests passed",),
    )

    assert result.details == ""


def test_details_are_trimmed():
    engine = SystemValidationEngine(
        require_human_approval=False
    )

    result = engine.validate_criterion(
        ValidationCriterion.FULL_REGRESSION,
        passed=True,
        evidence=("tests passed",),
        details="  Validation complete.  ",
    )

    assert (
        result.details
        == "Validation complete."
    )


def test_all_success_criteria_are_named():
    expected = {
        "reproducible_provenance",
        "predictive_causal_separation",
        "oos_is_separated",
        "changes_measurable",
        "changes_reversible",
        "portfolio_reconstructable",
        "quantum_classical_benchmark",
        "autonomous_research_controlled",
        "end_to_end_acceptance",
        "full_regression",
    }

    actual = {
        criterion.value
        for criterion
        in SystemValidationEngine.REQUIRED_CRITERIA
    }

    assert actual == expected