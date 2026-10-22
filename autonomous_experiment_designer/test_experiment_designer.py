import pytest

from autonomous_experiment_designer.experiment_designer import (
    ExperimentDesigner,
    ExperimentType,
    ValidationMethod,
)
from autonomous_experiment_designer.uncertainty_detector import (
    Severity,
    UncertaintyDetector,
    UncertaintyReport,
    UncertaintyType,
)


@pytest.fixture
def detector():
    return UncertaintyDetector()


@pytest.fixture
def designer():
    return ExperimentDesigner()


def test_default_designer_configuration(designer):
    assert designer is not None


def test_no_uncertainty_produces_no_plan(
    designer,
):
    report = UncertaintyReport(signals=())

    plan = designer.design(
        report,
        experiment_id="EXP-001",
    )

    assert plan is None


def test_confidence_gap_creates_confidence_validation_plan(
    designer,
    detector,
):
    report = detector.detect(
        confidence=(0.50, 0.80),
    )

    plan = designer.design(
        report,
        experiment_id="EXP-CONFIDENCE",
    )

    assert plan is not None
    assert plan.experiment_id == "EXP-CONFIDENCE"
    assert (
        plan.experiment_type
        == ExperimentType.CONFIDENCE_VALIDATION
    )
    assert (
        ValidationMethod.CROSS_VALIDATION
        in plan.validation_methods
    )
    assert (
        ValidationMethod.OUT_OF_SAMPLE
        in plan.validation_methods
    )


def test_regime_change_creates_regime_stability_plan(
    designer,
    detector,
):
    report = detector.detect(
        regime_distance=0.60,
    )

    plan = designer.design(
        report,
        experiment_id="EXP-REGIME",
    )

    assert plan is not None
    assert (
        plan.experiment_type
        == ExperimentType.REGIME_STABILITY
    )
    assert (
        ValidationMethod.WALK_FORWARD
        in plan.validation_methods
    )
    assert (
        ValidationMethod.STRESS_TEST
        in plan.validation_methods
    )


def test_performance_gap_creates_replication_plan(
    designer,
    detector,
):
    report = detector.detect(
        performance=(0.05, 0.25),
    )

    plan = designer.design(
        report,
        experiment_id="EXP-PERFORMANCE",
    )

    assert plan is not None
    assert (
        plan.experiment_type
        == ExperimentType.PERFORMANCE_REPLICATION
    )
    assert (
        ValidationMethod.HOLDOUT
        in plan.validation_methods
    )
    assert (
        ValidationMethod.OUT_OF_SAMPLE
        in plan.validation_methods
    )


def test_evidence_conflict_creates_reconciliation_plan(
    designer,
    detector,
):
    report = detector.detect(
        evidence_agreement=0.30,
    )

    plan = designer.design(
        report,
        experiment_id="EXP-EVIDENCE",
    )

    assert plan is not None
    assert (
        plan.experiment_type
        == ExperimentType.EVIDENCE_RECONCILIATION
    )


def test_multiple_signals_create_multi_factor_plan(
    designer,
    detector,
):
    report = detector.detect(
        confidence=(0.40, 0.80),
        regime_distance=0.70,
        performance=(0.00, 0.25),
    )

    plan = designer.design(
        report,
        experiment_id="EXP-MULTI",
    )

    assert plan is not None
    assert (
        plan.experiment_type
        == ExperimentType.MULTI_FACTOR_VALIDATION
    )
    assert (
        ValidationMethod.WALK_FORWARD
        in plan.validation_methods
    )
    assert (
        ValidationMethod.OUT_OF_SAMPLE
        in plan.validation_methods
    )
    assert (
        ValidationMethod.STRESS_TEST
        in plan.validation_methods
    )


def test_plan_requires_human_approval(
    designer,
    detector,
):
    report = detector.detect(
        regime_distance=0.60,
    )

    plan = designer.design(
        report,
        experiment_id="EXP-APPROVAL",
    )

    assert plan is not None
    assert plan.requires_human_approval is True


def test_plan_preserves_source_signal(
    designer,
    detector,
):
    report = detector.detect(
        confidence=(0.40, 0.80),
    )

    plan = designer.design(
        report,
        experiment_id="EXP-SOURCE",
    )

    assert plan is not None
    assert plan.source_signals == (
        UncertaintyType.CONFIDENCE_GAP,
    )


def test_plan_contains_default_datasets(
    designer,
    detector,
):
    report = detector.detect(
        regime_distance=0.60,
    )

    plan = designer.design(
        report,
        experiment_id="EXP-DATA",
    )

    assert plan is not None
    assert "historical_market_data" in plan.datasets
    assert "macro_economic_data" in plan.datasets
    assert "portfolio_performance_history" in plan.datasets


def test_plan_contains_default_agents(
    designer,
    detector,
):
    report = detector.detect(
        regime_distance=0.60,
    )

    plan = designer.design(
        report,
        experiment_id="EXP-AGENTS",
    )

    assert plan is not None
    assert "macro_agent" in plan.agents
    assert "technical_agent" in plan.agents
    assert "risk_agent" in plan.agents


def test_plan_contains_default_models(
    designer,
    detector,
):
    report = detector.detect(
        regime_distance=0.60,
    )

    plan = designer.design(
        report,
        experiment_id="EXP-MODELS",
    )

    assert plan is not None
    assert "baseline_model" in plan.models
    assert "challenger_model" in plan.models


def test_critical_signal_gets_highest_priority(
    designer,
    detector,
):
    report = detector.detect(
        regime_distance=0.90,
    )

    plan = designer.design(
        report,
        experiment_id="EXP-CRITICAL",
    )

    assert plan is not None
    assert plan.severity == Severity.CRITICAL
    assert plan.priority == 1


def test_high_signal_gets_high_priority(
    designer,
    detector,
):
    report = detector.detect(
        confidence=(0.50, 0.80),
    )

    plan = designer.design(
        report,
        experiment_id="EXP-HIGH",
    )

    assert plan is not None
    assert plan.priority == 2


def test_custom_configuration_is_used():
    designer = ExperimentDesigner(
        default_datasets=("dataset_a",),
        default_agents=("agent_a",),
        default_models=("model_a",),
    )

    detector = UncertaintyDetector()

    report = detector.detect(
        regime_distance=0.60,
    )

    plan = designer.design(
        report,
        experiment_id="EXP-CUSTOM",
    )

    assert plan is not None
    assert plan.datasets == ("dataset_a",)
    assert plan.agents == ("agent_a",)
    assert plan.models == ("model_a",)


@pytest.mark.parametrize(
    "field",
    [
        "default_datasets",
        "default_agents",
        "default_models",
    ],
)
def test_empty_configuration_is_rejected(field):
    kwargs = {
        "default_datasets": ("dataset",),
        "default_agents": ("agent",),
        "default_models": ("model",),
    }

    kwargs[field] = ()

    with pytest.raises(ValueError):
        ExperimentDesigner(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "default_datasets",
        "default_agents",
        "default_models",
    ],
)
def test_string_configuration_is_rejected(field):
    kwargs = {
        "default_datasets": ("dataset",),
        "default_agents": ("agent",),
        "default_models": ("model",),
    }

    kwargs[field] = "invalid"

    with pytest.raises(ValueError):
        ExperimentDesigner(**kwargs)


def test_duplicate_dataset_names_are_rejected():
    with pytest.raises(ValueError):
        ExperimentDesigner(
            default_datasets=(
                "dataset_a",
                "dataset_a",
            )
        )


def test_empty_dataset_name_is_rejected():
    with pytest.raises(ValueError):
        ExperimentDesigner(
            default_datasets=(
                "dataset_a",
                "",
            )
        )


def test_invalid_report_type_is_rejected(
    designer,
):
    with pytest.raises(TypeError):
        designer.design(
            "not-a-report",
            experiment_id="EXP-INVALID",
        )


@pytest.mark.parametrize(
    "experiment_id",
    [
        "",
        " ",
        None,
        123,
    ],
)
def test_invalid_experiment_id_is_rejected(
    designer,
    detector,
    experiment_id,
):
    report = detector.detect(
        regime_distance=0.60,
    )

    with pytest.raises(ValueError):
        designer.design(
            report,
            experiment_id=experiment_id,
        )


def test_plan_is_immutable(
    designer,
    detector,
):
    report = detector.detect(
        regime_distance=0.60,
    )

    plan = designer.design(
        report,
        experiment_id="EXP-IMMUTABLE",
    )

    assert plan is not None

    with pytest.raises(AttributeError):
        plan.priority = 99


def test_plan_objective_is_nonempty(
    designer,
    detector,
):
    report = detector.detect(
        regime_distance=0.60,
    )

    plan = designer.design(
        report,
        experiment_id="EXP-OBJECTIVE",
    )

    assert plan is not None
    assert plan.objective.strip()


def test_design_does_not_execute_experiment(
    designer,
    detector,
):
    report = detector.detect(
        regime_distance=0.60,
    )

    plan = designer.design(
        report,
        experiment_id="EXP-NO-EXECUTION",
    )

    assert plan is not None
    assert plan.requires_human_approval is True