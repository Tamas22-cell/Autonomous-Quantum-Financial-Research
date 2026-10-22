import pytest

from research_control_center.experiment_registry import ExperimentRegistry
from research_control_center.models import (
    ExperimentRecord,
    ExperimentStatus,
    ExperimentType,
)


def make_registry() -> ExperimentRegistry:
    return ExperimentRegistry()


def make_record(
    name: str = "BTC QAOA Experiment",
    experiment_type: ExperimentType = ExperimentType.QUANTUM,
    asset: str = "BTC",
    regime: str = "RISK_ON",
    horizon: str = "1D",
) -> ExperimentRecord:
    return ExperimentRecord(
        name=name,
        experiment_type=experiment_type,
        asset=asset,
        regime=regime,
        horizon=horizon,
        parameters={
            "depth": 3,
            "repetitions": 100,
        },
        random_seed=42,
        code_version="v5.0.10",
        data_version="market-data-v1",
    )


# ============================================================
# REGISTRATION
# ============================================================


def test_register_experiment():
    registry = make_registry()
    record = make_record()

    result = registry.register(record)

    assert result.experiment_id == record.experiment_id
    assert registry.count() == 1


def test_registered_experiment_is_research_only():
    registry = make_registry()

    result = registry.register(
        make_record()
    )

    assert result.metadata["research_only"] is True
    assert result.metadata["real_money_execution"] is False


def test_duplicate_registration_is_rejected():
    registry = make_registry()
    record = make_record()

    registry.register(record)

    with pytest.raises(ValueError):
        registry.register(record)


def test_invalid_registration_type():
    registry = make_registry()

    with pytest.raises(TypeError):
        registry.register("invalid")


def test_empty_experiment_name_is_rejected():
    registry = make_registry()

    record = make_record()
    record.name = "   "

    with pytest.raises(ValueError):
        registry.register(record)


# ============================================================
# CREATE
# ============================================================


def test_create_experiment():
    registry = make_registry()

    result = registry.create(
        name="Agent Benchmark",
        experiment_type=ExperimentType.AGENT,
        asset="SPY",
        regime="MIXED",
        horizon="7D",
        parameters={"model": "test-model"},
        random_seed=123,
        code_version="abc123",
        data_version="dataset-001",
    )

    assert registry.count() == 1
    assert result.name == "Agent Benchmark"
    assert result.asset == "SPY"


def test_create_rejects_empty_name():
    registry = make_registry()

    with pytest.raises(ValueError):
        registry.create(
            name="",
            experiment_type=ExperimentType.MODEL,
        )


def test_create_rejects_invalid_experiment_type():
    registry = make_registry()

    with pytest.raises(TypeError):
        registry.create(
            name="Invalid",
            experiment_type="MODEL",
        )


def test_create_rejects_boolean_seed():
    registry = make_registry()

    with pytest.raises(TypeError):
        registry.create(
            name="Invalid Seed",
            experiment_type=ExperimentType.MODEL,
            random_seed=True,
        )


def test_create_rejects_float_seed():
    registry = make_registry()

    with pytest.raises(TypeError):
        registry.create(
            name="Invalid Seed",
            experiment_type=ExperimentType.MODEL,
            random_seed=1.5,
        )


# ============================================================
# READ
# ============================================================


def test_get_registered_experiment():
    registry = make_registry()

    created = registry.register(
        make_record()
    )

    result = registry.get(
        created.experiment_id
    )

    assert result.experiment_id == created.experiment_id


def test_get_unknown_experiment():
    registry = make_registry()

    with pytest.raises(KeyError):
        registry.get("missing")


def test_get_returns_copy():
    registry = make_registry()

    created = registry.register(
        make_record()
    )

    result = registry.get(
        created.experiment_id
    )

    result.name = "CHANGED"

    original = registry.get(
        created.experiment_id
    )

    assert original.name != "CHANGED"


def test_list_all():
    registry = make_registry()

    registry.register(
        make_record(name="A")
    )

    registry.register(
        make_record(name="B")
    )

    assert len(registry.list_all()) == 2


# ============================================================
# STATUS
# ============================================================


def test_update_status_to_running():
    registry = make_registry()

    record = registry.register(
        make_record()
    )

    result = registry.update_status(
        record.experiment_id,
        ExperimentStatus.RUNNING,
    )

    assert result.status == ExperimentStatus.RUNNING
    assert result.completed_at is None


def test_update_status_to_completed():
    registry = make_registry()

    record = registry.register(
        make_record()
    )

    result = registry.update_status(
        record.experiment_id,
        ExperimentStatus.COMPLETED,
    )

    assert result.status == ExperimentStatus.COMPLETED
    assert result.completed_at is not None


def test_invalid_status_is_rejected():
    registry = make_registry()

    record = registry.register(
        make_record()
    )

    with pytest.raises(TypeError):
        registry.update_status(
            record.experiment_id,
            "COMPLETED",
        )


# ============================================================
# METRICS
# ============================================================


def test_update_metrics():
    registry = make_registry()

    record = registry.register(
        make_record()
    )

    result = registry.update_metrics(
        record.experiment_id,
        {
            "sharpe": 1.4,
            "return": 0.12,
        },
    )

    assert result.metrics["sharpe"] == 1.4
    assert result.metrics["return"] == 0.12


def test_metrics_are_merged():
    registry = make_registry()

    record = registry.register(
        make_record()
    )

    registry.update_metrics(
        record.experiment_id,
        {"sharpe": 1.2},
    )

    result = registry.update_metrics(
        record.experiment_id,
        {"alpha": 0.04},
    )

    assert result.metrics["sharpe"] == 1.2
    assert result.metrics["alpha"] == 0.04


def test_boolean_metric_is_rejected():
    registry = make_registry()

    record = registry.register(
        make_record()
    )

    with pytest.raises(TypeError):
        registry.update_metrics(
            record.experiment_id,
            {"success": True},
        )


def test_non_numeric_metric_is_rejected():
    registry = make_registry()

    record = registry.register(
        make_record()
    )

    with pytest.raises(TypeError):
        registry.update_metrics(
            record.experiment_id,
            {"sharpe": "high"},
        )


# ============================================================
# ARTIFACTS
# ============================================================


def test_add_artifact():
    registry = make_registry()

    record = registry.register(
        make_record()
    )

    result = registry.add_artifact(
        record.experiment_id,
        "reports/experiment.json",
    )

    assert (
        "reports/experiment.json"
        in result.artifacts
    )


def test_duplicate_artifact_is_not_duplicated():
    registry = make_registry()

    record = registry.register(
        make_record()
    )

    registry.add_artifact(
        record.experiment_id,
        "result.json",
    )

    result = registry.add_artifact(
        record.experiment_id,
        "result.json",
    )

    assert result.artifacts.count(
        "result.json"
    ) == 1


def test_empty_artifact_is_rejected():
    registry = make_registry()

    record = registry.register(
        make_record()
    )

    with pytest.raises(ValueError):
        registry.add_artifact(
            record.experiment_id,
            " ",
        )


# ============================================================
# REPRODUCIBILITY
# ============================================================


def test_complete_reproducibility_metadata():
    registry = make_registry()

    record = registry.register(
        make_record()
    )

    view = registry.reproducibility_view(
        record.experiment_id
    )

    assert view["reproducible"] is True
    assert view["missing_requirements"] == []


def test_missing_seed_is_not_reproducible():
    registry = make_registry()

    record = make_record()
    record.random_seed = None

    record = registry.register(record)

    view = registry.reproducibility_view(
        record.experiment_id
    )

    assert view["reproducible"] is False
    assert "random_seed" in view[
        "missing_requirements"
    ]


def test_missing_code_version_is_not_reproducible():
    registry = make_registry()

    record = make_record()
    record.code_version = None

    record = registry.register(record)

    assert registry.is_reproducible(
        record.experiment_id
    ) is False


def test_missing_data_version_is_not_reproducible():
    registry = make_registry()

    record = make_record()
    record.data_version = None

    record = registry.register(record)

    assert registry.is_reproducible(
        record.experiment_id
    ) is False


def test_reproducibility_view_is_research_only():
    registry = make_registry()

    record = registry.register(
        make_record()
    )

    view = registry.reproducibility_view(
        record.experiment_id
    )

    assert view["research_only"] is True
    assert view["real_money_execution"] is False


# ============================================================
# FILTERING
# ============================================================


def test_filter_by_asset():
    registry = make_registry()

    registry.register(
        make_record(
            name="BTC",
            asset="BTC",
        )
    )

    registry.register(
        make_record(
            name="SPY",
            asset="SPY",
        )
    )

    results = registry.filter(
        asset="BTC"
    )

    assert len(results) == 1
    assert results[0].asset == "BTC"


def test_filter_by_regime():
    registry = make_registry()

    registry.register(
        make_record(
            name="Risk On",
            regime="RISK_ON",
        )
    )

    registry.register(
        make_record(
            name="Risk Off",
            regime="RISK_OFF",
        )
    )

    results = registry.filter(
        regime="RISK_OFF"
    )

    assert len(results) == 1
    assert results[0].regime == "RISK_OFF"


def test_filter_by_horizon():
    registry = make_registry()

    registry.register(
        make_record(
            name="1D",
            horizon="1D",
        )
    )

    registry.register(
        make_record(
            name="30D",
            horizon="30D",
        )
    )

    results = registry.filter(
        horizon="30D"
    )

    assert len(results) == 1
    assert results[0].horizon == "30D"


def test_filter_by_type():
    registry = make_registry()

    registry.register(
        make_record(
            name="Quantum",
            experiment_type=ExperimentType.QUANTUM,
        )
    )

    registry.register(
        make_record(
            name="Classical",
            experiment_type=ExperimentType.CLASSICAL,
        )
    )

    results = registry.filter(
        experiment_type=ExperimentType.QUANTUM
    )

    assert len(results) == 1
    assert (
        results[0].experiment_type
        == ExperimentType.QUANTUM
    )


def test_filter_by_status():
    registry = make_registry()

    a = registry.register(
        make_record(name="A")
    )

    registry.register(
        make_record(name="B")
    )

    registry.update_status(
        a.experiment_id,
        ExperimentStatus.COMPLETED,
    )

    results = registry.filter(
        status=ExperimentStatus.COMPLETED
    )

    assert len(results) == 1
    assert (
        results[0].status
        == ExperimentStatus.COMPLETED
    )


def test_combined_filter():
    registry = make_registry()

    registry.register(
        make_record(
            name="Target",
            asset="BTC",
            regime="RISK_ON",
            horizon="1D",
        )
    )

    registry.register(
        make_record(
            name="Other",
            asset="BTC",
            regime="RISK_OFF",
            horizon="1D",
        )
    )

    results = registry.filter(
        asset="BTC",
        regime="RISK_ON",
        horizon="1D",
    )

    assert len(results) == 1
    assert results[0].name == "Target"


# ============================================================
# COMPARISON
# ============================================================


def test_compare_experiments():
    registry = make_registry()

    a = registry.register(
        make_record(name="A")
    )

    b_record = make_record(name="B")
    b_record.parameters["depth"] = 5

    b = registry.register(
        b_record
    )

    registry.update_metrics(
        a.experiment_id,
        {"score": 0.70},
    )

    registry.update_metrics(
        b.experiment_id,
        {"score": 0.85},
    )

    comparison = registry.compare(
        a.experiment_id,
        b.experiment_id,
    )

    assert (
        comparison[
            "parameter_comparison"
        ]["depth"]["equal"]
        is False
    )

    assert (
        comparison[
            "metric_comparison"
        ]["score"]["delta_b_minus_a"]
        == pytest.approx(0.15)
    )


def test_compare_identical_context():
    registry = make_registry()

    a = registry.register(
        make_record(name="A")
    )

    b = registry.register(
        make_record(name="B")
    )

    comparison = registry.compare(
        a.experiment_id,
        b.experiment_id,
    )

    assert comparison["same_asset"] is True
    assert comparison["same_regime"] is True
    assert comparison["same_horizon"] is True


# ============================================================
# AUDIT
# ============================================================


def test_registration_creates_audit_event():
    registry = make_registry()

    record = registry.register(
        make_record()
    )

    events = registry.audit_log(
        record.experiment_id
    )

    assert len(events) == 1
    assert (
        events[0]["event"]
        == "EXPERIMENT_REGISTERED"
    )


def test_updates_create_audit_history():
    registry = make_registry()

    record = registry.register(
        make_record()
    )

    registry.update_status(
        record.experiment_id,
        ExperimentStatus.RUNNING,
    )

    registry.update_metrics(
        record.experiment_id,
        {"score": 0.9},
    )

    registry.add_artifact(
        record.experiment_id,
        "report.json",
    )

    events = registry.audit_log(
        record.experiment_id
    )

    assert len(events) == 4


# ============================================================
# SNAPSHOT
# ============================================================


def test_snapshot_counts_experiments():
    registry = make_registry()

    registry.register(
        make_record(name="A")
    )

    incomplete = make_record(name="B")
    incomplete.random_seed = None

    registry.register(incomplete)

    snapshot = registry.snapshot()

    assert snapshot["total_experiments"] == 2
    assert snapshot["reproducible_experiments"] == 1
    assert snapshot["non_reproducible_experiments"] == 1


def test_snapshot_is_research_only():
    registry = make_registry()

    snapshot = registry.snapshot()

    assert snapshot["research_only"] is True
    assert snapshot["real_money_execution"] is False