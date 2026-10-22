from __future__ import annotations

import pytest

from alpha_discovery_laboratory.alpha_lineage import (
    AlphaExperimentAlreadyExistsError,
    AlphaExperimentNotFoundError,
    AlphaLineageAlreadyExistsError,
    AlphaLineageError,
    AlphaLineageNotFoundError,
    AlphaLineageRegistry,
)
from alpha_discovery_laboratory.models import (
    VERSION,
    AlphaDiscoveryExperiment,
    AlphaLineageRecord,
    PerformanceMetrics,
    StrategyCandidate,
    StrategyDirection,
    ValidationResult,
    ValidationStage,
    ValidationStatus,
)


# ============================================================
# Helpers
# ============================================================


def make_candidate(
    name: str = "btc_alpha",
    *,
    feature_ids=None,
    combination_ids=None,
    parent_candidate_ids=None,
) -> StrategyCandidate:
    return StrategyCandidate(
        name=name,
        feature_ids=list(
            feature_ids
            or ["feature_001"]
        ),
        combination_ids=list(
            combination_ids
            or ["combination_001"]
        ),
        direction=StrategyDirection.LONG,
        entry_rule={
            "operator": ">",
            "value": 1.0,
        },
        exit_rule={
            "operator": "<",
            "value": 0.0,
        },
        position_rule={
            "type": "risk_scaled",
        },
        parameters={
            "threshold": 1.0,
        },
        parent_candidate_ids=list(
            parent_candidate_ids
            or []
        ),
        metadata={
            "asset": "BTC",
        },
    )


def make_metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        total_return=0.20,
        annualized_return=0.18,
        volatility=0.15,
        sharpe_ratio=1.50,
        sortino_ratio=1.80,
        calmar_ratio=1.40,
        max_drawdown=0.12,
        win_rate=0.58,
        profit_factor=1.40,
        turnover=0.50,
        execution_cost=0.01,
        trade_count=100,
    )


def make_validation_result(
    candidate_id: str,
    stage: ValidationStage = ValidationStage.BACKTEST,
) -> ValidationResult:
    return ValidationResult(
        candidate_id=candidate_id,
        stage=stage,
        status=ValidationStatus.PASSED,
        metrics=make_metrics(),
        score=0.90,
        passed_checks=["quality"],
        failed_checks=[],
        rejection_reason=None,
        details="passed",
        metadata={
            "test": True,
        },
    )


def make_experiment(
    name: str = "alpha_experiment",
) -> AlphaDiscoveryExperiment:
    return AlphaDiscoveryExperiment(
        name=name,
        dataset_ids=[
            "dataset_001",
        ],
        feature_ids=[
            "feature_001",
        ],
        combination_ids=[
            "combination_001",
        ],
        candidate_ids=[
            "alpha_001",
        ],
        accepted_candidate_ids=[],
        rejected_candidate_ids=[],
        configuration={
            "mode": "research",
        },
        seed=42,
        run_id="run_001",
        metadata={
            "asset": "BTC",
        },
    )


def make_lineage_record(
    candidate_id: str = "alpha_001",
) -> AlphaLineageRecord:
    return AlphaLineageRecord(
        candidate_id=candidate_id,
        feature_ids=[
            "feature_001",
        ],
        combination_ids=[
            "combination_001",
        ],
        dataset_ids=[
            "dataset_001",
        ],
        dataset_fingerprints=[
            "dataset_fp_001",
        ],
        model_version_ids=[
            "model_v1",
        ],
        prompt_version_ids=[
            "prompt_v1",
        ],
        policy_version_ids=[
            "policy_v1",
        ],
        validation_fingerprints=[
            "validation_fp_001",
        ],
        experiment_id="experiment_001",
        run_id="run_001",
        parent_candidate_ids=[],
        metadata={
            "asset": "BTC",
        },
    )


def create_registered_experiment(
    registry: AlphaLineageRegistry,
):
    return registry.create_experiment(
        name="btc_alpha_experiment",
        dataset_ids=[
            "dataset_001",
        ],
        feature_ids=[
            "feature_001",
        ],
        combination_ids=[
            "combination_001",
        ],
        configuration={
            "validation": "strict",
        },
        seed=42,
        run_id="run_001",
        metadata={
            "asset": "BTC",
        },
    )


def create_registered_lineage(
    registry: AlphaLineageRegistry,
    *,
    name: str = "btc_alpha",
    experiment_id=None,
    run_id: str = "run_001",
    feature_ids=None,
    combination_ids=None,
    parent_candidate_ids=None,
):
    candidate = make_candidate(
        name,
        feature_ids=feature_ids,
        combination_ids=combination_ids,
        parent_candidate_ids=parent_candidate_ids,
    )

    validation = make_validation_result(
        candidate.candidate_id
    )

    record = registry.create_lineage(
        candidate=candidate,
        validation_results=[
            validation,
        ],
        dataset_ids=[
            "dataset_001",
        ],
        dataset_fingerprints=[
            "dataset_fp_001",
        ],
        model_version_ids=[
            "model_v1",
        ],
        prompt_version_ids=[
            "prompt_v1",
        ],
        policy_version_ids=[
            "policy_v1",
        ],
        experiment_id=experiment_id,
        run_id=run_id,
        metadata={
            "asset": "BTC",
        },
    )

    return candidate, record


# ============================================================
# Basic
# ============================================================


def test_version():
    assert AlphaLineageRegistry.VERSION == VERSION


def test_empty_registry():
    registry = AlphaLineageRegistry()

    assert registry.lineage_count() == 0
    assert registry.experiment_count() == 0
    assert registry.records() == []
    assert registry.experiments() == []
    assert registry.audit_log() == []


def test_summary_empty():
    registry = AlphaLineageRegistry()

    summary = registry.summary()

    assert summary["version"] == VERSION
    assert summary["lineage_count"] == 0
    assert summary["experiment_count"] == 0
    assert summary["audit_count"] == 0
    assert summary["research_only"] is True
    assert summary["real_money_execution"] is False


# ============================================================
# Experiment Registration
# ============================================================


def test_register_experiment():
    registry = AlphaLineageRegistry()
    experiment = make_experiment()

    result = registry.register_experiment(
        experiment
    )

    assert result is experiment
    assert registry.experiment_count() == 1


def test_experiment_exists():
    registry = AlphaLineageRegistry()
    experiment = make_experiment()

    registry.register_experiment(
        experiment
    )

    assert registry.experiment_exists(
        experiment.experiment_id
    )


def test_get_experiment():
    registry = AlphaLineageRegistry()
    experiment = make_experiment()

    registry.register_experiment(
        experiment
    )

    assert (
        registry.get_experiment(
            experiment.experiment_id
        )
        == experiment
    )


def test_get_missing_experiment():
    registry = AlphaLineageRegistry()

    with pytest.raises(
        AlphaExperimentNotFoundError
    ):
        registry.get_experiment(
            "alpha_experiment_missing"
        )


def test_register_duplicate_experiment_allowed():
    registry = AlphaLineageRegistry()
    experiment = make_experiment()

    first = registry.register_experiment(
        experiment
    )
    second = registry.register_experiment(
        experiment
    )

    assert first == second
    assert registry.experiment_count() == 1


def test_register_duplicate_experiment_disallowed():
    registry = AlphaLineageRegistry()
    experiment = make_experiment()

    registry.register_experiment(
        experiment
    )

    with pytest.raises(
        AlphaExperimentAlreadyExistsError
    ):
        registry.register_experiment(
            experiment,
            allow_existing=False,
        )


# ============================================================
# Experiment Creation
# ============================================================


def test_create_experiment():
    registry = AlphaLineageRegistry()

    experiment = create_registered_experiment(
        registry
    )

    assert registry.experiment_count() == 1
    assert registry.experiment_exists(
        experiment.experiment_id
    )


def test_create_experiment_properties():
    registry = AlphaLineageRegistry()

    experiment = registry.create_experiment(
        name="multi_asset_alpha",
        dataset_ids=[
            "dataset_btc",
            "dataset_spy",
        ],
        feature_ids=[
            "feature_momentum",
            "feature_volume",
        ],
        combination_ids=[
            "combination_001",
        ],
        candidate_ids=[
            "alpha_001",
        ],
        accepted_candidate_ids=[
            "alpha_001",
        ],
        rejected_candidate_ids=[],
        configuration={
            "validation": "strict",
        },
        seed=123,
        run_id="run_123",
        metadata={
            "scope": "multi_asset",
        },
    )

    assert experiment.name == "multi_asset_alpha"

    assert experiment.dataset_ids == [
        "dataset_btc",
        "dataset_spy",
    ]

    assert experiment.seed == 123
    assert experiment.run_id == "run_123"
    assert experiment.metadata["scope"] == "multi_asset"


def test_create_duplicate_experiment_allowed():
    registry = AlphaLineageRegistry()

    first = registry.create_experiment(
        name="same_experiment",
        dataset_ids=[
            "dataset_001",
        ],
        seed=42,
    )

    second = registry.create_experiment(
        name="same_experiment",
        dataset_ids=[
            "dataset_001",
        ],
        seed=42,
    )

    assert first.experiment_id == second.experiment_id
    assert registry.experiment_count() == 1


def test_create_duplicate_experiment_disallowed():
    registry = AlphaLineageRegistry()

    registry.create_experiment(
        name="same_experiment",
        dataset_ids=[
            "dataset_001",
        ],
        seed=42,
    )

    with pytest.raises(
        AlphaExperimentAlreadyExistsError
    ):
        registry.create_experiment(
            name="same_experiment",
            dataset_ids=[
                "dataset_001",
            ],
            seed=42,
            allow_existing=False,
        )


# ============================================================
# Lineage Registration
# ============================================================


def test_register_lineage():
    registry = AlphaLineageRegistry()
    record = make_lineage_record()

    result = registry.register_lineage(
        record
    )

    assert result is record
    assert registry.lineage_count() == 1


def test_lineage_exists():
    registry = AlphaLineageRegistry()
    record = make_lineage_record()

    registry.register_lineage(
        record
    )

    assert registry.exists(
        record.candidate_id
    )


def test_get_lineage():
    registry = AlphaLineageRegistry()
    record = make_lineage_record()

    registry.register_lineage(
        record
    )

    assert registry.get(
        record.candidate_id
    ) == record


def test_get_missing_lineage():
    registry = AlphaLineageRegistry()

    with pytest.raises(
        AlphaLineageNotFoundError
    ):
        registry.get(
            "alpha_missing"
        )


def test_get_by_fingerprint():
    registry = AlphaLineageRegistry()
    record = make_lineage_record()

    registry.register_lineage(
        record
    )

    assert (
        registry.get_by_fingerprint(
            record.fingerprint
        )
        == record
    )


def test_get_missing_fingerprint():
    registry = AlphaLineageRegistry()

    with pytest.raises(
        AlphaLineageNotFoundError
    ):
        registry.get_by_fingerprint(
            "missing_fingerprint"
        )


def test_duplicate_lineage_allowed():
    registry = AlphaLineageRegistry()
    record = make_lineage_record()

    first = registry.register_lineage(
        record
    )
    second = registry.register_lineage(
        record
    )

    assert first == second
    assert registry.lineage_count() == 1


def test_duplicate_lineage_disallowed():
    registry = AlphaLineageRegistry()
    record = make_lineage_record()

    registry.register_lineage(
        record
    )

    with pytest.raises(
        AlphaLineageAlreadyExistsError
    ):
        registry.register_lineage(
            record,
            allow_existing=False,
        )


# ============================================================
# Lineage Creation
# ============================================================


def test_create_lineage():
    registry = AlphaLineageRegistry()

    candidate, record = create_registered_lineage(
        registry
    )

    assert registry.lineage_count() == 1
    assert registry.exists(
        candidate.candidate_id
    )
    assert record.candidate_id == candidate.candidate_id


def test_create_lineage_feature_ids():
    registry = AlphaLineageRegistry()

    candidate, record = create_registered_lineage(
        registry,
        feature_ids=[
            "feature_momentum",
            "feature_volume",
        ],
    )

    assert record.feature_ids == candidate.feature_ids


def test_create_lineage_combination_ids():
    registry = AlphaLineageRegistry()

    candidate, record = create_registered_lineage(
        registry,
        combination_ids=[
            "combination_spread",
        ],
    )

    assert record.combination_ids == candidate.combination_ids


def test_create_lineage_dataset_data():
    registry = AlphaLineageRegistry()

    _, record = create_registered_lineage(
        registry
    )

    assert record.dataset_ids == [
        "dataset_001"
    ]
    assert record.dataset_fingerprints == [
        "dataset_fp_001"
    ]


def test_create_lineage_version_data():
    registry = AlphaLineageRegistry()

    _, record = create_registered_lineage(
        registry
    )

    assert record.model_version_ids == [
        "model_v1"
    ]
    assert record.prompt_version_ids == [
        "prompt_v1"
    ]
    assert record.policy_version_ids == [
        "policy_v1"
    ]


def test_create_lineage_validation_fingerprint():
    registry = AlphaLineageRegistry()
    candidate = make_candidate()

    validation = make_validation_result(
        candidate.candidate_id
    )

    record = registry.create_lineage(
        candidate=candidate,
        validation_results=[
            validation
        ],
    )

    assert (
        validation.fingerprint
        in record.validation_fingerprints
    )


def test_create_lineage_run_id():
    registry = AlphaLineageRegistry()

    _, record = create_registered_lineage(
        registry,
        run_id="run_special",
    )

    assert record.run_id == "run_special"


def test_create_lineage_metadata():
    registry = AlphaLineageRegistry()

    _, record = create_registered_lineage(
        registry
    )

    assert record.metadata["asset"] == "BTC"


# ============================================================
# Experiment ↔ Lineage
# ============================================================


def test_lineage_with_registered_experiment():
    registry = AlphaLineageRegistry()

    experiment = create_registered_experiment(
        registry
    )

    _, record = create_registered_lineage(
        registry,
        experiment_id=experiment.experiment_id,
    )

    assert record.experiment_id == experiment.experiment_id


def test_lineage_with_missing_experiment():
    registry = AlphaLineageRegistry()
    candidate = make_candidate()

    with pytest.raises(
        AlphaExperimentNotFoundError
    ):
        registry.create_lineage(
            candidate=candidate,
            experiment_id="alpha_experiment_missing",
        )


def test_validation_candidate_mismatch():
    registry = AlphaLineageRegistry()

    candidate = make_candidate(
        "candidate_one"
    )

    other = make_candidate(
        "candidate_two"
    )

    validation = make_validation_result(
        other.candidate_id
    )

    with pytest.raises(
        AlphaLineageError
    ):
        registry.create_lineage(
            candidate=candidate,
            validation_results=[
                validation
            ],
        )


# ============================================================
# Query by Experiment
# ============================================================


def test_by_experiment():
    registry = AlphaLineageRegistry()

    experiment = create_registered_experiment(
        registry
    )

    _, record = create_registered_lineage(
        registry,
        experiment_id=experiment.experiment_id,
    )

    result = registry.by_experiment(
        experiment.experiment_id
    )

    assert result == [record]


def test_by_missing_experiment_returns_empty():
    registry = AlphaLineageRegistry()

    assert registry.by_experiment(
        "experiment_missing"
    ) == []


# ============================================================
# Query by Run
# ============================================================


def test_by_run():
    registry = AlphaLineageRegistry()

    _, record = create_registered_lineage(
        registry,
        run_id="run_777",
    )

    result = registry.by_run(
        "run_777"
    )

    assert result == [record]


def test_by_missing_run_returns_empty():
    registry = AlphaLineageRegistry()

    assert registry.by_run(
        "run_missing"
    ) == []


# ============================================================
# Query by Dataset
# ============================================================


def test_by_dataset():
    registry = AlphaLineageRegistry()

    _, record = create_registered_lineage(
        registry
    )

    result = registry.by_dataset(
        "dataset_001"
    )

    assert result == [record]


def test_by_missing_dataset_returns_empty():
    registry = AlphaLineageRegistry()

    assert registry.by_dataset(
        "dataset_missing"
    ) == []


# ============================================================
# Query by Feature
# ============================================================


def test_by_feature():
    registry = AlphaLineageRegistry()

    _, record = create_registered_lineage(
        registry,
        feature_ids=[
            "feature_special"
        ],
    )

    result = registry.by_feature(
        "feature_special"
    )

    assert result == [record]


def test_by_missing_feature_returns_empty():
    registry = AlphaLineageRegistry()

    assert registry.by_feature(
        "feature_missing"
    ) == []


# ============================================================
# Query by Combination
# ============================================================


def test_by_combination():
    registry = AlphaLineageRegistry()

    _, record = create_registered_lineage(
        registry,
        combination_ids=[
            "combination_special"
        ],
    )

    result = registry.by_combination(
        "combination_special"
    )

    assert result == [record]


def test_by_missing_combination_returns_empty():
    registry = AlphaLineageRegistry()

    assert registry.by_combination(
        "combination_missing"
    ) == []


# ============================================================
# Parent / Child Lineage
# ============================================================


def test_children_of():
    registry = AlphaLineageRegistry()

    parent, parent_record = create_registered_lineage(
        registry,
        name="parent_alpha",
    )

    _, child_record = create_registered_lineage(
        registry,
        name="child_alpha",
        parent_candidate_ids=[
            parent.candidate_id
        ],
    )

    result = registry.children_of(
        parent.candidate_id
    )

    assert child_record in result
    assert parent_record not in result


def test_parents_of():
    registry = AlphaLineageRegistry()

    parent, parent_record = create_registered_lineage(
        registry,
        name="parent_alpha",
    )

    child, _ = create_registered_lineage(
        registry,
        name="child_alpha",
        parent_candidate_ids=[
            parent.candidate_id
        ],
    )

    result = registry.parents_of(
        child.candidate_id
    )

    assert result == [parent_record]


def test_children_of_missing_returns_empty():
    registry = AlphaLineageRegistry()

    assert registry.children_of(
        "alpha_missing"
    ) == []


def test_parents_of_missing_raises():
    registry = AlphaLineageRegistry()

    with pytest.raises(
        AlphaLineageNotFoundError
    ):
        registry.parents_of(
            "alpha_missing"
        )


# ============================================================
# Experiment Fingerprints
# ============================================================


def test_experiment_fingerprint():
    registry = AlphaLineageRegistry()

    experiment = create_registered_experiment(
        registry
    )

    fingerprint = registry.experiment_fingerprint(
        experiment.experiment_id
    )

    assert isinstance(
        fingerprint,
        str,
    )
    assert fingerprint


def test_missing_experiment_fingerprint():
    registry = AlphaLineageRegistry()

    with pytest.raises(
        AlphaExperimentNotFoundError
    ):
        registry.experiment_fingerprint(
            "experiment_missing"
        )


def test_experiment_set_fingerprint_empty():
    registry = AlphaLineageRegistry()

    first = registry.experiment_set_fingerprint()
    second = registry.experiment_set_fingerprint()

    assert first == second
    assert isinstance(first, str)
    assert first


def test_experiment_set_fingerprint_deterministic():
    first = AlphaLineageRegistry()
    second = AlphaLineageRegistry()

    first.create_experiment(
        name="exp_1",
        dataset_ids=["dataset_001"],
        seed=1,
    )

    first.create_experiment(
        name="exp_2",
        dataset_ids=["dataset_002"],
        seed=2,
    )

    second.create_experiment(
        name="exp_2",
        dataset_ids=["dataset_002"],
        seed=2,
    )

    second.create_experiment(
        name="exp_1",
        dataset_ids=["dataset_001"],
        seed=1,
    )

    assert (
        first.experiment_set_fingerprint()
        ==
        second.experiment_set_fingerprint()
    )


# ============================================================
# Lineage Fingerprints
# ============================================================


def test_lineage_set_fingerprint_empty():
    registry = AlphaLineageRegistry()

    first = registry.lineage_set_fingerprint()
    second = registry.lineage_set_fingerprint()

    assert first == second
    assert isinstance(first, str)
    assert first


def test_lineage_set_fingerprint_deterministic():
    first = AlphaLineageRegistry()
    second = AlphaLineageRegistry()

    record_one = make_lineage_record(
        "alpha_001"
    )

    record_two = make_lineage_record(
        "alpha_002"
    )

    first.register_lineage(
        record_one
    )
    first.register_lineage(
        record_two
    )

    second.register_lineage(
        record_two
    )
    second.register_lineage(
        record_one
    )

    assert (
        first.lineage_set_fingerprint()
        ==
        second.lineage_set_fingerprint()
    )


def test_full_fingerprint_changes():
    registry = AlphaLineageRegistry()

    before = registry.full_fingerprint()

    registry.register_lineage(
        make_lineage_record()
    )

    after = registry.full_fingerprint()

    assert before != after


# ============================================================
# Snapshot
# ============================================================


def test_snapshot_empty():
    registry = AlphaLineageRegistry()

    snapshot = registry.snapshot()

    assert snapshot.lineage_count == 0
    assert snapshot.experiment_count == 0
    assert snapshot.audit_count == 0
    assert snapshot.candidate_ids == []
    assert snapshot.experiment_ids == []


def test_snapshot_with_lineage():
    registry = AlphaLineageRegistry()
    record = make_lineage_record()

    registry.register_lineage(
        record
    )

    snapshot = registry.snapshot()

    assert snapshot.lineage_count == 1
    assert record.candidate_id in snapshot.candidate_ids


def test_snapshot_with_experiment():
    registry = AlphaLineageRegistry()

    experiment = create_registered_experiment(
        registry
    )

    snapshot = registry.snapshot()

    assert snapshot.experiment_count == 1
    assert (
        experiment.experiment_id
        in snapshot.experiment_ids
    )


def test_snapshot_fingerprint():
    registry = AlphaLineageRegistry()

    registry.register_lineage(
        make_lineage_record()
    )

    snapshot = registry.snapshot()

    assert isinstance(
        snapshot.fingerprint,
        str,
    )
    assert snapshot.fingerprint


# ============================================================
# Audit
# ============================================================


def test_register_lineage_creates_audit():
    registry = AlphaLineageRegistry()
    record = make_lineage_record()

    registry.register_lineage(
        record
    )

    audit = registry.audit_log()

    assert len(audit) == 1
    assert audit[0].object_id == record.candidate_id


def test_register_experiment_creates_audit():
    registry = AlphaLineageRegistry()
    experiment = make_experiment()

    registry.register_experiment(
        experiment
    )

    audit = registry.audit_log()

    assert len(audit) == 1
    assert audit[0].object_id == experiment.experiment_id


def test_create_lineage_creates_audit():
    registry = AlphaLineageRegistry()

    create_registered_lineage(
        registry
    )

    assert len(
        registry.audit_log()
    ) == 1


def test_create_experiment_creates_audit():
    registry = AlphaLineageRegistry()

    create_registered_experiment(
        registry
    )

    assert len(
        registry.audit_log()
    ) == 1


def test_audit_returns_copy():
    registry = AlphaLineageRegistry()

    registry.register_lineage(
        make_lineage_record()
    )

    audit = registry.audit_log()
    audit.clear()

    assert len(
        registry.audit_log()
    ) == 1


# ============================================================
# Summary
# ============================================================


def test_summary_with_lineage():
    registry = AlphaLineageRegistry()

    registry.register_lineage(
        make_lineage_record()
    )

    summary = registry.summary()

    assert summary["lineage_count"] == 1
    assert summary["experiment_count"] == 0
    assert summary["audit_count"] == 1


def test_summary_with_experiment():
    registry = AlphaLineageRegistry()

    create_registered_experiment(
        registry
    )

    summary = registry.summary()

    assert summary["lineage_count"] == 0
    assert summary["experiment_count"] == 1
    assert summary["audit_count"] == 1


def test_summary_with_both():
    registry = AlphaLineageRegistry()

    experiment = create_registered_experiment(
        registry
    )

    create_registered_lineage(
        registry,
        experiment_id=experiment.experiment_id,
    )

    summary = registry.summary()

    assert summary["lineage_count"] == 1
    assert summary["experiment_count"] == 1
    assert summary["audit_count"] == 2
    assert summary["research_only"] is True
    assert summary["real_money_execution"] is False


def test_summary_fingerprint():
    registry = AlphaLineageRegistry()

    summary = registry.summary()

    assert (
        summary["fingerprint"]
        == registry.full_fingerprint()
    )


# ============================================================
# Records / Experiments Copies
# ============================================================


def test_records_returns_copy():
    registry = AlphaLineageRegistry()

    registry.register_lineage(
        make_lineage_record()
    )

    records = registry.records()
    records.clear()

    assert registry.lineage_count() == 1


def test_experiments_returns_copy():
    registry = AlphaLineageRegistry()

    create_registered_experiment(
        registry
    )

    experiments = registry.experiments()
    experiments.clear()

    assert registry.experiment_count() == 1


# ============================================================
# Isolation
# ============================================================


def test_registries_are_independent():
    first = AlphaLineageRegistry()
    second = AlphaLineageRegistry()

    first.register_lineage(
        make_lineage_record()
    )

    assert first.lineage_count() == 1
    assert second.lineage_count() == 0


# ============================================================
# Reproducibility
# ============================================================


def test_identical_experiment_reproducible():
    first = AlphaLineageRegistry()
    second = AlphaLineageRegistry()

    exp_one = first.create_experiment(
        name="reproducible_test",
        dataset_ids=[
            "dataset_001"
        ],
        feature_ids=[
            "feature_001"
        ],
        seed=42,
        configuration={
            "window": 20
        },
    )

    exp_two = second.create_experiment(
        name="reproducible_test",
        dataset_ids=[
            "dataset_001"
        ],
        feature_ids=[
            "feature_001"
        ],
        seed=42,
        configuration={
            "window": 20
        },
    )

    assert exp_one.fingerprint == exp_two.fingerprint
    assert exp_one.experiment_id == exp_two.experiment_id


def test_experiment_seed_changes_identity():
    first = AlphaLineageRegistry()
    second = AlphaLineageRegistry()

    exp_one = first.create_experiment(
        name="seed_test",
        dataset_ids=[
            "dataset_001"
        ],
        seed=1,
    )

    exp_two = second.create_experiment(
        name="seed_test",
        dataset_ids=[
            "dataset_001"
        ],
        seed=2,
    )

    assert exp_one.fingerprint != exp_two.fingerprint


def test_dataset_changes_experiment_identity():
    first = AlphaLineageRegistry()
    second = AlphaLineageRegistry()

    exp_one = first.create_experiment(
        name="dataset_test",
        dataset_ids=[
            "dataset_001"
        ],
        seed=42,
    )

    exp_two = second.create_experiment(
        name="dataset_test",
        dataset_ids=[
            "dataset_002"
        ],
        seed=42,
    )

    assert exp_one.fingerprint != exp_two.fingerprint


def test_lineage_preserves_research_chain():
    registry = AlphaLineageRegistry()

    experiment = create_registered_experiment(
        registry
    )

    candidate, record = create_registered_lineage(
        registry,
        experiment_id=experiment.experiment_id,
        run_id="run_chain_001",
    )

    assert record.candidate_id == candidate.candidate_id
    assert record.feature_ids
    assert record.combination_ids
    assert record.dataset_ids
    assert record.dataset_fingerprints
    assert record.validation_fingerprints
    assert record.experiment_id == experiment.experiment_id
    assert record.run_id == "run_chain_001"


# ============================================================
# Exception Hierarchy
# ============================================================


def test_lineage_error_base():
    error = AlphaLineageError(
        "test"
    )

    assert isinstance(
        error,
        Exception,
    )


def test_lineage_not_found_error():
    error = AlphaLineageNotFoundError(
        "test"
    )

    assert isinstance(
        error,
        AlphaLineageError,
    )


def test_experiment_not_found_error():
    error = AlphaExperimentNotFoundError(
        "test"
    )

    assert isinstance(
        error,
        AlphaLineageError,
    )


# ============================================================
# Safety
# ============================================================


def test_research_safety_flags():
    registry = AlphaLineageRegistry()

    summary = registry.summary()

    assert summary["research_only"] is True
    assert summary["real_money_execution"] is False