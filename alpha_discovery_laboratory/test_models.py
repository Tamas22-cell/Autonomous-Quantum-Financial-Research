from __future__ import annotations

import pytest

from alpha_discovery_laboratory.models import (
    VERSION,
    AlphaDiscoveryError,
    AlphaDiscoveryExperiment,
    AlphaDiscoverySummary,
    AlphaLineageRecord,
    AlphaValidationError,
    CandidateStatus,
    CombinationOperator,
    FeatureCombination,
    FeatureTransform,
    FeatureType,
    MarketFeature,
    PerformanceMetrics,
    RejectionReason,
    StrategyCandidate,
    StrategyDirection,
    ValidationResult,
    ValidationStage,
    ValidationStatus,
    canonical_json,
    stable_hash,
)


# ============================================================
# Helpers
# ============================================================


def make_feature(
    name: str = "momentum_20",
) -> MarketFeature:
    return MarketFeature(
        name=name,
        feature_type=FeatureType.MOMENTUM,
        source="market_data",
        transform=FeatureTransform.ZSCORE,
        parameters={"window": 20},
        description="20-period momentum feature",
        dataset_id="dataset_001",
        dataset_fingerprint="dataset_fp_001",
        metadata={"asset": "BTC"},
    )


def make_combination() -> FeatureCombination:
    return FeatureCombination(
        name="momentum_volume_spread",
        feature_ids=[
            "feature_001",
            "feature_002",
        ],
        operator=CombinationOperator.SPREAD,
        weights=[0.6, 0.4],
        parameters={"window": 20},
        description="Momentum-volume spread",
        metadata={"asset": "BTC"},
    )


def make_candidate() -> StrategyCandidate:
    return StrategyCandidate(
        name="btc_momentum_alpha",
        feature_ids=[
            "feature_001",
            "feature_002",
        ],
        combination_ids=[
            "combination_001",
        ],
        direction=StrategyDirection.LONG_SHORT,
        entry_rule="signal > 1.0",
        exit_rule="signal < 0.0",
        position_rule="risk_scaled",
        parameters={
            "entry_threshold": 1.0,
            "exit_threshold": 0.0,
        },
        hypothesis_id="hypothesis_001",
        parent_candidate_ids=[],
        metadata={"asset": "BTC"},
    )


def make_metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        total_return=0.25,
        annualized_return=0.18,
        volatility=0.15,
        sharpe_ratio=1.20,
        sortino_ratio=1.50,
        calmar_ratio=1.10,
        max_drawdown=0.12,
        win_rate=0.55,
        profit_factor=1.40,
        turnover=0.30,
        execution_cost=0.01,
        trade_count=120,
        metadata={"period": "test"},
    )


# ============================================================
# Version
# ============================================================


def test_version():
    assert VERSION == "v6.0.3"


# ============================================================
# Enum Tests
# ============================================================


@pytest.mark.parametrize(
    "member",
    list(FeatureType),
)
def test_feature_type_values(member):
    assert isinstance(member.value, str)
    assert member.value


@pytest.mark.parametrize(
    "member",
    list(FeatureTransform),
)
def test_feature_transform_values(member):
    assert isinstance(member.value, str)
    assert member.value


@pytest.mark.parametrize(
    "member",
    list(CombinationOperator),
)
def test_combination_operator_values(member):
    assert isinstance(member.value, str)
    assert member.value


@pytest.mark.parametrize(
    "member",
    list(StrategyDirection),
)
def test_strategy_direction_values(member):
    assert isinstance(member.value, str)
    assert member.value


@pytest.mark.parametrize(
    "member",
    list(CandidateStatus),
)
def test_candidate_status_values(member):
    assert isinstance(member.value, str)
    assert member.value


@pytest.mark.parametrize(
    "member",
    list(ValidationStage),
)
def test_validation_stage_values(member):
    assert isinstance(member.value, str)
    assert member.value


@pytest.mark.parametrize(
    "member",
    list(ValidationStatus),
)
def test_validation_status_values(member):
    assert isinstance(member.value, str)
    assert member.value


@pytest.mark.parametrize(
    "member",
    list(RejectionReason),
)
def test_rejection_reason_values(member):
    assert isinstance(member.value, str)
    assert member.value


# ============================================================
# Hashing / Canonical Serialization
# ============================================================


def test_canonical_json_is_deterministic():
    first = {
        "b": 2,
        "a": 1,
    }

    second = {
        "a": 1,
        "b": 2,
    }

    assert canonical_json(first) == canonical_json(
        second
    )


def test_stable_hash_is_deterministic():
    first = stable_hash(
        {
            "b": 2,
            "a": 1,
        }
    )

    second = stable_hash(
        {
            "a": 1,
            "b": 2,
        }
    )

    assert first == second


def test_stable_hash_changes_with_content():
    first = stable_hash(
        {"value": 1}
    )

    second = stable_hash(
        {"value": 2}
    )

    assert first != second


def test_stable_hash_returns_string():
    result = stable_hash(
        {"value": 1}
    )

    assert isinstance(result, str)
    assert result


# ============================================================
# MarketFeature
# ============================================================


def test_market_feature_creation():
    feature = make_feature()

    assert feature.name == "momentum_20"
    assert (
        feature.feature_type
        == FeatureType.MOMENTUM
    )
    assert feature.source == "market_data"


def test_market_feature_id_prefix():
    feature = make_feature()

    assert feature.feature_id.startswith(
        "feature_"
    )


def test_market_feature_id_is_deterministic():
    first = make_feature()
    second = make_feature()

    assert (
        first.feature_id
        == second.feature_id
    )


def test_market_feature_fingerprint_deterministic():
    first = make_feature()
    second = make_feature()

    assert (
        first.fingerprint
        == second.fingerprint
    )


def test_market_feature_different_name_changes_id():
    first = make_feature("momentum_20")
    second = make_feature("momentum_50")

    assert (
        first.feature_id
        != second.feature_id
    )


def test_market_feature_metadata_preserved():
    feature = make_feature()

    assert feature.metadata["asset"] == "BTC"


def test_market_feature_dataset_reference():
    feature = make_feature()

    assert feature.dataset_id == "dataset_001"
    assert (
        feature.dataset_fingerprint
        == "dataset_fp_001"
    )


# ============================================================
# FeatureCombination
# ============================================================


def test_feature_combination_creation():
    combination = make_combination()

    assert (
        combination.operator
        == CombinationOperator.SPREAD
    )

    assert len(
        combination.feature_ids
    ) == 2


def test_feature_combination_id_prefix():
    combination = make_combination()

    assert (
        combination.combination_id
        .startswith("combination_")
    )


def test_feature_combination_deterministic():
    first = make_combination()
    second = make_combination()

    assert (
        first.fingerprint
        == second.fingerprint
    )

    assert (
        first.combination_id
        == second.combination_id
    )


def test_feature_combination_requires_two_features():
    with pytest.raises(
        AlphaValidationError
    ):
        FeatureCombination(
            name="invalid",
            feature_ids=[
                "feature_001"
            ],
            operator=(
                CombinationOperator.ADD
            ),
        )


def test_feature_combination_weight_length_validation():
    with pytest.raises(
        AlphaValidationError
    ):
        FeatureCombination(
            name="invalid_weights",
            feature_ids=[
                "feature_001",
                "feature_002",
            ],
            operator=(
                CombinationOperator
                .WEIGHTED_SUM
            ),
            weights=[1.0],
        )


def test_feature_combination_valid_weights():
    combination = FeatureCombination(
        name="weighted",
        feature_ids=[
            "feature_001",
            "feature_002",
        ],
        operator=(
            CombinationOperator.WEIGHTED_SUM
        ),
        weights=[
            0.7,
            0.3,
        ],
    )

    assert combination.weights == [
        0.7,
        0.3,
    ]


# ============================================================
# StrategyCandidate
# ============================================================


def test_strategy_candidate_creation():
    candidate = make_candidate()

    assert (
        candidate.direction
        == StrategyDirection.LONG_SHORT
    )

    assert (
        candidate.status
        == CandidateStatus.GENERATED
    )


def test_strategy_candidate_id_prefix():
    candidate = make_candidate()

    assert candidate.candidate_id.startswith(
        "alpha_"
    )


def test_strategy_candidate_id_deterministic():
    first = make_candidate()
    second = make_candidate()

    assert (
        first.candidate_id
        == second.candidate_id
    )


def test_strategy_candidate_fingerprint_deterministic():
    first = make_candidate()
    second = make_candidate()

    assert (
        first.fingerprint
        == second.fingerprint
    )


def test_candidate_initially_not_accepted():
    candidate = make_candidate()

    assert candidate.accepted is False


def test_candidate_initially_not_rejected():
    candidate = make_candidate()

    assert candidate.rejected is False


def test_candidate_set_status():
    candidate = make_candidate()

    candidate.set_status(
        CandidateStatus.BACKTESTING
    )

    assert (
        candidate.status
        == CandidateStatus.BACKTESTING
    )


def test_candidate_accept():
    candidate = make_candidate()

    candidate.accept()

    assert (
        candidate.status
        == CandidateStatus.ACCEPTED
    )

    assert candidate.accepted is True
    assert candidate.rejected is False


def test_candidate_reject():
    candidate = make_candidate()

    candidate.reject(
        RejectionReason.OVERFIT,
        "Detected overfitting.",
    )

    assert (
        candidate.status
        == CandidateStatus.REJECTED
    )

    assert candidate.rejected is True
    assert candidate.accepted is False

    assert (
        candidate.rejection_reason
        == RejectionReason.OVERFIT
    )

    assert (
        candidate.rejection_details
        == "Detected overfitting."
    )


def test_candidate_reject_updates_timestamp():
    candidate = make_candidate()

    previous = candidate.updated_at

    candidate.reject(
        RejectionReason.LOW_SHARPE,
        "Sharpe below threshold.",
    )

    assert candidate.updated_at >= previous


def test_candidate_accept_updates_timestamp():
    candidate = make_candidate()

    previous = candidate.updated_at

    candidate.accept()

    assert candidate.updated_at >= previous


# ============================================================
# PerformanceMetrics
# ============================================================


def test_performance_metrics_creation():
    metrics = make_metrics()

    assert metrics.total_return == 0.25
    assert metrics.sharpe_ratio == 1.20
    assert metrics.trade_count == 120


def test_performance_metrics_fingerprint():
    first = make_metrics()
    second = make_metrics()

    assert (
        first.fingerprint
        == second.fingerprint
    )


def test_performance_metrics_change_changes_fingerprint():
    first = make_metrics()

    second = PerformanceMetrics(
        total_return=0.10,
        annualized_return=0.08,
        volatility=0.15,
        sharpe_ratio=0.50,
        sortino_ratio=0.60,
        calmar_ratio=0.40,
        max_drawdown=0.20,
        win_rate=0.50,
        profit_factor=1.05,
        turnover=0.30,
        execution_cost=0.01,
        trade_count=120,
    )

    assert (
        first.fingerprint
        != second.fingerprint
    )


# ============================================================
# ValidationResult
# ============================================================


def test_validation_result_passed():
    result = ValidationResult(
        candidate_id="alpha_001",
        stage=ValidationStage.BACKTEST,
        status=ValidationStatus.PASSED,
        metrics=make_metrics(),
        score=0.90,
        passed_checks=[
            "sharpe",
            "drawdown",
        ],
    )

    assert result.successful is True


def test_validation_result_failed():
    result = ValidationResult(
        candidate_id="alpha_001",
        stage=ValidationStage.BACKTEST,
        status=ValidationStatus.FAILED,
        metrics=make_metrics(),
        score=0.20,
        failed_checks=[
            "sharpe",
        ],
        rejection_reason=(
            RejectionReason.LOW_SHARPE
        ),
    )

    assert result.successful is False


def test_validation_result_fingerprint():
    first = ValidationResult(
        candidate_id="alpha_001",
        stage=ValidationStage.BACKTEST,
        status=ValidationStatus.PASSED,
        metrics=make_metrics(),
        score=0.90,
    )

    second = ValidationResult(
        candidate_id="alpha_001",
        stage=ValidationStage.BACKTEST,
        status=ValidationStatus.PASSED,
        metrics=make_metrics(),
        score=0.90,
    )

    assert (
        first.fingerprint
        == second.fingerprint
    )


# ============================================================
# AlphaLineageRecord
# ============================================================


def test_alpha_lineage_record_creation():
    record = AlphaLineageRecord(
        candidate_id="alpha_001",
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
    )

    assert (
        record.candidate_id
        == "alpha_001"
    )

    assert (
        record.experiment_id
        == "experiment_001"
    )


def test_alpha_lineage_fingerprint():
    first = AlphaLineageRecord(
        candidate_id="alpha_001",
        feature_ids=[
            "feature_001",
        ],
        combination_ids=[],
    )

    second = AlphaLineageRecord(
        candidate_id="alpha_001",
        feature_ids=[
            "feature_001",
        ],
        combination_ids=[],
    )

    assert (
        first.fingerprint
        == second.fingerprint
    )


# ============================================================
# AlphaDiscoveryExperiment
# ============================================================


def test_alpha_discovery_experiment_creation():
    experiment = AlphaDiscoveryExperiment(
        name="BTC Alpha Research",
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
        configuration={
            "seed": 42,
        },
        seed=42,
        run_id="run_001",
    )

    assert (
        experiment.name
        == "BTC Alpha Research"
    )

    assert experiment.seed == 42


def test_alpha_discovery_experiment_id_prefix():
    experiment = AlphaDiscoveryExperiment(
        name="BTC Alpha Research",
        dataset_ids=[
            "dataset_001",
        ],
    )

    assert (
        experiment.experiment_id
        .startswith(
            "alpha_experiment_"
        )
    )


def test_alpha_discovery_experiment_deterministic():
    first = AlphaDiscoveryExperiment(
        name="BTC Alpha Research",
        dataset_ids=[
            "dataset_001",
        ],
        seed=42,
    )

    second = AlphaDiscoveryExperiment(
        name="BTC Alpha Research",
        dataset_ids=[
            "dataset_001",
        ],
        seed=42,
    )

    assert (
        first.experiment_id
        == second.experiment_id
    )

    assert (
        first.fingerprint
        == second.fingerprint
    )


def test_alpha_discovery_experiment_different_seed():
    first = AlphaDiscoveryExperiment(
        name="BTC Alpha Research",
        dataset_ids=[
            "dataset_001",
        ],
        seed=42,
    )

    second = AlphaDiscoveryExperiment(
        name="BTC Alpha Research",
        dataset_ids=[
            "dataset_001",
        ],
        seed=99,
    )

    assert (
        first.fingerprint
        != second.fingerprint
    )


# ============================================================
# AlphaDiscoverySummary
# ============================================================


def test_alpha_discovery_summary():
    summary = AlphaDiscoverySummary(
        experiment_id="experiment_001",
        feature_count=10,
        combination_count=20,
        candidate_count=30,
        accepted_count=5,
        rejected_count=25,
        validation_count=120,
    )

    assert summary.feature_count == 10
    assert summary.candidate_count == 30
    assert summary.accepted_count == 5
    assert summary.rejected_count == 25

    assert summary.research_only is True

    assert (
        summary.real_money_execution
        is False
    )


# ============================================================
# Exception Hierarchy
# ============================================================


def test_validation_error_is_alpha_discovery_error():
    error = AlphaValidationError(
        "test"
    )

    assert isinstance(
        error,
        AlphaDiscoveryError,
    )