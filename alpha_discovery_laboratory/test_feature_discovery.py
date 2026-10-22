from __future__ import annotations

import pytest

from alpha_discovery_laboratory.feature_discovery import (
    CombinationAlreadyExistsError,
    CombinationNotFoundError,
    FeatureAlreadyExistsError,
    FeatureDiscoveryEngine,
    FeatureNotFoundError,
)
from alpha_discovery_laboratory.models import (
    VERSION,
    AlphaValidationError,
    CombinationOperator,
    FeatureCombination,
    FeatureTransform,
    FeatureType,
    MarketFeature,
)


# ============================================================
# Helpers
# ============================================================


def make_engine() -> FeatureDiscoveryEngine:
    return FeatureDiscoveryEngine()


def make_feature(
    name: str = "momentum_20",
    feature_type: FeatureType = FeatureType.MOMENTUM,
    source: str = "market_data",
    dataset_id: str = "dataset_001",
) -> MarketFeature:
    return MarketFeature(
        name=name,
        feature_type=feature_type,
        source=source,
        transform=FeatureTransform.ZSCORE,
        parameters={"window": 20},
        description=f"Feature: {name}",
        dataset_id=dataset_id,
        dataset_fingerprint=f"{dataset_id}_fp",
        metadata={"asset": "BTC"},
    )


def register_two_features(
    engine: FeatureDiscoveryEngine,
) -> tuple[MarketFeature, MarketFeature]:
    first = make_feature(
        name="momentum_20",
        feature_type=FeatureType.MOMENTUM,
        source="market_data",
    )

    second = make_feature(
        name="volume_zscore",
        feature_type=FeatureType.VOLUME,
        source="market_data",
    )

    engine.register_feature(first)
    engine.register_feature(second)

    return first, second


def make_combination(
    first: MarketFeature,
    second: MarketFeature,
    name: str = "momentum_volume_spread",
) -> FeatureCombination:
    return FeatureCombination(
        name=name,
        feature_ids=[
            first.feature_id,
            second.feature_id,
        ],
        operator=CombinationOperator.SPREAD,
        weights=[],
        parameters={"window": 20},
        description="Momentum-volume spread",
        metadata={"asset": "BTC"},
    )


# ============================================================
# Basic Engine
# ============================================================


def test_version():
    assert FeatureDiscoveryEngine.VERSION == VERSION


def test_empty_engine():
    engine = make_engine()

    assert engine.feature_count() == 0
    assert engine.combination_count() == 0
    assert engine.audit_log() == []


def test_summary_empty():
    engine = make_engine()

    summary = engine.summary()

    assert summary["version"] == VERSION
    assert summary["feature_count"] == 0
    assert summary["combination_count"] == 0
    assert summary["research_only"] is True
    assert summary["real_money_execution"] is False


# ============================================================
# Feature Registration
# ============================================================


def test_register_feature():
    engine = make_engine()
    feature = make_feature()

    result = engine.register_feature(feature)

    assert result is feature
    assert engine.feature_count() == 1


def test_register_feature_exists():
    engine = make_engine()
    feature = make_feature()

    engine.register_feature(feature)

    assert engine.exists(feature.feature_id)


def test_get_feature():
    engine = make_engine()
    feature = make_feature()

    engine.register_feature(feature)

    result = engine.get(feature.feature_id)

    assert result == feature


def test_get_unknown_feature():
    engine = make_engine()

    with pytest.raises(FeatureNotFoundError):
        engine.get("feature_missing")


def test_get_by_fingerprint():
    engine = make_engine()
    feature = make_feature()

    engine.register_feature(feature)

    result = engine.get_by_fingerprint(
        feature.fingerprint
    )

    assert result == feature


def test_get_unknown_fingerprint():
    engine = make_engine()

    with pytest.raises(FeatureNotFoundError):
        engine.get_by_fingerprint(
            "missing_fingerprint"
        )


def test_register_same_feature_allow_existing():
    engine = make_engine()
    feature = make_feature()

    first = engine.register_feature(feature)
    second = engine.register_feature(feature)

    assert first == second
    assert engine.feature_count() == 1


def test_register_same_feature_disallow_existing():
    engine = make_engine()
    feature = make_feature()

    engine.register_feature(feature)

    with pytest.raises(
        FeatureAlreadyExistsError
    ):
        engine.register_feature(
            feature,
            allow_existing=False,
        )


def test_register_invalid_feature_type():
    engine = make_engine()

    with pytest.raises(
        AlphaValidationError
    ):
        engine.register_feature(
            "not-a-feature"  # type: ignore[arg-type]
        )


# ============================================================
# Feature Creation
# ============================================================


def test_create_feature():
    engine = make_engine()

    feature = engine.create_feature(
        name="btc_momentum",
        feature_type=FeatureType.MOMENTUM,
        source="market_data",
        transform=FeatureTransform.ZSCORE,
        parameters={"window": 20},
        dataset_id="dataset_001",
        dataset_fingerprint="fp_001",
    )

    assert engine.feature_count() == 1
    assert engine.exists(feature.feature_id)


def test_create_feature_preserves_properties():
    engine = make_engine()

    feature = engine.create_feature(
        name="btc_volume",
        feature_type=FeatureType.VOLUME,
        source="exchange",
        transform=FeatureTransform.RAW,
        parameters={"window": 10},
        description="BTC volume",
        dataset_id="dataset_002",
        dataset_fingerprint="fp_002",
        metadata={"asset": "BTC"},
    )

    assert feature.name == "btc_volume"
    assert feature.feature_type == FeatureType.VOLUME
    assert feature.source == "exchange"
    assert feature.dataset_id == "dataset_002"
    assert feature.metadata["asset"] == "BTC"


def test_create_feature_empty_name():
    engine = make_engine()

    with pytest.raises(
        AlphaValidationError
    ):
        engine.create_feature(
            name="",
            feature_type=FeatureType.MOMENTUM,
            source="market_data",
        )


def test_create_feature_empty_source():
    engine = make_engine()

    with pytest.raises(
        AlphaValidationError
    ):
        engine.create_feature(
            name="momentum",
            feature_type=FeatureType.MOMENTUM,
            source="",
        )


# ============================================================
# Feature Listing / Filtering
# ============================================================


def test_list_features():
    engine = make_engine()

    register_two_features(engine)

    features = engine.list_features()

    assert len(features) == 2


def test_features_by_type():
    engine = make_engine()

    first, _ = register_two_features(engine)

    results = engine.features_by_type(
        FeatureType.MOMENTUM
    )

    assert results == [first]


def test_features_by_source():
    engine = make_engine()

    register_two_features(engine)

    results = engine.features_by_source(
        "market_data"
    )

    assert len(results) == 2


def test_features_by_dataset():
    engine = make_engine()

    first = make_feature(
        name="momentum",
        dataset_id="dataset_A",
    )

    second = make_feature(
        name="volume",
        feature_type=FeatureType.VOLUME,
        dataset_id="dataset_B",
    )

    engine.register_feature(first)
    engine.register_feature(second)

    results = engine.features_by_dataset(
        "dataset_A"
    )

    assert results == [first]


# ============================================================
# Batch Feature Registration
# ============================================================


def test_register_features():
    engine = make_engine()

    features = [
        make_feature(
            name="momentum_20"
        ),
        make_feature(
            name="momentum_50"
        ),
        make_feature(
            name="volume_20",
            feature_type=FeatureType.VOLUME,
        ),
    ]

    result = engine.register_features(
        features
    )

    assert len(result) == 3
    assert engine.feature_count() == 3


def test_register_features_skips_duplicates():
    engine = make_engine()

    feature = make_feature()

    result = engine.register_features(
        [
            feature,
            feature,
        ]
    )

    assert len(result) == 2
    assert engine.feature_count() == 1


# ============================================================
# Combination Registration
# ============================================================


def test_register_combination():
    engine = make_engine()

    first, second = register_two_features(
        engine
    )

    combination = make_combination(
        first,
        second,
    )

    result = engine.register_combination(
        combination
    )

    assert result is combination
    assert engine.combination_count() == 1


def test_combination_exists():
    engine = make_engine()

    first, second = register_two_features(
        engine
    )

    combination = make_combination(
        first,
        second,
    )

    engine.register_combination(
        combination
    )

    assert engine.combination_exists(
        combination.combination_id
    )


def test_get_combination():
    engine = make_engine()

    first, second = register_two_features(
        engine
    )

    combination = make_combination(
        first,
        second,
    )

    engine.register_combination(
        combination
    )

    result = engine.get_combination(
        combination.combination_id
    )

    assert result == combination


def test_get_unknown_combination():
    engine = make_engine()

    with pytest.raises(
        CombinationNotFoundError
    ):
        engine.get_combination(
            "combination_missing"
        )


def test_register_duplicate_combination():
    engine = make_engine()

    first, second = register_two_features(
        engine
    )

    combination = make_combination(
        first,
        second,
    )

    engine.register_combination(
        combination
    )

    result = engine.register_combination(
        combination
    )

    assert result == combination
    assert engine.combination_count() == 1


def test_register_duplicate_combination_disallowed():
    engine = make_engine()

    first, second = register_two_features(
        engine
    )

    combination = make_combination(
        first,
        second,
    )

    engine.register_combination(
        combination
    )

    with pytest.raises(
        CombinationAlreadyExistsError
    ):
        engine.register_combination(
            combination,
            allow_existing=False,
        )


def test_register_invalid_combination_type():
    engine = make_engine()

    with pytest.raises(
        AlphaValidationError
    ):
        engine.register_combination(
            "invalid"  # type: ignore[arg-type]
        )


# ============================================================
# Combination Source Validation
# ============================================================


def test_combination_requires_registered_features():
    engine = make_engine()

    combination = FeatureCombination(
        name="invalid_sources",
        feature_ids=[
            "feature_missing_1",
            "feature_missing_2",
        ],
        operator=CombinationOperator.SPREAD,
    )

    with pytest.raises(
        FeatureNotFoundError
    ):
        engine.register_combination(
            combination
        )


# ============================================================
# Combination Creation
# ============================================================


def test_create_combination():
    engine = make_engine()

    first, second = register_two_features(
        engine
    )

    combination = engine.create_combination(
        name="created_spread",
        feature_ids=[
            first.feature_id,
            second.feature_id,
        ],
        operator=CombinationOperator.SPREAD,
        description="Test spread",
    )

    assert engine.combination_count() == 1
    assert engine.combination_exists(
        combination.combination_id
    )


def test_create_combination_with_weights():
    engine = make_engine()

    first, second = register_two_features(
        engine
    )

    combination = engine.create_combination(
        name="weighted_signal",
        feature_ids=[
            first.feature_id,
            second.feature_id,
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
# Combination Queries
# ============================================================


def test_list_combinations():
    engine = make_engine()

    first, second = register_two_features(
        engine
    )

    engine.register_combination(
        make_combination(
            first,
            second,
        )
    )

    assert len(
        engine.list_combinations()
    ) == 1


def test_combinations_for_feature():
    engine = make_engine()

    first, second = register_two_features(
        engine
    )

    combination = make_combination(
        first,
        second,
    )

    engine.register_combination(
        combination
    )

    results = engine.combinations_for_feature(
        first.feature_id
    )

    assert results == [combination]


def test_combinations_for_unused_feature():
    engine = make_engine()

    first, second = register_two_features(
        engine
    )

    third = make_feature(
        name="volatility",
        feature_type=FeatureType.VOLATILITY,
    )

    engine.register_feature(third)

    engine.register_combination(
        make_combination(
            first,
            second,
        )
    )

    assert (
        engine.combinations_for_feature(
            third.feature_id
        )
        == []
    )


# ============================================================
# Pairwise Discovery
# ============================================================


def test_discover_pairwise_two_features():
    engine = make_engine()

    register_two_features(engine)

    result = engine.discover_pairwise(
        operators=[
            CombinationOperator.SPREAD,
        ]
    )

    assert result.combination_count == 1
    assert engine.combination_count() == 1


def test_discover_pairwise_multiple_operators():
    engine = make_engine()

    register_two_features(engine)

    result = engine.discover_pairwise(
        operators=[
            CombinationOperator.SPREAD,
            CombinationOperator.RATIO,
            CombinationOperator.INTERACTION,
        ]
    )

    assert result.combination_count == 3
    assert engine.combination_count() == 3


def test_discover_pairwise_three_features():
    engine = make_engine()

    engine.register_feature(
        make_feature(
            name="momentum",
            feature_type=FeatureType.MOMENTUM,
        )
    )

    engine.register_feature(
        make_feature(
            name="volume",
            feature_type=FeatureType.VOLUME,
        )
    )

    engine.register_feature(
        make_feature(
            name="volatility",
            feature_type=FeatureType.VOLATILITY,
        )
    )

    result = engine.discover_pairwise(
        operators=[
            CombinationOperator.SPREAD,
        ]
    )

    assert result.combination_count == 3


def test_discover_pairwise_duplicate_run():
    engine = make_engine()

    register_two_features(engine)

    first = engine.discover_pairwise(
        operators=[
            CombinationOperator.SPREAD,
        ]
    )

    second = engine.discover_pairwise(
        operators=[
            CombinationOperator.SPREAD,
        ]
    )

    assert first.combination_count == 1
    assert second.combination_count == 0
    assert second.skipped_duplicates == 1
    assert engine.combination_count() == 1


def test_discover_pairwise_selected_features():
    engine = make_engine()

    first, second = register_two_features(
        engine
    )

    third = make_feature(
        name="volatility",
        feature_type=FeatureType.VOLATILITY,
    )

    engine.register_feature(third)

    result = engine.discover_pairwise(
        feature_ids=[
            first.feature_id,
            second.feature_id,
        ],
        operators=[
            CombinationOperator.SPREAD,
        ],
    )

    assert result.combination_count == 1


def test_discover_pairwise_unknown_feature():
    engine = make_engine()

    first, _ = register_two_features(
        engine
    )

    with pytest.raises(
        FeatureNotFoundError
    ):
        engine.discover_pairwise(
            feature_ids=[
                first.feature_id,
                "feature_missing",
            ]
        )


def test_discover_pairwise_requires_two_features():
    engine = make_engine()

    feature = make_feature()

    engine.register_feature(feature)

    with pytest.raises(
        AlphaValidationError
    ):
        engine.discover_pairwise(
            feature_ids=[
                feature.feature_id,
            ]
        )


def test_discover_pairwise_default_operators():
    engine = make_engine()

    register_two_features(engine)

    result = engine.discover_pairwise()

    assert result.combination_count == 3


# ============================================================
# Fingerprints
# ============================================================


def test_feature_set_fingerprint_empty():
    engine = make_engine()

    first = engine.feature_set_fingerprint()
    second = engine.feature_set_fingerprint()

    assert first == second
    assert isinstance(first, str)
    assert first


def test_feature_set_fingerprint_deterministic():
    first_engine = make_engine()
    second_engine = make_engine()

    first_feature = make_feature(
        "momentum"
    )

    second_feature = make_feature(
        "volume",
        feature_type=FeatureType.VOLUME,
    )

    first_engine.register_feature(
        first_feature
    )
    first_engine.register_feature(
        second_feature
    )

    second_engine.register_feature(
        second_feature
    )
    second_engine.register_feature(
        first_feature
    )

    assert (
        first_engine.feature_set_fingerprint()
        == second_engine.feature_set_fingerprint()
    )


def test_combination_set_fingerprint():
    engine = make_engine()

    first, second = register_two_features(
        engine
    )

    engine.register_combination(
        make_combination(
            first,
            second,
        )
    )

    fingerprint = (
        engine.combination_set_fingerprint()
    )

    assert isinstance(fingerprint, str)
    assert fingerprint


def test_full_fingerprint_changes():
    engine = make_engine()

    before = engine.full_fingerprint()

    engine.register_feature(
        make_feature()
    )

    after = engine.full_fingerprint()

    assert before != after


# ============================================================
# Snapshot
# ============================================================


def test_snapshot_empty():
    engine = make_engine()

    snapshot = engine.snapshot()

    assert snapshot.feature_count == 0
    assert snapshot.combination_count == 0
    assert snapshot.audit_count == 0


def test_snapshot_after_registration():
    engine = make_engine()

    first, second = register_two_features(
        engine
    )

    engine.register_combination(
        make_combination(
            first,
            second,
        )
    )

    snapshot = engine.snapshot()

    assert snapshot.feature_count == 2
    assert snapshot.combination_count == 1
    assert snapshot.audit_count == 3

    assert len(snapshot.feature_ids) == 2
    assert len(
        snapshot.combination_ids
    ) == 1


def test_snapshot_fingerprint():
    engine = make_engine()

    register_two_features(engine)

    snapshot = engine.snapshot()

    assert isinstance(
        snapshot.fingerprint,
        str,
    )

    assert snapshot.fingerprint


# ============================================================
# Audit
# ============================================================


def test_feature_registration_creates_audit():
    engine = make_engine()

    feature = make_feature()

    engine.register_feature(feature)

    audit = engine.audit_log()

    assert len(audit) == 1
    assert audit[0].action == "register"
    assert audit[0].object_type == "feature"
    assert (
        audit[0].object_id
        == feature.feature_id
    )


def test_combination_registration_creates_audit():
    engine = make_engine()

    first, second = register_two_features(
        engine
    )

    combination = make_combination(
        first,
        second,
    )

    engine.register_combination(
        combination
    )

    audit = engine.audit_log()

    assert len(audit) == 3

    assert (
        audit[-1].object_type
        == "combination"
    )

    assert (
        audit[-1].object_id
        == combination.combination_id
    )


def test_audit_returns_copy():
    engine = make_engine()

    engine.register_feature(
        make_feature()
    )

    audit = engine.audit_log()

    audit.clear()

    assert len(
        engine.audit_log()
    ) == 1


# ============================================================
# Summary
# ============================================================


def test_summary_after_data():
    engine = make_engine()

    first, second = register_two_features(
        engine
    )

    engine.register_combination(
        make_combination(
            first,
            second,
        )
    )

    summary = engine.summary()

    assert summary["version"] == VERSION
    assert summary["feature_count"] == 2
    assert summary["combination_count"] == 1
    assert summary["audit_count"] == 3

    assert summary["research_only"] is True

    assert (
        summary["real_money_execution"]
        is False
    )


# ============================================================
# Isolation
# ============================================================


def test_engines_are_independent():
    first = make_engine()
    second = make_engine()

    first.register_feature(
        make_feature()
    )

    assert first.feature_count() == 1
    assert second.feature_count() == 0