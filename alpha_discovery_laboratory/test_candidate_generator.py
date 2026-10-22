from __future__ import annotations

import pytest

from alpha_discovery_laboratory.candidate_generator import (
    CandidateAlreadyExistsError,
    CandidateGenerationError,
    CandidateNotFoundError,
    CandidateSourceError,
    StrategyCandidateGenerator,
)
from alpha_discovery_laboratory.feature_discovery import (
    CombinationNotFoundError,
    FeatureDiscoveryEngine,
)
from alpha_discovery_laboratory.models import (
    VERSION,
    AlphaValidationError,
    CandidateStatus,
    CombinationOperator,
    FeatureTransform,
    FeatureType,
    MarketFeature,
    RejectionReason,
    StrategyCandidate,
    StrategyDirection,
)


# ============================================================
# Helpers
# ============================================================


def make_feature_engine() -> FeatureDiscoveryEngine:
    return FeatureDiscoveryEngine()


def make_generator(
    feature_engine: FeatureDiscoveryEngine | None = None,
) -> StrategyCandidateGenerator:
    return StrategyCandidateGenerator(
        feature_discovery=feature_engine
    )


def make_feature(
    name: str,
    feature_type: FeatureType,
) -> MarketFeature:
    return MarketFeature(
        name=name,
        feature_type=feature_type,
        source="market_data",
        transform=FeatureTransform.ZSCORE,
        parameters={"window": 20},
        dataset_id="dataset_001",
        dataset_fingerprint="dataset_fp_001",
        metadata={"asset": "BTC"},
    )


def prepare_feature_engine() -> tuple[
    FeatureDiscoveryEngine,
    MarketFeature,
    MarketFeature,
]:
    engine = make_feature_engine()

    first = make_feature(
        "momentum_20",
        FeatureType.MOMENTUM,
    )

    second = make_feature(
        "volume_20",
        FeatureType.VOLUME,
    )

    engine.register_feature(first)
    engine.register_feature(second)

    return engine, first, second


def prepare_combination_engine():
    engine, first, second = prepare_feature_engine()

    combination = engine.create_combination(
        name="momentum_volume_spread",
        feature_ids=[
            first.feature_id,
            second.feature_id,
        ],
        operator=CombinationOperator.SPREAD,
    )

    return engine, first, second, combination


def make_registered_candidate_environment(
    name: str = "btc_alpha",
):
    engine, feature, _ = prepare_feature_engine()

    generator = make_generator(engine)

    candidate = generator.create_candidate(
        name=name,
        feature_ids=[
            feature.feature_id,
        ],
        combination_ids=[],
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
        metadata={
            "asset": "BTC",
        },
    )

    return engine, generator, candidate


def make_candidate_for_engine(
    engine: FeatureDiscoveryEngine,
    name: str = "btc_alpha",
) -> StrategyCandidate:
    feature = engine.list_features()[0]

    return StrategyCandidate(
        name=name,
        feature_ids=[
            feature.feature_id,
        ],
        combination_ids=[],
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
        metadata={
            "asset": "BTC",
        },
    )


# ============================================================
# Basic
# ============================================================


def test_version():
    assert (
        StrategyCandidateGenerator.VERSION
        == VERSION
    )


def test_empty_generator():
    generator = make_generator()

    assert generator.count() == 0
    assert generator.list_all() == []
    assert generator.audit_log() == []


def test_feature_discovery_reference():
    feature_engine = make_feature_engine()

    generator = make_generator(
        feature_engine
    )

    assert (
        generator.feature_discovery
        is feature_engine
    )


def test_generator_creates_default_feature_engine():
    generator = make_generator()

    assert isinstance(
        generator.feature_discovery,
        FeatureDiscoveryEngine,
    )


# ============================================================
# Registration
# ============================================================


def test_register_candidate():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    candidate = make_candidate_for_engine(
        engine
    )

    result = generator.register(
        candidate
    )

    assert result is candidate
    assert generator.count() == 1


def test_register_candidate_exists():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    candidate = make_candidate_for_engine(
        engine
    )

    generator.register(candidate)

    assert generator.exists(
        candidate.candidate_id
    )


def test_get_candidate():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    candidate = make_candidate_for_engine(
        engine
    )

    generator.register(candidate)

    assert (
        generator.get(
            candidate.candidate_id
        )
        == candidate
    )


def test_get_missing_candidate():
    generator = make_generator()

    with pytest.raises(
        CandidateNotFoundError
    ):
        generator.get(
            "alpha_missing"
        )


def test_get_by_fingerprint():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    candidate = make_candidate_for_engine(
        engine
    )

    generator.register(candidate)

    result = generator.get_by_fingerprint(
        candidate.fingerprint
    )

    assert result == candidate


def test_get_missing_fingerprint():
    generator = make_generator()

    with pytest.raises(
        CandidateNotFoundError
    ):
        generator.get_by_fingerprint(
            "missing_fingerprint"
        )


def test_duplicate_candidate_allowed():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    candidate = make_candidate_for_engine(
        engine
    )

    first = generator.register(candidate)
    second = generator.register(candidate)

    assert first == second
    assert generator.count() == 1


def test_duplicate_candidate_disallowed():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    candidate = make_candidate_for_engine(
        engine
    )

    generator.register(candidate)

    with pytest.raises(
        CandidateAlreadyExistsError
    ):
        generator.register(
            candidate,
            allow_existing=False,
        )


def test_invalid_candidate_type():
    generator = make_generator()

    with pytest.raises(
        AlphaValidationError
    ):
        generator.register(
            "invalid"  # type: ignore[arg-type]
        )


# ============================================================
# Candidate Creation
# ============================================================


def test_create_candidate():
    engine, feature, _ = (
        prepare_feature_engine()
    )

    generator = make_generator(engine)

    candidate = generator.create_candidate(
        name="btc_momentum",
        feature_ids=[
            feature.feature_id,
        ],
        direction=StrategyDirection.LONG,
    )

    assert generator.count() == 1
    assert generator.exists(
        candidate.candidate_id
    )


def test_create_candidate_properties():
    engine, feature, _ = (
        prepare_feature_engine()
    )

    generator = make_generator(engine)

    candidate = generator.create_candidate(
        name="btc_short",
        feature_ids=[
            feature.feature_id,
        ],
        direction=StrategyDirection.SHORT,
        entry_rule={
            "operator": "<",
            "value": -1.0,
        },
        exit_rule={
            "operator": ">",
            "value": 0.0,
        },
        position_rule={
            "type": "equal_weight",
        },
        parameters={
            "threshold": -1.0,
        },
        hypothesis_id="hypothesis_001",
        metadata={
            "asset": "BTC",
        },
    )

    assert candidate.name == "btc_short"

    assert (
        candidate.direction
        == StrategyDirection.SHORT
    )

    assert (
        candidate.hypothesis_id
        == "hypothesis_001"
    )

    assert (
        candidate.metadata["asset"]
        == "BTC"
    )


def test_create_candidate_empty_name():
    engine, feature, _ = (
        prepare_feature_engine()
    )

    generator = make_generator(engine)

    with pytest.raises(
        AlphaValidationError
    ):
        generator.create_candidate(
            name="",
            feature_ids=[
                feature.feature_id,
            ],
            direction=StrategyDirection.LONG,
        )


# ============================================================
# Source Validation
# ============================================================


def test_candidate_with_registered_feature():
    engine, feature, _ = (
        prepare_feature_engine()
    )

    generator = make_generator(engine)

    candidate = generator.create_candidate(
        name="feature_alpha",
        feature_ids=[
            feature.feature_id,
        ],
        direction=StrategyDirection.LONG,
    )

    assert generator.exists(
        candidate.candidate_id
    )


def test_candidate_with_missing_feature():
    engine, _, _ = (
        prepare_feature_engine()
    )

    generator = make_generator(engine)

    with pytest.raises(
        CandidateSourceError
    ):
        generator.create_candidate(
            name="invalid_feature",
            feature_ids=[
                "feature_missing",
            ],
            direction=StrategyDirection.LONG,
        )


def test_candidate_with_registered_combination():
    (
        engine,
        _,
        _,
        combination,
    ) = prepare_combination_engine()

    generator = make_generator(engine)

    candidate = generator.create_candidate(
        name="combination_alpha",
        combination_ids=[
            combination.combination_id,
        ],
        direction=(
            StrategyDirection.LONG_SHORT
        ),
    )

    assert generator.exists(
        candidate.candidate_id
    )


def test_candidate_with_missing_combination():
    engine, _, _ = (
        prepare_feature_engine()
    )

    generator = make_generator(engine)

    with pytest.raises(
        CandidateSourceError
    ):
        generator.create_candidate(
            name="invalid_combination",
            combination_ids=[
                "combination_missing",
            ],
            direction=(
                StrategyDirection.LONG_SHORT
            ),
        )


# ============================================================
# Listing / Status
# ============================================================


def test_list_all():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    first = make_candidate_for_engine(
        engine,
        "alpha_1",
    )

    second = make_candidate_for_engine(
        engine,
        "alpha_2",
    )

    generator.register(first)
    generator.register(second)

    assert len(
        generator.list_all()
    ) == 2


def test_by_status_generated():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    first = make_candidate_for_engine(
        engine,
        "alpha_1",
    )

    second = make_candidate_for_engine(
        engine,
        "alpha_2",
    )

    generator.register(first)
    generator.register(second)

    results = generator.by_status(
        CandidateStatus.GENERATED
    )

    assert len(results) == 2


def test_by_status_accepted():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    first = make_candidate_for_engine(
        engine,
        "alpha_1",
    )

    second = make_candidate_for_engine(
        engine,
        "alpha_2",
    )

    generator.register(first)
    generator.register(second)

    generator.accept(
        first.candidate_id
    )

    results = generator.by_status(
        CandidateStatus.ACCEPTED
    )

    assert results == [first]


def test_by_status_rejected():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    first = make_candidate_for_engine(
        engine,
        "alpha_1",
    )

    second = make_candidate_for_engine(
        engine,
        "alpha_2",
    )

    generator.register(first)
    generator.register(second)

    generator.reject(
        second.candidate_id,
        reason=RejectionReason.OVERFIT,
        details="Overfit detected",
    )

    results = generator.by_status(
        CandidateStatus.REJECTED
    )

    assert results == [second]


# ============================================================
# Lifecycle
# ============================================================


def test_accept_candidate():
    _, generator, candidate = (
        make_registered_candidate_environment()
    )

    result = generator.accept(
        candidate.candidate_id
    )

    assert result is candidate
    assert candidate.accepted is True

    assert (
        candidate.status
        == CandidateStatus.ACCEPTED
    )


def test_reject_candidate():
    _, generator, candidate = (
        make_registered_candidate_environment()
    )

    result = generator.reject(
        candidate.candidate_id,
        reason=RejectionReason.LOW_SHARPE,
        details="Sharpe below threshold",
    )

    assert result is candidate
    assert candidate.rejected is True

    assert (
        candidate.rejection_reason
        == RejectionReason.LOW_SHARPE
    )


def test_accept_missing_candidate():
    generator = make_generator()

    with pytest.raises(
        CandidateNotFoundError
    ):
        generator.accept(
            "alpha_missing"
        )


def test_reject_missing_candidate():
    generator = make_generator()

    with pytest.raises(
        CandidateNotFoundError
    ):
        generator.reject(
            "alpha_missing",
            reason=RejectionReason.OVERFIT,
            details="Missing",
        )


# ============================================================
# Generate From Features
# ============================================================


def test_generate_from_features():
    engine, first, _ = (
        prepare_feature_engine()
    )

    generator = make_generator(engine)

    result = generator.generate_from_features(
        feature_ids=[
            first.feature_id,
        ],
        directions=[
            StrategyDirection.LONG,
        ],
    )

    assert result.candidate_count == 1
    assert generator.count() == 1


def test_generate_from_features_default_directions():
    engine, first, _ = (
        prepare_feature_engine()
    )

    generator = make_generator(engine)

    result = generator.generate_from_features(
        feature_ids=[
            first.feature_id,
        ]
    )

    assert result.candidate_count == 2
    assert generator.count() == 2


def test_generate_from_multiple_features():
    engine, first, second = (
        prepare_feature_engine()
    )

    generator = make_generator(engine)

    result = generator.generate_from_features(
        feature_ids=[
            first.feature_id,
            second.feature_id,
        ],
        directions=[
            StrategyDirection.LONG,
        ],
    )

    assert result.candidate_count == 2
    assert generator.count() == 2


def test_generate_from_features_multiple_directions():
    engine, first, _ = (
        prepare_feature_engine()
    )

    generator = make_generator(engine)

    result = generator.generate_from_features(
        feature_ids=[
            first.feature_id,
        ],
        directions=[
            StrategyDirection.LONG,
            StrategyDirection.SHORT,
            StrategyDirection.LONG_SHORT,
        ],
    )

    assert result.candidate_count == 3
    assert generator.count() == 3


def test_generate_from_features_duplicate_run():
    engine, first, _ = (
        prepare_feature_engine()
    )

    generator = make_generator(engine)

    first_result = (
        generator.generate_from_features(
            feature_ids=[
                first.feature_id,
            ],
            directions=[
                StrategyDirection.LONG,
            ],
        )
    )

    second_result = (
        generator.generate_from_features(
            feature_ids=[
                first.feature_id,
            ],
            directions=[
                StrategyDirection.LONG,
            ],
        )
    )

    assert first_result.candidate_count == 1
    assert second_result.candidate_count == 0
    assert second_result.skipped_duplicates == 1
    assert generator.count() == 1


def test_generate_from_missing_feature():
    engine, _, _ = (
        prepare_feature_engine()
    )

    generator = make_generator(engine)

    with pytest.raises(
        CandidateSourceError
    ):
        generator.generate_from_features(
            feature_ids=[
                "feature_missing",
            ]
        )


def test_generate_features_with_empty_default_engine():
    generator = make_generator()

    with pytest.raises(
        CandidateSourceError
    ):
        generator.generate_from_features(
            feature_ids=[
                "feature_001",
            ]
        )


# ============================================================
# Generate From Combinations
# ============================================================


def test_generate_from_combinations():
    (
        engine,
        _,
        _,
        combination,
    ) = prepare_combination_engine()

    generator = make_generator(engine)

    result = (
        generator.generate_from_combinations(
            combination_ids=[
                combination.combination_id,
            ],
            directions=[
                StrategyDirection.LONG,
            ],
        )
    )

    assert result.candidate_count == 1
    assert generator.count() == 1


def test_generate_from_combinations_default_directions():
    (
        engine,
        _,
        _,
        combination,
    ) = prepare_combination_engine()

    generator = make_generator(engine)

    result = (
        generator.generate_from_combinations(
            combination_ids=[
                combination.combination_id,
            ]
        )
    )

    assert result.candidate_count == 3
    assert generator.count() == 3


def test_generate_from_combination_multiple_directions():
    (
        engine,
        _,
        _,
        combination,
    ) = prepare_combination_engine()

    generator = make_generator(engine)

    result = (
        generator.generate_from_combinations(
            combination_ids=[
                combination.combination_id,
            ],
            directions=[
                StrategyDirection.LONG,
                StrategyDirection.SHORT,
                StrategyDirection.LONG_SHORT,
            ],
        )
    )

    assert result.candidate_count == 3
    assert generator.count() == 3


def test_generate_from_combination_duplicate_run():
    (
        engine,
        _,
        _,
        combination,
    ) = prepare_combination_engine()

    generator = make_generator(engine)

    first = (
        generator.generate_from_combinations(
            combination_ids=[
                combination.combination_id,
            ],
            directions=[
                StrategyDirection.LONG,
            ],
        )
    )

    second = (
        generator.generate_from_combinations(
            combination_ids=[
                combination.combination_id,
            ],
            directions=[
                StrategyDirection.LONG,
            ],
        )
    )

    assert first.candidate_count == 1
    assert second.candidate_count == 0
    assert second.skipped_duplicates == 1
    assert generator.count() == 1


def test_generate_from_missing_combination():
    engine, _, _ = (
        prepare_feature_engine()
    )

    generator = make_generator(engine)

    with pytest.raises(
        CombinationNotFoundError
    ):
        generator.generate_from_combinations(
            combination_ids=[
                "combination_missing",
            ]
        )


def test_generate_combinations_with_empty_default_engine():
    generator = make_generator()

    with pytest.raises(
        CombinationNotFoundError
    ):
        generator.generate_from_combinations(
            combination_ids=[
                "combination_001",
            ]
        )


# ============================================================
# Batch Registration
# ============================================================


def test_register_many():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    candidates = [
        make_candidate_for_engine(
            engine,
            "alpha_1",
        ),
        make_candidate_for_engine(
            engine,
            "alpha_2",
        ),
        make_candidate_for_engine(
            engine,
            "alpha_3",
        ),
    ]

    result = generator.register_many(
        candidates
    )

    assert len(result) == 3
    assert generator.count() == 3


def test_register_many_duplicates():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    candidate = make_candidate_for_engine(
        engine
    )

    result = generator.register_many(
        [
            candidate,
            candidate,
        ]
    )

    assert len(result) == 2
    assert generator.count() == 1


# ============================================================
# Fingerprints
# ============================================================


def test_candidate_set_fingerprint_empty():
    generator = make_generator()

    first = (
        generator.candidate_set_fingerprint()
    )

    second = (
        generator.candidate_set_fingerprint()
    )

    assert first == second
    assert isinstance(first, str)
    assert first


def test_candidate_set_fingerprint_deterministic():
    first_engine, _, _ = (
        prepare_feature_engine()
    )

    second_engine, _, _ = (
        prepare_feature_engine()
    )

    first_generator = make_generator(
        first_engine
    )

    second_generator = make_generator(
        second_engine
    )

    first_candidate = (
        make_candidate_for_engine(
            first_engine,
            "alpha_1",
        )
    )

    second_candidate = (
        make_candidate_for_engine(
            first_engine,
            "alpha_2",
        )
    )

    first_generator.register(
        first_candidate
    )

    first_generator.register(
        second_candidate
    )

    second_candidate_copy = (
        make_candidate_for_engine(
            second_engine,
            "alpha_2",
        )
    )

    first_candidate_copy = (
        make_candidate_for_engine(
            second_engine,
            "alpha_1",
        )
    )

    second_generator.register(
        second_candidate_copy
    )

    second_generator.register(
        first_candidate_copy
    )

    assert (
        first_generator
        .candidate_set_fingerprint()
        ==
        second_generator
        .candidate_set_fingerprint()
    )


def test_candidate_set_fingerprint_selected():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    first = make_candidate_for_engine(
        engine,
        "alpha_1",
    )

    second = make_candidate_for_engine(
        engine,
        "alpha_2",
    )

    generator.register(first)
    generator.register(second)

    fingerprint = (
        generator.candidate_set_fingerprint(
            [
                first.candidate_id,
            ]
        )
    )

    assert isinstance(
        fingerprint,
        str,
    )

    assert fingerprint


def test_full_fingerprint_changes():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    before = generator.full_fingerprint()

    generator.register(
        make_candidate_for_engine(
            engine
        )
    )

    after = generator.full_fingerprint()

    assert before != after


# ============================================================
# Snapshot
# ============================================================


def test_snapshot_empty():
    generator = make_generator()

    snapshot = generator.snapshot()

    assert snapshot.candidate_count == 0
    assert snapshot.accepted_count == 0
    assert snapshot.rejected_count == 0
    assert snapshot.active_count == 0
    assert snapshot.audit_count == 0


def test_snapshot_generated_candidate():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    candidate = make_candidate_for_engine(
        engine
    )

    generator.register(candidate)

    snapshot = generator.snapshot()

    assert snapshot.candidate_count == 1
    assert snapshot.accepted_count == 0
    assert snapshot.rejected_count == 0
    assert snapshot.active_count == 1
    assert snapshot.audit_count == 1


def test_snapshot_accepted_and_rejected():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    first = make_candidate_for_engine(
        engine,
        "alpha_1",
    )

    second = make_candidate_for_engine(
        engine,
        "alpha_2",
    )

    third = make_candidate_for_engine(
        engine,
        "alpha_3",
    )

    generator.register(first)
    generator.register(second)
    generator.register(third)

    generator.accept(
        first.candidate_id
    )

    generator.reject(
        second.candidate_id,
        reason=RejectionReason.OVERFIT,
        details="Overfit",
    )

    snapshot = generator.snapshot()

    assert snapshot.candidate_count == 3
    assert snapshot.accepted_count == 1
    assert snapshot.rejected_count == 1
    assert snapshot.active_count == 1


def test_snapshot_candidate_ids():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    candidate = make_candidate_for_engine(
        engine
    )

    generator.register(candidate)

    snapshot = generator.snapshot()

    assert (
        candidate.candidate_id
        in snapshot.candidate_ids
    )


def test_snapshot_fingerprint():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    generator.register(
        make_candidate_for_engine(
            engine
        )
    )

    snapshot = generator.snapshot()

    assert isinstance(
        snapshot.fingerprint,
        str,
    )

    assert snapshot.fingerprint


# ============================================================
# Audit
# ============================================================


def test_register_creates_audit():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    candidate = make_candidate_for_engine(
        engine
    )

    generator.register(candidate)

    audit = generator.audit_log()

    assert len(audit) == 1
    assert audit[0].action == "register"

    assert (
        audit[0].candidate_id
        == candidate.candidate_id
    )


def test_accept_creates_audit():
    _, generator, candidate = (
        make_registered_candidate_environment()
    )

    generator.accept(
        candidate.candidate_id
    )

    audit = generator.audit_log()

    assert len(audit) == 2
    assert audit[-1].action == "accept"


def test_reject_creates_audit():
    _, generator, candidate = (
        make_registered_candidate_environment()
    )

    generator.reject(
        candidate.candidate_id,
        reason=RejectionReason.OVERFIT,
        details="Overfit",
    )

    audit = generator.audit_log()

    assert len(audit) == 2
    assert audit[-1].action == "reject"


def test_audit_returns_copy():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    generator.register(
        make_candidate_for_engine(
            engine
        )
    )

    audit = generator.audit_log()
    audit.clear()

    assert len(
        generator.audit_log()
    ) == 1


# ============================================================
# Summary
# ============================================================


def test_summary_empty():
    generator = make_generator()

    summary = generator.summary()

    assert summary["version"] == VERSION
    assert summary["candidate_count"] == 0
    assert summary["accepted_count"] == 0
    assert summary["rejected_count"] == 0
    assert summary["active_count"] == 0
    assert summary["research_only"] is True

    assert (
        summary["real_money_execution"]
        is False
    )


def test_summary_after_candidates():
    engine, _, _ = prepare_feature_engine()

    generator = make_generator(engine)

    first = make_candidate_for_engine(
        engine,
        "alpha_1",
    )

    second = make_candidate_for_engine(
        engine,
        "alpha_2",
    )

    generator.register(first)
    generator.register(second)

    generator.accept(
        first.candidate_id
    )

    generator.reject(
        second.candidate_id,
        reason=RejectionReason.OVERFIT,
        details="Overfit",
    )

    summary = generator.summary()

    assert summary["candidate_count"] == 2
    assert summary["accepted_count"] == 1
    assert summary["rejected_count"] == 1
    assert summary["active_count"] == 0


# ============================================================
# Isolation
# ============================================================


def test_generators_are_independent():
    first_engine, _, _ = (
        prepare_feature_engine()
    )

    second_engine, _, _ = (
        prepare_feature_engine()
    )

    first = make_generator(
        first_engine
    )

    second = make_generator(
        second_engine
    )

    first.register(
        make_candidate_for_engine(
            first_engine
        )
    )

    assert first.count() == 1
    assert second.count() == 0


# ============================================================
# Exception Hierarchy
# ============================================================


def test_candidate_generation_error():
    error = CandidateGenerationError(
        "test"
    )

    assert isinstance(
        error,
        Exception,
    )


def test_candidate_source_error():
    error = CandidateSourceError(
        "test"
    )

    assert isinstance(
        error,
        CandidateGenerationError,
    )