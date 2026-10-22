from __future__ import annotations

import pytest

from alpha_discovery_laboratory.alpha_discovery_lab import (
    AlphaDiscoveryExperimentError,
    AlphaDiscoveryLabError,
    AlphaDiscoveryLaboratory,
    AlphaResearchInput,
    CandidateEvidenceBundle,
)
from alpha_discovery_laboratory.models import (
    VERSION,
    CombinationOperator,
    FeatureTransform,
    FeatureType,
    PerformanceMetrics,
    RejectionReason,
    StrategyDirection,
)
from alpha_discovery_laboratory.overfit_detector import (
    OverfitEvidence,
    ParameterPerturbation,
    RegimePerformance,
)
from alpha_discovery_laboratory.validation_pipeline import (
    BacktestEvidence,
    CausalEvidence,
    OOSEvidence,
    RobustnessEvidence,
)


# ============================================================
# Helpers
# ============================================================


def make_metrics(
    *,
    total_return: float = 0.20,
    annualized_return: float = 0.18,
    volatility: float = 0.15,
    sharpe_ratio: float = 1.50,
    sortino_ratio: float = 1.80,
    calmar_ratio: float = 1.40,
    max_drawdown: float = 0.12,
    win_rate: float = 0.58,
    profit_factor: float = 1.40,
    turnover: float = 0.50,
    execution_cost: float = 0.01,
    trade_count: int = 100,
) -> PerformanceMetrics:
    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        turnover=turnover,
        execution_cost=execution_cost,
        trade_count=trade_count,
    )


def make_research_input(
    *,
    name: str = "btc_alpha_research",
    dataset_ids=None,
    dataset_fingerprints=None,
    feature_ids=None,
    combination_ids=None,
    directions=None,
    seed: int = 42,
    run_id: str = "run_001",
) -> AlphaResearchInput:
    return AlphaResearchInput(
        name=name,
        dataset_ids=list(
            dataset_ids
            or ["dataset_btc"]
        ),
        dataset_fingerprints=list(
            dataset_fingerprints
            or ["dataset_fp_btc"]
        ),
        feature_ids=list(
            feature_ids
            or []
        ),
        combination_ids=list(
            combination_ids
            or []
        ),
        directions=list(
            directions
            or [StrategyDirection.LONG]
        ),
        model_version_ids=[
            "model_v1",
        ],
        prompt_version_ids=[
            "prompt_v1",
        ],
        policy_version_ids=[
            "policy_v1",
        ],
        seed=seed,
        run_id=run_id,
        configuration={
            "validation": "strict",
        },
        metadata={
            "asset": "BTC",
        },
    )


def make_overfit_evidence(
    *,
    bad: bool = False,
) -> OverfitEvidence:
    if bad:
        oos_metrics = make_metrics(
            total_return=0.01,
            annualized_return=0.01,
            sharpe_ratio=0.10,
            max_drawdown=0.30,
        )
    else:
        oos_metrics = make_metrics(
            total_return=0.17,
            annualized_return=0.16,
            sharpe_ratio=1.25,
            max_drawdown=0.14,
        )

    perturbations = [
        ParameterPerturbation(
            name="window_1",
            baseline_value=20.0,
            perturbed_value=18.0,
            performance_score=0.82,
        ),
        ParameterPerturbation(
            name="window_2",
            baseline_value=20.0,
            perturbed_value=20.0,
            performance_score=0.80,
        ),
        ParameterPerturbation(
            name="window_3",
            baseline_value=20.0,
            perturbed_value=22.0,
            performance_score=0.78,
        ),
    ]

    regimes = [
        RegimePerformance(
            regime="risk_on",
            metrics=make_metrics(
                sharpe_ratio=1.30,
            ),
        ),
        RegimePerformance(
            regime="mixed",
            metrics=make_metrics(
                sharpe_ratio=1.20,
            ),
        ),
        RegimePerformance(
            regime="risk_off",
            metrics=make_metrics(
                sharpe_ratio=1.10,
            ),
        ),
    ]

    return OverfitEvidence(
        in_sample=make_metrics(
            total_return=0.20,
            annualized_return=0.18,
            sharpe_ratio=1.50,
            max_drawdown=0.12,
        ),
        out_of_sample=oos_metrics,
        parameter_perturbations=perturbations,
        regime_performance=regimes,
        bootstrap_scores=[
            0.80,
            0.82,
            0.78,
            0.81,
        ],
        metadata={
            "test": True,
        },
    )


def make_evidence_bundle(
    candidate_id: str,
    *,
    validation_failure: bool = False,
    overfit_failure: bool = False,
) -> CandidateEvidenceBundle:
    if validation_failure:
        backtest_metrics = make_metrics(
            sharpe_ratio=0.10,
            profit_factor=0.80,
        )
    else:
        backtest_metrics = make_metrics()

    return CandidateEvidenceBundle(
        candidate_id=candidate_id,
        backtest=BacktestEvidence(
            metrics=backtest_metrics,
            data_leakage_detected=False,
            lookahead_bias_detected=False,
            metadata={
                "test": True,
            },
        ),
        causal=CausalEvidence(
            causal_score=0.80,
            stable_effect=True,
            confounding_detected=False,
            metadata={
                "test": True,
            },
        ),
        oos=OOSEvidence(
            metrics=make_metrics(
                total_return=0.16,
                annualized_return=0.15,
                sharpe_ratio=1.10,
                max_drawdown=0.14,
            ),
            performance_retention=0.75,
            metadata={
                "test": True,
            },
        ),
        robustness=RobustnessEvidence(
            robustness_score=0.80,
            parameter_sensitivity=0.20,
            regime_stable=True,
            perturbation_stable=True,
            metadata={
                "test": True,
            },
        ),
        overfit=make_overfit_evidence(
            bad=overfit_failure
        ),
        metadata={
            "bundle": "test",
        },
    )


def build_candidate(
    lab: AlphaDiscoveryLaboratory,
    *,
    prefix: str = "btc",
):
    feature_one = lab.create_feature(
        name=f"{prefix}_momentum",
        feature_type=FeatureType.MOMENTUM,
        source="market_data",
        transform=FeatureTransform.RAW,
        dataset_id="dataset_btc",
        dataset_fingerprint="dataset_fp_btc",
    )

    feature_two = lab.create_feature(
        name=f"{prefix}_volume",
        feature_type=FeatureType.VOLUME,
        source="market_data",
        transform=FeatureTransform.RAW,
        dataset_id="dataset_btc",
        dataset_fingerprint="dataset_fp_btc",
    )

    discovery = lab.discover_features(
        feature_ids=[
            feature_one.feature_id,
            feature_two.feature_id,
        ],
        operators=[
            CombinationOperator.SPREAD,
        ],
        name_prefix=prefix,
    )

    assert discovery.combinations

    combination = discovery.combinations[0]

    generated = lab.generate_candidates(
        combination_ids=[
            combination.combination_id
        ],
        directions=[
            StrategyDirection.LONG
        ],
        name_prefix=f"{prefix}_alpha",
    )

    assert generated.candidates

    return (
        feature_one,
        feature_two,
        combination,
        generated.candidates[0],
    )


# ============================================================
# Basic Laboratory
# ============================================================


def test_version():
    assert (
        AlphaDiscoveryLaboratory.VERSION
        == VERSION
    )


def test_empty_summary():
    lab = AlphaDiscoveryLaboratory()

    summary = lab.summary()

    assert summary["version"] == VERSION
    assert summary["experiments"] == 0
    assert summary["features"] == 0
    assert (
        summary["feature_combinations"]
        == 0
    )
    assert (
        summary["strategy_candidates"]
        == 0
    )
    assert (
        summary["evaluated_candidates"]
        == 0
    )
    assert (
        summary["accepted_candidates"]
        == 0
    )
    assert (
        summary["rejected_candidates"]
        == 0
    )
    assert (
        summary["lineage_records"]
        == 0
    )
    assert summary["audit_records"] == 0
    assert summary["research_only"] is True
    assert (
        summary["real_money_execution"]
        is False
    )


def test_results_empty():
    lab = AlphaDiscoveryLaboratory()

    assert lab.results() == []


def test_audit_empty():
    lab = AlphaDiscoveryLaboratory()

    assert lab.audit_log() == []


# ============================================================
# Configuration Fingerprint
# ============================================================


def test_configuration_fingerprint():
    lab = AlphaDiscoveryLaboratory()

    fingerprint = (
        lab.configuration_fingerprint()
    )

    assert isinstance(
        fingerprint,
        str,
    )
    assert fingerprint


def test_configuration_fingerprint_deterministic():
    first = AlphaDiscoveryLaboratory()
    second = AlphaDiscoveryLaboratory()

    assert (
        first.configuration_fingerprint()
        ==
        second.configuration_fingerprint()
    )


def test_empty_summary_configuration_fingerprint():
    lab = AlphaDiscoveryLaboratory()

    assert (
        lab.summary()[
            "configuration_fingerprint"
        ]
        ==
        lab.configuration_fingerprint()
    )


# ============================================================
# Feature Creation
# ============================================================


def test_create_feature():
    lab = AlphaDiscoveryLaboratory()

    feature = lab.create_feature(
        name="btc_momentum",
        feature_type=FeatureType.MOMENTUM,
        source="market_data",
        dataset_id="dataset_btc",
    )

    assert feature.name == "btc_momentum"
    assert (
        feature.feature_type
        == FeatureType.MOMENTUM
    )


def test_create_feature_registered():
    lab = AlphaDiscoveryLaboratory()

    feature = lab.create_feature(
        name="btc_volume",
        feature_type=FeatureType.VOLUME,
        source="market_data",
    )

    assert (
        lab.feature_discovery.exists(
            feature.feature_id
        )
    )


def test_create_feature_summary():
    lab = AlphaDiscoveryLaboratory()

    lab.create_feature(
        name="btc_momentum",
        feature_type=FeatureType.MOMENTUM,
        source="market_data",
    )

    assert lab.summary()["features"] == 1


# ============================================================
# Feature Discovery
# ============================================================


def test_discover_features():
    lab = AlphaDiscoveryLaboratory()

    first = lab.create_feature(
        name="momentum",
        feature_type=FeatureType.MOMENTUM,
        source="market_data",
    )

    second = lab.create_feature(
        name="volume",
        feature_type=FeatureType.VOLUME,
        source="market_data",
    )

    result = lab.discover_features(
        feature_ids=[
            first.feature_id,
            second.feature_id,
        ],
        operators=[
            CombinationOperator.SPREAD
        ],
    )

    assert result.combination_count == 1


def test_discover_features_registered():
    lab = AlphaDiscoveryLaboratory()

    _, _, combination, _ = (
        build_candidate(lab)
    )

    assert (
        lab.feature_discovery
        .combination_exists(
            combination.combination_id
        )
    )


def test_discover_features_creates_audit():
    lab = AlphaDiscoveryLaboratory()

    first = lab.create_feature(
        name="momentum",
        feature_type=FeatureType.MOMENTUM,
        source="market_data",
    )

    second = lab.create_feature(
        name="volume",
        feature_type=FeatureType.VOLUME,
        source="market_data",
    )

    lab.discover_features(
        feature_ids=[
            first.feature_id,
            second.feature_id,
        ],
        operators=[
            CombinationOperator.SPREAD
        ],
    )

    assert len(
        lab.audit_log()
    ) >= 1


# ============================================================
# Candidate Generation
# ============================================================


def test_generate_candidates():
    lab = AlphaDiscoveryLaboratory()

    _, _, _, candidate = (
        build_candidate(lab)
    )

    assert candidate.candidate_id


def test_generate_candidate_registered():
    lab = AlphaDiscoveryLaboratory()

    _, _, _, candidate = (
        build_candidate(lab)
    )

    assert (
        lab.candidate_generator.exists(
            candidate.candidate_id
        )
    )


def test_generate_candidate_summary():
    lab = AlphaDiscoveryLaboratory()

    build_candidate(lab)

    assert (
        lab.summary()[
            "strategy_candidates"
        ]
        == 1
    )


# ============================================================
# Research Input
# ============================================================


def test_research_input():
    research_input = (
        make_research_input()
    )

    assert (
        research_input.name
        == "btc_alpha_research"
    )
    assert research_input.seed == 42
    assert (
        research_input.run_id
        == "run_001"
    )


def test_research_input_fingerprint():
    research_input = (
        make_research_input()
    )

    assert isinstance(
        research_input.fingerprint,
        str,
    )
    assert research_input.fingerprint


def test_research_input_reproducible():
    first = make_research_input()
    second = make_research_input()

    assert (
        first.fingerprint
        == second.fingerprint
    )


# ============================================================
# Experiment Creation
# ============================================================


def test_create_experiment():
    lab = AlphaDiscoveryLaboratory()

    research_input = (
        make_research_input()
    )

    experiment = lab.create_experiment(
        research_input
    )

    assert experiment.experiment_id


def test_create_experiment_registered():
    lab = AlphaDiscoveryLaboratory()

    experiment = lab.create_experiment(
        make_research_input()
    )

    assert (
        lab.lineage_registry
        .experiment_exists(
            experiment.experiment_id
        )
    )


def test_create_experiment_candidate_ids():
    lab = AlphaDiscoveryLaboratory()

    _, _, _, candidate = (
        build_candidate(lab)
    )

    experiment = lab.create_experiment(
        make_research_input(),
        candidate_ids=[
            candidate.candidate_id
        ],
    )

    assert (
        candidate.candidate_id
        in experiment.candidate_ids
    )


# ============================================================
# Candidate Evidence
# ============================================================


def test_candidate_evidence_bundle():
    lab = AlphaDiscoveryLaboratory()

    _, _, _, candidate = (
        build_candidate(lab)
    )

    evidence = make_evidence_bundle(
        candidate.candidate_id
    )

    assert (
        evidence.candidate_id
        == candidate.candidate_id
    )


def test_candidate_evidence_metadata():
    lab = AlphaDiscoveryLaboratory()

    _, _, _, candidate = (
        build_candidate(lab)
    )

    evidence = make_evidence_bundle(
        candidate.candidate_id
    )

    assert (
        evidence.metadata["bundle"]
        == "test"
    )


# ============================================================
# Successful Candidate Evaluation
# ============================================================


def test_evaluate_good_candidate():
    lab = AlphaDiscoveryLaboratory()

    _, _, _, candidate = (
        build_candidate(lab)
    )

    evidence = make_evidence_bundle(
        candidate.candidate_id
    )

    result = lab.evaluate_candidate(
        candidate=candidate,
        evidence=evidence,
        dataset_ids=[
            "dataset_btc"
        ],
        dataset_fingerprints=[
            "dataset_fp_btc"
        ],
        model_version_ids=[
            "model_v1"
        ],
        prompt_version_ids=[
            "prompt_v1"
        ],
        policy_version_ids=[
            "policy_v1"
        ],
        run_id="run_001",
    )

    assert result.accepted is True


def test_good_candidate_validation_accepted():
    lab = AlphaDiscoveryLaboratory()

    _, _, _, candidate = (
        build_candidate(lab)
    )

    result = lab.evaluate_candidate(
        candidate=candidate,
        evidence=make_evidence_bundle(
            candidate.candidate_id
        ),
    )

    assert (
        result.validation.accepted
        is True
    )


def test_good_candidate_overfit_checked():
    lab = AlphaDiscoveryLaboratory()

    _, _, _, candidate = (
        build_candidate(lab)
    )

    result = lab.evaluate_candidate(
        candidate=candidate,
        evidence=make_evidence_bundle(
            candidate.candidate_id
        ),
    )

    assert result.overfit is not None
    assert (
        result.overfit.accepted
        is True
    )


def test_good_candidate_lineage_created():
    lab = AlphaDiscoveryLaboratory()

    _, _, _, candidate = (
        build_candidate(lab)
    )

    result = lab.evaluate_candidate(
        candidate=candidate,
        evidence=make_evidence_bundle(
            candidate.candidate_id
        ),
        dataset_ids=[
            "dataset_btc"
        ],
    )

    assert result.lineage is not None
    assert (
        result.lineage.candidate_id
        == candidate.candidate_id
    )


def test_good_candidate_lineage_registered():
    lab = AlphaDiscoveryLaboratory()

    _, _, _, candidate = (
        build_candidate(lab)
    )

    lab.evaluate_candidate(
        candidate=candidate,
        evidence=make_evidence_bundle(
            candidate.candidate_id
        ),
    )

    assert (
        lab.lineage_registry.exists(
            candidate.candidate_id
        )
    )


# ============================================================
# Validation Rejection
# ============================================================


def test_validation_failure_rejected():
    lab = AlphaDiscoveryLaboratory()

    _, _, _, candidate = (
        build_candidate(lab)
    )

    result = lab.evaluate_candidate(
        candidate=candidate,
        evidence=make_evidence_bundle(
            candidate.candidate_id,
            validation_failure=True,
        ),
    )

    assert result.accepted is False
    assert result.validation.accepted is False


def test_validation_failure_has_reason():
    lab = AlphaDiscoveryLaboratory()

    _, _, _, candidate = (
        build_candidate(lab)
    )

    result = lab.evaluate_candidate(
        candidate=candidate,
        evidence=make_evidence_bundle(
            candidate.candidate_id,
            validation_failure=True,
        ),
    )

    assert (
        result.rejection_reason
        is not None
    )


def test_validation_failure_skips_overfit():
    lab = AlphaDiscoveryLaboratory()

    _, _, _, candidate = (
        build_candidate(lab)
    )

    result = lab.evaluate_candidate(
        candidate=candidate,
        evidence=make_evidence_bundle(
            candidate.candidate_id,
            validation_failure=True,
        ),
    )

    assert result.overfit is None


def test_validation_failure_still_has_lineage():
    lab = AlphaDiscoveryLaboratory()

    _, _, _, candidate = (
        build_candidate(lab)
    )

    result = lab.evaluate_candidate(
        candidate=candidate,
        evidence=make_evidence_bundle(
            candidate.candidate_id,
            validation_failure=True,
        ),
    )

    assert result.lineage is not None


# ============================================================
# Overfit Rejection
# ============================================================


def test_overfit_failure_rejected():
    lab = AlphaDiscoveryLaboratory()

    _, _, _, candidate = (
        build_candidate(lab)
    )

    result = lab.evaluate_candidate(
        candidate=candidate,
        evidence=make_evidence_bundle(
            candidate.candidate_id,
            overfit_failure=True,
        ),
    )

    assert result.accepted is False
    assert result.overfit is not None
    assert (
        result.overfit.overfit_detected
        is True
    )


def test_overfit_failure_reason():
    lab = AlphaDiscoveryLaboratory()

    _, _, _, candidate = (
        build_candidate(lab)
    )

    result = lab.evaluate_candidate(
        candidate=candidate,
        evidence=make_evidence_bundle(
            candidate.candidate_id,
            overfit_failure=True,
        ),
    )

    assert (
        result.rejection_reason
        == RejectionReason.OVERFIT
    )


def test_overfit_failure_candidate_rejected():
    lab = AlphaDiscoveryLaboratory()

    _, _, _, candidate = (
        build_candidate(lab)
    )

    lab.evaluate_candidate(
        candidate=candidate,
        evidence=make_evidence_bundle(
            candidate.candidate_id,
            overfit_failure=True,
        ),
    )

    assert candidate.rejected is True


# ============================================================
# Evidence Validation
# ============================================================


def test_evidence_candidate_mismatch():
    lab = AlphaDiscoveryLaboratory()

    _, _, _, candidate = (
        build_candidate(lab)
    )

    evidence = make_evidence_bundle(
        "alpha_wrong_candidate"
    )

    with pytest.raises(
        AlphaDiscoveryLabError
    ):
        lab.evaluate_candidate(
            candidate=candidate,
            evidence=evidence,
        )


# ============================================================
# Full Experiment
# ============================================================


def test_run_experiment():
    lab = AlphaDiscoveryLaboratory()

    feature_one, feature_two, combination, candidate = (
        build_candidate(lab)
    )

    research_input = make_research_input(
        feature_ids=[
            feature_one.feature_id,
            feature_two.feature_id,
        ],
        combination_ids=[
            combination.combination_id
        ],
    )

    evidence = {
        candidate.candidate_id:
            make_evidence_bundle(
                candidate.candidate_id
            )
    }

    result = lab.run_experiment(
        research_input=research_input,
        evidence_by_candidate=evidence,
    )

    assert result.experiment_id


def test_run_experiment_candidate_result():
    lab = AlphaDiscoveryLaboratory()

    feature_one, feature_two, combination, candidate = (
        build_candidate(lab)
    )

    result = lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                feature_one.feature_id,
                feature_two.feature_id,
            ],
            combination_ids=[
                combination.combination_id
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_evidence_bundle(
                    candidate.candidate_id
                )
        },
    )

    assert len(
        result.candidate_results
    ) == 1

    assert (
        result.candidate_results[0]
        .candidate_id
        == candidate.candidate_id
    )


def test_run_experiment_accepts_good_candidate():
    lab = AlphaDiscoveryLaboratory()

    feature_one, feature_two, combination, candidate = (
        build_candidate(lab)
    )

    result = lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                feature_one.feature_id,
                feature_two.feature_id,
            ],
            combination_ids=[
                combination.combination_id
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_evidence_bundle(
                    candidate.candidate_id
                )
        },
    )

    assert (
        candidate.candidate_id
        in result.accepted_candidate_ids
    )

    assert (
        candidate.candidate_id
        not in result.rejected_candidate_ids
    )


def test_run_experiment_rejects_bad_candidate():
    lab = AlphaDiscoveryLaboratory()

    feature_one, feature_two, combination, candidate = (
        build_candidate(lab)
    )

    result = lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                feature_one.feature_id,
                feature_two.feature_id,
            ],
            combination_ids=[
                combination.combination_id
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_evidence_bundle(
                    candidate.candidate_id,
                    validation_failure=True,
                )
        },
    )

    assert (
        candidate.candidate_id
        in result.rejected_candidate_ids
    )


# ============================================================
# Result Registry
# ============================================================


def test_result_registered():
    lab = AlphaDiscoveryLaboratory()

    feature_one, feature_two, combination, candidate = (
        build_candidate(lab)
    )

    result = lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                feature_one.feature_id,
                feature_two.feature_id,
            ],
            combination_ids=[
                combination.combination_id
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_evidence_bundle(
                    candidate.candidate_id
                )
        },
    )

    assert (
        lab.get_result(
            result.experiment_id
        )
        == result
    )


def test_results_after_experiment():
    lab = AlphaDiscoveryLaboratory()

    feature_one, feature_two, combination, candidate = (
        build_candidate(lab)
    )

    lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                feature_one.feature_id,
                feature_two.feature_id,
            ],
            combination_ids=[
                combination.combination_id
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_evidence_bundle(
                    candidate.candidate_id
                )
        },
    )

    assert len(
        lab.results()
    ) == 1


def test_missing_result():
    lab = AlphaDiscoveryLaboratory()

    with pytest.raises(
        AlphaDiscoveryExperimentError
    ):
        lab.get_result(
            "alpha_experiment_missing"
        )


# ============================================================
# Experiment Summary
# ============================================================


def test_summary_after_experiment():
    lab = AlphaDiscoveryLaboratory()

    feature_one, feature_two, combination, candidate = (
        build_candidate(lab)
    )

    lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                feature_one.feature_id,
                feature_two.feature_id,
            ],
            combination_ids=[
                combination.combination_id
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_evidence_bundle(
                    candidate.candidate_id
                )
        },
    )

    summary = lab.summary()

    assert summary["experiments"] == 1
    assert summary["features"] == 2
    assert (
        summary["feature_combinations"]
        == 1
    )
    assert (
        summary["strategy_candidates"]
        == 1
    )
    assert (
        summary["evaluated_candidates"]
        == 1
    )
    assert (
        summary["accepted_candidates"]
        == 1
    )
    assert (
        summary["rejected_candidates"]
        == 0
    )
    assert (
        summary["lineage_records"]
        == 1
    )


# ============================================================
# Audit
# ============================================================


def test_evaluation_creates_audit():
    lab = AlphaDiscoveryLaboratory()

    _, _, _, candidate = (
        build_candidate(lab)
    )

    before = len(
        lab.audit_log()
    )

    lab.evaluate_candidate(
        candidate=candidate,
        evidence=make_evidence_bundle(
            candidate.candidate_id
        ),
    )

    after = len(
        lab.audit_log()
    )

    assert after > before


def test_run_experiment_creates_audit():
    lab = AlphaDiscoveryLaboratory()

    feature_one, feature_two, combination, candidate = (
        build_candidate(lab)
    )

    lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                feature_one.feature_id,
                feature_two.feature_id,
            ],
            combination_ids=[
                combination.combination_id
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_evidence_bundle(
                    candidate.candidate_id
                )
        },
    )

    assert len(
        lab.audit_log()
    ) > 0


def test_audit_returns_copy():
    lab = AlphaDiscoveryLaboratory()

    build_candidate(lab)

    audit = lab.audit_log()
    audit.clear()

    assert len(
        lab.audit_log()
    ) > 0


# ============================================================
# Result Properties
# ============================================================


def test_lab_result_counts():
    lab = AlphaDiscoveryLaboratory()

    feature_one, feature_two, combination, candidate = (
        build_candidate(lab)
    )

    result = lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                feature_one.feature_id,
                feature_two.feature_id,
            ],
            combination_ids=[
                combination.combination_id
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_evidence_bundle(
                    candidate.candidate_id
                )
        },
    )

    assert result.accepted_count == 1
    assert result.rejected_count == 0


def test_lab_result_acceptance_rate():
    lab = AlphaDiscoveryLaboratory()

    feature_one, feature_two, combination, candidate = (
        build_candidate(lab)
    )

    result = lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                feature_one.feature_id,
                feature_two.feature_id,
            ],
            combination_ids=[
                combination.combination_id
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_evidence_bundle(
                    candidate.candidate_id
                )
        },
    )

    assert (
        result.acceptance_rate
        == pytest.approx(1.0)
    )


def test_lab_result_fingerprint():
    lab = AlphaDiscoveryLaboratory()

    feature_one, feature_two, combination, candidate = (
        build_candidate(lab)
    )

    result = lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                feature_one.feature_id,
                feature_two.feature_id,
            ],
            combination_ids=[
                combination.combination_id
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_evidence_bundle(
                    candidate.candidate_id
                )
        },
    )

    assert isinstance(
        result.fingerprint,
        str,
    )
    assert result.fingerprint


# ============================================================
# Lineage Integrity
# ============================================================


def test_experiment_lineage_link():
    lab = AlphaDiscoveryLaboratory()

    feature_one, feature_two, combination, candidate = (
        build_candidate(lab)
    )

    result = lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                feature_one.feature_id,
                feature_two.feature_id,
            ],
            combination_ids=[
                combination.combination_id
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_evidence_bundle(
                    candidate.candidate_id
                )
        },
    )

    lineage = (
        lab.lineage_registry.get(
            candidate.candidate_id
        )
    )

    assert (
        lineage.experiment_id
        == result.experiment_id
    )


def test_lineage_dataset_preserved():
    lab = AlphaDiscoveryLaboratory()

    feature_one, feature_two, combination, candidate = (
        build_candidate(lab)
    )

    lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                feature_one.feature_id,
                feature_two.feature_id,
            ],
            combination_ids=[
                combination.combination_id
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_evidence_bundle(
                    candidate.candidate_id
                )
        },
    )

    lineage = (
        lab.lineage_registry.get(
            candidate.candidate_id
        )
    )

    assert lineage.dataset_ids == [
        "dataset_btc"
    ]

    assert lineage.dataset_fingerprints == [
        "dataset_fp_btc"
    ]


def test_lineage_version_ids_preserved():
    lab = AlphaDiscoveryLaboratory()

    feature_one, feature_two, combination, candidate = (
        build_candidate(lab)
    )

    lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                feature_one.feature_id,
                feature_two.feature_id,
            ],
            combination_ids=[
                combination.combination_id
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_evidence_bundle(
                    candidate.candidate_id
                )
        },
    )

    lineage = (
        lab.lineage_registry.get(
            candidate.candidate_id
        )
    )

    assert lineage.model_version_ids == [
        "model_v1"
    ]
    assert lineage.prompt_version_ids == [
        "prompt_v1"
    ]
    assert lineage.policy_version_ids == [
        "policy_v1"
    ]


# ============================================================
# Isolation
# ============================================================


def test_laboratories_are_independent():
    first = AlphaDiscoveryLaboratory()
    second = AlphaDiscoveryLaboratory()

    first.create_feature(
        name="btc_momentum",
        feature_type=FeatureType.MOMENTUM,
        source="market_data",
    )

    assert (
        first.summary()["features"]
        == 1
    )

    assert (
        second.summary()["features"]
        == 0
    )


def test_results_returns_copy():
    lab = AlphaDiscoveryLaboratory()

    feature_one, feature_two, combination, candidate = (
        build_candidate(lab)
    )

    lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                feature_one.feature_id,
                feature_two.feature_id,
            ],
            combination_ids=[
                combination.combination_id
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_evidence_bundle(
                    candidate.candidate_id
                )
        },
    )

    results = lab.results()
    results.clear()

    assert len(
        lab.results()
    ) == 1


# ============================================================
# Safety
# ============================================================


def test_research_only():
    lab = AlphaDiscoveryLaboratory()

    summary = lab.summary()

    assert summary["research_only"] is True


def test_no_real_money_execution():
    lab = AlphaDiscoveryLaboratory()

    summary = lab.summary()

    assert (
        summary["real_money_execution"]
        is False
    )