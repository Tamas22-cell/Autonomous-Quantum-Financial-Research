from __future__ import annotations

import pytest

from alpha_discovery_laboratory.alpha_discovery_lab import (
    AlphaDiscoveryLaboratory,
    AlphaResearchInput,
    CandidateEvidenceBundle,
)
from alpha_discovery_laboratory.models import (
    CandidateStatus,
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


def make_overfit_evidence(
    *,
    bad: bool = False,
) -> OverfitEvidence:
    if bad:
        out_of_sample = make_metrics(
            total_return=0.01,
            annualized_return=0.01,
            sharpe_ratio=0.10,
            max_drawdown=0.30,
        )
    else:
        out_of_sample = make_metrics(
            total_return=0.17,
            annualized_return=0.16,
            sharpe_ratio=1.25,
            max_drawdown=0.14,
        )

    perturbations = [
        ParameterPerturbation(
            name="lookback_18",
            baseline_value=20.0,
            perturbed_value=18.0,
            performance_score=0.82,
        ),
        ParameterPerturbation(
            name="lookback_20",
            baseline_value=20.0,
            perturbed_value=20.0,
            performance_score=0.80,
        ),
        ParameterPerturbation(
            name="lookback_22",
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
        out_of_sample=out_of_sample,
        parameter_perturbations=perturbations,
        regime_performance=regimes,
        bootstrap_scores=[
            0.80,
            0.82,
            0.78,
            0.81,
        ],
        metadata={
            "stage": "e2e",
        },
    )


def make_bundle(
    candidate_id: str,
    *,
    validation_failure: bool = False,
    overfit_failure: bool = False,
) -> CandidateEvidenceBundle:
    if validation_failure:
        backtest = make_metrics(
            sharpe_ratio=0.10,
            profit_factor=0.80,
        )
    else:
        backtest = make_metrics()

    return CandidateEvidenceBundle(
        candidate_id=candidate_id,
        backtest=BacktestEvidence(
            metrics=backtest,
            data_leakage_detected=False,
            lookahead_bias_detected=False,
            metadata={
                "stage": "backtest",
            },
        ),
        causal=CausalEvidence(
            causal_score=0.80,
            stable_effect=True,
            confounding_detected=False,
            metadata={
                "stage": "causal",
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
                "stage": "oos",
            },
        ),
        robustness=RobustnessEvidence(
            robustness_score=0.80,
            parameter_sensitivity=0.20,
            regime_stable=True,
            perturbation_stable=True,
            metadata={
                "stage": "robustness",
            },
        ),
        overfit=make_overfit_evidence(
            bad=overfit_failure
        ),
        metadata={
            "pipeline": "e2e",
        },
    )


def build_research_pipeline(
    *,
    directions=None,
):
    lab = AlphaDiscoveryLaboratory()

    momentum = lab.create_feature(
        name="btc_momentum",
        feature_type=FeatureType.MOMENTUM,
        source="market_data",
        transform=FeatureTransform.RAW,
        dataset_id="btc_daily",
        dataset_fingerprint="btc_daily_fp",
        metadata={
            "asset": "BTC",
        },
    )

    volume = lab.create_feature(
        name="btc_volume",
        feature_type=FeatureType.VOLUME,
        source="market_data",
        transform=FeatureTransform.RAW,
        dataset_id="btc_daily",
        dataset_fingerprint="btc_daily_fp",
        metadata={
            "asset": "BTC",
        },
    )

    discovery = lab.discover_features(
        feature_ids=[
            momentum.feature_id,
            volume.feature_id,
        ],
        operators=[
            CombinationOperator.SPREAD,
        ],
        name_prefix="btc_discovery",
        metadata={
            "pipeline": "e2e",
        },
    )

    combination = discovery.combinations[0]

    generated = lab.generate_candidates(
        combination_ids=[
            combination.combination_id,
        ],
        directions=(
            directions
            or [StrategyDirection.LONG]
        ),
        name_prefix="btc_alpha",
        metadata={
            "pipeline": "e2e",
        },
    )

    return (
        lab,
        momentum,
        volume,
        combination,
        generated,
    )


def make_research_input(
    *,
    feature_ids,
    combination_ids,
    directions=None,
) -> AlphaResearchInput:
    return AlphaResearchInput(
        name="btc_alpha_discovery_e2e",
        dataset_ids=[
            "btc_daily",
        ],
        dataset_fingerprints=[
            "btc_daily_fp",
        ],
        feature_ids=list(feature_ids),
        combination_ids=list(
            combination_ids
        ),
        directions=list(
            directions
            or [StrategyDirection.LONG]
        ),
        model_version_ids=[
            "model_v6_0_3",
        ],
        prompt_version_ids=[
            "prompt_v6_0_3",
        ],
        policy_version_ids=[
            "policy_v6_0_3",
        ],
        seed=42,
        run_id="run_v603_e2e",
        configuration={
            "research_only": True,
            "validation": "strict",
        },
        metadata={
            "suite": "end_to_end",
        },
    )


# ============================================================
# E2E — Discovery
# ============================================================


def test_e2e_features_registered():
    (
        lab,
        momentum,
        volume,
        _,
        _,
    ) = build_research_pipeline()

    assert (
        lab.feature_discovery.exists(
            momentum.feature_id
        )
    )

    assert (
        lab.feature_discovery.exists(
            volume.feature_id
        )
    )


def test_e2e_feature_combination_registered():
    (
        lab,
        _,
        _,
        combination,
        _,
    ) = build_research_pipeline()

    assert (
        lab.feature_discovery
        .combination_exists(
            combination.combination_id
        )
    )


def test_e2e_candidate_generated():
    (
        _,
        _,
        _,
        _,
        generated,
    ) = build_research_pipeline()

    assert generated.candidate_count == 1


def test_e2e_candidate_registered():
    (
        lab,
        _,
        _,
        _,
        generated,
    ) = build_research_pipeline()

    candidate = generated.candidates[0]

    assert (
        lab.candidate_generator.exists(
            candidate.candidate_id
        )
    )


# ============================================================
# E2E — Successful Candidate
# ============================================================


def test_e2e_successful_candidate_pipeline():
    (
        lab,
        momentum,
        volume,
        combination,
        generated,
    ) = build_research_pipeline()

    candidate = generated.candidates[0]

    research_input = make_research_input(
        feature_ids=[
            momentum.feature_id,
            volume.feature_id,
        ],
        combination_ids=[
            combination.combination_id,
        ],
    )

    result = lab.run_experiment(
        research_input=research_input,
        evidence_by_candidate={
            candidate.candidate_id:
                make_bundle(
                    candidate.candidate_id
                )
        },
    )

    assert result.accepted_count == 1
    assert result.rejected_count == 0

    assert (
        candidate.candidate_id
        in result.accepted_candidate_ids
    )


def test_e2e_successful_validation_all_stages():
    (
        lab,
        momentum,
        volume,
        combination,
        generated,
    ) = build_research_pipeline()

    candidate = generated.candidates[0]

    result = lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                momentum.feature_id,
                volume.feature_id,
            ],
            combination_ids=[
                combination.combination_id,
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_bundle(
                    candidate.candidate_id
                )
        },
    )

    candidate_result = (
        result.candidate_results[0]
    )

    assert candidate_result.accepted is True
    assert (
        candidate_result.validation.accepted
        is True
    )
    assert (
        candidate_result
        .validation
        .passed_stage_count
        == 4
    )


def test_e2e_successful_overfit_gate():
    (
        lab,
        momentum,
        volume,
        combination,
        generated,
    ) = build_research_pipeline()

    candidate = generated.candidates[0]

    result = lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                momentum.feature_id,
                volume.feature_id,
            ],
            combination_ids=[
                combination.combination_id,
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_bundle(
                    candidate.candidate_id
                )
        },
    )

    overfit = (
        result
        .candidate_results[0]
        .overfit
    )

    assert overfit is not None
    assert overfit.overfit_detected is False
    assert overfit.unstable_detected is False
    assert overfit.accepted is True


def test_e2e_successful_candidate_status():
    (
        lab,
        momentum,
        volume,
        combination,
        generated,
    ) = build_research_pipeline()

    candidate = generated.candidates[0]

    lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                momentum.feature_id,
                volume.feature_id,
            ],
            combination_ids=[
                combination.combination_id,
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_bundle(
                    candidate.candidate_id
                )
        },
    )

    assert (
        candidate.status
        == CandidateStatus.ACCEPTED
    )


# ============================================================
# E2E — Validation Rejection
# ============================================================


def test_e2e_validation_rejection():
    (
        lab,
        momentum,
        volume,
        combination,
        generated,
    ) = build_research_pipeline()

    candidate = generated.candidates[0]

    result = lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                momentum.feature_id,
                volume.feature_id,
            ],
            combination_ids=[
                combination.combination_id,
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_bundle(
                    candidate.candidate_id,
                    validation_failure=True,
                )
        },
    )

    candidate_result = (
        result.candidate_results[0]
    )

    assert candidate_result.accepted is False
    assert (
        candidate_result.validation.accepted
        is False
    )
    assert candidate_result.overfit is None
    assert candidate.rejected is True


def test_e2e_validation_rejection_recorded():
    (
        lab,
        momentum,
        volume,
        combination,
        generated,
    ) = build_research_pipeline()

    candidate = generated.candidates[0]

    result = lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                momentum.feature_id,
                volume.feature_id,
            ],
            combination_ids=[
                combination.combination_id,
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_bundle(
                    candidate.candidate_id,
                    validation_failure=True,
                )
        },
    )

    assert result.accepted_count == 0
    assert result.rejected_count == 1

    assert (
        candidate.candidate_id
        in result.rejected_candidate_ids
    )


# ============================================================
# E2E — Overfit Rejection
# ============================================================


def test_e2e_overfit_rejection():
    (
        lab,
        momentum,
        volume,
        combination,
        generated,
    ) = build_research_pipeline()

    candidate = generated.candidates[0]

    result = lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                momentum.feature_id,
                volume.feature_id,
            ],
            combination_ids=[
                combination.combination_id,
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_bundle(
                    candidate.candidate_id,
                    overfit_failure=True,
                )
        },
    )

    candidate_result = (
        result.candidate_results[0]
    )

    assert (
        candidate_result.validation.accepted
        is True
    )
    assert candidate_result.overfit is not None
    assert (
        candidate_result
        .overfit
        .overfit_detected
        is True
    )
    assert candidate_result.accepted is False
    assert (
        candidate_result.rejection_reason
        == RejectionReason.OVERFIT
    )


def test_e2e_overfit_rejection_applied():
    (
        lab,
        momentum,
        volume,
        combination,
        generated,
    ) = build_research_pipeline()

    candidate = generated.candidates[0]

    lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                momentum.feature_id,
                volume.feature_id,
            ],
            combination_ids=[
                combination.combination_id,
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_bundle(
                    candidate.candidate_id,
                    overfit_failure=True,
                )
        },
    )

    assert candidate.rejected is True
    assert (
        candidate.rejection_reason
        == RejectionReason.OVERFIT
    )


# ============================================================
# E2E — Lineage
# ============================================================


def test_e2e_lineage_created():
    (
        lab,
        momentum,
        volume,
        combination,
        generated,
    ) = build_research_pipeline()

    candidate = generated.candidates[0]

    result = lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                momentum.feature_id,
                volume.feature_id,
            ],
            combination_ids=[
                combination.combination_id,
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_bundle(
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
        lineage.candidate_id
        == candidate.candidate_id
    )

    assert (
        lineage.experiment_id
        == result.experiment_id
    )


def test_e2e_lineage_preserves_features():
    (
        lab,
        momentum,
        volume,
        combination,
        generated,
    ) = build_research_pipeline()

    candidate = generated.candidates[0]

    lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                momentum.feature_id,
                volume.feature_id,
            ],
            combination_ids=[
                combination.combination_id,
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_bundle(
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
        combination.combination_id
        in lineage.combination_ids
    )


def test_e2e_lineage_preserves_dataset():
    (
        lab,
        momentum,
        volume,
        combination,
        generated,
    ) = build_research_pipeline()

    candidate = generated.candidates[0]

    lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                momentum.feature_id,
                volume.feature_id,
            ],
            combination_ids=[
                combination.combination_id,
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_bundle(
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
        "btc_daily"
    ]

    assert lineage.dataset_fingerprints == [
        "btc_daily_fp"
    ]


def test_e2e_lineage_preserves_versions():
    (
        lab,
        momentum,
        volume,
        combination,
        generated,
    ) = build_research_pipeline()

    candidate = generated.candidates[0]

    lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                momentum.feature_id,
                volume.feature_id,
            ],
            combination_ids=[
                combination.combination_id,
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_bundle(
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
        "model_v6_0_3"
    ]
    assert lineage.prompt_version_ids == [
        "prompt_v6_0_3"
    ]
    assert lineage.policy_version_ids == [
        "policy_v6_0_3"
    ]


# ============================================================
# E2E — Experiment Registry
# ============================================================


def test_e2e_experiment_registered():
    (
        lab,
        momentum,
        volume,
        combination,
        generated,
    ) = build_research_pipeline()

    candidate = generated.candidates[0]

    result = lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                momentum.feature_id,
                volume.feature_id,
            ],
            combination_ids=[
                combination.combination_id,
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_bundle(
                    candidate.candidate_id
                )
        },
    )

    assert (
        lab.lineage_registry
        .experiment_exists(
            result.experiment_id
        )
    )


def test_e2e_result_registered():
    (
        lab,
        momentum,
        volume,
        combination,
        generated,
    ) = build_research_pipeline()

    candidate = generated.candidates[0]

    result = lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                momentum.feature_id,
                volume.feature_id,
            ],
            combination_ids=[
                combination.combination_id,
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_bundle(
                    candidate.candidate_id
                )
        },
    )

    stored = lab.get_result(
        result.experiment_id
    )

    assert stored == result


# ============================================================
# E2E — Multiple Candidates
# ============================================================


def test_e2e_multiple_candidates():
    (
        lab,
        momentum,
        volume,
        combination,
        generated,
    ) = build_research_pipeline(
        directions=[
            StrategyDirection.LONG,
            StrategyDirection.SHORT,
        ]
    )

    assert generated.candidate_count == 2

    first = generated.candidates[0]
    second = generated.candidates[1]

    evidence = {
        first.candidate_id:
            make_bundle(
                first.candidate_id
            ),
        second.candidate_id:
            make_bundle(
                second.candidate_id,
                validation_failure=True,
            ),
    }

    result = lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                momentum.feature_id,
                volume.feature_id,
            ],
            combination_ids=[
                combination.combination_id,
            ],
            directions=[
                StrategyDirection.LONG,
                StrategyDirection.SHORT,
            ],
        ),
        evidence_by_candidate=evidence,
    )

    assert len(
        result.candidate_results
    ) == 2

    assert result.accepted_count == 1
    assert result.rejected_count == 1


def test_e2e_multiple_candidate_lineage():
    (
        lab,
        momentum,
        volume,
        combination,
        generated,
    ) = build_research_pipeline(
        directions=[
            StrategyDirection.LONG,
            StrategyDirection.SHORT,
        ]
    )

    first = generated.candidates[0]
    second = generated.candidates[1]

    lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                momentum.feature_id,
                volume.feature_id,
            ],
            combination_ids=[
                combination.combination_id,
            ],
            directions=[
                StrategyDirection.LONG,
                StrategyDirection.SHORT,
            ],
        ),
        evidence_by_candidate={
            first.candidate_id:
                make_bundle(
                    first.candidate_id
                ),
            second.candidate_id:
                make_bundle(
                    second.candidate_id,
                    validation_failure=True,
                ),
        },
    )

    assert (
        lab.lineage_registry.lineage_count()
        == 2
    )


# ============================================================
# E2E — Reproducibility
# ============================================================


def test_e2e_configuration_fingerprint_stable():
    first = AlphaDiscoveryLaboratory()
    second = AlphaDiscoveryLaboratory()

    assert (
        first.configuration_fingerprint()
        ==
        second.configuration_fingerprint()
    )


def test_e2e_feature_fingerprint_reproducible():
    first_lab = AlphaDiscoveryLaboratory()
    second_lab = AlphaDiscoveryLaboratory()

    first = first_lab.create_feature(
        name="btc_momentum",
        feature_type=FeatureType.MOMENTUM,
        source="market_data",
        transform=FeatureTransform.RAW,
        dataset_id="btc_daily",
        dataset_fingerprint="btc_daily_fp",
    )

    second = second_lab.create_feature(
        name="btc_momentum",
        feature_type=FeatureType.MOMENTUM,
        source="market_data",
        transform=FeatureTransform.RAW,
        dataset_id="btc_daily",
        dataset_fingerprint="btc_daily_fp",
    )

    assert (
        first.fingerprint
        == second.fingerprint
    )

    assert (
        first.feature_id
        == second.feature_id
    )


def test_e2e_research_input_reproducible():
    first = make_research_input(
        feature_ids=[
            "feature_one",
            "feature_two",
        ],
        combination_ids=[
            "combination_one",
        ],
    )

    second = make_research_input(
        feature_ids=[
            "feature_one",
            "feature_two",
        ],
        combination_ids=[
            "combination_one",
        ],
    )

    assert (
        first.fingerprint
        == second.fingerprint
    )


# ============================================================
# E2E — Auditability
# ============================================================


def test_e2e_audit_history_created():
    (
        lab,
        momentum,
        volume,
        combination,
        generated,
    ) = build_research_pipeline()

    candidate = generated.candidates[0]

    lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                momentum.feature_id,
                volume.feature_id,
            ],
            combination_ids=[
                combination.combination_id,
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_bundle(
                    candidate.candidate_id
                )
        },
    )

    assert len(
        lab.audit_log()
    ) > 0

    assert len(
        lab.validation_pipeline.audit_log()
    ) > 0

    assert len(
        lab.overfit_detector.audit_log()
    ) > 0

    assert len(
        lab.lineage_registry.audit_log()
    ) > 0


# ============================================================
# E2E — Summary
# ============================================================


def test_e2e_summary():
    (
        lab,
        momentum,
        volume,
        combination,
        generated,
    ) = build_research_pipeline()

    candidate = generated.candidates[0]

    lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                momentum.feature_id,
                volume.feature_id,
            ],
            combination_ids=[
                combination.combination_id,
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_bundle(
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
# E2E — Safety Boundary
# ============================================================


def test_e2e_research_only():
    lab = AlphaDiscoveryLaboratory()

    summary = lab.summary()

    assert summary["research_only"] is True


def test_e2e_no_real_money_execution():
    lab = AlphaDiscoveryLaboratory()

    summary = lab.summary()

    assert (
        summary["real_money_execution"]
        is False
    )


def test_e2e_complete_research_safety_boundary():
    (
        lab,
        momentum,
        volume,
        combination,
        generated,
    ) = build_research_pipeline()

    candidate = generated.candidates[0]

    result = lab.run_experiment(
        research_input=make_research_input(
            feature_ids=[
                momentum.feature_id,
                volume.feature_id,
            ],
            combination_ids=[
                combination.combination_id,
            ],
        ),
        evidence_by_candidate={
            candidate.candidate_id:
                make_bundle(
                    candidate.candidate_id
                )
        },
    )

    assert result.accepted_count == 1

    summary = lab.summary()

    assert summary["research_only"] is True
    assert (
        summary["real_money_execution"]
        is False
    )