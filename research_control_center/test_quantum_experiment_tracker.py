import pytest

from research_control_center.models import (
    QuantumClassicalExperiment,
)
from research_control_center.quantum_experiment_tracker import (
    QuantumExperimentTracker,
    QuantumExperimentTrackerConfig,
)


def make_tracker() -> QuantumExperimentTracker:
    return QuantumExperimentTracker()


def make_experiment(
    problem_name: str = "Portfolio Selection",
) -> QuantumClassicalExperiment:
    return QuantumClassicalExperiment(
        problem_name=problem_name,
        classical_method="MeanVariance",
        quantum_method="QAOA",
        asset_universe=[
            "NVDA",
            "MSFT",
            "META",
        ],
        classical_score=0.80,
        quantum_score=0.85,
        classical_runtime_seconds=2.0,
        quantum_runtime_seconds=4.0,
        classical_objective=0.90,
        quantum_objective=0.85,
        approximation_ratio=0.94,
        qubits=6,
        circuit_depth=24,
        repetitions=5,
    )


# ============================================================
# CONFIG
# ============================================================


def test_default_config():
    config = QuantumExperimentTrackerConfig()

    assert (
        config.minimum_approximation_ratio
        == 0.80
    )
    assert (
        config.maximum_runtime_ratio
        == 10.0
    )
    assert (
        config.minimum_score_improvement
        == 0.0
    )


def test_custom_config():
    config = QuantumExperimentTrackerConfig(
        minimum_approximation_ratio=0.90,
        maximum_runtime_ratio=5.0,
        minimum_score_improvement=0.05,
    )

    assert (
        config.minimum_approximation_ratio
        == 0.90
    )
    assert (
        config.maximum_runtime_ratio
        == 5.0
    )
    assert (
        config.minimum_score_improvement
        == 0.05
    )


@pytest.mark.parametrize(
    "field",
    [
        "minimum_approximation_ratio",
        "maximum_runtime_ratio",
        "minimum_score_improvement",
    ],
)
def test_boolean_config_rejected(field):
    with pytest.raises(TypeError):
        QuantumExperimentTrackerConfig(
            **{field: True}
        )


def test_invalid_approximation_config_rejected():
    with pytest.raises(ValueError):
        QuantumExperimentTrackerConfig(
            minimum_approximation_ratio=1.1
        )


def test_invalid_runtime_ratio_config_rejected():
    with pytest.raises(ValueError):
        QuantumExperimentTrackerConfig(
            maximum_runtime_ratio=0.0
        )


# ============================================================
# REGISTRATION
# ============================================================


def test_register_experiment():
    tracker = make_tracker()

    experiment = make_experiment()

    result = tracker.register(
        experiment
    )

    assert (
        result.experiment_id
        == experiment.experiment_id
    )
    assert tracker.count() == 1


def test_registration_is_research_only():
    tracker = make_tracker()

    result = tracker.register(
        make_experiment()
    )

    assert (
        result.metadata["research_only"]
        is True
    )
    assert (
        result.metadata[
            "real_money_execution"
        ]
        is False
    )


def test_duplicate_registration_rejected():
    tracker = make_tracker()

    experiment = make_experiment()

    tracker.register(
        experiment
    )

    with pytest.raises(ValueError):
        tracker.register(
            experiment
        )


def test_invalid_registration_type_rejected():
    tracker = make_tracker()

    with pytest.raises(TypeError):
        tracker.register(
            "invalid"
        )


def test_empty_problem_name_rejected():
    tracker = make_tracker()

    experiment = make_experiment()
    experiment.problem_name = ""

    with pytest.raises(ValueError):
        tracker.register(
            experiment
        )


def test_empty_classical_method_rejected():
    tracker = make_tracker()

    experiment = make_experiment()
    experiment.classical_method = " "

    with pytest.raises(ValueError):
        tracker.register(
            experiment
        )


def test_empty_quantum_method_rejected():
    tracker = make_tracker()

    experiment = make_experiment()
    experiment.quantum_method = ""

    with pytest.raises(ValueError):
        tracker.register(
            experiment
        )


def test_empty_asset_universe_rejected():
    tracker = make_tracker()

    experiment = make_experiment()
    experiment.asset_universe = []

    with pytest.raises(ValueError):
        tracker.register(
            experiment
        )


def test_duplicate_assets_removed():
    tracker = make_tracker()

    experiment = make_experiment()

    experiment.asset_universe = [
        "BTC",
        "BTC",
        "ETH",
    ]

    result = tracker.register(
        experiment
    )

    assert result.asset_universe == [
        "BTC",
        "ETH",
    ]


def test_invalid_runtime_rejected():
    tracker = make_tracker()

    experiment = make_experiment()

    experiment.quantum_runtime_seconds = (
        -1.0
    )

    with pytest.raises(ValueError):
        tracker.register(
            experiment
        )


def test_invalid_approximation_ratio_rejected():
    tracker = make_tracker()

    experiment = make_experiment()

    experiment.approximation_ratio = 1.1

    with pytest.raises(ValueError):
        tracker.register(
            experiment
        )


def test_invalid_qubits_rejected():
    tracker = make_tracker()

    experiment = make_experiment()

    experiment.qubits = -1

    with pytest.raises(ValueError):
        tracker.register(
            experiment
        )


def test_invalid_repetitions_rejected():
    tracker = make_tracker()

    experiment = make_experiment()

    experiment.repetitions = 0

    with pytest.raises(ValueError):
        tracker.register(
            experiment
        )


# ============================================================
# CREATE
# ============================================================


def test_create_experiment():
    tracker = make_tracker()

    result = tracker.create(
        problem_name="QUBO Portfolio",
        classical_method="MinimumVariance",
        quantum_method="QAOA",
        asset_universe=[
            "NVDA",
            "MSFT",
        ],
        classical_score=0.80,
        quantum_score=0.82,
        classical_runtime_seconds=1.0,
        quantum_runtime_seconds=3.0,
        classical_objective=0.90,
        quantum_objective=0.86,
        approximation_ratio=0.95,
        qubits=4,
        circuit_depth=16,
        repetitions=3,
    )

    assert (
        result.problem_name
        == "QUBO Portfolio"
    )
    assert (
        result.quantum_method
        == "QAOA"
    )
    assert tracker.count() == 1


# ============================================================
# READ
# ============================================================


def test_get_experiment():
    tracker = make_tracker()

    created = tracker.register(
        make_experiment()
    )

    result = tracker.get(
        created.experiment_id
    )

    assert (
        result.experiment_id
        == created.experiment_id
    )


def test_unknown_experiment_rejected():
    tracker = make_tracker()

    with pytest.raises(KeyError):
        tracker.get(
            "missing"
        )


def test_get_returns_copy():
    tracker = make_tracker()

    created = tracker.register(
        make_experiment()
    )

    result = tracker.get(
        created.experiment_id
    )

    result.problem_name = "CHANGED"

    original = tracker.get(
        created.experiment_id
    )

    assert (
        original.problem_name
        != "CHANGED"
    )


def test_list_all():
    tracker = make_tracker()

    tracker.register(
        make_experiment(
            "Experiment A"
        )
    )

    tracker.register(
        make_experiment(
            "Experiment B"
        )
    )

    assert (
        len(tracker.list_all())
        == 2
    )


# ============================================================
# SCORE IMPROVEMENT
# ============================================================


def test_score_improvement():
    experiment = make_experiment()

    result = (
        QuantumExperimentTracker
        .score_improvement(
            experiment
        )
    )

    assert result == pytest.approx(
        0.05
    )


def test_missing_score_returns_none():
    experiment = make_experiment()

    experiment.quantum_score = None

    result = (
        QuantumExperimentTracker
        .score_improvement(
            experiment
        )
    )

    assert result is None


# ============================================================
# RUNTIME
# ============================================================


def test_runtime_ratio():
    experiment = make_experiment()

    result = (
        QuantumExperimentTracker
        .runtime_ratio(
            experiment
        )
    )

    assert result == pytest.approx(
        2.0
    )


def test_zero_zero_runtime_ratio():
    experiment = make_experiment()

    experiment.classical_runtime_seconds = (
        0.0
    )
    experiment.quantum_runtime_seconds = (
        0.0
    )

    result = (
        QuantumExperimentTracker
        .runtime_ratio(
            experiment
        )
    )

    assert result == 1.0


def test_zero_classical_runtime_gives_inf():
    experiment = make_experiment()

    experiment.classical_runtime_seconds = (
        0.0
    )
    experiment.quantum_runtime_seconds = (
        1.0
    )

    result = (
        QuantumExperimentTracker
        .runtime_ratio(
            experiment
        )
    )

    assert result == float("inf")


# ============================================================
# OBJECTIVE
# ============================================================


def test_objective_gap():
    experiment = make_experiment()

    result = (
        QuantumExperimentTracker
        .objective_gap(
            experiment
        )
    )

    assert result == pytest.approx(
        -0.05
    )


def test_missing_objective_returns_none():
    experiment = make_experiment()

    experiment.quantum_objective = None

    result = (
        QuantumExperimentTracker
        .objective_gap(
            experiment
        )
    )

    assert result is None


# ============================================================
# APPROXIMATION RATIO
# ============================================================


def test_calculate_approximation_ratio():
    tracker = make_tracker()

    experiment = make_experiment()

    experiment.approximation_ratio = None

    created = tracker.register(
        experiment
    )

    ratio = (
        tracker
        .calculate_approximation_ratio(
            created.experiment_id
        )
    )

    assert ratio == pytest.approx(
        0.85 / 0.90
    )


def test_zero_objectives_give_perfect_ratio():
    tracker = make_tracker()

    experiment = make_experiment()

    experiment.approximation_ratio = None
    experiment.classical_objective = 0.0
    experiment.quantum_objective = 0.0

    created = tracker.register(
        experiment
    )

    ratio = (
        tracker
        .calculate_approximation_ratio(
            created.experiment_id
        )
    )

    assert ratio == 1.0


def test_missing_objective_cannot_calculate_ratio():
    tracker = make_tracker()

    experiment = make_experiment()

    experiment.approximation_ratio = None
    experiment.quantum_objective = None

    created = tracker.register(
        experiment
    )

    ratio = (
        tracker
        .calculate_approximation_ratio(
            created.experiment_id
        )
    )

    assert ratio is None


# ============================================================
# COMPETITIVENESS
# ============================================================


def test_competitive_experiment():
    tracker = make_tracker()

    created = tracker.register(
        make_experiment()
    )

    report = (
        tracker
        .competitiveness_report(
            created.experiment_id
        )
    )

    assert (
        report["score_requirement"]
        is True
    )
    assert (
        report[
            "approximation_requirement"
        ]
        is True
    )
    assert (
        report["runtime_requirement"]
        is True
    )
    assert report["competitive"] is True


def test_non_competitive_low_approximation():
    tracker = make_tracker()

    experiment = make_experiment()

    experiment.approximation_ratio = 0.50

    created = tracker.register(
        experiment
    )

    report = (
        tracker
        .competitiveness_report(
            created.experiment_id
        )
    )

    assert (
        report["competitive"]
        is False
    )


def test_non_competitive_runtime():
    tracker = make_tracker()

    experiment = make_experiment()

    experiment.classical_runtime_seconds = (
        1.0
    )
    experiment.quantum_runtime_seconds = (
        20.0
    )

    created = tracker.register(
        experiment
    )

    report = (
        tracker
        .competitiveness_report(
            created.experiment_id
        )
    )

    assert (
        report["runtime_requirement"]
        is False
    )
    assert (
        report["competitive"]
        is False
    )


def test_competitiveness_is_research_only():
    tracker = make_tracker()

    created = tracker.register(
        make_experiment()
    )

    report = (
        tracker
        .competitiveness_report(
            created.experiment_id
        )
    )

    assert (
        report["research_only"]
        is True
    )
    assert (
        report[
            "real_money_execution"
        ]
        is False
    )


# ============================================================
# FILTERING
# ============================================================


def test_filter_by_problem_name():
    tracker = make_tracker()

    tracker.register(
        make_experiment(
            "Portfolio A"
        )
    )

    tracker.register(
        make_experiment(
            "Portfolio B"
        )
    )

    results = tracker.filter(
        problem_name="Portfolio A"
    )

    assert len(results) == 1
    assert (
        results[0].problem_name
        == "Portfolio A"
    )


def test_filter_by_quantum_method():
    tracker = make_tracker()

    first = make_experiment(
        "A"
    )

    second = make_experiment(
        "B"
    )

    second.quantum_method = "VQE"

    tracker.register(first)
    tracker.register(second)

    results = tracker.filter(
        quantum_method="VQE"
    )

    assert len(results) == 1
    assert (
        results[0].quantum_method
        == "VQE"
    )


def test_filter_by_asset():
    tracker = make_tracker()

    tracker.register(
        make_experiment()
    )

    results = tracker.filter(
        asset="NVDA"
    )

    assert len(results) == 1


# ============================================================
# QUALITY / RANKING
# ============================================================


def test_quality_score_is_bounded():
    tracker = make_tracker()

    score = (
        tracker
        .experiment_quality_score(
            make_experiment()
        )
    )

    assert 0.0 <= score <= 1.0


def test_rank_best_experiment_first():
    tracker = make_tracker()

    good = make_experiment(
        "Good"
    )

    good.quantum_score = 0.95
    good.approximation_ratio = 0.99
    good.classical_runtime_seconds = 2.0
    good.quantum_runtime_seconds = 2.0

    weak = make_experiment(
        "Weak"
    )

    weak.quantum_score = 0.50
    weak.approximation_ratio = 0.40
    weak.classical_runtime_seconds = 1.0
    weak.quantum_runtime_seconds = 10.0

    tracker.register(weak)
    tracker.register(good)

    ranking = tracker.rank()

    assert (
        ranking[0].problem_name
        == "Good"
    )


def test_top_limits_results():
    tracker = make_tracker()

    tracker.register(
        make_experiment("A")
    )
    tracker.register(
        make_experiment("B")
    )
    tracker.register(
        make_experiment("C")
    )

    assert len(
        tracker.top(2)
    ) == 2


def test_invalid_top_limit_rejected():
    tracker = make_tracker()

    with pytest.raises(ValueError):
        tracker.top(0)


# ============================================================
# COMPARISON
# ============================================================


def test_compare_experiments():
    tracker = make_tracker()

    weak = make_experiment(
        "Weak"
    )
    weak.quantum_score = 0.60
    weak.approximation_ratio = 0.50
    weak.quantum_runtime_seconds = 10.0

    good = make_experiment(
        "Good"
    )
    good.quantum_score = 0.95
    good.approximation_ratio = 0.99
    good.quantum_runtime_seconds = 2.0

    weak = tracker.register(
        weak
    )
    good = tracker.register(
        good
    )

    comparison = tracker.compare(
        weak.experiment_id,
        good.experiment_id,
    )

    assert (
        comparison[
            "quality_score_b"
        ]
        > comparison[
            "quality_score_a"
        ]
    )


# ============================================================
# SUMMARY
# ============================================================


def test_summary():
    tracker = make_tracker()

    tracker.register(
        make_experiment(
            "Competitive"
        )
    )

    weak = make_experiment(
        "Weak"
    )

    weak.quantum_score = 0.40
    weak.approximation_ratio = 0.40

    tracker.register(
        weak
    )

    summary = tracker.summary()

    assert (
        summary["total_experiments"]
        == 2
    )
    assert (
        summary[
            "competitive_experiments"
        ]
        == 1
    )
    assert (
        summary[
            "non_competitive_experiments"
        ]
        == 1
    )


# ============================================================
# AUDIT
# ============================================================


def test_registration_creates_audit_event():
    tracker = make_tracker()

    created = tracker.register(
        make_experiment()
    )

    events = tracker.audit_log(
        created.experiment_id
    )

    assert len(events) == 1
    assert (
        events[0]["event"]
        == "QUANTUM_EXPERIMENT_REGISTERED"
    )


def test_ratio_calculation_is_audited():
    tracker = make_tracker()

    experiment = make_experiment()
    experiment.approximation_ratio = None

    created = tracker.register(
        experiment
    )

    tracker.calculate_approximation_ratio(
        created.experiment_id
    )

    events = tracker.audit_log(
        created.experiment_id
    )

    assert len(events) == 2
    assert (
        events[1]["event"]
        == "APPROXIMATION_RATIO_CALCULATED"
    )


def test_audit_is_research_only():
    tracker = make_tracker()

    created = tracker.register(
        make_experiment()
    )

    event = tracker.audit_log(
        created.experiment_id
    )[0]

    assert (
        event["research_only"]
        is True
    )
    assert (
        event[
            "real_money_execution"
        ]
        is False
    )


# ============================================================
# SNAPSHOT
# ============================================================


def test_snapshot():
    tracker = make_tracker()

    tracker.register(
        make_experiment()
    )

    snapshot = tracker.snapshot()

    assert (
        len(snapshot["experiments"])
        == 1
    )
    assert (
        len(snapshot["ranking"])
        == 1
    )
    assert (
        snapshot["summary"][
            "total_experiments"
        ]
        == 1
    )


def test_snapshot_is_research_only():
    tracker = make_tracker()

    snapshot = tracker.snapshot()

    assert (
        snapshot["research_only"]
        is True
    )
    assert (
        snapshot[
            "real_money_execution"
        ]
        is False
    )