import pytest

from .outcome_attribution_meta_learning import (
    OutcomeAttributionError,
    OutcomeAttributionMetaLearning,
    OutcomeLabel,
)


def test_version_and_safety_flags() -> None:
    service = OutcomeAttributionMetaLearning()

    assert service.VERSION == "v7.0.11"
    assert service.RESEARCH_ONLY is True
    assert service.REAL_MONEY_EXECUTION is False
    assert service.AUTOMATIC_TRADING is False


def test_positive_outcome_is_attributed() -> None:
    service = OutcomeAttributionMetaLearning()

    record = service.attribute(
        "outcome-001",
        {
            "macro_signal": 2.0,
            "risk_signal": 1.0,
        },
        realized_outcome=0.12,
        baseline_outcome=0.02,
    )

    assert record.label == OutcomeLabel.POSITIVE
    assert record.signal_contributions[
        "macro_signal"
    ] == pytest.approx(0.0666666667)
    assert record.signal_contributions[
        "risk_signal"
    ] == pytest.approx(0.0333333333)


def test_negative_outcome_is_attributed() -> None:
    service = OutcomeAttributionMetaLearning()

    record = service.attribute(
        "outcome-002",
        {"signal_a": 1.0},
        realized_outcome=-0.05,
    )

    assert record.label == OutcomeLabel.NEGATIVE
    assert record.signal_contributions["signal_a"] == -0.05


def test_baseline_is_removed_before_attribution() -> None:
    service = OutcomeAttributionMetaLearning()

    record = service.attribute(
        "outcome-003",
        {
            "signal_a": 1.0,
            "signal_b": 1.0,
        },
        realized_outcome=0.10,
        baseline_outcome=0.04,
    )

    assert sum(
        record.signal_contributions.values()
    ) == pytest.approx(0.06)


def test_meta_learning_updates_weights() -> None:
    service = OutcomeAttributionMetaLearning(
        learning_rate=0.5,
    )

    before = service.weight("signal_a")

    service.attribute(
        "outcome-004",
        {"signal_a": 1.0},
        realized_outcome=0.20,
    )

    after = service.weight("signal_a")

    assert before == 1.0
    assert after > before
    assert service.learning_updates()[0].observation_count == 1


def test_zero_scores_use_equal_attribution() -> None:
    service = OutcomeAttributionMetaLearning()

    record = service.attribute(
        "outcome-005",
        {
            "signal_a": 0.0,
            "signal_b": 0.0,
        },
        realized_outcome=0.10,
    )

    assert record.signal_contributions[
        "signal_a"
    ] == pytest.approx(0.05)
    assert record.signal_contributions[
        "signal_b"
    ] == pytest.approx(0.05)


def test_duplicate_outcome_is_rejected() -> None:
    service = OutcomeAttributionMetaLearning()

    service.attribute(
        "outcome-006",
        {"signal_a": 1.0},
        realized_outcome=0.10,
    )

    with pytest.raises(OutcomeAttributionError):
        service.attribute(
            "outcome-006",
            {"signal_a": 1.0},
            realized_outcome=0.20,
        )


def test_invalid_input_is_rejected() -> None:
    service = OutcomeAttributionMetaLearning()

    with pytest.raises(OutcomeAttributionError):
        service.attribute(
            "outcome-007",
            {},
            realized_outcome=0.10,
        )

    with pytest.raises(OutcomeAttributionError):
        OutcomeAttributionMetaLearning(
            learning_rate=0.0,
        )


def test_summary_and_snapshot() -> None:
    service = OutcomeAttributionMetaLearning()

    service.attribute(
        "outcome-008",
        {"signal_a": 1.0},
        realized_outcome=0.05,
    )

    summary = service.summary()
    snapshot = service.snapshot()

    assert summary["version"] == "v7.0.11"
    assert summary["outcome_count"] == 1
    assert summary["signal_count"] == 1
    assert summary["meta_learning_enabled"] is True

    assert snapshot["version"] == "v7.0.11"
    assert snapshot["research_only"] is True
    assert snapshot["real_money_execution"] is False