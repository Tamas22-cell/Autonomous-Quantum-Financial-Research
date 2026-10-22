import pytest

from autonomous_experiment_designer.uncertainty_detector import (
    Severity,
    UncertaintyDetector,
    UncertaintyType,
)


def test_default_detector():
    detector = UncertaintyDetector()

    assert detector.confidence_gap_threshold == 0.20
    assert detector.regime_change_threshold == 0.30
    assert detector.performance_gap_threshold == 0.10
    assert detector.evidence_conflict_threshold == 0.40


def test_no_uncertainty_returns_empty_report():
    report = UncertaintyDetector().detect(
        confidence=(0.90, 0.90),
        regime_distance=0.10,
        performance=(0.20, 0.21),
        evidence_agreement=0.90,
    )

    assert report.has_uncertainty is False
    assert report.signal_count == 0
    assert report.highest_severity is None


def test_confidence_gap_detected():
    signal = UncertaintyDetector().detect_confidence_gap(
        confidence=0.50,
        target_confidence=0.80,
    )

    assert signal is not None
    assert signal.signal_type == UncertaintyType.CONFIDENCE_GAP
    assert signal.score == pytest.approx(0.30)
    assert signal.severity == Severity.HIGH


def test_confidence_gap_below_threshold_is_ignored():
    signal = UncertaintyDetector().detect_confidence_gap(
        confidence=0.70,
        target_confidence=0.80,
    )

    assert signal is None


def test_regime_change_detected():
    signal = UncertaintyDetector().detect_regime_change(
        regime_distance=0.60,
    )

    assert signal is not None
    assert signal.signal_type == UncertaintyType.REGIME_CHANGE
    assert signal.score == pytest.approx(0.60)


def test_regime_change_below_threshold_is_ignored():
    signal = UncertaintyDetector().detect_regime_change(
        regime_distance=0.20,
    )

    assert signal is None


def test_performance_gap_detected():
    signal = UncertaintyDetector().detect_performance_gap(
        observed_performance=0.05,
        expected_performance=0.25,
    )

    assert signal is not None
    assert signal.signal_type == UncertaintyType.PERFORMANCE_GAP
    assert signal.score == pytest.approx(0.20)


def test_performance_gap_below_threshold_is_ignored():
    signal = UncertaintyDetector().detect_performance_gap(
        observed_performance=0.18,
        expected_performance=0.20,
    )

    assert signal is None


def test_evidence_conflict_detected():
    signal = UncertaintyDetector().detect_evidence_conflict(
        agreement_score=0.40,
    )

    assert signal is not None
    assert signal.signal_type == UncertaintyType.EVIDENCE_CONFLICT
    assert signal.score == pytest.approx(0.60)


def test_evidence_conflict_below_threshold_is_ignored():
    signal = UncertaintyDetector().detect_evidence_conflict(
        agreement_score=0.80,
    )

    assert signal is None


def test_detect_combines_multiple_signals():
    report = UncertaintyDetector().detect(
        confidence=(0.40, 0.80),
        regime_distance=0.70,
        performance=(0.00, 0.25),
        evidence_agreement=0.20,
    )

    assert report.signal_count == 4
    assert report.has_uncertainty is True


def test_highest_severity_is_selected():
    report = UncertaintyDetector().detect(
        confidence=(0.70, 0.90),
        regime_distance=0.90,
    )

    assert report.highest_severity == Severity.CRITICAL


def test_signals_are_sorted_by_severity():
    report = UncertaintyDetector().detect(
        confidence=(0.70, 0.90),
        regime_distance=0.90,
        performance=(0.00, 0.20),
    )

    order = {
        Severity.LOW: 0,
        Severity.MEDIUM: 1,
        Severity.HIGH: 2,
        Severity.CRITICAL: 3,
    }

    values = [
        order[signal.severity]
        for signal in report.signals
    ]

    assert values == sorted(values, reverse=True)


@pytest.mark.parametrize(
    "value",
    [0, -0.1, float("nan"), float("inf"), True, "0.2"],
)
def test_invalid_confidence_threshold(value):
    with pytest.raises(ValueError):
        UncertaintyDetector(
            confidence_gap_threshold=value
        )


@pytest.mark.parametrize(
    "value",
    [0, -0.1, float("nan"), float("inf"), True, "0.2"],
)
def test_invalid_regime_threshold(value):
    with pytest.raises(ValueError):
        UncertaintyDetector(
            regime_change_threshold=value
        )


def test_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        UncertaintyDetector().detect_confidence_gap(
            confidence=1.2,
            target_confidence=0.8,
        )


def test_target_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        UncertaintyDetector().detect_confidence_gap(
            confidence=0.5,
            target_confidence=1.2,
        )


def test_negative_regime_distance_rejected():
    with pytest.raises(ValueError):
        UncertaintyDetector().detect_regime_change(
            regime_distance=-0.1
        )


def test_agreement_score_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        UncertaintyDetector().detect_evidence_conflict(
            agreement_score=1.2
        )


def test_invalid_confidence_tuple_rejected():
    with pytest.raises(ValueError):
        UncertaintyDetector().detect(
            confidence=(0.5,),
        )


def test_invalid_performance_tuple_rejected():
    with pytest.raises(ValueError):
        UncertaintyDetector().detect(
            performance=(0.5,),
        )


def test_report_is_immutable():
    report = UncertaintyDetector().detect(
        regime_distance=0.6,
    )

    with pytest.raises(AttributeError):
        report.signals = ()


def test_signal_is_immutable():
    signal = UncertaintyDetector().detect_regime_change(
        regime_distance=0.6,
    )

    assert signal is not None

    with pytest.raises(AttributeError):
        signal.score = 1.0


def test_critical_signal():
    signal = UncertaintyDetector().detect_regime_change(
        regime_distance=0.95,
    )

    assert signal is not None
    assert signal.severity == Severity.CRITICAL


def test_boundary_threshold_is_detected():
    detector = UncertaintyDetector(
        regime_change_threshold=0.30,
    )

    signal = detector.detect_regime_change(
        regime_distance=0.30,
    )

    assert signal is not None


def test_detect_returns_tuple_signals():
    report = UncertaintyDetector().detect(
        regime_distance=0.50,
    )

    assert isinstance(report.signals, tuple)