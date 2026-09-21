from __future__ import annotations

from v7_data_fabric.statistical_validation import (
    StatisticalValidationFacade,
)


def _passed_results():
    from research_benchmarking.models import (
        ValidationResult,
        ValidationStage,
        ValidationStatus,
    )

    return [
        ValidationResult(
            stage=stage,
            status=ValidationStatus.PASSED,
            score=1.0,
            message=f"{stage.value} passed",
        )
        for stage in ValidationStage
    ]


def test_facade_version_and_safety_flags():
    facade = StatisticalValidationFacade()

    assert facade.version == "v7.0.6"
    assert facade.RESEARCH_ONLY is True
    assert facade.REAL_MONEY_EXECUTION is False
    assert facade.ANTI_OVERFITTING_ENABLED is True


def test_snapshot_contains_engine_configuration():
    facade = StatisticalValidationFacade()

    snapshot = facade.snapshot()

    assert snapshot["component"] == (
        "statistical_validation"
    )
    assert snapshot["version"] == "v7.0.6"
    assert snapshot["research_only"] is True
    assert snapshot["real_money_execution"] is False
    assert snapshot["anti_overfitting_enabled"] is True
    assert snapshot["automatic_alpha_approval"] is False
    assert isinstance(snapshot["required_stages"], list)


def test_analysis_delegates_to_benchmark_engine():
    facade = StatisticalValidationFacade()

    analysis = facade.analyze(_passed_results())

    assert analysis["component"] == (
        "statistical_validation"
    )
    assert analysis["version"] == "v7.0.6"
    assert analysis["research_only"] is True
    assert analysis["real_money_execution"] is False
    assert analysis["anti_overfitting_enabled"] is True
    assert analysis["automatic_alpha_approval"] is False
    assert "summary" in analysis
    assert "decision" in analysis
    assert facade.analysis_count == 1


def test_last_analysis_and_summary():
    facade = StatisticalValidationFacade()

    assert facade.last_analysis is None

    facade.analyze(_passed_results())

    assert facade.last_analysis is not None

    summary = facade.summary()

    assert summary["component"] == (
        "statistical_validation"
    )
    assert summary["version"] == "v7.0.6"
    assert summary["analysis_count"] == 1
    assert summary["has_last_analysis"] is True
    assert summary["research_only"] is True
    assert summary["real_money_execution"] is False
    assert summary["automatic_alpha_approval"] is False


def test_empty_results_are_rejected():
    facade = StatisticalValidationFacade()

    try:
        facade.analyze([])
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Empty validation results must be rejected."
        )