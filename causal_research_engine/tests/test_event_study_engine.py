"""
Tests for the Event Study Engine.
Version: v5.0.5
"""

import pytest

from causal_research_engine.event_study_engine import (
    EventStudyEngine,
    EventStudyEngineError,
)


def test_analyze_calculates_pre_and_post_means():
    engine = EventStudyEngine()

    result = engine.analyze(
        event_name="BTC_ETF_APPROVAL",
        pre_event_values=[1.0, 2.0, 3.0],
        post_event_values=[4.0, 5.0, 6.0],
    )

    assert result.event_name == "BTC_ETF_APPROVAL"
    assert result.pre_event_mean == 2.0
    assert result.post_event_mean == 5.0
    assert result.abnormal_effect == 3.0
    assert result.cumulative_abnormal_effect == 3.0


def test_analyze_with_benchmark():
    engine = EventStudyEngine()

    result = engine.analyze(
        event_name="MARKET_EVENT",
        pre_event_values=[2.0, 3.0],
        post_event_values=[6.0, 7.0],
        benchmark_effect=1.0,
    )

    assert result.pre_event_mean == 2.5
    assert result.post_event_mean == 6.5
    assert result.abnormal_effect == 3.0


def test_analyze_preserves_statistics():
    engine = EventStudyEngine()

    result = engine.analyze(
        event_name="FED_EVENT",
        pre_event_values=[1.0, 2.0],
        post_event_values=[2.0, 4.0],
        p_value=0.03,
        confidence=0.95,
        notes="Event-study test.",
    )

    assert result.p_value == 0.03
    assert result.confidence == 0.95
    assert result.notes == "Event-study test."


def test_event_name_is_trimmed():
    engine = EventStudyEngine()

    result = engine.analyze(
        event_name="  BTC_EVENT  ",
        pre_event_values=[1.0],
        post_event_values=[2.0],
    )

    assert result.event_name == "BTC_EVENT"


def test_empty_event_name_rejected():
    engine = EventStudyEngine()

    with pytest.raises(
        EventStudyEngineError,
        match="event_name is required",
    ):
        engine.analyze(
            event_name=" ",
            pre_event_values=[1.0],
            post_event_values=[2.0],
        )


def test_empty_pre_event_values_rejected():
    engine = EventStudyEngine()

    with pytest.raises(
        EventStudyEngineError,
        match="pre_event_values cannot be empty",
    ):
        engine.analyze(
            event_name="TEST_EVENT",
            pre_event_values=[],
            post_event_values=[1.0, 2.0],
        )


def test_empty_post_event_values_rejected():
    engine = EventStudyEngine()

    with pytest.raises(
        EventStudyEngineError,
        match="post_event_values cannot be empty",
    ):
        engine.analyze(
            event_name="TEST_EVENT",
            pre_event_values=[1.0, 2.0],
            post_event_values=[],
        )


def test_non_numeric_pre_event_value_rejected():
    engine = EventStudyEngine()

    with pytest.raises(
        EventStudyEngineError,
        match="contains a non-numeric value",
    ):
        engine.analyze(
            event_name="TEST_EVENT",
            pre_event_values=[1.0, "bad"],
            post_event_values=[2.0],
        )


def test_non_numeric_post_event_value_rejected():
    engine = EventStudyEngine()

    with pytest.raises(
        EventStudyEngineError,
        match="contains a non-numeric value",
    ):
        engine.analyze(
            event_name="TEST_EVENT",
            pre_event_values=[1.0],
            post_event_values=[2.0, "bad"],
        )


def test_intervention_effect():
    engine = EventStudyEngine()

    result = engine.intervention_effect(
        event_name="INTERVENTION",
        pre_event_values=[2.0, 4.0],
        post_event_values=[7.0, 9.0],
        benchmark_effect=1.0,
    )

    assert result.pre_event_mean == 3.0
    assert result.post_event_mean == 8.0
    assert result.abnormal_effect == 4.0
    assert result.cumulative_abnormal_effect == 4.0


def test_cumulative_effect():
    engine = EventStudyEngine()

    result = engine.cumulative_effect(
        event_name="EVENT_WINDOW",
        abnormal_returns=[0.02, 0.03, -0.01, 0.04],
    )

    assert result.event_name == "EVENT_WINDOW"
    assert result.abnormal_effect == pytest.approx(0.02)
    assert result.cumulative_abnormal_effect == pytest.approx(0.08)


def test_cumulative_effect_with_statistics():
    engine = EventStudyEngine()

    result = engine.cumulative_effect(
        event_name="BTC_EVENT",
        abnormal_returns=[0.01, 0.02],
        p_value=0.04,
        confidence=0.90,
        notes="Cumulative event-window analysis.",
    )

    assert result.cumulative_abnormal_effect == pytest.approx(0.03)
    assert result.p_value == 0.04
    assert result.confidence == 0.90
    assert result.notes == (
        "Cumulative event-window analysis."
    )


def test_cumulative_effect_empty_values_rejected():
    engine = EventStudyEngine()

    with pytest.raises(
        EventStudyEngineError,
        match="abnormal_returns cannot be empty",
    ):
        engine.cumulative_effect(
            event_name="EMPTY_EVENT",
            abnormal_returns=[],
        )