from __future__ import annotations

import asyncio

import pytest

from production_research_infrastructure.event_bus import (
    DeliveryStatus,
    EventBus,
    EventBusError,
    EventPriority,
    EventType,
    EventValidationError,
    ResearchEvent,
    SubscriberError,
    Subscription,
)


def make_event(
    event_type: EventType = EventType.RESEARCH_REQUESTED,
    *,
    run_id: str = "run_test_001",
    correlation_id: str = "corr_test_001",
) -> ResearchEvent:
    return ResearchEvent(
        event_type=event_type,
        payload={
            "asset": "BTC",
            "horizon": "1D",
        },
        source="test_suite",
        run_id=run_id,
        correlation_id=correlation_id,
    )


# ============================================================
# ResearchEvent
# ============================================================


def test_event_creation() -> None:
    event = make_event()

    assert event.event_type == EventType.RESEARCH_REQUESTED
    assert event.event_id.startswith("event_")
    assert event.run_id == "run_test_001"


def test_event_default_priority() -> None:
    event = make_event()

    assert event.priority == EventPriority.NORMAL


def test_event_custom_priority() -> None:
    event = ResearchEvent(
        event_type=EventType.RISK_ASSESSED,
        priority=EventPriority.CRITICAL,
    )

    assert event.priority == EventPriority.CRITICAL


def test_event_research_safety() -> None:
    event = make_event()

    assert event.metadata["research_only"] is True
    assert event.metadata["real_money_execution"] is False


def test_event_requires_event_type_enum() -> None:
    with pytest.raises(TypeError):
        ResearchEvent(
            event_type="RESEARCH_REQUESTED",  # type: ignore[arg-type]
        )


def test_event_requires_dict_payload() -> None:
    with pytest.raises(TypeError):
        ResearchEvent(
            event_type=EventType.RESEARCH_REQUESTED,
            payload=["BTC"],  # type: ignore[arg-type]
        )


def test_event_requires_priority_enum() -> None:
    with pytest.raises(TypeError):
        ResearchEvent(
            event_type=EventType.RESEARCH_REQUESTED,
            priority=20,  # type: ignore[arg-type]
        )


def test_event_rejects_empty_run_id() -> None:
    with pytest.raises(EventValidationError):
        ResearchEvent(
            event_type=EventType.RESEARCH_REQUESTED,
            run_id=" ",
        )


def test_event_trims_identifiers() -> None:
    event = ResearchEvent(
        event_type=EventType.RESEARCH_REQUESTED,
        source="  orchestrator  ",
        run_id="  run_1  ",
        correlation_id="  corr_1  ",
    )

    assert event.source == "orchestrator"
    assert event.run_id == "run_1"
    assert event.correlation_id == "corr_1"


# ============================================================
# Child events / causation
# ============================================================


def test_child_event_inherits_run_id() -> None:
    parent = make_event()

    child = parent.child(
        EventType.RESEARCH_STARTED
    )

    assert child.run_id == parent.run_id


def test_child_event_sets_causation_id() -> None:
    parent = make_event()

    child = parent.child(
        EventType.RESEARCH_STARTED
    )

    assert child.causation_id == parent.event_id


def test_child_event_inherits_correlation_id() -> None:
    parent = make_event()

    child = parent.child(
        EventType.RESEARCH_STARTED
    )

    assert (
        child.correlation_id
        == parent.correlation_id
    )


def test_child_event_uses_parent_id_as_correlation_when_missing() -> None:
    parent = ResearchEvent(
        event_type=EventType.RESEARCH_REQUESTED,
        run_id="run_1",
    )

    child = parent.child(
        EventType.RESEARCH_STARTED
    )

    assert child.correlation_id == parent.event_id


def test_child_event_can_override_source() -> None:
    parent = make_event()

    child = parent.child(
        EventType.AGENT_STARTED,
        source="macro_agent",
    )

    assert child.source == "macro_agent"


def test_child_event_can_override_priority() -> None:
    parent = make_event()

    child = parent.child(
        EventType.RISK_ASSESSED,
        priority=EventPriority.CRITICAL,
    )

    assert child.priority == EventPriority.CRITICAL


# ============================================================
# Subscription
# ============================================================


def test_subscription_creation() -> None:
    def handler(event: ResearchEvent) -> None:
        return None

    subscription = Subscription(
        event_type=EventType.RESEARCH_REQUESTED,
        handler=handler,
        subscriber_name="research_handler",
    )

    assert subscription.subscription_id.startswith(
        "subscription_"
    )


def test_subscription_research_safety() -> None:
    def handler(event: ResearchEvent) -> None:
        return None

    subscription = Subscription(
        event_type=EventType.RESEARCH_REQUESTED,
        handler=handler,
        subscriber_name="handler",
    )

    assert subscription.metadata["research_only"] is True
    assert (
        subscription.metadata["real_money_execution"]
        is False
    )


def test_subscription_rejects_non_callable() -> None:
    with pytest.raises(TypeError):
        Subscription(
            event_type=EventType.RESEARCH_REQUESTED,
            handler="invalid",  # type: ignore[arg-type]
            subscriber_name="handler",
        )


def test_subscription_rejects_empty_name() -> None:
    def handler(event: ResearchEvent) -> None:
        return None

    with pytest.raises(SubscriberError):
        Subscription(
            event_type=EventType.RESEARCH_REQUESTED,
            handler=handler,
            subscriber_name=" ",
        )


# ============================================================
# EventBus subscriptions
# ============================================================


def test_event_bus_starts_empty() -> None:
    bus = EventBus()

    assert bus.subscription_count() == 0
    assert bus.event_history() == []
    assert bus.delivery_history() == []


def test_subscribe_handler() -> None:
    bus = EventBus()

    def handler(event: ResearchEvent) -> None:
        return None

    subscription = bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        handler,
    )

    assert bus.subscription_count() == 1
    assert subscription.event_type == (
        EventType.RESEARCH_REQUESTED
    )


def test_subscribe_custom_name() -> None:
    bus = EventBus()

    def handler(event: ResearchEvent) -> None:
        return None

    subscription = bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        handler,
        subscriber_name="custom_handler",
    )

    assert (
        subscription.subscriber_name
        == "custom_handler"
    )


def test_subscribers_by_event_type() -> None:
    bus = EventBus()

    def handler(event: ResearchEvent) -> None:
        return None

    bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        handler,
    )
    bus.subscribe(
        EventType.AGENT_STARTED,
        handler,
    )

    assert (
        bus.subscription_count(
            EventType.RESEARCH_REQUESTED
        )
        == 1
    )
    assert (
        bus.subscription_count(
            EventType.AGENT_STARTED
        )
        == 1
    )


def test_unsubscribe() -> None:
    bus = EventBus()

    def handler(event: ResearchEvent) -> None:
        return None

    subscription = bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        handler,
    )

    result = bus.unsubscribe(
        subscription.subscription_id
    )

    assert result is True
    assert bus.subscription_count() == 0


def test_unsubscribe_unknown_returns_false() -> None:
    bus = EventBus()

    assert (
        bus.unsubscribe("subscription_missing")
        is False
    )


def test_unsubscribe_rejects_empty_id() -> None:
    bus = EventBus()

    with pytest.raises(SubscriberError):
        bus.unsubscribe("")


def test_subscribe_rejects_invalid_event_type() -> None:
    bus = EventBus()

    def handler(event: ResearchEvent) -> None:
        return None

    with pytest.raises(TypeError):
        bus.subscribe(
            "RESEARCH_REQUESTED",  # type: ignore[arg-type]
            handler,
        )


# ============================================================
# Synchronous publishing
# ============================================================


def test_publish_to_sync_handler() -> None:
    bus = EventBus()
    received = []

    def handler(event: ResearchEvent) -> str:
        received.append(event.event_id)
        return "ok"

    bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        handler,
    )

    event = make_event()

    report = bus.publish(event)

    assert received == [event.event_id]
    assert report.status == DeliveryStatus.DELIVERED
    assert report.success_count == 1
    assert report.failure_count == 0


def test_publish_returns_handler_result() -> None:
    bus = EventBus()

    def handler(event: ResearchEvent) -> dict:
        return {
            "decision": "RESEARCH",
        }

    bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        handler,
    )

    report = bus.publish(make_event())

    assert report.handler_results[0].result == {
        "decision": "RESEARCH"
    }


def test_publish_without_subscribers_is_delivered() -> None:
    bus = EventBus()

    report = bus.publish(make_event())

    assert report.status == DeliveryStatus.DELIVERED
    assert report.subscriber_count == 0


def test_publish_records_event_history() -> None:
    bus = EventBus()
    event = make_event()

    bus.publish(event)

    assert bus.event_history() == [event]


def test_publish_records_delivery_history() -> None:
    bus = EventBus()

    bus.publish(make_event())

    assert len(bus.delivery_history()) == 1


def test_sequential_publish_multiple_handlers() -> None:
    bus = EventBus()
    order = []

    def first(event: ResearchEvent) -> None:
        order.append("first")

    def second(event: ResearchEvent) -> None:
        order.append("second")

    bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        first,
    )
    bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        second,
    )

    report = bus.publish(
        make_event(),
        parallel=False,
    )

    assert order == ["first", "second"]
    assert report.success_count == 2


# ============================================================
# Async / parallel publishing
# ============================================================


def test_publish_async_handler() -> None:
    bus = EventBus()

    async def handler(
        event: ResearchEvent,
    ) -> str:
        await asyncio.sleep(0)
        return "async-ok"

    bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        handler,
    )

    report = asyncio.run(
        bus.publish_async(make_event())
    )

    assert report.status == DeliveryStatus.DELIVERED
    assert (
        report.handler_results[0].result
        == "async-ok"
    )


def test_parallel_publish_multiple_async_handlers() -> None:
    bus = EventBus()

    async def first(
        event: ResearchEvent,
    ) -> str:
        await asyncio.sleep(0.01)
        return "first"

    async def second(
        event: ResearchEvent,
    ) -> str:
        await asyncio.sleep(0.01)
        return "second"

    bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        first,
    )
    bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        second,
    )

    report = asyncio.run(
        bus.publish_async(
            make_event(),
            parallel=True,
        )
    )

    assert report.success_count == 2
    assert {
        item.result
        for item in report.handler_results
    } == {"first", "second"}


def test_publish_method_rejected_inside_active_loop() -> None:
    bus = EventBus()

    async def scenario() -> None:
        with pytest.raises(EventBusError):
            bus.publish(make_event())

    asyncio.run(scenario())


# ============================================================
# Failure isolation
# ============================================================


def test_handler_failure_is_captured() -> None:
    bus = EventBus()

    def broken(event: ResearchEvent) -> None:
        raise RuntimeError("agent failure")

    bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        broken,
    )

    report = bus.publish(make_event())

    assert report.status == DeliveryStatus.FAILED
    assert report.failure_count == 1
    assert "RuntimeError" in (
        report.handler_results[0].error or ""
    )


def test_one_failure_does_not_stop_other_handler() -> None:
    bus = EventBus()

    def broken(event: ResearchEvent) -> None:
        raise RuntimeError("failed")

    def healthy(event: ResearchEvent) -> str:
        return "healthy"

    bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        broken,
    )
    bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        healthy,
    )

    report = bus.publish(make_event())

    assert report.status == DeliveryStatus.PARTIAL
    assert report.success_count == 1
    assert report.failure_count == 1


def test_all_handler_failures_mark_delivery_failed() -> None:
    bus = EventBus()

    def broken_one(event: ResearchEvent) -> None:
        raise RuntimeError("one")

    def broken_two(event: ResearchEvent) -> None:
        raise ValueError("two")

    bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        broken_one,
    )
    bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        broken_two,
    )

    report = bus.publish(make_event())

    assert report.status == DeliveryStatus.FAILED
    assert report.failure_count == 2


# ============================================================
# History / correlation
# ============================================================


def test_get_event() -> None:
    bus = EventBus()
    event = make_event()

    bus.publish(event)

    assert bus.get_event(event.event_id) is event


def test_get_unknown_event_returns_none() -> None:
    bus = EventBus()

    assert bus.get_event("event_missing") is None


def test_get_event_rejects_empty_id() -> None:
    bus = EventBus()

    with pytest.raises(EventValidationError):
        bus.get_event("")


def test_filter_event_history_by_type() -> None:
    bus = EventBus()

    bus.publish(
        make_event(
            EventType.RESEARCH_REQUESTED
        )
    )
    bus.publish(
        make_event(
            EventType.AGENT_STARTED
        )
    )

    result = bus.event_history(
        EventType.AGENT_STARTED
    )

    assert len(result) == 1
    assert result[0].event_type == (
        EventType.AGENT_STARTED
    )


def test_events_for_run() -> None:
    bus = EventBus()

    bus.publish(
        make_event(run_id="run_A")
    )
    bus.publish(
        make_event(run_id="run_B")
    )
    bus.publish(
        make_event(run_id="run_A")
    )

    result = bus.events_for_run("run_A")

    assert len(result) == 2


def test_events_for_correlation() -> None:
    bus = EventBus()

    bus.publish(
        make_event(
            correlation_id="corr_A"
        )
    )
    bus.publish(
        make_event(
            correlation_id="corr_B"
        )
    )

    result = bus.events_for_correlation(
        "corr_A"
    )

    assert len(result) == 1


# ============================================================
# Audit / summary / clear
# ============================================================


def test_subscribe_creates_audit_event() -> None:
    bus = EventBus()

    def handler(event: ResearchEvent) -> None:
        return None

    bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        handler,
    )

    assert any(
        item["event"] == "SUBSCRIBED"
        for item in bus.audit_log()
    )


def test_publish_creates_audit_events() -> None:
    bus = EventBus()

    bus.publish(make_event())

    events = [
        item["event"]
        for item in bus.audit_log()
    ]

    assert "EVENT_PUBLISHED" in events
    assert "EVENT_DELIVERY_COMPLETED" in events


def test_summary_empty_bus() -> None:
    bus = EventBus()

    summary = bus.summary()

    assert summary["subscriptions"] == 0
    assert summary["events"] == 0
    assert summary["deliveries"] == 0
    assert summary["research_only"] is True
    assert summary["real_money_execution"] is False


def test_summary_counts_successful_delivery() -> None:
    bus = EventBus()

    bus.publish(make_event())

    summary = bus.summary()

    assert summary["events"] == 1
    assert summary["deliveries"] == 1
    assert summary["delivered"] == 1
    assert summary["failed"] == 0


def test_summary_counts_partial_delivery() -> None:
    bus = EventBus()

    def healthy(event: ResearchEvent) -> None:
        return None

    def broken(event: ResearchEvent) -> None:
        raise RuntimeError("failure")

    bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        healthy,
    )
    bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        broken,
    )

    bus.publish(make_event())

    summary = bus.summary()

    assert summary["partial"] == 1


def test_clear_history() -> None:
    bus = EventBus()

    bus.publish(make_event())

    bus.clear_history()

    assert bus.event_history() == []
    assert bus.delivery_history() == []


def test_clear_history_preserves_subscriptions() -> None:
    bus = EventBus()

    def handler(event: ResearchEvent) -> None:
        return None

    bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        handler,
    )

    bus.publish(make_event())
    bus.clear_history()

    assert bus.subscription_count() == 1


def test_handler_result_duration_non_negative() -> None:
    bus = EventBus()

    def handler(event: ResearchEvent) -> str:
        return "ok"

    bus.subscribe(
        EventType.RESEARCH_REQUESTED,
        handler,
    )

    report = bus.publish(make_event())

    assert (
        report.handler_results[0]
        .duration_seconds
        >= 0.0
    )