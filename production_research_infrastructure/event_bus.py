from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class EventBusError(RuntimeError):
    """Base exception for event-bus failures."""


class EventValidationError(EventBusError):
    """Raised when an event is invalid."""


class SubscriberError(EventBusError):
    """Raised when a subscriber cannot be registered."""


class EventType(str, Enum):
    RESEARCH_REQUESTED = "RESEARCH_REQUESTED"
    RESEARCH_STARTED = "RESEARCH_STARTED"
    RESEARCH_COMPLETED = "RESEARCH_COMPLETED"
    RESEARCH_FAILED = "RESEARCH_FAILED"

    AGENT_REQUESTED = "AGENT_REQUESTED"
    AGENT_STARTED = "AGENT_STARTED"
    AGENT_COMPLETED = "AGENT_COMPLETED"
    AGENT_FAILED = "AGENT_FAILED"

    DATASET_REGISTERED = "DATASET_REGISTERED"
    DATASET_UPDATED = "DATASET_UPDATED"

    HYPOTHESIS_GENERATED = "HYPOTHESIS_GENERATED"
    STRATEGY_GENERATED = "STRATEGY_GENERATED"
    VALIDATION_COMPLETED = "VALIDATION_COMPLETED"
    RISK_ASSESSED = "RISK_ASSESSED"
    SHADOW_ALLOCATED = "SHADOW_ALLOCATED"
    OUTCOME_OBSERVED = "OUTCOME_OBSERVED"

    EXPERIMENT_STARTED = "EXPERIMENT_STARTED"
    EXPERIMENT_COMPLETED = "EXPERIMENT_COMPLETED"

    ARTIFACT_CREATED = "ARTIFACT_CREATED"
    RUN_RECOVERY_REQUESTED = "RUN_RECOVERY_REQUESTED"

    SYSTEM = "SYSTEM"


class EventPriority(int, Enum):
    LOW = 10
    NORMAL = 20
    HIGH = 30
    CRITICAL = 40


class DeliveryStatus(str, Enum):
    PENDING = "PENDING"
    DELIVERED = "DELIVERED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


@dataclass
class ResearchEvent:
    event_type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    run_id: Optional[str] = None
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
    event_id: str = field(
        default_factory=lambda: generate_id("event")
    )
    created_at: datetime = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, EventType):
            raise TypeError(
                "event_type must be an EventType"
            )

        if not isinstance(self.payload, dict):
            raise TypeError(
                "payload must be a dictionary"
            )

        if not isinstance(self.priority, EventPriority):
            raise TypeError(
                "priority must be an EventPriority"
            )

        for name in (
            "source",
            "run_id",
            "correlation_id",
            "causation_id",
        ):
            value = getattr(self, name)

            if value is not None:
                if (
                    not isinstance(value, str)
                    or not value.strip()
                ):
                    raise EventValidationError(
                        f"{name} must be a non-empty string"
                    )

                setattr(self, name, value.strip())

        self.metadata.setdefault(
            "research_only",
            True,
        )
        self.metadata.setdefault(
            "real_money_execution",
            False,
        )

    def child(
        self,
        event_type: EventType,
        *,
        payload: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        priority: Optional[EventPriority] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ResearchEvent":
        return ResearchEvent(
            event_type=event_type,
            payload=payload or {},
            source=source or self.source,
            run_id=self.run_id,
            correlation_id=(
                self.correlation_id
                or self.event_id
            ),
            causation_id=self.event_id,
            priority=priority or self.priority,
            metadata=metadata or {},
        )


@dataclass
class HandlerResult:
    subscriber_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime = field(default_factory=utc_now)

    @property
    def duration_seconds(self) -> float:
        return max(
            0.0,
            (
                self.completed_at
                - self.started_at
            ).total_seconds(),
        )


@dataclass
class EventDeliveryReport:
    event: ResearchEvent
    status: DeliveryStatus
    handler_results: List[HandlerResult] = field(
        default_factory=list
    )
    published_at: datetime = field(default_factory=utc_now)
    completed_at: datetime = field(default_factory=utc_now)

    @property
    def subscriber_count(self) -> int:
        return len(self.handler_results)

    @property
    def success_count(self) -> int:
        return sum(
            1
            for item in self.handler_results
            if item.success
        )

    @property
    def failure_count(self) -> int:
        return sum(
            1
            for item in self.handler_results
            if not item.success
        )

    @property
    def successful(self) -> bool:
        return self.status == DeliveryStatus.DELIVERED


EventHandler = Callable[
    [ResearchEvent],
    Union[Any, Awaitable[Any]],
]


@dataclass
class Subscription:
    event_type: EventType
    handler: EventHandler
    subscriber_name: str
    subscription_id: str = field(
        default_factory=lambda: generate_id(
            "subscription"
        )
    )
    created_at: datetime = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, EventType):
            raise TypeError(
                "event_type must be an EventType"
            )

        if not callable(self.handler):
            raise TypeError(
                "handler must be callable"
            )

        if (
            not isinstance(self.subscriber_name, str)
            or not self.subscriber_name.strip()
        ):
            raise SubscriberError(
                "subscriber_name must be a non-empty string"
            )

        self.subscriber_name = (
            self.subscriber_name.strip()
        )

        self.metadata.setdefault(
            "research_only",
            True,
        )
        self.metadata.setdefault(
            "real_money_execution",
            False,
        )


class EventBus:
    """
    In-process event bus for production research execution.

    Supports:
    - typed research events,
    - sync and async subscribers,
    - sequential or parallel delivery,
    - correlation / causation IDs,
    - isolated subscriber failures,
    - delivery reports,
    - event history and audit trail.

    The bus coordinates research workloads only.
    It has no broker or real-money execution capability.
    """

    def __init__(self) -> None:
        self._subscriptions: Dict[
            EventType,
            List[Subscription],
        ] = {}
        self._event_history: List[ResearchEvent] = []
        self._delivery_history: List[
            EventDeliveryReport
        ] = []
        self._audit_events: List[
            Dict[str, Any]
        ] = []

    # ========================================================
    # Audit
    # ========================================================

    def _audit(
        self,
        event: str,
        **details: Any,
    ) -> None:
        self._audit_events.append(
            {
                "event": event,
                "created_at": utc_now(),
                "details": details,
            }
        )

    # ========================================================
    # Subscriptions
    # ========================================================

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
        *,
        subscriber_name: Optional[str] = None,
    ) -> Subscription:
        if not isinstance(event_type, EventType):
            raise TypeError(
                "event_type must be an EventType"
            )

        if not callable(handler):
            raise TypeError(
                "handler must be callable"
            )

        name = (
            subscriber_name
            or getattr(
                handler,
                "__name__",
                handler.__class__.__name__,
            )
        )

        subscription = Subscription(
            event_type=event_type,
            handler=handler,
            subscriber_name=name,
        )

        subscriptions = self._subscriptions.setdefault(
            event_type,
            [],
        )

        subscriptions.append(subscription)

        self._audit(
            "SUBSCRIBED",
            subscription_id=(
                subscription.subscription_id
            ),
            event_type=event_type.value,
            subscriber_name=(
                subscription.subscriber_name
            ),
        )

        return subscription

    def unsubscribe(
        self,
        subscription_id: str,
    ) -> bool:
        if (
            not isinstance(subscription_id, str)
            or not subscription_id.strip()
        ):
            raise SubscriberError(
                "subscription_id must be "
                "a non-empty string"
            )

        for event_type, subscriptions in (
            self._subscriptions.items()
        ):
            for subscription in list(
                subscriptions
            ):
                if (
                    subscription.subscription_id
                    == subscription_id
                ):
                    subscriptions.remove(
                        subscription
                    )

                    self._audit(
                        "UNSUBSCRIBED",
                        subscription_id=(
                            subscription_id
                        ),
                        event_type=(
                            event_type.value
                        ),
                        subscriber_name=(
                            subscription
                            .subscriber_name
                        ),
                    )

                    return True

        return False

    def subscribers(
        self,
        event_type: EventType,
    ) -> List[Subscription]:
        if not isinstance(event_type, EventType):
            raise TypeError(
                "event_type must be an EventType"
            )

        return list(
            self._subscriptions.get(
                event_type,
                [],
            )
        )

    def subscription_count(
        self,
        event_type: Optional[EventType] = None,
    ) -> int:
        if event_type is not None:
            return len(
                self.subscribers(event_type)
            )

        return sum(
            len(items)
            for items in self._subscriptions.values()
        )

    # ========================================================
    # Handler execution
    # ========================================================

    async def _invoke(
        self,
        subscription: Subscription,
        event: ResearchEvent,
    ) -> HandlerResult:
        started_at = utc_now()

        try:
            result = subscription.handler(event)

            if inspect.isawaitable(result):
                result = await result

            return HandlerResult(
                subscriber_name=(
                    subscription.subscriber_name
                ),
                success=True,
                result=result,
                started_at=started_at,
                completed_at=utc_now(),
            )

        except Exception as exc:
            return HandlerResult(
                subscriber_name=(
                    subscription.subscriber_name
                ),
                success=False,
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
                started_at=started_at,
                completed_at=utc_now(),
            )

    @staticmethod
    def _delivery_status(
        results: List[HandlerResult],
    ) -> DeliveryStatus:
        if not results:
            return DeliveryStatus.DELIVERED

        success_count = sum(
            1
            for item in results
            if item.success
        )

        if success_count == len(results):
            return DeliveryStatus.DELIVERED

        if success_count == 0:
            return DeliveryStatus.FAILED

        return DeliveryStatus.PARTIAL

    # ========================================================
    # Publishing
    # ========================================================

    async def publish_async(
        self,
        event: ResearchEvent,
        *,
        parallel: bool = True,
    ) -> EventDeliveryReport:
        if not isinstance(event, ResearchEvent):
            raise TypeError(
                "event must be a ResearchEvent"
            )

        published_at = utc_now()

        self._event_history.append(event)

        subscriptions = self.subscribers(
            event.event_type
        )

        self._audit(
            "EVENT_PUBLISHED",
            event_id=event.event_id,
            event_type=event.event_type.value,
            subscriber_count=len(
                subscriptions
            ),
            parallel=parallel,
            run_id=event.run_id,
            correlation_id=(
                event.correlation_id
            ),
            causation_id=event.causation_id,
        )

        if parallel and subscriptions:
            results = list(
                await asyncio.gather(
                    *[
                        self._invoke(
                            subscription,
                            event,
                        )
                        for subscription
                        in subscriptions
                    ]
                )
            )
        else:
            results = []

            for subscription in subscriptions:
                result = await self._invoke(
                    subscription,
                    event,
                )
                results.append(result)

        report = EventDeliveryReport(
            event=event,
            status=self._delivery_status(
                results
            ),
            handler_results=results,
            published_at=published_at,
            completed_at=utc_now(),
        )

        self._delivery_history.append(
            report
        )

        self._audit(
            "EVENT_DELIVERY_COMPLETED",
            event_id=event.event_id,
            event_type=event.event_type.value,
            status=report.status.value,
            success_count=(
                report.success_count
            ),
            failure_count=(
                report.failure_count
            ),
        )

        return report

    def publish(
        self,
        event: ResearchEvent,
        *,
        parallel: bool = True,
    ) -> EventDeliveryReport:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.publish_async(
                    event,
                    parallel=parallel,
                )
            )

        raise EventBusError(
            "publish() cannot be called from "
            "an active asyncio event loop; "
            "use await publish_async() instead"
        )

    # ========================================================
    # History / reporting
    # ========================================================

    def event_history(
        self,
        event_type: Optional[EventType] = None,
    ) -> List[ResearchEvent]:
        if event_type is None:
            return list(
                self._event_history
            )

        if not isinstance(event_type, EventType):
            raise TypeError(
                "event_type must be an EventType"
            )

        return [
            event
            for event in self._event_history
            if event.event_type == event_type
        ]

    def delivery_history(
        self,
    ) -> List[EventDeliveryReport]:
        return list(
            self._delivery_history
        )

    def get_event(
        self,
        event_id: str,
    ) -> Optional[ResearchEvent]:
        if (
            not isinstance(event_id, str)
            or not event_id.strip()
        ):
            raise EventValidationError(
                "event_id must be a non-empty string"
            )

        for event in self._event_history:
            if event.event_id == event_id:
                return event

        return None

    def events_for_run(
        self,
        run_id: str,
    ) -> List[ResearchEvent]:
        if (
            not isinstance(run_id, str)
            or not run_id.strip()
        ):
            raise EventValidationError(
                "run_id must be a non-empty string"
            )

        return [
            event
            for event in self._event_history
            if event.run_id == run_id
        ]

    def events_for_correlation(
        self,
        correlation_id: str,
    ) -> List[ResearchEvent]:
        if (
            not isinstance(
                correlation_id,
                str,
            )
            or not correlation_id.strip()
        ):
            raise EventValidationError(
                "correlation_id must be "
                "a non-empty string"
            )

        return [
            event
            for event in self._event_history
            if (
                event.correlation_id
                == correlation_id
            )
        ]

    def audit_log(
        self,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "event": item["event"],
                "created_at": item[
                    "created_at"
                ],
                "details": dict(
                    item["details"]
                ),
            }
            for item in self._audit_events
        ]

    def summary(
        self,
    ) -> Dict[str, Any]:
        delivered = sum(
            1
            for report in self._delivery_history
            if (
                report.status
                == DeliveryStatus.DELIVERED
            )
        )

        partial = sum(
            1
            for report in self._delivery_history
            if (
                report.status
                == DeliveryStatus.PARTIAL
            )
        )

        failed = sum(
            1
            for report in self._delivery_history
            if (
                report.status
                == DeliveryStatus.FAILED
            )
        )

        return {
            "component": "event_bus",
            "subscriptions": (
                self.subscription_count()
            ),
            "events": len(
                self._event_history
            ),
            "deliveries": len(
                self._delivery_history
            ),
            "delivered": delivered,
            "partial": partial,
            "failed": failed,
            "audit_events": len(
                self._audit_events
            ),
            "research_only": True,
            "real_money_execution": False,
        }

    def clear_history(
        self,
    ) -> None:
        self._event_history.clear()
        self._delivery_history.clear()

        self._audit(
            "HISTORY_CLEARED"
        )