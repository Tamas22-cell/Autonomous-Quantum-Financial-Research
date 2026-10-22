from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Set

from research_control_center.models import (
    DecisionLineage,
    DecisionLineageEvent,
    utc_now,
)


class DecisionLineageConfig:
    """
    Configuration for Research Control Center v5.0.10
    Decision Lineage Engine.
    """

    DEFAULT_STAGE_ORDER = (
        "RAW_DATA",
        "AGENT_EVIDENCE",
        "HYPOTHESIS",
        "STRATEGY",
        "VALIDATION",
        "RISK",
        "SHADOW_ALLOCATION",
        "OUTCOME",
    )

    def __init__(
        self,
        enforce_stage_order: bool = True,
        require_parent: bool = True,
        allow_branching: bool = True,
    ) -> None:
        if not isinstance(
            enforce_stage_order,
            bool,
        ):
            raise TypeError(
                "enforce_stage_order must be boolean"
            )

        if not isinstance(
            require_parent,
            bool,
        ):
            raise TypeError(
                "require_parent must be boolean"
            )

        if not isinstance(
            allow_branching,
            bool,
        ):
            raise TypeError(
                "allow_branching must be boolean"
            )

        self.enforce_stage_order = (
            enforce_stage_order
        )
        self.require_parent = (
            require_parent
        )
        self.allow_branching = (
            allow_branching
        )

        self.stage_order = (
            self.DEFAULT_STAGE_ORDER
        )


class DecisionLineageEngine:
    """
    Research Control Center v5.0.10
    Decision Lineage Engine.

    Tracks:

    RAW_DATA
        ->
    AGENT_EVIDENCE
        ->
    HYPOTHESIS
        ->
    STRATEGY
        ->
    VALIDATION
        ->
    RISK
        ->
    SHADOW_ALLOCATION
        ->
    OUTCOME

    Responsibilities:
    - create lineage graphs
    - append auditable decision events
    - enforce parent-child relationships
    - enforce research pipeline stage ordering
    - support branching
    - trace ancestry
    - trace descendants
    - inspect stage coverage
    - detect broken references
    - generate audit reports
    - preserve research-only safety metadata

    This component does not execute trades.
    """

    def __init__(
        self,
        config: Optional[
            DecisionLineageConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or DecisionLineageConfig()
        )

        self._lineages: Dict[
            str,
            DecisionLineage,
        ] = {}

        self._audit_log: List[dict] = []

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def _validate_string(
        value: str,
        name: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                f"{name} must be a string"
            )

        value = value.strip()

        if not value:
            raise ValueError(
                f"{name} cannot be empty"
            )

        return value

    @staticmethod
    def _validate_optional_string(
        value: Optional[str],
        name: str,
    ) -> Optional[str]:
        if value is None:
            return None

        return (
            DecisionLineageEngine
            ._validate_string(
                value,
                name,
            )
        )

    @staticmethod
    def _validate_dict(
        value: Optional[dict],
        name: str,
    ) -> dict:
        if value is None:
            return {}

        if not isinstance(value, dict):
            raise TypeError(
                f"{name} must be a dictionary"
            )

        return deepcopy(value)

    def _validate_stage(
        self,
        stage: str,
    ) -> str:
        stage = self._validate_string(
            stage,
            "stage",
        ).upper()

        if stage not in self.config.stage_order:
            raise ValueError(
                f"unknown lineage stage: {stage}"
            )

        return stage

    def _get_internal(
        self,
        lineage_id: str,
    ) -> DecisionLineage:
        lineage_id = self._validate_string(
            lineage_id,
            "lineage_id",
        )

        try:
            return self._lineages[
                lineage_id
            ]
        except KeyError as exc:
            raise KeyError(
                f"unknown lineage_id: "
                f"{lineage_id}"
            ) from exc

    @staticmethod
    def _event_map(
        lineage: DecisionLineage,
    ) -> Dict[str, DecisionLineageEvent]:
        return {
            event.event_id: event
            for event in lineage.events
        }

    @staticmethod
    def _entity_map(
        lineage: DecisionLineage,
    ) -> Dict[str, DecisionLineageEvent]:
        return {
            event.entity_id: event
            for event in lineage.events
        }

    def _audit(
        self,
        event: str,
        lineage: DecisionLineage,
        details: Optional[dict] = None,
    ) -> None:
        self._audit_log.append(
            {
                "timestamp": utc_now(),
                "event": event,
                "lineage_id": (
                    lineage.lineage_id
                ),
                "root_id": lineage.root_id,
                "details": deepcopy(
                    details or {}
                ),
                "research_only": True,
                "real_money_execution": False,
            }
        )

    # =========================================================
    # CREATE LINEAGE
    # =========================================================

    def create(
        self,
        root_id: str,
        *,
        metadata: Optional[dict] = None,
    ) -> DecisionLineage:
        root_id = self._validate_string(
            root_id,
            "root_id",
        )

        metadata = self._validate_dict(
            metadata,
            "metadata",
        )

        lineage = DecisionLineage(
            root_id=root_id,
            metadata={
                **metadata,
                "engine": (
                    "v5.0.10-decision-lineage"
                ),
                "auditable": True,
                "research_only": True,
                "real_money_execution": False,
            },
        )

        self._lineages[
            lineage.lineage_id
        ] = lineage

        self._audit(
            "LINEAGE_CREATED",
            lineage,
        )

        return deepcopy(lineage)

    # =========================================================
    # READ
    # =========================================================

    def get(
        self,
        lineage_id: str,
    ) -> DecisionLineage:
        return deepcopy(
            self._get_internal(
                lineage_id
            )
        )

    def list_all(
        self,
    ) -> List[DecisionLineage]:
        return [
            deepcopy(lineage)
            for lineage
            in self._lineages.values()
        ]

    def count(self) -> int:
        return len(self._lineages)

    # =========================================================
    # EVENT LOOKUP
    # =========================================================

    def get_event(
        self,
        lineage_id: str,
        event_id: str,
    ) -> DecisionLineageEvent:
        lineage = self._get_internal(
            lineage_id
        )

        event_id = self._validate_string(
            event_id,
            "event_id",
        )

        for event in lineage.events:
            if event.event_id == event_id:
                return deepcopy(event)

        raise KeyError(
            f"unknown event_id: {event_id}"
        )

    def get_event_by_entity(
        self,
        lineage_id: str,
        entity_id: str,
    ) -> DecisionLineageEvent:
        lineage = self._get_internal(
            lineage_id
        )

        entity_id = self._validate_string(
            entity_id,
            "entity_id",
        )

        for event in lineage.events:
            if event.entity_id == entity_id:
                return deepcopy(event)

        raise KeyError(
            f"unknown entity_id: {entity_id}"
        )

    # =========================================================
    # STAGE ORDER
    # =========================================================

    def _stage_index(
        self,
        stage: str,
    ) -> int:
        return self.config.stage_order.index(
            stage
        )

    def _validate_parent_stage(
        self,
        parent: DecisionLineageEvent,
        child_stage: str,
    ) -> None:
        if not self.config.enforce_stage_order:
            return

        parent_index = self._stage_index(
            parent.stage
        )

        child_index = self._stage_index(
            child_stage
        )

        if child_index <= parent_index:
            raise ValueError(
                "child stage must occur after "
                "parent stage"
            )

    # =========================================================
    # ADD EVENT
    # =========================================================

    def add_event(
        self,
        lineage_id: str,
        *,
        stage: str,
        entity_type: str,
        entity_id: str,
        parent_entity_id: Optional[
            str
        ] = None,
        summary: Optional[str] = None,
        evidence: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> DecisionLineageEvent:
        lineage = self._get_internal(
            lineage_id
        )

        stage = self._validate_stage(
            stage
        )

        entity_type = self._validate_string(
            entity_type,
            "entity_type",
        )

        entity_id = self._validate_string(
            entity_id,
            "entity_id",
        )

        parent_entity_id = (
            self._validate_optional_string(
                parent_entity_id,
                "parent_entity_id",
            )
        )

        summary = (
            self._validate_optional_string(
                summary,
                "summary",
            )
        )

        evidence = self._validate_dict(
            evidence,
            "evidence",
        )

        metadata = self._validate_dict(
            metadata,
            "metadata",
        )

        entity_map = self._entity_map(
            lineage
        )

        if entity_id in entity_map:
            raise ValueError(
                "entity already exists in lineage: "
                f"{entity_id}"
            )

        parent = None

        if parent_entity_id is not None:
            parent = entity_map.get(
                parent_entity_id
            )

            if parent is None:
                raise ValueError(
                    "parent entity does not exist: "
                    f"{parent_entity_id}"
                )

            self._validate_parent_stage(
                parent,
                stage,
            )

            if not self.config.allow_branching:
                children = [
                    event
                    for event in lineage.events
                    if event.parent_entity_id
                    == parent_entity_id
                ]

                if children:
                    raise ValueError(
                        "branching is disabled and "
                        "parent already has a child"
                    )

        elif (
            self.config.require_parent
            and len(lineage.events) > 0
        ):
            raise ValueError(
                "parent_entity_id is required "
                "for non-root events"
            )

        event = DecisionLineageEvent(
            stage=stage,
            entity_type=entity_type,
            entity_id=entity_id,
            parent_entity_id=(
                parent_entity_id
            ),
            summary=summary,
            evidence=evidence,
            metadata={
                **metadata,
                "research_only": True,
                "real_money_execution": False,
            },
        )

        lineage.add_event(event)

        self._audit(
            "LINEAGE_EVENT_ADDED",
            lineage,
            {
                "event_id": event.event_id,
                "stage": event.stage,
                "entity_type": (
                    event.entity_type
                ),
                "entity_id": (
                    event.entity_id
                ),
                "parent_entity_id": (
                    event.parent_entity_id
                ),
            },
        )

        return deepcopy(event)

    # =========================================================
    # PIPELINE HELPERS
    # =========================================================

    def add_raw_data(
        self,
        lineage_id: str,
        entity_id: str,
        *,
        summary: Optional[str] = None,
        evidence: Optional[dict] = None,
    ) -> DecisionLineageEvent:
        return self.add_event(
            lineage_id,
            stage="RAW_DATA",
            entity_type="RAW_DATA",
            entity_id=entity_id,
            summary=summary,
            evidence=evidence,
        )

    def add_agent_evidence(
        self,
        lineage_id: str,
        entity_id: str,
        parent_entity_id: str,
        *,
        summary: Optional[str] = None,
        evidence: Optional[dict] = None,
    ) -> DecisionLineageEvent:
        return self.add_event(
            lineage_id,
            stage="AGENT_EVIDENCE",
            entity_type="AGENT_EVIDENCE",
            entity_id=entity_id,
            parent_entity_id=parent_entity_id,
            summary=summary,
            evidence=evidence,
        )

    def add_hypothesis(
        self,
        lineage_id: str,
        entity_id: str,
        parent_entity_id: str,
        *,
        summary: Optional[str] = None,
        evidence: Optional[dict] = None,
    ) -> DecisionLineageEvent:
        return self.add_event(
            lineage_id,
            stage="HYPOTHESIS",
            entity_type="HYPOTHESIS",
            entity_id=entity_id,
            parent_entity_id=parent_entity_id,
            summary=summary,
            evidence=evidence,
        )

    def add_strategy(
        self,
        lineage_id: str,
        entity_id: str,
        parent_entity_id: str,
        *,
        summary: Optional[str] = None,
        evidence: Optional[dict] = None,
    ) -> DecisionLineageEvent:
        return self.add_event(
            lineage_id,
            stage="STRATEGY",
            entity_type="STRATEGY",
            entity_id=entity_id,
            parent_entity_id=parent_entity_id,
            summary=summary,
            evidence=evidence,
        )

    def add_validation(
        self,
        lineage_id: str,
        entity_id: str,
        parent_entity_id: str,
        *,
        summary: Optional[str] = None,
        evidence: Optional[dict] = None,
    ) -> DecisionLineageEvent:
        return self.add_event(
            lineage_id,
            stage="VALIDATION",
            entity_type="VALIDATION",
            entity_id=entity_id,
            parent_entity_id=parent_entity_id,
            summary=summary,
            evidence=evidence,
        )

    def add_risk(
        self,
        lineage_id: str,
        entity_id: str,
        parent_entity_id: str,
        *,
        summary: Optional[str] = None,
        evidence: Optional[dict] = None,
    ) -> DecisionLineageEvent:
        return self.add_event(
            lineage_id,
            stage="RISK",
            entity_type="RISK",
            entity_id=entity_id,
            parent_entity_id=parent_entity_id,
            summary=summary,
            evidence=evidence,
        )

    def add_shadow_allocation(
        self,
        lineage_id: str,
        entity_id: str,
        parent_entity_id: str,
        *,
        summary: Optional[str] = None,
        evidence: Optional[dict] = None,
    ) -> DecisionLineageEvent:
        return self.add_event(
            lineage_id,
            stage="SHADOW_ALLOCATION",
            entity_type="SHADOW_ALLOCATION",
            entity_id=entity_id,
            parent_entity_id=parent_entity_id,
            summary=summary,
            evidence=evidence,
        )

    def add_outcome(
        self,
        lineage_id: str,
        entity_id: str,
        parent_entity_id: str,
        *,
        summary: Optional[str] = None,
        evidence: Optional[dict] = None,
    ) -> DecisionLineageEvent:
        return self.add_event(
            lineage_id,
            stage="OUTCOME",
            entity_type="OUTCOME",
            entity_id=entity_id,
            parent_entity_id=parent_entity_id,
            summary=summary,
            evidence=evidence,
        )

    # =========================================================
    # CHILDREN
    # =========================================================

    def children(
        self,
        lineage_id: str,
        entity_id: str,
    ) -> List[DecisionLineageEvent]:
        lineage = self._get_internal(
            lineage_id
        )

        entity_id = self._validate_string(
            entity_id,
            "entity_id",
        )

        return [
            deepcopy(event)
            for event in lineage.events
            if event.parent_entity_id
            == entity_id
        ]

    # =========================================================
    # ANCESTRY
    # =========================================================

    def ancestry(
        self,
        lineage_id: str,
        entity_id: str,
    ) -> List[DecisionLineageEvent]:
        lineage = self._get_internal(
            lineage_id
        )

        entity_id = self._validate_string(
            entity_id,
            "entity_id",
        )

        entity_map = self._entity_map(
            lineage
        )

        if entity_id not in entity_map:
            raise KeyError(
                f"unknown entity_id: "
                f"{entity_id}"
            )

        result: List[
            DecisionLineageEvent
        ] = []

        current = entity_map[
            entity_id
        ]

        visited: Set[str] = set()

        while (
            current.parent_entity_id
            is not None
        ):
            if current.entity_id in visited:
                raise RuntimeError(
                    "cycle detected in lineage"
                )

            visited.add(
                current.entity_id
            )

            parent = entity_map.get(
                current.parent_entity_id
            )

            if parent is None:
                raise RuntimeError(
                    "broken parent reference: "
                    f"{current.parent_entity_id}"
                )

            result.append(
                deepcopy(parent)
            )

            current = parent

        result.reverse()

        return result

    # =========================================================
    # DESCENDANTS
    # =========================================================

    def descendants(
        self,
        lineage_id: str,
        entity_id: str,
    ) -> List[DecisionLineageEvent]:
        lineage = self._get_internal(
            lineage_id
        )

        entity_map = self._entity_map(
            lineage
        )

        entity_id = self._validate_string(
            entity_id,
            "entity_id",
        )

        if entity_id not in entity_map:
            raise KeyError(
                f"unknown entity_id: "
                f"{entity_id}"
            )

        result: List[
            DecisionLineageEvent
        ] = []

        visited: Set[str] = set()

        def walk(
            parent_entity_id: str,
        ) -> None:
            if parent_entity_id in visited:
                raise RuntimeError(
                    "cycle detected in lineage"
                )

            visited.add(
                parent_entity_id
            )

            for event in lineage.events:
                if (
                    event.parent_entity_id
                    == parent_entity_id
                ):
                    result.append(
                        deepcopy(event)
                    )

                    walk(
                        event.entity_id
                    )

        walk(entity_id)

        return result

    # =========================================================
    # PATH
    # =========================================================

    def path_to(
        self,
        lineage_id: str,
        entity_id: str,
    ) -> List[DecisionLineageEvent]:
        lineage = self._get_internal(
            lineage_id
        )

        target = self.get_event_by_entity(
            lineage_id,
            entity_id,
        )

        ancestors = self.ancestry(
            lineage_id,
            entity_id,
        )

        return [
            *ancestors,
            deepcopy(target),
        ]

    # =========================================================
    # STAGE COVERAGE
    # =========================================================

    def stage_coverage(
        self,
        lineage_id: str,
    ) -> dict:
        lineage = self._get_internal(
            lineage_id
        )

        present = {
            event.stage
            for event in lineage.events
        }

        stages = {
            stage: stage in present
            for stage
            in self.config.stage_order
        }

        completed = sum(
            1
            for value in stages.values()
            if value
        )

        total = len(
            self.config.stage_order
        )

        return {
            "stages": stages,
            "completed_stages": completed,
            "total_stages": total,
            "coverage_ratio": (
                completed / total
                if total
                else 0.0
            ),
            "complete": (
                completed == total
            ),
        }

    # =========================================================
    # INTEGRITY
    # =========================================================

    def integrity_report(
        self,
        lineage_id: str,
    ) -> dict:
        lineage = self._get_internal(
            lineage_id
        )

        entity_map = self._entity_map(
            lineage
        )

        duplicate_entities: List[str] = []
        seen: Set[str] = set()

        for event in lineage.events:
            if event.entity_id in seen:
                duplicate_entities.append(
                    event.entity_id
                )

            seen.add(
                event.entity_id
            )

        broken_parents: List[str] = []

        for event in lineage.events:
            if (
                event.parent_entity_id
                is not None
                and event.parent_entity_id
                not in entity_map
            ):
                broken_parents.append(
                    event.entity_id
                )

        invalid_stage_links: List[str] = []

        for event in lineage.events:
            if event.parent_entity_id is None:
                continue

            parent = entity_map.get(
                event.parent_entity_id
            )

            if parent is None:
                continue

            if (
                self._stage_index(
                    event.stage
                )
                <= self._stage_index(
                    parent.stage
                )
            ):
                invalid_stage_links.append(
                    event.entity_id
                )

        root_events = [
            event
            for event in lineage.events
            if event.parent_entity_id
            is None
        ]

        valid = (
            not duplicate_entities
            and not broken_parents
            and not invalid_stage_links
            and len(root_events) <= 1
        )

        return {
            "lineage_id": (
                lineage.lineage_id
            ),
            "root_id": (
                lineage.root_id
            ),
            "event_count": (
                len(lineage.events)
            ),
            "root_event_count": (
                len(root_events)
            ),
            "duplicate_entities": (
                duplicate_entities
            ),
            "broken_parent_references": (
                broken_parents
            ),
            "invalid_stage_links": (
                invalid_stage_links
            ),
            "valid": valid,
            "research_only": True,
            "real_money_execution": False,
        }

    # =========================================================
    # PIPELINE REPORT
    # =========================================================

    def pipeline_report(
        self,
        lineage_id: str,
    ) -> dict:
        lineage = self._get_internal(
            lineage_id
        )

        coverage = self.stage_coverage(
            lineage_id
        )

        integrity = self.integrity_report(
            lineage_id
        )

        by_stage: Dict[
            str,
            List[dict],
        ] = {
            stage: []
            for stage
            in self.config.stage_order
        }

        for event in lineage.events:
            by_stage[event.stage].append(
                {
                    "event_id": (
                        event.event_id
                    ),
                    "entity_type": (
                        event.entity_type
                    ),
                    "entity_id": (
                        event.entity_id
                    ),
                    "parent_entity_id": (
                        event.parent_entity_id
                    ),
                    "summary": (
                        event.summary
                    ),
                    "evidence": deepcopy(
                        event.evidence
                    ),
                }
            )

        return {
            "lineage_id": (
                lineage.lineage_id
            ),
            "root_id": (
                lineage.root_id
            ),
            "events": (
                len(lineage.events)
            ),
            "by_stage": by_stage,
            "coverage": coverage,
            "integrity": integrity,
            "auditable": True,
            "research_only": True,
            "real_money_execution": False,
        }

    # =========================================================
    # SUMMARY
    # =========================================================

    def summary(self) -> dict:
        lineages = self.list_all()

        total_events = sum(
            len(lineage.events)
            for lineage in lineages
        )

        complete_lineages = 0
        valid_lineages = 0

        for lineage in lineages:
            coverage = self.stage_coverage(
                lineage.lineage_id
            )

            integrity = self.integrity_report(
                lineage.lineage_id
            )

            if coverage["complete"]:
                complete_lineages += 1

            if integrity["valid"]:
                valid_lineages += 1

        return {
            "total_lineages": (
                len(lineages)
            ),
            "total_events": (
                total_events
            ),
            "complete_lineages": (
                complete_lineages
            ),
            "valid_lineages": (
                valid_lineages
            ),
            "generated_at": utc_now(),
            "auditable": True,
            "research_only": True,
            "real_money_execution": False,
        }

    # =========================================================
    # AUDIT LOG
    # =========================================================

    def audit_log(
        self,
        lineage_id: Optional[
            str
        ] = None,
    ) -> List[dict]:
        if lineage_id is None:
            return deepcopy(
                self._audit_log
            )

        return [
            deepcopy(event)
            for event in self._audit_log
            if event["lineage_id"]
            == lineage_id
        ]

    # =========================================================
    # SNAPSHOT
    # =========================================================

    def snapshot(self) -> dict:
        return {
            "lineages": self.list_all(),
            "summary": self.summary(),
            "generated_at": utc_now(),
            "auditable": True,
            "research_only": True,
            "real_money_execution": False,
        }