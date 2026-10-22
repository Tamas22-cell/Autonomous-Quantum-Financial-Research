"""
Agent Tournament System - Blind Evaluator
==========================================

v5.0.6 of the Multi-Agent Financial Research Platform.

Responsibilities:
- anonymize tournament competitors
- guarantee identical market snapshots
- store blind submissions
- prevent evaluator access to competitor identity
- reveal identities only after evaluation
- maintain deterministic blind mappings
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping

from .models import (
    AgentCompetitor,
    MarketSnapshot,
    TournamentTask,
    generate_id,
)


@dataclass(frozen=True)
class BlindCompetitor:
    """
    Anonymous representation of a tournament competitor.

    The evaluator sees only blind_id, never the real competitor_id.
    """

    blind_id: str


@dataclass(frozen=True)
class BlindSubmission:
    """
    Anonymous tournament submission.

    competitor identity is intentionally excluded.
    """

    submission_id: str
    blind_id: str
    task_id: str
    snapshot_id: str

    prediction: Any
    confidence: float

    latency_ms: float = 0.0
    cost_usd: float = 0.0


class BlindEvaluationError(RuntimeError):
    """Raised when blind evaluation integrity is violated."""


class BlindEvaluator:
    """
    Handles identity masking and blind tournament submissions.

    Workflow:

        competitors
            ↓
        register_competitors()
            ↓
        blind IDs
            ↓
        identical MarketSnapshot
            ↓
        anonymous submissions
            ↓
        evaluation
            ↓
        reveal identities
    """

    def __init__(self) -> None:
        self._blind_to_competitor: Dict[str, str] = {}
        self._competitor_to_blind: Dict[str, str] = {}

        self._snapshot_hashes: Dict[str, str] = {}
        self._submissions: Dict[str, BlindSubmission] = {}

    # ------------------------------------------------------------------
    # Competitor registration
    # ------------------------------------------------------------------

    def register_competitors(
        self,
        competitors: Iterable[AgentCompetitor],
    ) -> List[BlindCompetitor]:
        """
        Register competitors and generate anonymous IDs.

        Existing competitors retain their original blind ID.
        """

        blind_competitors: List[BlindCompetitor] = []

        for competitor in competitors:
            competitor_id = competitor.competitor_id

            if competitor_id in self._competitor_to_blind:
                blind_id = self._competitor_to_blind[competitor_id]

            else:
                blind_id = generate_id("blind")

                self._competitor_to_blind[competitor_id] = blind_id
                self._blind_to_competitor[blind_id] = competitor_id

            blind_competitors.append(
                BlindCompetitor(blind_id=blind_id)
            )

        return blind_competitors

    # ------------------------------------------------------------------
    # Snapshot integrity
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """
        Convert values into deterministic JSON-compatible structures.
        """

        if hasattr(value, "isoformat"):
            return value.isoformat()

        if isinstance(value, Mapping):
            return {
                str(key): BlindEvaluator._normalize_value(val)
                for key, val in sorted(
                    value.items(),
                    key=lambda item: str(item[0]),
                )
            }

        if isinstance(value, (list, tuple)):
            return [
                BlindEvaluator._normalize_value(item)
                for item in value
            ]

        if isinstance(value, set):
            normalized = [
                BlindEvaluator._normalize_value(item)
                for item in value
            ]

            return sorted(
                normalized,
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    default=str,
                ),
            )

        return value

    @classmethod
    def snapshot_hash(cls, snapshot: MarketSnapshot) -> str:
        """
        Produce deterministic SHA-256 fingerprint of a MarketSnapshot.
        """

        payload = cls._normalize_value(asdict(snapshot))

        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()

    def register_snapshot(
        self,
        snapshot: MarketSnapshot,
    ) -> str:
        """
        Register the canonical snapshot used by a tournament task.
        """

        fingerprint = self.snapshot_hash(snapshot)

        existing = self._snapshot_hashes.get(snapshot.snapshot_id)

        if existing is not None and existing != fingerprint:
            raise BlindEvaluationError(
                "Snapshot integrity violation: "
                f"snapshot_id={snapshot.snapshot_id!r} "
                "was already registered with different content."
            )

        self._snapshot_hashes[snapshot.snapshot_id] = fingerprint

        return fingerprint

    def verify_snapshot(
        self,
        snapshot: MarketSnapshot,
    ) -> bool:
        """
        Verify that a snapshot is identical to its registered version.
        """

        expected = self._snapshot_hashes.get(snapshot.snapshot_id)

        if expected is None:
            raise BlindEvaluationError(
                f"Snapshot {snapshot.snapshot_id!r} "
                "has not been registered."
            )

        actual = self.snapshot_hash(snapshot)

        if actual != expected:
            raise BlindEvaluationError(
                "Snapshot mismatch detected. "
                "Competitors must receive identical market snapshots."
            )

        return True

    # ------------------------------------------------------------------
    # Task validation
    # ------------------------------------------------------------------

    def validate_task(
        self,
        task: TournamentTask,
        snapshot: MarketSnapshot,
    ) -> bool:
        """
        Ensure that a tournament task matches its MarketSnapshot.
        """

        if task.snapshot_id != snapshot.snapshot_id:
            raise BlindEvaluationError(
                "TournamentTask snapshot_id does not match "
                "the supplied MarketSnapshot."
            )

        if task.asset != snapshot.asset:
            raise BlindEvaluationError(
                "TournamentTask asset does not match snapshot asset."
            )

        if task.regime != snapshot.regime:
            raise BlindEvaluationError(
                "TournamentTask regime does not match snapshot regime."
            )

        if task.horizon != snapshot.horizon:
            raise BlindEvaluationError(
                "TournamentTask horizon does not match snapshot horizon."
            )

        self.verify_snapshot(snapshot)

        return True

    # ------------------------------------------------------------------
    # Blind submissions
    # ------------------------------------------------------------------

    def submit(
        self,
        *,
        blind_id: str,
        task: TournamentTask,
        snapshot: MarketSnapshot,
        prediction: Any,
        confidence: float,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
    ) -> BlindSubmission:
        """
        Store an anonymous tournament submission.
        """

        if blind_id not in self._blind_to_competitor:
            raise BlindEvaluationError(
                f"Unknown blind competitor: {blind_id!r}"
            )

        self.validate_task(task, snapshot)

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0.0 and 1.0"
            )

        if latency_ms < 0.0:
            raise ValueError(
                "latency_ms cannot be negative"
            )

        if cost_usd < 0.0:
            raise ValueError(
                "cost_usd cannot be negative"
            )

        for existing in self._submissions.values():
            if (
                existing.blind_id == blind_id
                and existing.task_id == task.task_id
            ):
                raise BlindEvaluationError(
                    "Duplicate submission detected for "
                    f"blind_id={blind_id!r}, task_id={task.task_id!r}."
                )

        submission = BlindSubmission(
            submission_id=generate_id("submission"),
            blind_id=blind_id,
            task_id=task.task_id,
            snapshot_id=snapshot.snapshot_id,
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )

        self._submissions[submission.submission_id] = submission

        return submission

    # ------------------------------------------------------------------
    # Submission retrieval
    # ------------------------------------------------------------------

    def get_submission(
        self,
        submission_id: str,
    ) -> BlindSubmission:
        """
        Return one blind submission.
        """

        try:
            return self._submissions[submission_id]
        except KeyError as exc:
            raise BlindEvaluationError(
                f"Unknown submission: {submission_id!r}"
            ) from exc

    def submissions_for_task(
        self,
        task_id: str,
    ) -> List[BlindSubmission]:
        """
        Return anonymous submissions belonging to one task.
        """

        return [
            submission
            for submission in self._submissions.values()
            if submission.task_id == task_id
        ]

    # ------------------------------------------------------------------
    # Identity reveal
    # ------------------------------------------------------------------

    def reveal_competitor(
        self,
        blind_id: str,
    ) -> str:
        """
        Reveal the real competitor ID after blind evaluation.
        """

        try:
            return self._blind_to_competitor[blind_id]
        except KeyError as exc:
            raise BlindEvaluationError(
                f"Unknown blind ID: {blind_id!r}"
            ) from exc

    def reveal_submission(
        self,
        submission_id: str,
    ) -> Dict[str, Any]:
        """
        Reveal competitor identity for an evaluated submission.
        """

        submission = self.get_submission(submission_id)

        return {
            "submission_id": submission.submission_id,
            "competitor_id": self.reveal_competitor(
                submission.blind_id
            ),
            "task_id": submission.task_id,
            "snapshot_id": submission.snapshot_id,
            "prediction": submission.prediction,
            "confidence": submission.confidence,
            "latency_ms": submission.latency_ms,
            "cost_usd": submission.cost_usd,
        }

    # ------------------------------------------------------------------
    # Tournament diagnostics
    # ------------------------------------------------------------------

    @property
    def competitor_count(self) -> int:
        return len(self._competitor_to_blind)

    @property
    def submission_count(self) -> int:
        return len(self._submissions)

    def blind_mapping(self) -> Dict[str, str]:
        """
        Return a copy of the internal blind mapping.

        Intended for governance/audit use after evaluation,
        not for the scoring layer.
        """

        return dict(self._blind_to_competitor)


__all__ = [
    "BlindCompetitor",
    "BlindEvaluationError",
    "BlindEvaluator",
    "BlindSubmission",
]