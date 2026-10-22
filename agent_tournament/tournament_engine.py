"""
Agent Tournament System - Tournament Engine
============================================

v5.0.6 of the Multi-Agent Financial Research Platform.

Coordinates:

- Champion / Challenger registration
- blind competitor evaluation
- asset x regime x horizon tournaments
- scoring
- ranking
- winner selection
- tournament lifecycle

Promotion/demotion governance is intentionally handled by
a separate governance layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence

from .blind_evaluator import BlindEvaluator, BlindSubmission
from .models import (
    AgentCompetitor,
    CompetitorResult,
    CompetitorRole,
    MarketSnapshot,
    TournamentResult,
    TournamentStatus,
    TournamentTask,
    generate_id,
)
from .scoring_engine import (
    ScoredObservation,
    ScoringEngine,
    ScoringError,
)


class TournamentError(RuntimeError):
    """Raised when tournament execution fails."""


@dataclass(frozen=True)
class TournamentConfiguration:
    """
    Configuration for one tournament.

    minimum_observations controls the amount of evidence required
    before a competitor can receive a valid aggregate score.
    """

    minimum_observations: int = 1

    def __post_init__(self) -> None:
        if self.minimum_observations < 1:
            raise ValueError(
                "minimum_observations must be at least 1."
            )


class TournamentEngine:
    """
    Central coordinator for champion/challenger tournaments.

    The engine never scores competitors by their real identity.

    Real competitor IDs are revealed only after the blind scoring
    stage has completed.
    """

    def __init__(
        self,
        *,
        scoring_engine: Optional[ScoringEngine] = None,
        blind_evaluator: Optional[BlindEvaluator] = None,
        configuration: Optional[TournamentConfiguration] = None,
    ) -> None:
        self.scoring_engine = scoring_engine or ScoringEngine()
        self.blind_evaluator = blind_evaluator or BlindEvaluator()

        self.configuration = (
            configuration or TournamentConfiguration()
        )

        self._competitors: Dict[str, AgentCompetitor] = {}

    # ------------------------------------------------------------------
    # Competitor management
    # ------------------------------------------------------------------

    def register_competitors(
        self,
        competitors: Iterable[AgentCompetitor],
    ) -> None:
        """
        Register tournament competitors.

        Exactly one Champion is required for tournament execution.
        """

        competitors_list = list(competitors)

        if not competitors_list:
            raise TournamentError(
                "At least one competitor must be registered."
            )

        for competitor in competitors_list:
            if competitor.competitor_id in self._competitors:
                raise TournamentError(
                    "Duplicate competitor_id: "
                    f"{competitor.competitor_id!r}"
                )

            self._competitors[
                competitor.competitor_id
            ] = competitor

        self.blind_evaluator.register_competitors(
            competitors_list
        )

    def get_competitor(
        self,
        competitor_id: str,
    ) -> AgentCompetitor:
        try:
            return self._competitors[competitor_id]

        except KeyError as exc:
            raise TournamentError(
                f"Unknown competitor: {competitor_id!r}"
            ) from exc

    def champion(self) -> AgentCompetitor:
        champions = [
            competitor
            for competitor in self._competitors.values()
            if competitor.role == CompetitorRole.CHAMPION
        ]

        if len(champions) != 1:
            raise TournamentError(
                "Exactly one Champion must be registered. "
                f"Found {len(champions)}."
            )

        return champions[0]

    def challengers(self) -> List[AgentCompetitor]:
        return [
            competitor
            for competitor in self._competitors.values()
            if competitor.role == CompetitorRole.CHALLENGER
        ]

    # ------------------------------------------------------------------
    # Tournament validation
    # ------------------------------------------------------------------

    def _validate_tournament_dimensions(
        self,
        task: TournamentTask,
        snapshot: MarketSnapshot,
    ) -> None:
        """
        Validate asset x regime x horizon consistency.
        """

        if task.asset != snapshot.asset:
            raise TournamentError(
                "Task asset does not match snapshot asset."
            )

        if task.regime != snapshot.regime:
            raise TournamentError(
                "Task regime does not match snapshot regime."
            )

        if task.horizon != snapshot.horizon:
            raise TournamentError(
                "Task horizon does not match snapshot horizon."
            )

        if task.snapshot_id != snapshot.snapshot_id:
            raise TournamentError(
                "Task snapshot_id does not match snapshot."
            )

    # ------------------------------------------------------------------
    # Snapshot preparation
    # ------------------------------------------------------------------

    def prepare_snapshot(
        self,
        snapshot: MarketSnapshot,
    ) -> str:
        """
        Register the canonical snapshot before submissions begin.
        """

        return self.blind_evaluator.register_snapshot(snapshot)

    # ------------------------------------------------------------------
    # Submission validation
    # ------------------------------------------------------------------

    def _validate_submissions(
        self,
        submissions: Sequence[BlindSubmission],
    ) -> None:
        """
        Ensure every registered competitor submitted exactly once
        for the tournament task.
        """

        if not submissions:
            raise TournamentError(
                "Tournament contains no submissions."
            )

        submitted_blind_ids = [
            submission.blind_id
            for submission in submissions
        ]

        if len(submitted_blind_ids) != len(
            set(submitted_blind_ids)
        ):
            raise TournamentError(
                "Duplicate blind competitor submissions detected."
            )

        expected_count = len(self._competitors)

        if len(submissions) != expected_count:
            raise TournamentError(
                "Tournament requires one submission from every "
                f"registered competitor. Expected {expected_count}, "
                f"received {len(submissions)}."
            )

    # ------------------------------------------------------------------
    # Blind scoring
    # ------------------------------------------------------------------

    def _group_observations(
        self,
        observations: Iterable[ScoredObservation],
    ) -> Dict[str, List[ScoredObservation]]:
        grouped: Dict[str, List[ScoredObservation]] = {}

        for observation in observations:
            blind_id = observation.submission.blind_id

            grouped.setdefault(
                blind_id,
                [],
            ).append(observation)

        return grouped

    def _score_competitors(
        self,
        *,
        observations: Iterable[ScoredObservation],
        task: TournamentTask,
    ) -> List[CompetitorResult]:
        """
        Score competitors while identities remain blind.
        """

        grouped = self._group_observations(observations)

        if not grouped:
            raise TournamentError(
                "No scored observations were supplied."
            )

        blind_results = []

        for blind_id, competitor_observations in grouped.items():

            if (
                len(competitor_observations)
                < self.configuration.minimum_observations
            ):
                raise TournamentError(
                    f"Blind competitor {blind_id!r} has "
                    f"{len(competitor_observations)} observations; "
                    f"{self.configuration.minimum_observations} required."
                )

            try:
                metrics = self.scoring_engine.evaluate(
                    competitor_observations
                )

            except ScoringError as exc:
                raise TournamentError(
                    f"Scoring failed for blind competitor "
                    f"{blind_id!r}: {exc}"
                ) from exc

            blind_results.append(
                (
                    blind_id,
                    metrics,
                )
            )

        # Rank while identities are still hidden.
        blind_results.sort(
            key=lambda item: item[1].composite_score,
            reverse=True,
        )

        results: List[CompetitorResult] = []

        for rank, (blind_id, metrics) in enumerate(
            blind_results,
            start=1,
        ):
            competitor_id = (
                self.blind_evaluator.reveal_competitor(
                    blind_id
                )
            )

            results.append(
                CompetitorResult(
                    competitor_id=competitor_id,
                    asset=task.asset,
                    regime=task.regime,
                    horizon=task.horizon,
                    metrics=metrics,
                    rank=rank,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Tournament execution
    # ------------------------------------------------------------------

    def run_tournament(
        self,
        *,
        task: TournamentTask,
        snapshot: MarketSnapshot,
        observations: Iterable[ScoredObservation],
    ) -> TournamentResult:
        """
        Execute one complete blind tournament.

        Expected workflow:

        1. validate Champion / Challengers
        2. validate task dimensions
        3. verify canonical snapshot
        4. validate submissions
        5. score blindly
        6. rank competitors
        7. reveal identities
        8. return TournamentResult
        """

        champion = self.champion()
        challengers = self.challengers()

        if not challengers:
            raise TournamentError(
                "At least one Challenger is required."
            )

        self._validate_tournament_dimensions(
            task,
            snapshot,
        )

        self.blind_evaluator.verify_snapshot(snapshot)

        observations_list = list(observations)

        submissions = [
            observation.submission
            for observation in observations_list
        ]

        # Multiple observations per competitor may be present.
        unique_submissions = {
            submission.submission_id: submission
            for submission in submissions
        }

        if not unique_submissions:
            raise TournamentError(
                "No tournament submissions found."
            )

        started_at = datetime.now(timezone.utc)

        tournament_id = generate_id("tournament")

        try:
            results = self._score_competitors(
                observations=observations_list,
                task=task,
            )

            if len(results) != len(self._competitors):
                raise TournamentError(
                    "Not every registered competitor received "
                    "a tournament score."
                )

            winner_id = results[0].competitor_id

            completed_at = datetime.now(timezone.utc)

            return TournamentResult(
                tournament_id=tournament_id,
                champion_id=champion.competitor_id,
                challenger_ids=[
                    challenger.competitor_id
                    for challenger in challengers
                ],
                results=results,
                winner_id=winner_id,
                status=TournamentStatus.COMPLETED,
                started_at=started_at,
                completed_at=completed_at,
                metadata={
                    "asset": task.asset,
                    "regime": task.regime,
                    "horizon": task.horizon,
                    "task_id": task.task_id,
                    "snapshot_id": snapshot.snapshot_id,
                    "competitor_count": len(
                        self._competitors
                    ),
                },
            )

        except Exception as exc:
            if isinstance(exc, TournamentError):
                raise

            raise TournamentError(
                f"Tournament execution failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def competitor_count(self) -> int:
        return len(self._competitors)

    def leaderboard(
        self,
        tournament_result: TournamentResult,
    ) -> List[CompetitorResult]:
        """
        Return tournament results ordered by rank.
        """

        return sorted(
            tournament_result.results,
            key=lambda result: (
                result.rank
                if result.rank is not None
                else 10**9
            ),
        )


__all__ = [
    "TournamentConfiguration",
    "TournamentEngine",
    "TournamentError",
]