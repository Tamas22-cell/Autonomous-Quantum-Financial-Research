"""
Performance Matrix for the v5.0.6 Agent Tournament System.

The matrix stores and aggregates tournament performance across:

    Asset x Regime x Horizon x Competitor

This allows the tournament system to answer questions such as:

    - Which agent performs best on BTC?
    - Which agent performs best during high-volatility regimes?
    - Which agent performs best on a 1D / 7D / 30D horizon?
    - Is a challenger consistently better than the current champion?
    - Is an agent strong globally but weak in a specific regime?

The module intentionally operates on CompetitorResult objects produced by
the TournamentEngine so it remains compatible with the existing v5.0.6
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .models import CompetitorResult, TournamentResult


class PerformanceMatrixError(Exception):
    """
    Raised when invalid data is supplied to the performance matrix.
    """


@dataclass(frozen=True)
class PerformanceDimension:
    """
    Unique Asset x Regime x Horizon dimension.
    """

    asset: str
    regime: str
    horizon: str

    def __post_init__(self) -> None:
        if not self.asset:
            raise PerformanceMatrixError(
                "PerformanceDimension.asset cannot be empty."
            )

        if not self.regime:
            raise PerformanceMatrixError(
                "PerformanceDimension.regime cannot be empty."
            )

        if not self.horizon:
            raise PerformanceMatrixError(
                "PerformanceDimension.horizon cannot be empty."
            )

    @property
    def key(self) -> Tuple[str, str, str]:
        """
        Return the dimension as a dictionary-compatible tuple key.
        """

        return (
            self.asset,
            self.regime,
            self.horizon,
        )


@dataclass
class PerformanceRecord:
    """
    Single competitor performance observation.
    """

    competitor_id: str
    tournament_id: str
    asset: str
    regime: str
    horizon: str

    composite_score: float
    prediction_score: float
    calibration_score: float
    consistency_score: float

    latency_ms: float
    cost_usd: float

    sample_count: int = 0
    rank: Optional[int] = None

    metadata: Dict[str, object] = field(
        default_factory=dict
    )

    @property
    def dimension(self) -> PerformanceDimension:
        return PerformanceDimension(
            asset=self.asset,
            regime=self.regime,
            horizon=self.horizon,
        )


@dataclass
class AggregatedPerformance:
    """
    Aggregated historical performance for one competitor
    inside one Asset x Regime x Horizon dimension.
    """

    competitor_id: str
    asset: str
    regime: str
    horizon: str

    tournament_count: int

    average_composite_score: float
    average_prediction_score: float
    average_calibration_score: float
    average_consistency_score: float

    average_latency_ms: float
    average_cost_usd: float

    total_sample_count: int

    best_composite_score: float
    worst_composite_score: float

    win_count: int
    win_rate: float


@dataclass
class MatrixLeaderboardEntry:
    """
    Ranked competitor inside a specific matrix dimension.
    """

    competitor_id: str
    rank: int
    performance: AggregatedPerformance


class PerformanceMatrix:
    """
    Historical Asset x Regime x Horizon performance matrix.

    The matrix accepts TournamentResult objects produced by
    TournamentEngine and maintains historical performance records
    for every competitor.
    """

    def __init__(self) -> None:
        self._records: List[PerformanceRecord] = []
        self._processed_tournaments: set[str] = set()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_tournament_result(
        self,
        tournament_result: TournamentResult,
    ) -> None:
        """
        Add every competitor result from a completed tournament.

        The same tournament cannot be added twice.
        """

        if not tournament_result.tournament_id:
            raise PerformanceMatrixError(
                "Tournament result must have a tournament_id."
            )

        if tournament_result.tournament_id in self._processed_tournaments:
            raise PerformanceMatrixError(
                "Tournament has already been added to the "
                "performance matrix: "
                f"{tournament_result.tournament_id}"
            )

        if not tournament_result.results:
            raise PerformanceMatrixError(
                "Tournament result contains no competitor results."
            )

        new_records: List[PerformanceRecord] = []

        for result in tournament_result.results:
            record = self._build_record(
                tournament_id=tournament_result.tournament_id,
                result=result,
            )

            new_records.append(record)

        self._records.extend(new_records)

        self._processed_tournaments.add(
            tournament_result.tournament_id
        )

    def add_competitor_result(
        self,
        tournament_id: str,
        result: CompetitorResult,
    ) -> None:
        """
        Add a single competitor result manually.

        Primarily useful for testing, replay systems, historical imports,
        or future persistence adapters.
        """

        if not tournament_id:
            raise PerformanceMatrixError(
                "tournament_id cannot be empty."
            )

        record = self._build_record(
            tournament_id=tournament_id,
            result=result,
        )

        self._records.append(record)

    @staticmethod
    def _build_record(
        tournament_id: str,
        result: CompetitorResult,
    ) -> PerformanceRecord:
        """
        Convert CompetitorResult into an internal PerformanceRecord.
        """

        if not result.competitor_id:
            raise PerformanceMatrixError(
                "CompetitorResult.competitor_id cannot be empty."
            )

        metrics = result.metrics

        return PerformanceRecord(
            competitor_id=result.competitor_id,
            tournament_id=tournament_id,
            asset=result.asset,
            regime=result.regime,
            horizon=result.horizon,
            composite_score=metrics.composite_score,
            prediction_score=metrics.prediction_score,
            calibration_score=metrics.calibration_score,
            consistency_score=metrics.consistency_score,
            latency_ms=metrics.latency_ms,
            cost_usd=metrics.cost_usd,
            sample_count=metrics.sample_count,
            rank=result.rank,
            metadata=dict(result.metadata),
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def records(self) -> List[PerformanceRecord]:
        """
        Return a copy of all historical performance records.
        """

        return list(self._records)

    def dimensions(self) -> List[PerformanceDimension]:
        """
        Return all unique Asset x Regime x Horizon dimensions.
        """

        unique_dimensions = {
            record.dimension
            for record in self._records
        }

        return sorted(
            unique_dimensions,
            key=lambda item: (
                item.asset,
                item.regime,
                item.horizon,
            ),
        )

    def competitor_ids(self) -> List[str]:
        """
        Return all competitors currently represented in the matrix.
        """

        return sorted(
            {
                record.competitor_id
                for record in self._records
            }
        )

    def get_records(
        self,
        competitor_id: Optional[str] = None,
        asset: Optional[str] = None,
        regime: Optional[str] = None,
        horizon: Optional[str] = None,
    ) -> List[PerformanceRecord]:
        """
        Filter historical records.

        Any parameter left as None acts as a wildcard.
        """

        records = self._records

        if competitor_id is not None:
            records = [
                record
                for record in records
                if record.competitor_id == competitor_id
            ]

        if asset is not None:
            records = [
                record
                for record in records
                if record.asset == asset
            ]

        if regime is not None:
            records = [
                record
                for record in records
                if record.regime == regime
            ]

        if horizon is not None:
            records = [
                record
                for record in records
                if record.horizon == horizon
            ]

        return list(records)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(
        self,
        competitor_id: str,
        asset: str,
        regime: str,
        horizon: str,
    ) -> AggregatedPerformance:
        """
        Aggregate historical performance for one competitor
        in one matrix dimension.
        """

        records = self.get_records(
            competitor_id=competitor_id,
            asset=asset,
            regime=regime,
            horizon=horizon,
        )

        if not records:
            raise PerformanceMatrixError(
                "No performance records found for "
                f"competitor={competitor_id}, "
                f"asset={asset}, "
                f"regime={regime}, "
                f"horizon={horizon}."
            )

        tournament_count = len(records)

        average_composite_score = self._mean(
            record.composite_score
            for record in records
        )

        average_prediction_score = self._mean(
            record.prediction_score
            for record in records
        )

        average_calibration_score = self._mean(
            record.calibration_score
            for record in records
        )

        average_consistency_score = self._mean(
            record.consistency_score
            for record in records
        )

        average_latency_ms = self._mean(
            record.latency_ms
            for record in records
        )

        average_cost_usd = self._mean(
            record.cost_usd
            for record in records
        )

        total_sample_count = sum(
            record.sample_count
            for record in records
        )

        composite_scores = [
            record.composite_score
            for record in records
        ]

        win_count = sum(
            1
            for record in records
            if record.rank == 1
        )

        win_rate = (
            win_count / tournament_count
            if tournament_count
            else 0.0
        )

        return AggregatedPerformance(
            competitor_id=competitor_id,
            asset=asset,
            regime=regime,
            horizon=horizon,
            tournament_count=tournament_count,
            average_composite_score=average_composite_score,
            average_prediction_score=average_prediction_score,
            average_calibration_score=average_calibration_score,
            average_consistency_score=average_consistency_score,
            average_latency_ms=average_latency_ms,
            average_cost_usd=average_cost_usd,
            total_sample_count=total_sample_count,
            best_composite_score=max(composite_scores),
            worst_composite_score=min(composite_scores),
            win_count=win_count,
            win_rate=win_rate,
        )

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    def leaderboard(
        self,
        asset: str,
        regime: str,
        horizon: str,
    ) -> List[MatrixLeaderboardEntry]:
        """
        Rank all competitors that have historical observations
        for the requested Asset x Regime x Horizon dimension.

        Ranking is based on average composite score.
        """

        matching_records = self.get_records(
            asset=asset,
            regime=regime,
            horizon=horizon,
        )

        if not matching_records:
            return []

        competitor_ids = sorted(
            {
                record.competitor_id
                for record in matching_records
            }
        )

        performances = [
            self.aggregate(
                competitor_id=competitor_id,
                asset=asset,
                regime=regime,
                horizon=horizon,
            )
            for competitor_id in competitor_ids
        ]

        performances.sort(
            key=lambda item: (
                item.average_composite_score,
                item.average_prediction_score,
                item.average_calibration_score,
                item.average_consistency_score,
                -item.average_latency_ms,
                -item.average_cost_usd,
            ),
            reverse=True,
        )

        return [
            MatrixLeaderboardEntry(
                competitor_id=performance.competitor_id,
                rank=index,
                performance=performance,
            )
            for index, performance in enumerate(
                performances,
                start=1,
            )
        ]

    # ------------------------------------------------------------------
    # Champion / Challenger comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        champion_id: str,
        challenger_id: str,
        asset: str,
        regime: str,
        horizon: str,
    ) -> Dict[str, object]:
        """
        Compare champion and challenger performance inside
        one Asset x Regime x Horizon dimension.
        """

        champion = self.aggregate(
            competitor_id=champion_id,
            asset=asset,
            regime=regime,
            horizon=horizon,
        )

        challenger = self.aggregate(
            competitor_id=challenger_id,
            asset=asset,
            regime=regime,
            horizon=horizon,
        )

        score_delta = (
            challenger.average_composite_score
            - champion.average_composite_score
        )

        prediction_delta = (
            challenger.average_prediction_score
            - champion.average_prediction_score
        )

        calibration_delta = (
            challenger.average_calibration_score
            - champion.average_calibration_score
        )

        consistency_delta = (
            challenger.average_consistency_score
            - champion.average_consistency_score
        )

        latency_delta_ms = (
            challenger.average_latency_ms
            - champion.average_latency_ms
        )

        cost_delta_usd = (
            challenger.average_cost_usd
            - champion.average_cost_usd
        )

        if score_delta > 0:
            preferred_competitor_id = challenger_id

        elif score_delta < 0:
            preferred_competitor_id = champion_id

        else:
            preferred_competitor_id = None

        return {
            "champion_id": champion_id,
            "challenger_id": challenger_id,
            "asset": asset,
            "regime": regime,
            "horizon": horizon,
            "champion_score": (
                champion.average_composite_score
            ),
            "challenger_score": (
                challenger.average_composite_score
            ),
            "score_delta": score_delta,
            "prediction_delta": prediction_delta,
            "calibration_delta": calibration_delta,
            "consistency_delta": consistency_delta,
            "latency_delta_ms": latency_delta_ms,
            "cost_delta_usd": cost_delta_usd,
            "preferred_competitor_id": (
                preferred_competitor_id
            ),
        }

    # ------------------------------------------------------------------
    # Global competitor performance
    # ------------------------------------------------------------------

    def global_average_score(
        self,
        competitor_id: str,
    ) -> float:
        """
        Calculate a competitor's average composite score
        across every recorded dimension.
        """

        records = self.get_records(
            competitor_id=competitor_id
        )

        if not records:
            raise PerformanceMatrixError(
                "No performance records found for competitor: "
                f"{competitor_id}"
            )

        return self._mean(
            record.composite_score
            for record in records
        )

    def best_dimension(
        self,
        competitor_id: str,
    ) -> AggregatedPerformance:
        """
        Return the competitor's strongest historical dimension.
        """

        performances = self._competitor_dimension_aggregates(
            competitor_id
        )

        if not performances:
            raise PerformanceMatrixError(
                "No performance records found for competitor: "
                f"{competitor_id}"
            )

        return max(
            performances,
            key=lambda item: item.average_composite_score,
        )

    def worst_dimension(
        self,
        competitor_id: str,
    ) -> AggregatedPerformance:
        """
        Return the competitor's weakest historical dimension.
        """

        performances = self._competitor_dimension_aggregates(
            competitor_id
        )

        if not performances:
            raise PerformanceMatrixError(
                "No performance records found for competitor: "
                f"{competitor_id}"
            )

        return min(
            performances,
            key=lambda item: item.average_composite_score,
        )

    def _competitor_dimension_aggregates(
        self,
        competitor_id: str,
    ) -> List[AggregatedPerformance]:
        """
        Build aggregated performance objects for every dimension
        in which the competitor has observations.
        """

        records = self.get_records(
            competitor_id=competitor_id
        )

        dimensions = {
            (
                record.asset,
                record.regime,
                record.horizon,
            )
            for record in records
        }

        return [
            self.aggregate(
                competitor_id=competitor_id,
                asset=asset,
                regime=regime,
                horizon=horizon,
            )
            for asset, regime, horizon in dimensions
        ]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _mean(
        values: Iterable[float],
    ) -> float:
        """
        Calculate arithmetic mean without external dependencies.
        """

        values_list = list(values)

        if not values_list:
            return 0.0

        return sum(values_list) / len(values_list)

    def __len__(self) -> int:
        """
        Number of historical performance records.
        """

        return len(self._records)

    def __bool__(self) -> bool:
        """
        True when the matrix contains at least one record.
        """

        return bool(self._records)


__all__ = [
    "AggregatedPerformance",
    "MatrixLeaderboardEntry",
    "PerformanceDimension",
    "PerformanceMatrix",
    "PerformanceMatrixError",
    "PerformanceRecord",
]