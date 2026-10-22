"""
Agent Tournament System - v5.0.6

Champion / Challenger evaluation and governance framework.

Main capabilities:

- Blind competitor evaluation
- Identical market snapshot evaluation
- Prediction quality scoring
- Calibration scoring
- Consistency scoring
- Latency and cost scoring
- Asset x Regime x Horizon performance matrices
- Champion / Challenger historical comparison
- Promotion governance
- Explicit approval workflow
- Complete audit history
- Champion rollback
"""

# ==============================================================
# Core models
# ==============================================================

from .models import (
    AgentCompetitor,
    AuditRecord,
    BlindSubmission as ModelBlindSubmission,
    CompetitorResult,
    CompetitorRole,
    DecisionType,
    EvaluationMetrics,
    MarketSnapshot,
    PromotionDecision,
    RollbackRecord,
    TournamentResult,
    TournamentStatus,
    TournamentTask,
    generate_id,
    utc_now,
)

# ==============================================================
# Blind evaluation
# ==============================================================

from .blind_evaluator import (
    BlindCompetitor,
    BlindEvaluationError,
    BlindEvaluator,
    BlindSubmission,
)

# ==============================================================
# Scoring
# ==============================================================

from .scoring_engine import (
    ScoredObservation,
    ScoringEngine,
    ScoringError,
    ScoringWeights,
)

# ==============================================================
# Tournament execution
# ==============================================================

from .tournament_engine import (
    TournamentConfiguration,
    TournamentEngine,
    TournamentError,
)

# ==============================================================
# Performance matrix
# ==============================================================

from .performance_matrix import (
    AggregatedPerformance,
    MatrixLeaderboardEntry,
    PerformanceDimension,
    PerformanceMatrix,
    PerformanceMatrixError,
    PerformanceRecord,
)

# ==============================================================
# Promotion governance
# ==============================================================

from .promotion_engine import (
    PromotionEngine,
    PromotionEngineError,
    PromotionEvaluation,
    PromotionPolicy,
)

# ==============================================================
# Audit and rollback governance
# ==============================================================

from .audit_rollback import (
    AuditRollbackEngine,
    AuditRollbackError,
)


__version__ = "5.0.6"


__all__ = [
    # Version
    "__version__",

    # Core models
    "AgentCompetitor",
    "AuditRecord",
    "CompetitorResult",
    "CompetitorRole",
    "DecisionType",
    "EvaluationMetrics",
    "MarketSnapshot",
    "ModelBlindSubmission",
    "PromotionDecision",
    "RollbackRecord",
    "TournamentResult",
    "TournamentStatus",
    "TournamentTask",
    "generate_id",
    "utc_now",

    # Blind evaluation
    "BlindCompetitor",
    "BlindEvaluationError",
    "BlindEvaluator",
    "BlindSubmission",

    # Scoring
    "ScoredObservation",
    "ScoringEngine",
    "ScoringError",
    "ScoringWeights",

    # Tournament
    "TournamentConfiguration",
    "TournamentEngine",
    "TournamentError",

    # Performance matrix
    "AggregatedPerformance",
    "MatrixLeaderboardEntry",
    "PerformanceDimension",
    "PerformanceMatrix",
    "PerformanceMatrixError",
    "PerformanceRecord",

    # Promotion governance
    "PromotionEngine",
    "PromotionEngineError",
    "PromotionEvaluation",
    "PromotionPolicy",

    # Audit / rollback
    "AuditRollbackEngine",
    "AuditRollbackError",
]