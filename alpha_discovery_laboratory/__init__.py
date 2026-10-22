"""
Alpha Discovery Laboratory
==========================

v6.0.3 component of the Multi-Agent Financial Research Platform.

Research pipeline:

Feature Discovery
    ->
Feature Combination Discovery
    ->
Strategy Candidate Generation
    ->
Backtest Validation
    ->
Causal Validation
    ->
Out-of-Sample Validation
    ->
Robustness Validation
    ->
Overfit / Stability Detection
    ->
Alpha Lineage
    ->
Reproducible Experiment Records

Research-only architecture.
No autonomous real-money execution.
"""

# ============================================================
# Core Models
# ============================================================

from .models import (
    VERSION,
    AlphaDiscoveryError,
    AlphaValidationError,
    FeatureType,
    FeatureTransform,
    CombinationOperator,
    StrategyDirection,
    CandidateStatus,
    ValidationStage,
    ValidationStatus,
    RejectionReason,
    MarketFeature,
    FeatureCombination,
    StrategyCandidate,
    PerformanceMetrics,
    ValidationResult,
    AlphaLineageRecord,
    AlphaDiscoveryExperiment,
    AlphaDiscoverySummary,
    utc_now,
    canonical_json,
    stable_hash,
)

# ============================================================
# Feature Discovery
# ============================================================

from .feature_discovery import (
    FeatureDiscoveryError,
    FeatureAlreadyExistsError,
    FeatureNotFoundError,
    CombinationAlreadyExistsError,
    CombinationNotFoundError,
    FeatureDiscoveryAuditRecord,
    FeatureDiscoveryResult,
    FeatureDiscoverySnapshot,
    FeatureDiscoveryEngine,
)

# ============================================================
# Candidate Generation
# ============================================================

from .candidate_generator import (
    CandidateGenerationError,
    CandidateAlreadyExistsError,
    CandidateNotFoundError,
    CandidateSourceError,
    CandidateGenerationAuditRecord,
    CandidateGenerationResult,
    CandidateGeneratorSnapshot,
    StrategyCandidateGenerator,
)

# ============================================================
# Validation Pipeline
# ============================================================

from .validation_pipeline import (
    ValidationPipelineError,
    ValidationStageError,
    ValidationConfigurationError,
    ValidationThresholds,
    BacktestEvidence,
    CausalEvidence,
    OOSEvidence,
    RobustnessEvidence,
    ValidationPipelineAuditRecord,
    ValidationPipelineResult,
    AlphaValidationPipeline,
)

# ============================================================
# Alpha Lineage
# ============================================================

from .alpha_lineage import (
    AlphaLineageError,
    AlphaLineageNotFoundError,
    AlphaExperimentNotFoundError,
    AlphaLineageAlreadyExistsError,
    AlphaExperimentAlreadyExistsError,
    AlphaLineageAuditRecord,
    AlphaLineageSnapshot,
    AlphaLineageRegistry,
)

# ============================================================
# Overfit Detection
# ============================================================

from .overfit_detector import (
    OverfitDetectionError,
    OverfitConfigurationError,
    OverfitThresholds,
    ParameterPerturbation,
    RegimePerformance,
    OverfitEvidence,
    OverfitDetectionResult,
    OverfitAuditRecord,
    OverfitDetector,
)

# ============================================================
# Alpha Discovery Laboratory
# ============================================================

from .alpha_discovery_lab import (
    AlphaDiscoveryLabError,
    AlphaDiscoveryLabValidationError,
    AlphaDiscoveryExperimentError,
    AlphaResearchInput,
    CandidateEvidenceBundle,
    CandidateResearchResult,
    AlphaDiscoveryLabResult,
    AlphaDiscoveryLabAuditRecord,
    AlphaDiscoveryLaboratory,
)


# ============================================================
# Package Version
# ============================================================

__version__ = VERSION


# ============================================================
# Public API
# ============================================================

__all__ = [
    # Version
    "VERSION",
    "__version__",

    # Helpers
    "utc_now",
    "canonical_json",
    "stable_hash",

    # Core Exceptions
    "AlphaDiscoveryError",
    "AlphaValidationError",

    # Enums
    "FeatureType",
    "FeatureTransform",
    "CombinationOperator",
    "StrategyDirection",
    "CandidateStatus",
    "ValidationStage",
    "ValidationStatus",
    "RejectionReason",

    # Core Models
    "MarketFeature",
    "FeatureCombination",
    "StrategyCandidate",
    "PerformanceMetrics",
    "ValidationResult",
    "AlphaLineageRecord",
    "AlphaDiscoveryExperiment",
    "AlphaDiscoverySummary",

    # Feature Discovery
    "FeatureDiscoveryError",
    "FeatureAlreadyExistsError",
    "FeatureNotFoundError",
    "CombinationAlreadyExistsError",
    "CombinationNotFoundError",
    "FeatureDiscoveryAuditRecord",
    "FeatureDiscoveryResult",
    "FeatureDiscoverySnapshot",
    "FeatureDiscoveryEngine",

    # Candidate Generation
    "CandidateGenerationError",
    "CandidateAlreadyExistsError",
    "CandidateNotFoundError",
    "CandidateSourceError",
    "CandidateGenerationAuditRecord",
    "CandidateGenerationResult",
    "CandidateGeneratorSnapshot",
    "StrategyCandidateGenerator",

    # Validation
    "ValidationPipelineError",
    "ValidationStageError",
    "ValidationConfigurationError",
    "ValidationThresholds",
    "BacktestEvidence",
    "CausalEvidence",
    "OOSEvidence",
    "RobustnessEvidence",
    "ValidationPipelineAuditRecord",
    "ValidationPipelineResult",
    "AlphaValidationPipeline",

    # Lineage
    "AlphaLineageError",
    "AlphaLineageNotFoundError",
    "AlphaExperimentNotFoundError",
    "AlphaLineageAlreadyExistsError",
    "AlphaExperimentAlreadyExistsError",
    "AlphaLineageAuditRecord",
    "AlphaLineageSnapshot",
    "AlphaLineageRegistry",

    # Overfit Detection
    "OverfitDetectionError",
    "OverfitConfigurationError",
    "OverfitThresholds",
    "ParameterPerturbation",
    "RegimePerformance",
    "OverfitEvidence",
    "OverfitDetectionResult",
    "OverfitAuditRecord",
    "OverfitDetector",

    # Laboratory
    "AlphaDiscoveryLabError",
    "AlphaDiscoveryLabValidationError",
    "AlphaDiscoveryExperimentError",
    "AlphaResearchInput",
    "CandidateEvidenceBundle",
    "CandidateResearchResult",
    "AlphaDiscoveryLabResult",
    "AlphaDiscoveryLabAuditRecord",
    "AlphaDiscoveryLaboratory",
]