"""
Production Research Infrastructure — v6.0.1

Infrastructure layer for reproducible, versioned,
event-driven and fully traceable quantitative research.

Research-only system.
No live-trading authorization is provided by this package.
"""

# ============================================================
# Data contracts
# ============================================================

from .data_contracts import (
    SCHEMA_VERSION,
    ContractCompatibility,
    ContractValidationResult,
    ContractVersion,
    ContractVersionError,
    DataContract,
    DataContractError,
    DataContractValidator,
    DataEnvelope,
    DataField,
    DataQualityStatus,
    DatasetReference,
    DatasetType,
    ValidationIssue,
)

# ============================================================
# Dataset registry
# ============================================================

from .dataset_registry import (
    DatasetAlreadyExistsError,
    DatasetContractNotFoundError,
    DatasetNotFoundError,
    DatasetRegistry,
    DatasetRegistryError,
    DatasetRegistrySnapshot,
    DatasetValidationError,
    DatasetVersionRecord,
)

# ============================================================
# Event bus
# ============================================================

from .event_bus import (
    DeliveryStatus,
    EventBus,
    EventBusError,
    EventDeliveryReport,
    EventPriority,
    EventType,
    EventValidationError,
    HandlerResult,
    ResearchEvent,
    SubscriberError,
    Subscription,
)

# ============================================================
# Parallel execution
# ============================================================

from .parallel_executor import (
    ExecutionStatus,
    ExecutionTask,
    ExecutionValidationError,
    ParallelExecutionReport,
    ParallelExecutor,
    ParallelExecutorError,
    TaskExecutionResult,
)

# ============================================================
# Research execution engine
# ============================================================

from .execution_engine import (
    ResearchExecutionEngine,
    ResearchExecutionError,
    ResearchExecutionRequest,
    ResearchExecutionValidationError,
    ResearchRun,
    ResearchRunStatus,
)

# ============================================================
# Version registry
# ============================================================

from .version_registry import (
    SemanticVersion,
    VersionAlreadyExistsError,
    VersionNotFoundError,
    VersionRecord,
    VersionRegistry,
    VersionRegistryError,
    VersionRegistrySnapshot,
    VersionStatus,
    VersionType,
    VersionValidationError,
)

# ============================================================
# Run manager
# ============================================================

from .run_manager import (
    ManagedRun,
    ManagedRunStatus,
    RunAlreadyExistsError,
    RunIdentity,
    RunManager,
    RunManagerError,
    RunManagerSnapshot,
    RunNotFoundError,
    RunStateTransitionError,
    RunTransition,
    RunValidationError,
    TERMINAL_STATUSES,
)

# ============================================================
# Cache manager
# ============================================================

from .cache_manager import (
    CacheEntry,
    CacheEntryNotFoundError,
    CacheKey,
    CacheLookupResult,
    CacheManager,
    CacheManagerError,
    CacheManagerSnapshot,
    CacheStatus,
    CacheValidationError,
)

# ============================================================
# Recovery manager
# ============================================================

from .recovery_manager import (
    CheckpointNotFoundError,
    FailureType,
    RecoveryAttempt,
    RecoveryCheckpoint,
    RecoveryManager,
    RecoveryManagerError,
    RecoveryManagerSnapshot,
    RecoveryNotFoundError,
    RecoveryRecord,
    RecoveryStateError,
    RecoveryStatus,
    RecoveryValidationError,
    RetryPolicy,
    TERMINAL_RECOVERY_STATUSES,
)

# ============================================================
# Artifact registry
# ============================================================

from .artifact_registry import (
    ArtifactAlreadyExistsError,
    ArtifactNotFoundError,
    ArtifactRecord,
    ArtifactReference,
    ArtifactRegistry,
    ArtifactRegistryError,
    ArtifactRegistrySnapshot,
    ArtifactStatus,
    ArtifactType,
    ArtifactValidationError,
)

# ============================================================
# Lineage engine
# ============================================================

from .lineage_engine import (
    LineageCycleError,
    LineageEdge,
    LineageEdgeAlreadyExistsError,
    LineageEngine,
    LineageEngineError,
    LineageNode,
    LineageNodeNotFoundError,
    LineageNodeType,
    LineagePath,
    LineageRelation,
    LineageSnapshot,
    LineageValidationError,
)


__version__ = "6.0.1"


__all__ = [
    # Data contracts
    "SCHEMA_VERSION",
    "ContractCompatibility",
    "ContractValidationResult",
    "ContractVersion",
    "ContractVersionError",
    "DataContract",
    "DataContractError",
    "DataContractValidator",
    "DataEnvelope",
    "DataField",
    "DataQualityStatus",
    "DatasetReference",
    "DatasetType",
    "ValidationIssue",

    # Dataset registry
    "DatasetAlreadyExistsError",
    "DatasetContractNotFoundError",
    "DatasetNotFoundError",
    "DatasetRegistry",
    "DatasetRegistryError",
    "DatasetRegistrySnapshot",
    "DatasetValidationError",
    "DatasetVersionRecord",

    # Event bus
    "DeliveryStatus",
    "EventBus",
    "EventBusError",
    "EventDeliveryReport",
    "EventPriority",
    "EventType",
    "EventValidationError",
    "HandlerResult",
    "ResearchEvent",
    "SubscriberError",
    "Subscription",

    # Parallel executor
    "ExecutionStatus",
    "ExecutionTask",
    "ExecutionValidationError",
    "ParallelExecutionReport",
    "ParallelExecutor",
    "ParallelExecutorError",
    "TaskExecutionResult",

    # Execution engine
    "ResearchExecutionEngine",
    "ResearchExecutionError",
    "ResearchExecutionRequest",
    "ResearchExecutionValidationError",
    "ResearchRun",
    "ResearchRunStatus",

    # Version registry
    "SemanticVersion",
    "VersionAlreadyExistsError",
    "VersionNotFoundError",
    "VersionRecord",
    "VersionRegistry",
    "VersionRegistryError",
    "VersionRegistrySnapshot",
    "VersionStatus",
    "VersionType",
    "VersionValidationError",

    # Run manager
    "ManagedRun",
    "ManagedRunStatus",
    "RunAlreadyExistsError",
    "RunIdentity",
    "RunManager",
    "RunManagerError",
    "RunManagerSnapshot",
    "RunNotFoundError",
    "RunStateTransitionError",
    "RunTransition",
    "RunValidationError",
    "TERMINAL_STATUSES",

    # Cache manager
    "CacheEntry",
    "CacheEntryNotFoundError",
    "CacheKey",
    "CacheLookupResult",
    "CacheManager",
    "CacheManagerError",
    "CacheManagerSnapshot",
    "CacheStatus",
    "CacheValidationError",

    # Recovery manager
    "CheckpointNotFoundError",
    "FailureType",
    "RecoveryAttempt",
    "RecoveryCheckpoint",
    "RecoveryManager",
    "RecoveryManagerError",
    "RecoveryManagerSnapshot",
    "RecoveryNotFoundError",
    "RecoveryRecord",
    "RecoveryStateError",
    "RecoveryStatus",
    "RecoveryValidationError",
    "RetryPolicy",
    "TERMINAL_RECOVERY_STATUSES",

    # Artifact registry
    "ArtifactAlreadyExistsError",
    "ArtifactNotFoundError",
    "ArtifactRecord",
    "ArtifactReference",
    "ArtifactRegistry",
    "ArtifactRegistryError",
    "ArtifactRegistrySnapshot",
    "ArtifactStatus",
    "ArtifactType",
    "ArtifactValidationError",

    # Lineage engine
    "LineageCycleError",
    "LineageEdge",
    "LineageEdgeAlreadyExistsError",
    "LineageEngine",
    "LineageEngineError",
    "LineageNode",
    "LineageNodeNotFoundError",
    "LineageNodeType",
    "LineagePath",
    "LineageRelation",
    "LineageSnapshot",
    "LineageValidationError",
]