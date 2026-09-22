"""
v7.0.1 Data Fabric
v7.0.2 Knowledge Graph Bridge
v7.0.3 Research Relationships
v7.0.4 Research Event Lineage
v7.0.5 Autonomous Experiment Engine
v7.0.6 Statistical Validation
v7.0.7 Causal Research and Alpha Discovery
v7.0.8 Portfolio Intelligence and Risk Governance
v7.0.9 Shadow Fund and Digital Twin

Research-only architecture.
No real-money execution is supported.
"""

from .facade import (
    DataFabric,
    DataFabricError,
)

from .knowledge_graph_bridge import (
    KnowledgeGraphBridge,
    KnowledgeGraphBridgeError,
)

from .relationship_manager import (
    RelationshipApprovalRequiredError,
    RelationshipManagerError,
    ResearchRelationshipManager,
)

from .lineage_service import (
    LineageServiceError,
    ResearchLineageService,
)

from .experiment_engine import (
    AutonomousExperimentEngine,
    ExperimentApprovalRequiredError,
    ExperimentEngineError,
    ExperimentRecord,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
)

from .statistical_validation import (
    StatisticalValidationError,
    StatisticalValidationFacade,
)

from .causal_alpha_discovery import (
    AlphaCandidate,
    CandidateStatus,
    CausalAlphaDiscovery,
    CausalAlphaDiscoveryError,
    CausalHypothesis,
    EvidenceAssessment,
    EvidenceClass,
    HypothesisStatus,
    MechanismMapping,
)

from .portfolio_risk_governance import (
    PortfolioRecommendation,
    PortfolioResearchSignal,
    PortfolioRiskGovernance,
    PortfolioRiskGovernanceError,
    RecommendationStatus,
    RiskAssessment,
    RiskLimits,
    RiskSeverity,
    SignalAction,
)

from .shadow_fund_digital_twin import (
    ScenarioResult,
    ScenarioSeverity,
    ShadowFundDigitalTwin,
    ShadowFundDigitalTwinError,
    ShadowPosition,
    StressScenario,
    TwinSnapshot,
    VirtualFill,
    VirtualOrderSide,
)


__version__ = "v7.0.9"


__all__ = [
    "DataFabric",
    "DataFabricError",
    "KnowledgeGraphBridge",
    "KnowledgeGraphBridgeError",
    "RelationshipApprovalRequiredError",
    "RelationshipManagerError",
    "ResearchRelationshipManager",
    "LineageServiceError",
    "ResearchLineageService",
    "AutonomousExperimentEngine",
    "ExperimentApprovalRequiredError",
    "ExperimentEngineError",
    "ExperimentRecord",
    "ExperimentResult",
    "ExperimentSpec",
    "ExperimentStatus",
    "StatisticalValidationError",
    "StatisticalValidationFacade",
    "AlphaCandidate",
    "CandidateStatus",
    "CausalAlphaDiscovery",
    "CausalAlphaDiscoveryError",
    "CausalHypothesis",
    "EvidenceAssessment",
    "EvidenceClass",
    "HypothesisStatus",
    "MechanismMapping",
    "PortfolioRecommendation",
    "PortfolioResearchSignal",
    "PortfolioRiskGovernance",
    "PortfolioRiskGovernanceError",
    "RecommendationStatus",
    "RiskAssessment",
    "RiskLimits",
    "RiskSeverity",
    "SignalAction",
    "ScenarioResult",
    "ScenarioSeverity",
    "ShadowFundDigitalTwin",
    "ShadowFundDigitalTwinError",
    "ShadowPosition",
    "StressScenario",
    "TwinSnapshot",
    "VirtualFill",
    "VirtualOrderSide",
    "__version__",
]