from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from research_control_center.alpha_library import AlphaLibrary
from research_control_center.backtest_diagnostics import BacktestDiagnosticsEngine
from research_control_center.decision_lineage import DecisionLineageEngine
from research_control_center.experiment_registry import ExperimentRegistry
from research_control_center.leaderboard_engine import LeaderboardEngine
from research_control_center.quantum_experiment_tracker import QuantumExperimentTracker
from research_control_center.shadow_fund_dashboard import ShadowFundDashboard
from research_control_center.models import (
    ResearchControlCenterSnapshot,
    utc_now,
)


class ResearchControlCenter:
    """
    AI Multi-Agent Financial Research Platform
    Research Control Center v5.0.10

    Unified research observability and governance layer.

    Components:
    1. Experiment Registry
    2. Agent / Model Leaderboard
    3. Alpha Library
    4. Backtest / OOS Diagnostics
    5. Shadow Fund Dashboard
    6. Quantum-vs-Classical Experiment Tracking
    7. Decision Lineage

    The Control Center is intentionally research-only.

    It does NOT:
    - place broker orders
    - authorize real-money execution
    - bypass validation
    - bypass risk governance
    """

    VERSION = "v5.0.10"

    def __init__(
        self,
        *,
        experiment_registry: Optional[
            ExperimentRegistry
        ] = None,
        leaderboard: Optional[
            LeaderboardEngine
        ] = None,
        alpha_library: Optional[
            AlphaLibrary
        ] = None,
        backtest_diagnostics: Optional[
            BacktestDiagnosticsEngine
        ] = None,
        shadow_fund: Optional[
            ShadowFundDashboard
        ] = None,
        quantum_tracker: Optional[
            QuantumExperimentTracker
        ] = None,
        decision_lineage: Optional[
            DecisionLineageEngine
        ] = None,
    ) -> None:
        self.experiment_registry = (
            experiment_registry
            or ExperimentRegistry()
        )

        self.leaderboard = (
            leaderboard
            or LeaderboardEngine()
        )

        self.alpha_library = (
            alpha_library
            or AlphaLibrary()
        )

        self.backtest_diagnostics = (
            backtest_diagnostics
            or BacktestDiagnosticsEngine()
        )

        self.shadow_fund = (
            shadow_fund
            or ShadowFundDashboard()
        )

        self.quantum_tracker = (
            quantum_tracker
            or QuantumExperimentTracker()
        )

        self.decision_lineage = (
            decision_lineage
            or DecisionLineageEngine()
        )

        self._audit_log = []

        self._audit(
            "CONTROL_CENTER_INITIALIZED",
            {
                "version": self.VERSION,
            },
        )

    # =========================================================
    # AUDIT
    # =========================================================

    def _audit(
        self,
        event: str,
        details: Optional[dict] = None,
    ) -> None:
        self._audit_log.append(
            {
                "timestamp": utc_now(),
                "event": event,
                "details": deepcopy(
                    details or {}
                ),
                "version": self.VERSION,
                "research_only": True,
                "real_money_execution": False,
            }
        )

    def audit_log(self) -> list:
        return deepcopy(
            self._audit_log
        )

    # =========================================================
    # COMPONENT ACCESS
    # =========================================================

    def components(self) -> Dict[str, Any]:
        """
        Return the live component instances.

        This is intentionally different from snapshot().
        Components are operational research services.
        """

        return {
            "experiment_registry": (
                self.experiment_registry
            ),
            "leaderboard": (
                self.leaderboard
            ),
            "alpha_library": (
                self.alpha_library
            ),
            "backtest_diagnostics": (
                self.backtest_diagnostics
            ),
            "shadow_fund": (
                self.shadow_fund
            ),
            "quantum_tracker": (
                self.quantum_tracker
            ),
            "decision_lineage": (
                self.decision_lineage
            ),
        }

    def component_names(self) -> tuple:
        return tuple(
            self.components().keys()
        )

    # =========================================================
    # HEALTH
    # =========================================================

    def health(self) -> dict:
        components = self.components()

        component_status = {
            name: (
                component is not None
            )
            for name, component
            in components.items()
        }

        healthy_components = sum(
            1
            for healthy
            in component_status.values()
            if healthy
        )

        total_components = len(
            component_status
        )

        healthy = (
            healthy_components
            == total_components
        )

        return {
            "version": self.VERSION,
            "healthy": healthy,
            "healthy_components": (
                healthy_components
            ),
            "total_components": (
                total_components
            ),
            "components": (
                component_status
            ),
            "research_only": True,
            "real_money_execution": False,
        }

    # =========================================================
    # EXPERIMENT REGISTRY SNAPSHOT
    # =========================================================

    def _experiment_snapshot(self):
        registry = (
            self.experiment_registry
        )

        if hasattr(
            registry,
            "list_all",
        ):
            return deepcopy(
                registry.list_all()
            )

        if hasattr(
            registry,
            "snapshot",
        ):
            result = registry.snapshot()

            if isinstance(result, dict):
                if "experiments" in result:
                    return deepcopy(
                        result["experiments"]
                    )

            return deepcopy(result)

        return []

    # =========================================================
    # LEADERBOARD SNAPSHOT
    # =========================================================

    def _leaderboard_snapshot(self):
        leaderboard = self.leaderboard

        if hasattr(
            leaderboard,
            "rank",
        ):
            try:
                return deepcopy(
                    leaderboard.rank()
                )
            except TypeError:
                pass

        if hasattr(
            leaderboard,
            "list_all",
        ):
            return deepcopy(
                leaderboard.list_all()
            )

        if hasattr(
            leaderboard,
            "snapshot",
        ):
            result = leaderboard.snapshot()

            if isinstance(result, dict):
                for key in (
                    "leaderboard",
                    "entries",
                    "ranking",
                ):
                    if key in result:
                        return deepcopy(
                            result[key]
                        )

            return deepcopy(result)

        return []

    # =========================================================
    # ALPHA LIBRARY SNAPSHOT
    # =========================================================

    def _alpha_snapshot(self):
        library = self.alpha_library

        if hasattr(
            library,
            "list_all",
        ):
            return deepcopy(
                library.list_all()
            )

        if hasattr(
            library,
            "snapshot",
        ):
            result = library.snapshot()

            if isinstance(result, dict):
                for key in (
                    "alpha_library",
                    "alphas",
                    "records",
                ):
                    if key in result:
                        return deepcopy(
                            result[key]
                        )

            return deepcopy(result)

        return []

    # =========================================================
    # BACKTEST SNAPSHOT
    # =========================================================

    def _backtest_snapshot(self):
        diagnostics = (
            self.backtest_diagnostics
        )

        if hasattr(
            diagnostics,
            "list_all",
        ):
            return deepcopy(
                diagnostics.list_all()
            )

        if hasattr(
            diagnostics,
            "snapshot",
        ):
            result = diagnostics.snapshot()

            if isinstance(result, dict):
                for key in (
                    "backtest_diagnostics",
                    "diagnostics",
                    "records",
                ):
                    if key in result:
                        return deepcopy(
                            result[key]
                        )

            return deepcopy(result)

        return []

    # =========================================================
    # SHADOW FUND SNAPSHOT
    # =========================================================

    def _shadow_snapshot(self):
        dashboard = self.shadow_fund

        if hasattr(
            dashboard,
            "snapshot",
        ):
            return deepcopy(
                dashboard.snapshot()
            )

        return None

    # =========================================================
    # QUANTUM SNAPSHOT
    # =========================================================

    def _quantum_snapshot(self):
        tracker = self.quantum_tracker

        if hasattr(
            tracker,
            "list_all",
        ):
            return deepcopy(
                tracker.list_all()
            )

        if hasattr(
            tracker,
            "snapshot",
        ):
            result = tracker.snapshot()

            if isinstance(result, dict):
                if "experiments" in result:
                    return deepcopy(
                        result["experiments"]
                    )

            return deepcopy(result)

        return []

    # =========================================================
    # LINEAGE SNAPSHOT
    # =========================================================

    def _lineage_snapshot(self):
        engine = self.decision_lineage

        if hasattr(
            engine,
            "list_all",
        ):
            return deepcopy(
                engine.list_all()
            )

        if hasattr(
            engine,
            "snapshot",
        ):
            result = engine.snapshot()

            if isinstance(result, dict):
                if "lineages" in result:
                    return deepcopy(
                        result["lineages"]
                    )

            return deepcopy(result)

        return []

    # =========================================================
    # MODEL SNAPSHOT
    # =========================================================

    def model_snapshot(
        self,
    ) -> ResearchControlCenterSnapshot:
        """
        Build the canonical v5.0.10 model snapshot.
        """

        experiments = (
            self._experiment_snapshot()
        )

        leaderboard = (
            self._leaderboard_snapshot()
        )

        alpha_library = (
            self._alpha_snapshot()
        )

        backtest_diagnostics = (
            self._backtest_snapshot()
        )

        shadow_fund = (
            self._shadow_snapshot()
        )

        quantum_experiments = (
            self._quantum_snapshot()
        )

        decision_lineages = (
            self._lineage_snapshot()
        )

        snapshot = (
            ResearchControlCenterSnapshot(
                experiments=experiments,
                leaderboard=leaderboard,
                alpha_library=alpha_library,
                backtest_diagnostics=(
                    backtest_diagnostics
                ),
                shadow_fund=shadow_fund,
                quantum_classical_experiments=(
                    quantum_experiments
                ),
                decision_lineages=(
                    decision_lineages
                ),
                metadata={
                    "version": (
                        self.VERSION
                    ),
                    "research_only": True,
                    "real_money_execution": False,
                    "auditable": True,
                },
            )
        )

        return snapshot

    # =========================================================
    # COUNTS
    # =========================================================

    def counts(self) -> dict:
        model = self.model_snapshot()

        return {
            "experiments": len(
                model.experiments
            ),
            "leaderboard_entries": len(
                model.leaderboard
            ),
            "alpha_records": len(
                model.alpha_library
            ),
            "backtest_diagnostics": len(
                model.backtest_diagnostics
            ),
            "quantum_experiments": len(
                model.quantum_classical_experiments
            ),
            "decision_lineages": len(
                model.decision_lineages
            ),
            "shadow_fund_available": (
                model.shadow_fund
                is not None
            ),
        }

    # =========================================================
    # RESEARCH SUMMARY
    # =========================================================

    def summary(self) -> dict:
        counts = self.counts()
        health = self.health()

        return {
            "version": self.VERSION,
            "status": (
                "HEALTHY"
                if health["healthy"]
                else "DEGRADED"
            ),
            "components": (
                health["components"]
            ),
            "counts": counts,
            "research_only": True,
            "real_money_execution": False,
            "auditable": True,
            "generated_at": utc_now(),
        }

    # =========================================================
    # FULL DASHBOARD
    # =========================================================

    def dashboard(self) -> dict:
        """
        Build the complete Research Control Center view.
        """

        model = self.model_snapshot()

        shadow_dashboard = None

        if hasattr(
            self.shadow_fund,
            "dashboard",
        ):
            shadow_dashboard = deepcopy(
                self.shadow_fund.dashboard()
            )
        elif model.shadow_fund is not None:
            shadow_dashboard = deepcopy(
                model.shadow_fund
            )

        leaderboard_summary = None

        if hasattr(
            self.leaderboard,
            "summary",
        ):
            leaderboard_summary = deepcopy(
                self.leaderboard.summary()
            )

        alpha_summary = None

        if hasattr(
            self.alpha_library,
            "summary",
        ):
            alpha_summary = deepcopy(
                self.alpha_library.summary()
            )

        backtest_summary = None

        if hasattr(
            self.backtest_diagnostics,
            "summary",
        ):
            backtest_summary = deepcopy(
                self.backtest_diagnostics.summary()
            )

        quantum_summary = None

        if hasattr(
            self.quantum_tracker,
            "summary",
        ):
            quantum_summary = deepcopy(
                self.quantum_tracker.summary()
            )

        lineage_summary = None

        if hasattr(
            self.decision_lineage,
            "summary",
        ):
            lineage_summary = deepcopy(
                self.decision_lineage.summary()
            )

        registry_summary = None

        if hasattr(
            self.experiment_registry,
            "summary",
        ):
            registry_summary = deepcopy(
                self.experiment_registry.summary()
            )

        result = {
            "version": self.VERSION,
            "health": self.health(),
            "summary": self.summary(),
            "experiments": deepcopy(
                model.experiments
            ),
            "experiment_summary": (
                registry_summary
            ),
            "leaderboard": deepcopy(
                model.leaderboard
            ),
            "leaderboard_summary": (
                leaderboard_summary
            ),
            "alpha_library": deepcopy(
                model.alpha_library
            ),
            "alpha_summary": (
                alpha_summary
            ),
            "backtest_diagnostics": deepcopy(
                model.backtest_diagnostics
            ),
            "backtest_summary": (
                backtest_summary
            ),
            "shadow_fund": (
                shadow_dashboard
            ),
            "quantum_classical_experiments": deepcopy(
                model.quantum_classical_experiments
            ),
            "quantum_summary": (
                quantum_summary
            ),
            "decision_lineages": deepcopy(
                model.decision_lineages
            ),
            "decision_lineage_summary": (
                lineage_summary
            ),
            "generated_at": utc_now(),
            "auditable": True,
            "research_only": True,
            "real_money_execution": False,
        }

        self._audit(
            "CONTROL_CENTER_DASHBOARD_GENERATED",
            {
                "counts": self.counts(),
            },
        )

        return result

    # =========================================================
    # SAFETY REPORT
    # =========================================================

    def safety_report(self) -> dict:
        """
        Explicitly state execution boundaries.

        v5.0.10 is an observability, research,
        diagnostics and governance layer.
        """

        return {
            "version": self.VERSION,
            "research_only": True,
            "shadow_execution_only": True,
            "real_money_execution": False,
            "broker_order_allowed": False,
            "live_trade_authorization": False,
            "risk_bypass_allowed": False,
            "validation_bypass_allowed": False,
            "control_center_can_execute_orders": False,
        }

    # =========================================================
    # SYSTEM REPORT
    # =========================================================

    def system_report(self) -> dict:
        """
        Compact machine-readable state report.
        """

        return {
            "version": self.VERSION,
            "health": self.health(),
            "counts": self.counts(),
            "safety": self.safety_report(),
            "audit_events": len(
                self._audit_log
            ),
            "generated_at": utc_now(),
        }

    # =========================================================
    # SNAPSHOT
    # =========================================================

    def snapshot(self) -> dict:
        """
        Immutable-style unified snapshot for
        reporting, persistence and E2E tests.
        """

        model = self.model_snapshot()

        return {
            "model": deepcopy(model),
            "health": self.health(),
            "counts": self.counts(),
            "safety": self.safety_report(),
            "audit_log": self.audit_log(),
            "version": self.VERSION,
            "generated_at": utc_now(),
            "auditable": True,
            "research_only": True,
            "real_money_execution": False,
        }