from __future__ import annotations

from production_research_infrastructure import (
    ArtifactRegistry,
    ArtifactStatus,
    ArtifactType,
    CacheManager,
    EventBus,
    EventType,
    ExecutionStatus,
    ExecutionTask,
    FailureType,
    LineageEngine,
    LineageNodeType,
    LineageRelation,
    ManagedRunStatus,
    ParallelExecutor,
    RecoveryManager,
    RecoveryStatus,
    ResearchExecutionEngine,
    ResearchExecutionRequest,
    RetryPolicy,
    RunIdentity,
    RunManager,
    VersionRegistry,
    VersionStatus,
)


# ============================================================
# Helpers
# ============================================================


def build_infrastructure():
    return {
        "events": EventBus(),
        "parallel": ParallelExecutor(max_concurrency=4),
        "versions": VersionRegistry(),
        "runs": RunManager(),
        "cache": CacheManager(),
        "recovery": RecoveryManager(),
        "artifacts": ArtifactRegistry(),
        "lineage": LineageEngine(),
    }


def create_versions(versions: VersionRegistry):
    model = versions.register_model(
        name="crypto_agent",
        version="1.0.0",
        content={
            "model": "research-model",
            "temperature": 0.1,
        },
        status=VersionStatus.ACTIVE,
    )

    prompt = versions.register_prompt(
        name="crypto_research",
        version="1.0.0",
        content={
            "template": "Analyze market regime."
        },
        status=VersionStatus.ACTIVE,
    )

    policy = versions.register_research_policy(
        name="risk_policy",
        version="1.0.0",
        content={
            "max_drawdown": 0.15,
            "research_only": True,
        },
        status=VersionStatus.ACTIVE,
    )

    return model, prompt, policy


def configuration_fingerprint(
    versions: VersionRegistry,
    model,
    prompt,
    policy,
) -> str:
    return versions.configuration_fingerprint(
        model_version_ids=[
            model.version_id
        ],
        prompt_version_ids=[
            prompt.version_id
        ],
        policy_version_ids=[
            policy.version_id
        ],
    )


def create_managed_run(
    runs: RunManager,
    *,
    configuration_fingerprint_value: str,
):
    identity = RunIdentity(
        research_name="btc_regime_research",
        configuration_fingerprint=(
            configuration_fingerprint_value
        ),
        dataset_fingerprints=[
            "btc_dataset_fingerprint"
        ],
        parameters={
            "code_version": "v6.0.1",
            "asset": "BTC",
            "horizon": "7D",
        },
        metadata={
            "research_only": True,
            "real_money_execution": False,
        },
    )

    return runs.create_run(identity)


def completed_task_count(report) -> int:
    return sum(
        1
        for result in report.task_results
        if result.status == ExecutionStatus.COMPLETED
    )


# ============================================================
# Infrastructure construction
# ============================================================


def test_all_infrastructure_components_construct():
    infra = build_infrastructure()

    assert infra["events"] is not None
    assert infra["parallel"] is not None
    assert infra["versions"] is not None
    assert infra["runs"] is not None
    assert infra["cache"] is not None
    assert infra["recovery"] is not None
    assert infra["artifacts"] is not None
    assert infra["lineage"] is not None


def test_core_components_report_safe_mode():
    infra = build_infrastructure()

    assert (
        infra["versions"].summary()[
            "real_money_execution"
        ]
        is False
    )

    assert (
        infra["runs"].summary()[
            "real_money_execution"
        ]
        is False
    )

    assert (
        infra["cache"].summary()[
            "real_money_execution"
        ]
        is False
    )

    assert (
        infra["recovery"].summary()[
            "real_money_execution"
        ]
        is False
    )

    assert (
        infra["artifacts"].summary()[
            "real_money_execution"
        ]
        is False
    )

    assert (
        infra["lineage"].summary()[
            "real_money_execution"
        ]
        is False
    )


# ============================================================
# Version Registry
# ============================================================


def test_version_configuration_fingerprint():
    versions = VersionRegistry()

    model, prompt, policy = create_versions(
        versions
    )

    fingerprint = configuration_fingerprint(
        versions,
        model,
        prompt,
        policy,
    )

    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 64


def test_same_configuration_same_run_identity():
    versions = VersionRegistry()

    model, prompt, policy = create_versions(
        versions
    )

    config = configuration_fingerprint(
        versions,
        model,
        prompt,
        policy,
    )

    first = RunIdentity(
        research_name="btc_research",
        configuration_fingerprint=config,
        dataset_fingerprints=[
            "dataset_fp"
        ],
        parameters={
            "code_version": "v6.0.1"
        },
    )

    second = RunIdentity(
        research_name="btc_research",
        configuration_fingerprint=config,
        dataset_fingerprints=[
            "dataset_fp"
        ],
        parameters={
            "code_version": "v6.0.1"
        },
    )

    assert (
        first.reproducible_run_id
        == second.reproducible_run_id
    )

    assert (
        first.fingerprint
        == second.fingerprint
    )


def test_configuration_change_changes_run_identity():
    versions = VersionRegistry()

    model, prompt, policy = create_versions(
        versions
    )

    first_config = configuration_fingerprint(
        versions,
        model,
        prompt,
        policy,
    )

    model_v2 = versions.register_model(
        name="crypto_agent",
        version="1.1.0",
        content={
            "model": "research-model-v2"
        },
        status=VersionStatus.ACTIVE,
    )

    second_config = (
        versions.configuration_fingerprint(
            model_version_ids=[
                model_v2.version_id
            ],
            prompt_version_ids=[
                prompt.version_id
            ],
            policy_version_ids=[
                policy.version_id
            ],
        )
    )

    first_identity = RunIdentity(
        research_name="btc_research",
        configuration_fingerprint=(
            first_config
        ),
        dataset_fingerprints=[
            "dataset_fp"
        ],
        parameters={
            "code_version": "v6.0.1"
        },
    )

    second_identity = RunIdentity(
        research_name="btc_research",
        configuration_fingerprint=(
            second_config
        ),
        dataset_fingerprints=[
            "dataset_fp"
        ],
        parameters={
            "code_version": "v6.0.1"
        },
    )

    assert first_config != second_config

    assert (
        first_identity.reproducible_run_id
        != second_identity.reproducible_run_id
    )


# ============================================================
# Managed Run Lifecycle
# ============================================================


def test_managed_run_lifecycle():
    versions = VersionRegistry()
    runs = RunManager()

    model, prompt, policy = create_versions(
        versions
    )

    config = configuration_fingerprint(
        versions,
        model,
        prompt,
        policy,
    )

    run = create_managed_run(
        runs,
        configuration_fingerprint_value=config,
    )

    assert (
        run.status
        == ManagedRunStatus.CREATED
    )

    runs.start(run.run_id)

    assert (
        run.status
        == ManagedRunStatus.RUNNING
    )

    runs.complete(run.run_id)

    assert (
        run.status
        == ManagedRunStatus.COMPLETED
    )

    stored_run = runs.get(
        run.run_id
    )

    assert (
        stored_run.status
        == ManagedRunStatus.COMPLETED
    )


# ============================================================
# Cache Integration
# ============================================================


def test_cache_miss_then_hit():
    cache = CacheManager()

    key = cache.create_key(
        namespace="research",
        payload={
            "asset": "BTC",
            "horizon": "7D",
        },
        version="v6.0.1",
    )

    first = cache.lookup(key)

    assert first.hit is False

    cache.set(
        key,
        {
            "decision": "HOLD",
            "score": 0.61,
        },
    )

    second = cache.lookup(key)

    assert second.hit is True

    assert (
        second.value["decision"]
        == "HOLD"
    )


def test_cache_key_is_reproducible():
    cache = CacheManager()

    first = cache.create_key(
        namespace="research",
        payload={
            "b": 2,
            "a": 1,
        },
        version="v6.0.1",
    )

    second = cache.create_key(
        namespace="research",
        payload={
            "a": 1,
            "b": 2,
        },
        version="v6.0.1",
    )

    assert (
        first.cache_id
        == second.cache_id
    )


# ============================================================
# Event-driven Parallel Execution
# ============================================================


def test_execution_engine_runs_parallel_tasks():
    event_bus = EventBus()

    parallel = ParallelExecutor(
        max_concurrency=4
    )

    engine = ResearchExecutionEngine(
        event_bus=event_bus,
        parallel_executor=parallel,
    )

    request = ResearchExecutionRequest(
        name="multi_agent_research",
        tasks=[
            ExecutionTask(
                name="macro",
                callable=lambda: {
                    "signal": "NEUTRAL"
                },
                agent_name="macro_agent",
            ),
            ExecutionTask(
                name="crypto",
                callable=lambda: {
                    "signal": "BULLISH"
                },
                agent_name="crypto_agent",
            ),
            ExecutionTask(
                name="risk",
                callable=lambda: {
                    "signal": "NEUTRAL"
                },
                agent_name="risk_agent",
            ),
        ],
        parallel=True,
        correlation_id="managed_run_001",
    )

    run = engine.execute(request)

    assert run.successful is True

    assert (
        run.execution_report
        is not None
    )

    assert (
        completed_task_count(
            run.execution_report
        )
        == 3
    )


def test_execution_engine_task_results_preserve_order():
    engine = ResearchExecutionEngine(
        event_bus=EventBus(),
        parallel_executor=ParallelExecutor(
            max_concurrency=3
        ),
    )

    request = ResearchExecutionRequest(
        name="ordered_execution",
        tasks=[
            ExecutionTask(
                name="first",
                callable=lambda: 1,
            ),
            ExecutionTask(
                name="second",
                callable=lambda: 2,
            ),
            ExecutionTask(
                name="third",
                callable=lambda: 3,
            ),
        ],
        parallel=True,
    )

    run = engine.execute(request)

    assert (
        run.execution_report
        is not None
    )

    results = (
        run.execution_report.task_results
    )

    assert [
        result.result
        for result in results
    ] == [1, 2, 3]


def test_execution_engine_emits_events():
    event_bus = EventBus()

    engine = ResearchExecutionEngine(
        event_bus=event_bus,
        parallel_executor=ParallelExecutor(
            max_concurrency=2
        ),
    )

    request = ResearchExecutionRequest(
        name="event_test",
        tasks=[
            ExecutionTask(
                name="task",
                callable=lambda: 42,
            )
        ],
        correlation_id="corr_001",
    )

    engine.execute(request)

    events = event_bus.event_history()

    event_types = {
        event.event_type
        for event in events
    }

    assert (
        EventType.RESEARCH_REQUESTED
        in event_types
    )

    assert (
        EventType.RESEARCH_STARTED
        in event_types
    )

    assert (
        EventType.RESEARCH_COMPLETED
        in event_types
    )


def test_execution_events_keep_correlation_id():
    event_bus = EventBus()

    engine = ResearchExecutionEngine(
        event_bus=event_bus,
        parallel_executor=ParallelExecutor(
            max_concurrency=2
        ),
    )

    correlation_id = (
        "managed_research_run_001"
    )

    request = ResearchExecutionRequest(
        name="correlation_test",
        correlation_id=correlation_id,
        tasks=[
            ExecutionTask(
                name="task",
                callable=lambda: "OK",
            )
        ],
    )

    engine.execute(request)

    events = (
        event_bus.events_for_correlation(
            correlation_id
        )
    )

    assert len(events) >= 3

    assert all(
        event.correlation_id
        == correlation_id
        for event in events
    )


# ============================================================
# Recovery Integration
# ============================================================


def test_failure_checkpoint_recovery_flow():
    recovery = RecoveryManager()

    checkpoint = (
        recovery.create_checkpoint(
            run_id="run_failure_001",
            stage="agent_execution",
            state={
                "completed_agents": [
                    "macro",
                    "crypto",
                ],
                "pending_agents": [
                    "risk"
                ],
            },
            metadata={
                "research_only": True
            },
        )
    )

    policy = RetryPolicy(
        max_attempts=3,
        base_delay_seconds=0.0,
        backoff_multiplier=1.0,
        max_delay_seconds=0.0,
        retry_on=[
            FailureType.AGENT_ERROR
        ],
    )

    record = recovery.create_recovery(
        run_id="run_failure_001",
        failure_type=(
            FailureType.AGENT_ERROR
        ),
        error="risk agent failed",
        retry_policy=policy,
        checkpoint_id=(
            checkpoint.checkpoint_id
        ),
    )

    assert (
        record.status
        == RecoveryStatus.READY
    )

    recovery.begin_attempt(
        record.recovery_id
    )

    recovery.complete_attempt(
        record.recovery_id
    )

    assert (
        record.status
        == RecoveryStatus.RECOVERED
    )

    restored = (
        recovery.restore_checkpoint(
            checkpoint.checkpoint_id
        )
    )

    assert (
        restored["completed_agents"]
        == ["macro", "crypto"]
    )

    assert (
        restored["pending_agents"]
        == ["risk"]
    )


# ============================================================
# Artifact Integration
# ============================================================


def test_research_result_becomes_artifact():
    artifacts = ArtifactRegistry()

    content = {
        "decision": "HOLD",
        "confidence": 0.71,
    }

    artifact = artifacts.create(
        name="btc_research_result",
        artifact_type=(
            ArtifactType.RESEARCH_REPORT
        ),
        content=content,
        run_id="run_001",
        experiment_id="experiment_001",
        dataset_ids=[
            "dataset_btc_v1"
        ],
        version_ids=[
            "model_v1",
            "prompt_v1",
            "policy_v1",
        ],
    )

    artifacts.verify(
        artifact.artifact_id
    )

    assert (
        artifact.status
        == ArtifactStatus.VERIFIED
    )

    assert artifacts.verify_content(
        artifact.artifact_id,
        {
            "confidence": 0.71,
            "decision": "HOLD",
        },
    )


def test_artifact_content_fingerprint_is_stable():
    artifacts = ArtifactRegistry()

    artifact = artifacts.create(
        name="stable_artifact",
        artifact_type=(
            ArtifactType.METRIC
        ),
        content={
            "b": 2,
            "a": 1,
        },
        run_id="run_stable",
    )

    assert artifacts.verify_content(
        artifact.artifact_id,
        {
            "a": 1,
            "b": 2,
        },
    )


# ============================================================
# Lineage
# ============================================================


def test_full_lineage_from_dataset_to_artifact():
    lineage = LineageEngine()

    lineage.create_node(
        node_id="dataset_btc_v1",
        node_type=LineageNodeType.DATASET,
    )

    lineage.create_node(
        node_id="model_v1",
        node_type=(
            LineageNodeType.MODEL_VERSION
        ),
    )

    lineage.create_node(
        node_id="prompt_v1",
        node_type=(
            LineageNodeType.PROMPT_VERSION
        ),
    )

    lineage.create_node(
        node_id="policy_v1",
        node_type=(
            LineageNodeType.POLICY_VERSION
        ),
    )

    lineage.create_node(
        node_id="run_001",
        node_type=LineageNodeType.RUN,
    )

    lineage.create_node(
        node_id="experiment_001",
        node_type=(
            LineageNodeType.EXPERIMENT
        ),
    )

    lineage.create_node(
        node_id="artifact_001",
        node_type=(
            LineageNodeType.ARTIFACT
        ),
    )

    for source in (
        "dataset_btc_v1",
        "model_v1",
        "prompt_v1",
        "policy_v1",
    ):
        lineage.connect(
            source,
            "run_001",
            LineageRelation.USES,
        )

    lineage.connect(
        "run_001",
        "experiment_001",
        LineageRelation.PRODUCES,
    )

    lineage.connect(
        "experiment_001",
        "artifact_001",
        LineageRelation.PRODUCES,
    )

    ancestors = {
        node.node_id
        for node in lineage.ancestors(
            "artifact_001"
        )
    }

    assert ancestors == {
        "dataset_btc_v1",
        "model_v1",
        "prompt_v1",
        "policy_v1",
        "run_001",
        "experiment_001",
    }


def test_lineage_path_is_reproducible():
    lineage = LineageEngine()

    lineage.create_node(
        node_id="dataset",
        node_type=LineageNodeType.DATASET,
    )

    lineage.create_node(
        node_id="run",
        node_type=LineageNodeType.RUN,
    )

    lineage.create_node(
        node_id="artifact",
        node_type=LineageNodeType.ARTIFACT,
    )

    lineage.connect(
        "dataset",
        "run",
        LineageRelation.USES,
    )

    lineage.connect(
        "run",
        "artifact",
        LineageRelation.PRODUCES,
    )

    first = lineage.find_path(
        "dataset",
        "artifact",
    )

    second = lineage.find_path(
        "dataset",
        "artifact",
    )

    assert first is not None
    assert second is not None

    assert (
        first.node_ids
        == second.node_ids
    )

    assert (
        first.edge_ids
        == second.edge_ids
    )


# ============================================================
# COMPLETE v6.0.1 PRODUCTION RESEARCH PIPELINE
# ============================================================


def test_complete_v601_production_research_pipeline():
    versions = VersionRegistry()
    runs = RunManager()
    cache = CacheManager()
    events = EventBus()
    recovery = RecoveryManager()
    artifacts = ArtifactRegistry()
    lineage = LineageEngine()

    # --------------------------------------------------------
    # 1. Versioned research configuration
    # --------------------------------------------------------

    model, prompt, policy = create_versions(
        versions
    )

    config_fingerprint = (
        configuration_fingerprint(
            versions,
            model,
            prompt,
            policy,
        )
    )

    assert (
        len(config_fingerprint)
        == 64
    )

    # --------------------------------------------------------
    # 2. Reproducible managed run
    # --------------------------------------------------------

    identity = RunIdentity(
        research_name=(
            "btc_multi_agent_research"
        ),
        configuration_fingerprint=(
            config_fingerprint
        ),
        dataset_fingerprints=[
            "btc_prices_v3_fp",
            "macro_v2_fp",
        ],
        parameters={
            "code_version": "v6.0.1",
            "asset": "BTC",
            "regime": "MIXED",
            "horizon": "7D",
        },
        seed=42,
        namespace=(
            "production_research"
        ),
        metadata={
            "research_only": True,
            "real_money_execution": False,
        },
    )

    managed_run = runs.create_run(
        identity
    )

    managed_run_id = (
        managed_run.run_id
    )

    runs.start(
        managed_run_id
    )

    assert (
        managed_run.status
        == ManagedRunStatus.RUNNING
    )

    # --------------------------------------------------------
    # 3. Deterministic cache lookup
    # --------------------------------------------------------

    cache_key = cache.create_key(
        namespace="research_execution",
        payload={
            "run_id": managed_run_id,
            "configuration": (
                config_fingerprint
            ),
            "datasets": [
                "btc_prices_v3_fp",
                "macro_v2_fp",
            ],
        },
        version="v6.0.1",
    )

    assert (
        cache.lookup(
            cache_key
        ).hit
        is False
    )

    # --------------------------------------------------------
    # 4. Event-driven parallel execution
    # --------------------------------------------------------

    execution_engine = (
        ResearchExecutionEngine(
            event_bus=events,
            parallel_executor=(
                ParallelExecutor(
                    max_concurrency=4
                )
            ),
        )
    )

    request = ResearchExecutionRequest(
        name="btc_multi_agent_research",
        correlation_id=managed_run_id,
        parallel=True,
        tasks=[
            ExecutionTask(
                name="macro_analysis",
                agent_name="macro_agent",
                callable=lambda: {
                    "signal": "NEUTRAL",
                    "confidence": 0.64,
                },
            ),
            ExecutionTask(
                name="crypto_analysis",
                agent_name="crypto_agent",
                callable=lambda: {
                    "signal": "BULLISH",
                    "confidence": 0.76,
                },
            ),
            ExecutionTask(
                name="risk_analysis",
                agent_name="risk_agent",
                callable=lambda: {
                    "signal": "NEUTRAL",
                    "confidence": 0.81,
                },
            ),
        ],
        metadata={
            "managed_run_id": (
                managed_run_id
            ),
            "configuration_fingerprint": (
                config_fingerprint
            ),
        },
    )

    execution_run = (
        execution_engine.execute(
            request
        )
    )

    assert (
        execution_run.successful
        is True
    )

    assert (
        execution_run.execution_report
        is not None
    )

    assert (
        completed_task_count(
            execution_run.execution_report
        )
        == 3
    )

    # --------------------------------------------------------
    # 5. Research result
    # --------------------------------------------------------

    result = {
        "asset": "BTC",
        "decision": "HOLD",
        "market_regime": "MIXED",
        "confidence": 0.73,
        "agents_completed": 3,
        "risk_override": False,
        "research_only": True,
        "real_money_execution": False,
    }

    # --------------------------------------------------------
    # 6. Cache result
    # --------------------------------------------------------

    cache.set(
        cache_key,
        result,
    )

    cached = cache.lookup(
        cache_key
    )

    assert cached.hit is True

    assert (
        cached.value["decision"]
        == "HOLD"
    )

    # --------------------------------------------------------
    # 7. Complete managed run
    # --------------------------------------------------------

    runs.complete(
        managed_run_id
    )

    assert (
        managed_run.status
        == ManagedRunStatus.COMPLETED
    )

    stored_completed_run = runs.get(
        managed_run_id
    )

    assert (
        stored_completed_run.status
        == ManagedRunStatus.COMPLETED
    )

    # --------------------------------------------------------
    # 8. Register verified artifact
    # --------------------------------------------------------

    artifact = artifacts.create(
        name="btc_research_report",
        artifact_type=(
            ArtifactType.RESEARCH_REPORT
        ),
        content=result,
        run_id=managed_run_id,
        experiment_id=(
            "experiment_btc_001"
        ),
        dataset_ids=[
            "dataset_btc_prices_v3",
            "dataset_macro_v2",
        ],
        version_ids=[
            model.version_id,
            prompt.version_id,
            policy.version_id,
        ],
    )

    artifacts.verify(
        artifact.artifact_id
    )

    assert (
        artifact.status
        == ArtifactStatus.VERIFIED
    )

    # --------------------------------------------------------
    # 9. Build provenance graph
    # --------------------------------------------------------

    lineage.create_node(
        node_id=(
            "dataset_btc_prices_v3"
        ),
        node_type=(
            LineageNodeType.DATASET
        ),
        fingerprint=(
            "btc_prices_v3_fp"
        ),
    )

    lineage.create_node(
        node_id="dataset_macro_v2",
        node_type=(
            LineageNodeType.DATASET
        ),
        fingerprint="macro_v2_fp",
    )

    lineage.create_node(
        node_id=model.version_id,
        node_type=(
            LineageNodeType.MODEL_VERSION
        ),
        fingerprint=model.fingerprint,
    )

    lineage.create_node(
        node_id=prompt.version_id,
        node_type=(
            LineageNodeType.PROMPT_VERSION
        ),
        fingerprint=prompt.fingerprint,
    )

    lineage.create_node(
        node_id=policy.version_id,
        node_type=(
            LineageNodeType.POLICY_VERSION
        ),
        fingerprint=policy.fingerprint,
    )

    lineage.create_node(
        node_id=managed_run_id,
        node_type=(
            LineageNodeType.RUN
        ),
        fingerprint=(
            identity.fingerprint
        ),
    )

    lineage.create_node(
        node_id="experiment_btc_001",
        node_type=(
            LineageNodeType.EXPERIMENT
        ),
    )

    lineage.create_node(
        node_id=artifact.artifact_id,
        node_type=(
            LineageNodeType.ARTIFACT
        ),
        fingerprint=(
            artifact.content_fingerprint
        ),
    )

    for dependency in (
        "dataset_btc_prices_v3",
        "dataset_macro_v2",
        model.version_id,
        prompt.version_id,
        policy.version_id,
    ):
        lineage.connect(
            dependency,
            managed_run_id,
            LineageRelation.USES,
        )

    lineage.connect(
        managed_run_id,
        "experiment_btc_001",
        LineageRelation.PRODUCES,
    )

    lineage.connect(
        "experiment_btc_001",
        artifact.artifact_id,
        LineageRelation.PRODUCES,
    )

    # --------------------------------------------------------
    # 10. Verify provenance
    # --------------------------------------------------------

    ancestors = {
        node.node_id
        for node in lineage.ancestors(
            artifact.artifact_id
        )
    }

    assert (
        "dataset_btc_prices_v3"
        in ancestors
    )

    assert (
        "dataset_macro_v2"
        in ancestors
    )

    assert (
        model.version_id
        in ancestors
    )

    assert (
        prompt.version_id
        in ancestors
    )

    assert (
        policy.version_id
        in ancestors
    )

    assert (
        managed_run_id
        in ancestors
    )

    assert (
        "experiment_btc_001"
        in ancestors
    )

    path = lineage.find_path(
        "dataset_btc_prices_v3",
        artifact.artifact_id,
    )

    assert path is not None

    assert (
        path.length
        == 3
    )

    # --------------------------------------------------------
    # 11. Verify event history
    # --------------------------------------------------------

    event_types = {
        event.event_type
        for event in (
            events.event_history()
        )
    }

    assert (
        EventType.RESEARCH_REQUESTED
        in event_types
    )

    assert (
        EventType.RESEARCH_STARTED
        in event_types
    )

    assert (
        EventType.RESEARCH_COMPLETED
        in event_types
    )

    correlated_events = (
        events.events_for_correlation(
            managed_run_id
        )
    )

    assert (
        len(correlated_events)
        >= 3
    )

    # --------------------------------------------------------
    # 12. Safety verification
    # --------------------------------------------------------

    assert (
        result["research_only"]
        is True
    )

    assert (
        result[
            "real_money_execution"
        ]
        is False
    )

    assert (
        artifacts.summary()[
            "real_money_execution"
        ]
        is False
    )

    assert (
        lineage.summary()[
            "real_money_execution"
        ]
        is False
    )

    assert (
        recovery.summary()[
            "real_money_execution"
        ]
        is False
    )

    # --------------------------------------------------------
    # 13. Infrastructure integrity
    # --------------------------------------------------------

    assert (
        artifacts.count()
        == 1
    )

    assert (
        lineage.node_count()
        == 8
    )

    assert (
        lineage.edge_count()
        == 7
    )

    assert (
        len(
            lineage.full_graph_fingerprint()
        )
        == 64
    )

    final_run = runs.get(
        managed_run_id
    )

    assert (
        final_run.status
        == ManagedRunStatus.COMPLETED
    )

    assert (
        runs.exists(
            managed_run_id
        )
        is True
    )

    # Exact RunManager API:
    # is_reproducible(run_id, identity)

    assert (
        runs.is_reproducible(
            managed_run_id,
            identity,
        )
        is True
    )