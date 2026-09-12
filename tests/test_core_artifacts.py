from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

import danids.policy.artifacts as artifact_module
from danids.adaptation.actions import DeployedState
from danids.adaptation.memory import (
    SOURCE_PARTITION_SELECTOR_VERSION,
    ReplayAuditMemory,
    deterministic_audit_exemplars,
    deterministic_replay_exemplars,
    initialize_source_replay_audit_memory,
)
from danids.continual.initial_state import threshold_state_digest
from danids.continual.memory import subset_learning_batch
from danids.continual.supervision import row_positions_digest
from danids.data.manifests import generate_split_manifest
from danids.data.materialized import materialize_dataset
from danids.data.preprocessing import NumericPreprocessor
from danids.data.registry import DatasetSpec
from danids.data.schema import FeatureContract
from danids.data.types import LearningBatch, PartitionKind, ValidationSet
from danids.evaluation.threshold import ThresholdSelection
from danids.models.mlp import StaticMLP
from danids.models.training import predict_scores
from danids.policy.artifacts import (
    ADMINISTRATIVE_ROUTES_FILENAME,
    CORE_RUN_ARTIFACT_VERSION,
    MANIFEST_FILENAME,
    MEMORY_LOG_FILENAME,
    POLICY_WINDOWS_FILENAME,
    SUMMARY_FILENAME,
    AdministrativeWindowRoute,
    CoreRunProvenance,
    CoreWindowTrace,
    SourceMemorySelectionEvidence,
    build_source_memory_selection_evidence,
    validate_core_run_artifacts,
    write_core_run_artifacts,
)
from danids.policy.core import CorePolicyObservation, DANIDSCoreController
from danids.policy.executor import CoreMemoryIdentity
from danids.policy.health_artifact import (
    POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
    POLICY_HEALTH_FEATURES,
    PolicyHealthVector,
    PredictedHealthState,
)
from danids.policy.references import (
    FixedReferenceRows,
    FixedReferenceSelection,
    HealthDecisionBinding,
    R1ReferenceState,
    build_r1_reference_state,
)

FEATURES = ("f0", "f1", "f2")
SOURCE_SCOPE = "source-scope"
CURRENT_SCOPE = "current-scope"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _learning(
    size: int,
    *,
    offset: int = 0,
    kind: PartitionKind = PartitionKind.INITIAL_TRAIN,
) -> LearningBatch:
    labels = np.asarray(np.arange(size) % 2, dtype=np.int8)
    return LearningBatch(
        np.arange(size * len(FEATURES), dtype=np.float32).reshape(size, len(FEATURES)),
        labels,
        np.where(labels == 1, "attack", "Benign").astype(object),
        pd.DataFrame({"timestamp": np.arange(offset, offset + size)}),
        np.arange(offset, offset + size, dtype=np.int64),
        FEATURES,
        kind,
    )


def _validation(size: int, *, offset: int) -> ValidationSet:
    source = _learning(size, offset=offset)
    return ValidationSet(
        source.features,
        source.binary_labels,
        source.native_attack_labels,
        source.metadata,
        source.row_positions,
        source.feature_columns,
    )


def _state(initial: LearningBatch) -> DeployedState:
    preprocessor = NumericPreprocessor().fit(initial)
    torch.manual_seed(9)
    model = StaticMLP(len(FEATURES), dropout=0.0)
    model.eval()
    threshold = ThresholdSelection(
        threshold=1.0,
        target_fpr=0.001,
        validation_fpr=0.0,
        validation_tpr=0.0,
        tp=0,
        fp=0,
        tn=6,
        fn=6,
    )
    return DeployedState(model, preprocessor, threshold)


def _memory(initial: LearningBatch) -> ReplayAuditMemory:
    replay = deterministic_replay_exemplars(
        initial,
        domain_id=SOURCE_SCOPE,
        capacity=400,
        seed=7,
    )
    audit = deterministic_audit_exemplars(
        initial,
        domain_id=SOURCE_SCOPE,
        capacity=100,
        seed=8,
        excluded_positions=replay.batch.row_positions,
    )
    memory = ReplayAuditMemory()
    memory.add_replay(replay)
    memory.add_audit(audit)
    return memory


def _reference(initial: LearningBatch, deployed: DeployedState) -> R1ReferenceState:
    validation = _validation(12, offset=len(initial))
    selection = FixedReferenceSelection(
        0,
        len(initial),
        len(initial),
        len(initial) + len(validation),
        "synthetic-study3-reference",
    )
    training = subset_learning_batch(initial, selection.expected_training_positions())
    fixed = FixedReferenceRows.capture(
        SOURCE_SCOPE,
        training,
        validation,
        deployed.preprocessor,
        selection=selection,
    )
    scores = predict_scores(
        deployed.model,
        deployed.preprocessor,
        validation,
        device=torch.device("cpu"),
    )
    recall = float(np.mean(scores[validation.binary_labels == 1] >= 1.0))
    return build_r1_reference_state(
        fixed,
        deployed,
        HealthDecisionBinding(_sha("health"), _sha("thresholds")),
        origin_reference_recall=recall,
        origin_recall_floor=max(0.0, recall - 0.10),
        conformal_alpha=0.10,
        device=torch.device("cpu"),
        batch_size=128,
    )


def _source_evidence(
    initial: LearningBatch,
    memory: ReplayAuditMemory,
    deployed: DeployedState,
) -> SourceMemorySelectionEvidence:
    manifest = memory.manifest()
    replay = manifest["replay_domains"][0]["row_positions"]
    audit = manifest["audit_domains"][0]["row_positions"]
    replay_labels = tuple(int(initial.binary_labels[position]) for position in replay)
    audit_labels = tuple(int(initial.binary_labels[position]) for position in audit)
    attack_support = audit_labels.count(1)
    return SourceMemorySelectionEvidence(
        source_scope=SOURCE_SCOPE,
        selector_version="task006-source-memory-selection-v1",
        initial_train_start=0,
        initial_train_stop=len(initial),
        initial_train_labels_digest=_sha("initial-labels"),
        source_dataset_fingerprint=_sha("U"),
        benign_support=int(np.sum(initial.binary_labels == 0)),
        attack_support=int(np.sum(initial.binary_labels == 1)),
        replay_seed=7,
        audit_seed=8,
        replay_positions=tuple(replay),
        replay_binary_labels=replay_labels,
        audit_positions=tuple(audit),
        audit_binary_labels=audit_labels,
        audit_true_positives_at_learned_state=0,
        audit_learned_recall=(0.0 if attack_support else None),
        detector_model_digest=deployed.model_digest,
        detector_threshold_digest=threshold_state_digest(deployed.threshold),
        preprocessor_digest=deployed.preprocessor_digest,
    )


def _provenance(
    deployed: DeployedState,
    evidence: SourceMemorySelectionEvidence,
) -> CoreRunProvenance:
    return CoreRunProvenance(
        experiment_id="synthetic-core-artifact",
        artifact_mode="synthetic_test",
        sequence=("U", "T", "C", "B"),
        seed=42,
        config_sha256=_sha("config"),
        code_commit="1" * 40,
        study1_source_run="synthetic-study1",
        dataset_fingerprints={domain: _sha(domain) for domain in ("U", "T", "C", "B")},
        study3_canonical_dataset_sha256=_sha("study3"),
        health_fit_component_digest=_sha("fit"),
        health_calibration_component_digest=_sha("calibration"),
        health_model_artifact_identity=_sha("health"),
        health_model_digest=_sha("health"),
        health_model_serialized_sha256=_sha("health-bytes"),
        health_thresholds_digest=_sha("thresholds"),
        health_feature_contract_digest=POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
        initial_model_digest=deployed.model_digest,
        initial_preprocessor_digest=deployed.preprocessor_digest,
        initial_threshold_digest=threshold_state_digest(deployed.threshold),
        source_initial_train_start=0,
        source_initial_train_stop=evidence.initial_train_stop,
        source_initial_train_labels_digest=evidence.initial_train_labels_digest,
        source_replay_selection_seed=7,
        source_audit_selection_seed=8,
    )


def _observation(
    deployed: DeployedState,
    reference: R1ReferenceState,
    memory: ReplayAuditMemory,
) -> CorePolicyObservation:
    identity = CoreMemoryIdentity.from_memory(memory)
    return CorePolicyObservation(
        health=PolicyHealthVector.from_mapping(
            {name: float(index) for index, name in enumerate(POLICY_HEALTH_FEATURES)}
        ),
        harm_probability=0.0,
        predicted_state=PredictedHealthState.SAFE,
        health_model_digest=_sha("health"),
        health_model_artifact_identity=_sha("health"),
        health_feature_contract_digest=POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
        r1_reference_state_digest=reference.state_digest,
        prediction_token=_sha("prediction-0"),
        remaining_label_budget=100,
        query_count=0,
        query_pending=False,
        released_label_count=0,
        released_labels_digest=None,
        deployed_model_digest=deployed.model_digest,
        deployed_threshold_digest=threshold_state_digest(deployed.threshold),
        preprocessor_digest=deployed.preprocessor_digest,
        replay_row_count=identity.replay_row_count,
        replay_digest=identity.replay_digest,
        audit_escrow_row_count=0,
        audit_escrow_digest=None,
        active_historical_audit_panels=identity.audit_panel_count,
        audit_memory_digest=identity.audit_digest,
    )


def _bundle(tmp_path: Path) -> Path:
    initial = _learning(5_000)
    deployed = _state(initial)
    memory = _memory(initial)
    reference = _reference(initial, deployed)
    evidence = _source_evidence(initial, memory, deployed)
    provenance = _provenance(deployed, evidence)
    observation = _observation(deployed, reference, memory)
    controller = DANIDSCoreController(
        health_model_digest=provenance.health_model_digest,
        health_model_artifact_identity=provenance.health_model_artifact_identity,
    )
    decision = controller.observe(observation)
    trace = CoreWindowTrace(0, observation, (decision,), controller.state)
    route = AdministrativeWindowRoute(
        prediction_index=0,
        sequence=provenance.sequence,
        seed=provenance.seed,
        domain_stage=0,
        current_domain="U",
        opaque_scope_token=CURRENT_SCOPE,
        partition_kind=PartitionKind.ONLINE_STREAM.value,
        window_id=0,
        row_start=5_012,
        row_stop=5_062,
        online_stream_start=5_012,
        online_stream_stop=5_100,
        permanent_holdout_start=5_100,
        permanent_holdout_stop=5_200,
    )
    return write_core_run_artifacts(
        tmp_path / "core-run",
        provenance=provenance,
        windows=(trace,),
        administrative_routes=(route,),
        memory=memory,
        source_memory_selection=evidence,
        references=(reference,),
    )


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _rehash_bundle(root: Path) -> None:
    manifest_path = root / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in manifest["files"]
    }
    manifest["files"] = files
    manifest["bundle_digest"] = hashlib.sha256(_canonical(files)).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _rewrite(root: Path, filename: str, mutate: Any) -> None:
    path = root / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rehash_bundle(root)


def test_core_artifact_round_trip_and_write_once(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    summary = validate_core_run_artifacts(root)
    assert summary["policy_window_count"] == 1
    assert summary["predicted_state_counts"]["PREDICTED_SAFE"] == 1
    assert summary["historical_replay_rows"] == 400
    assert summary["historical_audit_rows"] == 100
    with pytest.raises(FileExistsError, match="overwrite"):
        _bundle(tmp_path)


def test_recomputed_summary_rejects_corruption(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    _rewrite(
        root,
        SUMMARY_FILENAME,
        lambda payload: payload.__setitem__("policy_window_count", 99),
    )
    with pytest.raises(ValueError, match="summary differs"):
        validate_core_run_artifacts(root)


def test_policy_window_rejects_evaluator_truth_even_when_rehashed(tmp_path: Path) -> None:
    root = _bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        window = payload["windows"][0]
        window["observation"]["health_state"] = "HARMFUL"
        body = {key: value for key, value in window.items() if key != "trace_digest"}
        window["trace_digest"] = hashlib.sha256(_canonical(body)).hexdigest()

    _rewrite(root, POLICY_WINDOWS_FILENAME, mutate)
    with pytest.raises(ValueError, match=r"missing or additional|evaluator-only"):
        validate_core_run_artifacts(root)


def test_source_memory_selection_corruption_is_rejected(tmp_path: Path) -> None:
    root = _bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        evidence = payload["source_selection"]
        evidence["replay_binary_labels"][0] = 1 - evidence["replay_binary_labels"][0]
        body = {key: value for key, value in evidence.items() if key != "evidence_digest"}
        evidence["evidence_digest"] = hashlib.sha256(_canonical(body)).hexdigest()

    _rewrite(root, MEMORY_LOG_FILENAME, mutate)
    with pytest.raises(ValueError, match="stratified allocation"):
        validate_core_run_artifacts(root)


def test_route_stage_cannot_switch_scope_or_return(tmp_path: Path) -> None:
    root = _bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        first = payload["routes"][0]
        payload["routes"] = [
            first,
            {
                **first,
                "prediction_index": 1,
                "window_id": 0,
                "opaque_scope_token": "other-scope",
            },
        ]

    _rewrite(root, ADMINISTRATIVE_ROUTES_FILENAME, mutate)
    with pytest.raises(ValueError, match=r"routes do not match|multiple supervision scopes"):
        validate_core_run_artifacts(root)


def test_interleaved_domain_routes_are_rejected() -> None:
    sequence = ("U", "T", "C", "B")

    def route(index: int, stage: int, domain: str, scope: str) -> dict[str, Any]:
        return AdministrativeWindowRoute(
            prediction_index=index,
            sequence=sequence,
            seed=42,
            domain_stage=stage,
            current_domain=domain,
            opaque_scope_token=scope,
            partition_kind=PartitionKind.ONLINE_STREAM.value,
            window_id=0,
            row_start=0,
            row_stop=50,
            online_stream_start=0,
            online_stream_stop=100,
            permanent_holdout_start=100,
            permanent_holdout_stop=200,
        ).to_dict()

    payload = {
        "version": CORE_RUN_ARTIFACT_VERSION,
        "routes": [
            route(0, 0, "U", "scope-u"),
            route(1, 1, "T", "scope-t"),
            route(2, 0, "U", "scope-u"),
        ],
    }
    with pytest.raises(ValueError, match="not chronological"):
        artifact_module._validate_routes(  # pyright: ignore[reportPrivateUsage]
            payload, {"sequence": list(sequence), "seed": 42}, 3
        )


def test_query_rows_are_recomputed_from_exact_routed_window() -> None:
    route = AdministrativeWindowRoute(
        prediction_index=0,
        sequence=("U", "T", "C", "B"),
        seed=42,
        domain_stage=0,
        current_domain="U",
        opaque_scope_token=CURRENT_SCOPE,
        partition_kind=PartitionKind.ONLINE_STREAM.value,
        window_id=0,
        row_start=100,
        row_stop=200,
        online_stream_start=100,
        online_stream_stop=300,
        permanent_holdout_start=300,
        permanent_holdout_stop=400,
    ).to_dict()
    selection = artifact_module._recompute_query_from_positions(
        tuple(range(100, 200)),
        seed=42,
        scope=CURRENT_SCOPE,
        window_id=0,
        query_ordinal=0,
    ).to_dict()
    replacement = next(
        position for position in range(100, 200) if position not in selection["selected_positions"]
    )
    changed = sorted([*selection["selected_positions"][:-1], replacement])
    selection["selected_positions"] = changed
    selection["selected_positions_digest"] = row_positions_digest(changed)
    payload = {
        "version": "task006-sha256-label-blind-v1",
        "selections": [selection],
    }
    windows = [{"observation": {"query_count": 0}, "decisions": [{"query": {}}]}]
    with pytest.raises(ValueError, match="exact SHA-256 recomputation"):
        artifact_module._validate_queries(payload, windows, [route])


def _historical_activation_case(
    *, final_query: bool, activation_window: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    scope = "scope-a"
    routes = [
        {
            "opaque_scope_token": scope,
            "window_id": 0,
            "prediction_index": 0,
            "domain_stage": 1,
            "row_start": 0,
            "row_stop": 50,
        },
        {
            "opaque_scope_token": scope,
            "window_id": 1,
            "prediction_index": 1,
            "domain_stage": 1,
            "row_start": 50,
            "row_stop": 100,
        },
        {
            "opaque_scope_token": "scope-b",
            "window_id": 0,
            "prediction_index": 2,
            "domain_stage": 2,
            "row_start": 100,
            "row_stop": 150,
        },
    ]
    query_prediction = 1 if final_query else 0
    row_start = int(routes[query_prediction]["row_start"])
    positions = list(range(row_start, row_start + 25))
    queries = [
        {
            "opaque_scope_token": scope,
            "window_id": query_prediction,
            "status": artifact_module.QuerySelectionStatus.SELECTED.value,
            "selected_positions": positions,
            "selected_positions_digest": row_positions_digest(positions),
        }
    ]
    allocations = {
        "version": artifact_module.SCARCE_LABEL_ALLOCATION_VERSION,
        "allocations": [
            {
                "scope_id": scope,
                "releases": [
                    {
                        "query_window": query_prediction,
                        "release_window": query_prediction + 1,
                        "queried_positions": positions,
                        "queried_positions_digest": row_positions_digest(positions),
                    }
                ],
                "historical_activation": {"activation_window": activation_window},
            }
        ],
    }
    return allocations, queries, routes


@pytest.mark.parametrize(
    ("final_query", "activation_window"),
    [(False, 1), (True, 2)],
)
def test_historical_activation_accepts_both_frozen_boundary_modes(
    monkeypatch: pytest.MonkeyPatch,
    final_query: bool,
    activation_window: int,
) -> None:
    monkeypatch.setattr(artifact_module, "validate_allocation_manifest", lambda _: None)
    payload, queries, routes = _historical_activation_case(
        final_query=final_query,
        activation_window=activation_window,
    )
    artifact_module._validate_allocations(payload, queries, routes)


@pytest.mark.parametrize(
    ("final_query", "activation_window"),
    [(False, 2), (True, 1), (True, 3)],
)
def test_historical_activation_rejects_wrong_boundary_mode(
    monkeypatch: pytest.MonkeyPatch,
    final_query: bool,
    activation_window: int,
) -> None:
    monkeypatch.setattr(artifact_module, "validate_allocation_manifest", lambda _: None)
    payload, queries, routes = _historical_activation_case(
        final_query=final_query,
        activation_window=activation_window,
    )
    with pytest.raises(ValueError, match="pending-query boundary mode"):
        artifact_module._validate_allocations(payload, queries, routes)


def test_trace_dataclass_rejects_unresolved_query_state() -> None:
    initial = _learning(5_000)
    deployed = _state(initial)
    memory = _memory(initial)
    reference = _reference(initial, deployed)
    observation = replace(
        _observation(deployed, reference, memory),
        harm_probability=0.999,
        predicted_state=PredictedHealthState.HARMFUL,
    )
    controller = DANIDSCoreController(
        health_model_digest=_sha("health"),
        health_model_artifact_identity=_sha("health"),
    )
    decision = controller.observe(observation)
    with pytest.raises(ValueError, match="unresolved query"):
        CoreWindowTrace(0, observation, (decision,), controller.state)


def test_source_memory_uses_frozen_selector_identity() -> None:
    assert SOURCE_PARTITION_SELECTOR_VERSION == "task006-source-binary-stratified-v1"
    assert CORE_RUN_ARTIFACT_VERSION == "task006-core-run-artifacts-v1"
    assert row_positions_digest((1, 2, 3)) != row_positions_digest((1, 2, 4))


def test_source_evidence_builder_is_bounded_and_deterministic(tmp_path: Path) -> None:
    positions = np.arange(1_000)
    source_path = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "timestamp": positions,
            "f0": positions.astype(np.float32),
            "f1": (positions % 17).astype(np.float32),
            "f2": (positions % 29).astype(np.float32),
            "label": (positions % 2).astype(np.int8),
            "attack": np.where(positions % 2, "attack", "Benign"),
        }
    ).to_csv(source_path, index=False)
    spec = DatasetSpec(
        dataset_id="U",
        name="synthetic",
        path=source_path,
        binary_label_column="label",
        native_attack_column="attack",
        timestamp_column="timestamp",
        metadata_columns=("timestamp",),
        optional_metadata_columns=(),
    )
    contract = FeatureContract("source-evidence-test-v1", FEATURES)
    split = generate_split_manifest(
        spec,
        contract,
        role="initial",
        split_version="task001-chronological-v1",
        seed=42,
    )
    cached = materialize_dataset(
        spec,
        contract,
        split,
        cache_root=tmp_path / "cache",
        chunk_rows=73,
    )
    partition = cached.partition(PartitionKind.INITIAL_TRAIN)
    preprocessor = NumericPreprocessor().fit_source(partition, batch_size=97)
    torch.manual_seed(17)
    model = StaticMLP(len(FEATURES), dropout=0.0)
    model.eval()
    deployed = DeployedState(
        model,
        preprocessor,
        ThresholdSelection(1.0, 0.001, 0.0, 0.0, 0, 0, 1, 1),
    )
    memory = initialize_source_replay_audit_memory(
        partition,
        domain_id=SOURCE_SCOPE,
        replay_seed=7,
        audit_seed=8,
        scan_rows=61,
    )
    before = (deployed.model_digest, deployed.preprocessor_digest, deployed.threshold_digest)
    evidence = build_source_memory_selection_evidence(
        partition,
        memory,
        deployed,
        source_scope=SOURCE_SCOPE,
        replay_seed=7,
        audit_seed=8,
        device=torch.device("cpu"),
        scan_rows=37,
    )
    repeated = build_source_memory_selection_evidence(
        partition,
        memory,
        deployed,
        source_scope=SOURCE_SCOPE,
        replay_seed=7,
        audit_seed=8,
        device=torch.device("cpu"),
        scan_rows=113,
    )
    assert evidence == repeated
    assert evidence.benign_support == evidence.attack_support == 300
    assert len(evidence.replay_positions) == 400
    assert len(evidence.audit_positions) == 100
    assert evidence.source_dataset_fingerprint == split.source.sha256
    assert before == (
        deployed.model_digest,
        deployed.preprocessor_digest,
        deployed.threshold_digest,
    )
