from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from danids.adaptation.actions import (
    InterventionAction,
    InterventionRecord,
    replay_flat_positions_digest,
    replay_scoped_positions_digest,
)
from danids.config.rdx_training_evidence import (
    RDX004_METHOD,
    RDX004_ROTATIONS,
    RDX004_SEEDS,
    RDX004Budget,
    RDX004RunIdentity,
)
from danids.config.rdx_training_execution import RDX004ExecutionIdentity
from danids.continual.supervision import row_positions_digest
from danids.evaluation import rdx_training_execution as validation


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _seal(root: Path) -> dict[str, object]:
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in sorted(validation.RDX004_RUN_OUTPUT_FILES - {validation.MANIFEST_FILENAME})
    }
    manifest: dict[str, object] = {
        "version": validation.RDX004_RUN_ARTIFACT_VERSION,
        "files": files,
        "bundle_digest": _canonical_digest(files),
    }
    _write_json(root / validation.MANIFEST_FILENAME, manifest)
    return manifest


def _identity(*, smoke: bool) -> RDX004ExecutionIdentity:
    scientific = RDX004RunIdentity(
        budget=RDX004Budget.B400,
        rotation=RDX004_ROTATIONS[0],
        seed=RDX004_SEEDS[0],
    )
    return RDX004ExecutionIdentity(scientific_identity=scientific, smoke=smoke)


def _smoke_events(identity: RDX004ExecutionIdentity) -> list[dict[str, object]]:
    events = [
        {
            "stage": 1,
            "current_domain": identity.rotation[1],
            "selection_window_id": ordinal,
            "prediction_index": 10 + ordinal,
            "release_prediction_index": 11 + ordinal,
            "released": True,
        }
        for ordinal in range(4)
    ]
    events.extend(
        [
            {
                "stage": 2,
                "current_domain": identity.rotation[2],
                "selection_window_id": 0,
                "prediction_index": 100,
                "release_prediction_index": 101,
                "released": True,
            },
            {
                "stage": 2,
                "current_domain": identity.rotation[2],
                "selection_window_id": 1,
                "prediction_index": 101,
                "release_prediction_index": 102,
                "released": False,
            },
        ]
    )
    return events


def _smoke_windows(identity: RDX004ExecutionIdentity) -> list[dict[str, str]]:
    decision_id = "d" * 64
    result: list[dict[str, str]] = []
    for stage, count, base in ((1, 5, 10), (2, 2, 100)):
        for window_id in range(count):
            evaluated = stage == 2 and window_id == 1
            result.append(
                {
                    "prediction_index": str(base + window_id),
                    "seed": str(identity.seed),
                    "stage": str(stage),
                    "current_domain": identity.rotation[stage],
                    "window_id": str(window_id),
                    "row_start": str(window_id * 50_000),
                    "row_stop": str((window_id + 1) * 50_000),
                    "partition_kind": "online_stream",
                    "prediction_before_truth": "True",
                    "policy_health_evaluator_free": "True",
                    "evaluator_health_state": "HARMFUL",
                    "model_digest_at_prediction": "a" * 64,
                    "model_digest_after_action": "a" * 64,
                    "oracle_decision_evaluated": str(evaluated),
                    "decision_id": decision_id if evaluated else "",
                    "selected_action": "A0_NO_OP",
                    "accepted_update_after_prediction": "False",
                    "evaluator_truth_revealed_after_action": "False",
                    "permanent_holdout_used_for_choice": "False",
                }
            )
    return result


def _smoke_oracle_record() -> dict[str, object]:
    return {
        "prediction_index": 101,
        "decision_id": "d" * 64,
        "stage": 2,
        "current_domain": RDX004_ROTATIONS[0][2],
        "window_id": 1,
        "current_evaluator_state": "HARMFUL",
        "successor_window_id": 2,
        "selected_action": "A0_NO_OP",
        "candidates": [
            {
                "action": action.value,
                "incoming_model_digest": "a" * 64,
                "deployed_model_digest": "a" * 64,
                "successor_window_id": 2,
                "successor_row_start": 100_000,
                "successor_row_stop": 150_000,
            }
            for action in InterventionAction
        ],
    }


def _pipeline(identity: RDX004ExecutionIdentity) -> dict[str, object]:
    return {
        "window_size": 50_000,
        "manifest_partition_ranges": {
            domain: {
                "initial_train": None,
                "validation": None,
                "online_stream": [0, 300_000],
                "permanent_holdout": None,
            }
            for domain in identity.rotation
        },
    }


def _holdout_rows(
    identity: RDX004ExecutionIdentity, windows: list[dict[str, str]]
) -> list[dict[str, str]]:
    calls = [
        (1, "source_initial", 0, "", identity.rotation[:1]),
        (2, "domain_end", 1, "14", identity.rotation[:2]),
        (3, "domain_end", 2, "101", identity.rotation[:3]),
        (4, "final", 2, "101", identity.rotation[:3]),
    ]
    model = windows[0]["model_digest_at_prediction"]
    return [
        {
            "event_index": str(event_index),
            "event": event,
            "stage": str(stage),
            "prediction_index": prediction,
            "holdout_dataset_id": domain,
            "partition_kind": "permanent_holdout",
            "evaluation_only": "True",
            "used_for_query": "False",
            "used_for_training": "False",
            "used_for_adaptation": "False",
            "used_for_action_choice": "False",
            "used_for_oracle_choice": "False",
            "model_digest": model,
        }
        for event_index, event, stage, prediction, domains in calls
        for domain in domains
    ]


def _intervention_candidate(
    identity: RDX004ExecutionIdentity,
    action: InterventionAction,
    *,
    target: tuple[int, ...],
    replay_scopes: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    scoped = [] if replay_scopes is None else replay_scopes
    replay_positions: list[int] = []
    for item in scoped:
        positions = item["row_positions"]
        assert isinstance(positions, list)
        replay_positions.extend(int(position) for position in positions)
    replay = tuple(sorted(replay_positions))
    optimizer_steps = validation._expected_optimizer_steps(action, len(target))
    before = "1" * 64
    after = "2" * 64
    threshold = "3" * 64
    preprocessor = "4" * 64
    record = InterventionRecord(
        sequence=identity.sequence_token,
        seed=identity.seed,
        domain_stage=2,
        current_domain=identity.rotation[2],
        window_id=1,
        policy_health_information="{}",
        remaining_label_budget=200,
        labels_requested=0,
        labels_available=len(target),
        action_attempted=action.value,
        action_rank=action.rank,
        action_cost_proxy=action.rank,
        threshold_before=0.5,
        candidate_threshold=0.5,
        threshold_after=0.5,
        threshold_digest_before=threshold,
        threshold_digest_candidate=threshold,
        threshold_digest_after=threshold,
        model_digest_before=before,
        model_digest_candidate=after,
        model_digest_after=after,
        preprocessor_digest_before=preprocessor,
        preprocessor_digest_after=preprocessor,
        optimizer_steps=optimizer_steps,
        target_rows=len(target),
        target_row_positions=json.dumps(target, separators=(",", ":")),
        target_row_positions_digest=row_positions_digest(target),
        calibration_rows=0,
        calibration_row_positions="[]",
        calibration_row_positions_digest="",
        replay_rows=len(replay),
        replay_row_positions=json.dumps(replay, separators=(",", ":")),
        replay_row_positions_digest=(replay_flat_positions_digest(replay) if replay else ""),
        audit_domains_checked="[]",
        audit_result="SAFE",
        audit_decisions="[]",
        accepted=True,
        rolled_back=False,
        rejection_reason="",
        wall_clock_update_seconds=0.0,
        replay_scoped_row_positions=json.dumps(scoped, sort_keys=True, separators=(",", ":")),
        replay_scoped_row_positions_digest=(
            replay_scoped_positions_digest(scoped) if scoped else ""
        ),
    ).to_dict()
    available_rows = [
        {"domain_id": item["scope_id"], "positions": item["row_positions"]} for item in scoped
    ]
    return {
        "feasible": True,
        "execution_succeeded": True,
        "audit_admissible": True,
        "audit_state": "SAFE",
        "successor_evaluator_state": "SAFE",
        "confirmed_success": True,
        "optimizer_steps": optimizer_steps,
        "target_rows": len(target),
        "replay_rows": len(replay),
        "incoming_model_digest": before,
        "deployed_model_digest": after,
        "deployed_threshold_digest": threshold,
        "target_positions_digest_consumed": row_positions_digest(target),
        "historical_replay_positions_digest_available": (
            _canonical_digest(available_rows) if available_rows else "5" * 64
        ),
        "historical_replay_positions_digest_consumed": (
            replay_scoped_positions_digest(scoped) if scoped else ""
        ),
        "intervention_record": record,
    }


def _validate_test_candidate(
    candidate: dict[str, object],
    identity: RDX004ExecutionIdentity,
    action: InterventionAction,
    target: tuple[int, ...],
    *,
    replay_rows: int = 0,
    replay_scope_ids: set[str] | None = None,
    replay_scoped_positions: list[dict[str, object]] | None = None,
) -> None:
    validation._validate_candidate_intervention(
        candidate,
        action=action,
        identity=identity,
        stage=2,
        domain=identity.rotation[2],
        window_id=1,
        available_target=target,
        current_audit=set(),
        expected_replay_available=replay_rows,
        expected_replay_scope_ids=set() if replay_scope_ids is None else replay_scope_ids,
        expected_replay_scoped_positions=replay_scoped_positions,
    )


def _minimal_contract(identity: RDX004ExecutionIdentity) -> dict[str, object]:
    return {
        "version": validation.RDX004_RUN_ARTIFACT_VERSION,
        "identity": identity.to_dict(),
        "method": RDX004_METHOD,
        "protocol": {},
        "execution_config": {},
        "preflight": {},
        "starting_state": {},
        "information_boundary": {},
        "treatment": {},
        "executor": {},
        "selector": {},
        "pipeline": {},
        "code": {},
    }


def _write_syntactic_bundle(root: Path, contract: dict[str, object] | None = None) -> None:
    root.mkdir()
    for name in validation.RDX004_RUN_OUTPUT_FILES - {validation.MANIFEST_FILENAME}:
        path = root / name
        if name == validation.CONFIG_FILENAME:
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write("{}\n")
        elif path.suffix == ".csv":
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write("\n")
        else:
            _write_json(path, contract if name == validation.EXECUTION_CONTRACT_FILENAME else {})
    _seal(root)


def test_exact_sealed_bundle_accepts_only_the_thirteen_frozen_files(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_syntactic_bundle(root)

    manifest, bundle_digest = validation._seal_manifest_identity(
        root,
        expected_files=validation.RDX004_RUN_OUTPUT_FILES,
        manifest_name=validation.MANIFEST_FILENAME,
        expected_version=validation.RDX004_RUN_ARTIFACT_VERSION,
    )

    assert len(validation.RDX004_RUN_OUTPUT_FILES) == 13
    assert bundle_digest == manifest["bundle_digest"]
    (root / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(validation.RDX004RunArtifactError, match="file set differs"):
        validation._seal_manifest_identity(
            root,
            expected_files=validation.RDX004_RUN_OUTPUT_FILES,
            manifest_name=validation.MANIFEST_FILENAME,
            expected_version=validation.RDX004_RUN_ARTIFACT_VERSION,
        )


def test_public_validator_rejects_payload_tampering_before_semantic_loading(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _write_syntactic_bundle(root)
    (root / validation.SUMMARY_FILENAME).write_text("tampered\n", encoding="utf-8")

    with pytest.raises(validation.RDX004RunArtifactError, match="manifest differs"):
        validation.validate_rdx004_run(root)


def test_public_validator_rejects_smoke_unless_explicitly_expected(tmp_path: Path) -> None:
    root = tmp_path / "smoke"
    identity = _identity(smoke=True)
    _write_syntactic_bundle(root, _minimal_contract(identity))

    with pytest.raises(validation.RDX004RunArtifactError, match="unexpected smoke"):
        validation.validate_rdx004_run(root)
    assert (
        validation._identity_from_contract(_minimal_contract(identity), expected_smoke=True)
        == identity
    )


def test_identity_reconstruction_rejects_coherently_resealed_run_id_tampering() -> None:
    identity = _identity(smoke=False)
    contract = _minimal_contract(identity)
    raw_identity = dict(identity.to_dict())
    raw_identity["run_id"] = "RDX004_FORGED"
    contract["identity"] = raw_identity

    with pytest.raises(validation.RDX004RunArtifactError, match="differs from recomputation"):
        validation._identity_from_contract(contract, expected_smoke=False)


def test_oracle_identity_and_holdout_use_tampering_are_rejected() -> None:
    identity = _identity(smoke=False)
    oracle: dict[str, object] = {
        "version": validation.RDX004_RUN_ARTIFACT_VERSION,
        "run_id": "RDX004_FORGED",
        "budget": identity.budget.value,
        "smoke": False,
        "oracle_version": "forged",
        "information_policy": "forged",
        "horizon": "forged",
        "records": [{}],
    }
    with pytest.raises(validation.RDX004RunArtifactError, match="identity/semantics"):
        validation._validate_oracle_counterfactuals(oracle, identity, [], {})

    smoke_identity = _identity(smoke=True)
    windows = _smoke_windows(smoke_identity)
    holdouts = _holdout_rows(smoke_identity, windows)
    holdouts[0]["used_for_training"] = "True"
    with pytest.raises(validation.RDX004RunArtifactError, match="evaluation-only"):
        validation._validate_holdout_rows(
            holdouts,
            identity=smoke_identity,
            window_rows=windows,
        )


def test_smoke_window_and_holdout_plan_rejects_coverage_tampering() -> None:
    identity = _identity(smoke=True)
    events = _smoke_events(identity)
    windows = _smoke_windows(identity)
    oracle = _smoke_oracle_record()
    holdouts = _holdout_rows(identity, windows)

    pipeline = _pipeline(identity)
    validation._validate_window_rows(windows, identity, events, [oracle], pipeline)
    validation._validate_oracle_window_bindings([oracle], windows, identity, events, pipeline)
    validation._validate_holdout_rows(
        holdouts,
        identity=identity,
        window_rows=windows,
    )

    with pytest.raises(validation.RDX004RunArtifactError, match="seven-window smoke coverage"):
        validation._validate_window_rows(windows[:-1], identity, events, [oracle], pipeline)
    with pytest.raises(validation.RDX004RunArtifactError, match="coverage differs"):
        validation._validate_holdout_rows(
            holdouts[:-1],
            identity=identity,
            window_rows=windows,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("current_evaluator_state", "SAFE", "current-window binding"),
        ("candidate_incoming_model", "b" * 64, "model/successor binding"),
        ("candidate_successor_row_start", 100_001, "model/successor binding"),
    ],
)
def test_oracle_route_rejects_current_model_and_successor_tampering(
    field: str, value: object, message: str
) -> None:
    identity = _identity(smoke=True)
    events = _smoke_events(identity)
    windows = _smoke_windows(identity)
    oracle = _smoke_oracle_record()
    if field == "current_evaluator_state":
        oracle[field] = value
    else:
        candidates = oracle["candidates"]
        assert isinstance(candidates, list)
        candidate = candidates[0]
        assert isinstance(candidate, dict)
        candidate[
            "incoming_model_digest"
            if field == "candidate_incoming_model"
            else "successor_row_start"
        ] = value
    with pytest.raises(validation.RDX004RunArtifactError, match=message):
        validation._validate_oracle_window_bindings(
            [oracle], windows, identity, events, _pipeline(identity)
        )


def test_oracle_route_requires_every_frozen_decision_window() -> None:
    identity = _identity(smoke=True)
    with pytest.raises(validation.RDX004RunArtifactError, match="coverage differs"):
        validation._validate_oracle_window_bindings(
            [], _smoke_windows(identity), identity, _smoke_events(identity), _pipeline(identity)
        )


def test_non_smoke_requires_exact_full_online_stream_coverage() -> None:
    identity = _identity(smoke=False)
    pipeline = _pipeline(identity)
    events = [
        {
            "stage": stage,
            "current_domain": identity.rotation[stage],
            "selection_window_id": ordinal,
            "prediction_index": stage * 100 + ordinal,
            "release_prediction_index": stage * 100 + ordinal + 1,
            "released": True,
        }
        for stage in (1, 2, 3)
        for ordinal in range(4)
    ]
    plan = validation._expected_window_plan(identity, pipeline, events)
    decision_by_prediction = {
        item.prediction_index: f"{item.prediction_index:064x}"
        for item in plan
        if item.oracle_evaluated
    }
    windows = [
        {
            "prediction_index": str(item.prediction_index),
            "seed": str(identity.seed),
            "stage": str(item.stage),
            "current_domain": item.domain,
            "window_id": str(item.window_id),
            "row_start": str(item.row_start),
            "row_stop": str(item.row_stop),
            "partition_kind": "online_stream",
            "prediction_before_truth": "True",
            "policy_health_evaluator_free": "True",
            "evaluator_health_state": "SAFE",
            "model_digest_at_prediction": "a" * 64,
            "model_digest_after_action": "a" * 64,
            "oracle_decision_evaluated": str(item.oracle_evaluated),
            "decision_id": decision_by_prediction.get(item.prediction_index, ""),
            "selected_action": "A0_NO_OP",
            "accepted_update_after_prediction": "False",
            "evaluator_truth_revealed_after_action": "False",
            "permanent_holdout_used_for_choice": "False",
        }
        for item in plan
    ]
    oracle = [
        {
            "prediction_index": item.prediction_index,
            "decision_id": decision_by_prediction[item.prediction_index],
            "selected_action": "A0_NO_OP",
        }
        for item in plan
        if item.oracle_evaluated
    ]
    validation._validate_window_rows(windows, identity, events, oracle, pipeline)
    with pytest.raises(validation.RDX004RunArtifactError, match="full online-stream coverage"):
        validation._validate_window_rows(windows[:-1], identity, events, oracle, pipeline)


def test_smoke_rejects_exact_online_row_bound_tampering() -> None:
    identity = _identity(smoke=True)
    windows = _smoke_windows(identity)
    windows[0]["row_start"] = "1"
    with pytest.raises(validation.RDX004RunArtifactError, match="seven-window smoke coverage"):
        validation._validate_window_rows(
            windows,
            identity,
            _smoke_events(identity),
            [_smoke_oracle_record()],
            _pipeline(identity),
        )


@pytest.mark.parametrize(
    "action",
    [InterventionAction.HEAD_UPDATE, InterventionAction.FULL_FINE_TUNE],
)
def test_target_only_optimizer_candidates_require_the_exact_released_batch(
    action: InterventionAction,
) -> None:
    identity = _identity(smoke=True)
    target = tuple(range(1_000, 1_095))
    candidate = _intervention_candidate(identity, action, target=target)

    _validate_test_candidate(candidate, identity, action, target)

    tampered = deepcopy(candidate)
    raw_record = tampered["intervention_record"]
    assert isinstance(raw_record, dict)
    record = dict(raw_record)
    record["target_row_positions"] = json.dumps(target[:-1], separators=(",", ":"))
    tampered["intervention_record"] = record
    with pytest.raises(validation.RDX004RunArtifactError, match="semantics differ"):
        _validate_test_candidate(tampered, identity, action, target)


def test_replay_candidate_requires_all_and_only_historical_scope_rows() -> None:
    identity = _identity(smoke=True)
    target = tuple(range(2_000, 2_095))
    source_positions = list(range(400))
    prior_positions = list(range(1_000, 1_380))
    replay_scopes: list[dict[str, object]] = [
        {
            "scope_id": identity.rotation[0],
            "row_positions": source_positions,
            "row_positions_digest": row_positions_digest(source_positions),
        },
        {
            "scope_id": "opaque-stage-1",
            "row_positions": prior_positions,
            "row_positions_digest": row_positions_digest(prior_positions),
        },
    ]
    candidate = _intervention_candidate(
        identity,
        InterventionAction.REPLAY_UPDATE,
        target=target,
        replay_scopes=replay_scopes,
    )
    replay_scope_ids = {identity.rotation[0], "opaque-stage-1"}
    available_rows = [
        {"domain_id": item["scope_id"], "positions": item["row_positions"]}
        for item in replay_scopes
    ]
    _validate_test_candidate(
        candidate,
        identity,
        InterventionAction.REPLAY_UPDATE,
        target,
        replay_rows=780,
        replay_scope_ids=replay_scope_ids,
        replay_scoped_positions=available_rows,
    )

    tampered = deepcopy(candidate)
    tampered["historical_replay_positions_digest_available"] = "0" * 64
    with pytest.raises(validation.RDX004RunArtifactError, match="exact historical replay"):
        _validate_test_candidate(
            tampered,
            identity,
            InterventionAction.REPLAY_UPDATE,
            target,
            replay_rows=780,
            replay_scope_ids=replay_scope_ids,
            replay_scoped_positions=available_rows,
        )

    substituted = deepcopy(candidate)
    substituted_record = substituted["intervention_record"]
    assert isinstance(substituted_record, dict)
    substituted_scopes = deepcopy(replay_scopes)
    substituted_source = substituted_scopes[0]["row_positions"]
    assert isinstance(substituted_source, list)
    substituted_source[-1] = 999
    substituted_scopes[0]["row_positions_digest"] = row_positions_digest(substituted_source)
    substituted_flat = tuple(
        sorted(
            int(position)
            for scope in substituted_scopes
            for position in scope["row_positions"]  # type: ignore[union-attr]
        )
    )
    substituted_record["replay_row_positions"] = json.dumps(substituted_flat, separators=(",", ":"))
    substituted_record["replay_row_positions_digest"] = replay_flat_positions_digest(
        substituted_flat
    )
    substituted_record["replay_scoped_row_positions"] = json.dumps(
        substituted_scopes, sort_keys=True, separators=(",", ":")
    )
    substituted_record["replay_scoped_row_positions_digest"] = replay_scoped_positions_digest(
        substituted_scopes
    )
    substituted["historical_replay_positions_digest_consumed"] = substituted_record[
        "replay_scoped_row_positions_digest"
    ]
    substituted_available = [
        {"domain_id": item["scope_id"], "positions": item["row_positions"]}
        for item in substituted_scopes
    ]
    substituted["historical_replay_positions_digest_available"] = _canonical_digest(
        substituted_available
    )
    with pytest.raises(validation.RDX004RunArtifactError, match="exact historical replay"):
        _validate_test_candidate(
            substituted,
            identity,
            InterventionAction.REPLAY_UPDATE,
            target,
            replay_rows=780,
            replay_scope_ids=replay_scope_ids,
            replay_scoped_positions=available_rows,
        )


def test_smoke_requires_every_model_changing_candidate_attempt() -> None:
    identity = _identity(smoke=True)
    omitted = {
        "feasible": False,
        "execution_succeeded": False,
        "target_rows": 0,
        "replay_rows": 0,
        "optimizer_steps": 0,
        "target_positions_digest_consumed": None,
        "historical_replay_positions_digest_consumed": None,
        "intervention_record": None,
    }

    with pytest.raises(validation.RDX004RunArtifactError, match="required intervention attempt"):
        validation._validate_candidate_intervention(
            omitted,
            action=InterventionAction.HEAD_UPDATE,
            identity=identity,
            stage=2,
            domain=identity.rotation[2],
            window_id=1,
            available_target=tuple(range(100)),
            current_audit=set(),
            expected_replay_available=780,
            expected_replay_scope_ids={identity.rotation[0], "opaque-stage-1"},
        )


def test_candidate_memory_identity_rejects_replay_or_audit_digest_tampering() -> None:
    memory = validation._HistoricalMemoryEvidence(
        replay_scoped_positions=(),
        replay_rows=400,
        replay_digest="1" * 64,
        audit_scoped_positions=(),
        audit_rows=100,
        audit_digest="2" * 64,
        scope_ids=frozenset({"source"}),
    )
    candidate: dict[str, object] = {
        "historical_replay_rows_available": 400,
        "historical_replay_positions_digest_available": "1" * 64,
        "audit_rows_available": 100,
        "audit_positions_digest_available": "2" * 64,
    }
    validation._validate_candidate_memory_identity(candidate, memory)
    for field in (
        "historical_replay_positions_digest_available",
        "audit_positions_digest_available",
    ):
        tampered = dict(candidate)
        tampered[field] = "0" * 64
        with pytest.raises(validation.RDX004RunArtifactError, match="replay/audit identity"):
            validation._validate_candidate_memory_identity(tampered, memory)
