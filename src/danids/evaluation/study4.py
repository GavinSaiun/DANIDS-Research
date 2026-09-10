"""Artifact-only validation and aggregation for Study-4 E4 executions."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from danids.adaptation.actions import PolicyObservation
from danids.config.continual import FROZEN_ROTATIONS
from danids.config.study4 import STUDY4_CONFIRMATORY_SEEDS, Study4Method
from danids.continual.supervision import row_positions_digest
from danids.data.types import PredictionView
from danids.health.states import HealthState, classify_health
from danids.policy.allocation import validate_allocation_manifest
from danids.policy.artifacts import validate_core_run_artifacts
from danids.policy.health_artifact import (
    HEALTH_MODEL_FILENAME,
    HEALTH_MODEL_MANIFEST_FILENAME,
    POLICY_HEALTH_FEATURES,
    FrozenHealthModel,
    PolicyHealthVector,
    validate_health_model_artifact,
)
from danids.policy.query import (
    CORE_QUERY_BATCH_SIZE,
    CoreQuerySelection,
    QuerySelectionStatus,
    select_core_query,
)

STUDY4_RUN_ARTIFACT_VERSION = "task006-e4-run-v1"
STUDY4_EVALUATION_VERSION = "task006-e4-evaluation-v1"
CORE_BUNDLE_DIRNAME = "core_artifacts"
HEALTH_BUNDLE_DIRNAME = "health_artifact"

CONFIG_FILENAME = "config.resolved.yaml"
PROVENANCE_FILENAME = "provenance.json"
INITIAL_STATE_FILENAME = "initial_state_identity.json"
QUERY_FILENAME = "query_log.json"
DECISION_FILENAME = "decision_log.json"
INTERVENTION_FILENAME = "intervention_log.csv"
MEMORY_FILENAME = "replay_audit_manifests.json"
REFERENCE_FILENAME = "reference_state_history.json"
WINDOW_FILENAME = "window_metrics.csv"
HOLDOUT_FILENAME = "holdout_metrics.csv"
NATIVE_FILENAME = "native_attack_metrics.csv"
RESOURCE_FILENAME = "resource_metrics.json"
SUMMARY_FILENAME = "summary.json"
MANIFEST_FILENAME = "artifact_manifest.json"

_RUN_FILES = (
    CONFIG_FILENAME,
    PROVENANCE_FILENAME,
    INITIAL_STATE_FILENAME,
    f"{HEALTH_BUNDLE_DIRNAME}/{HEALTH_MODEL_MANIFEST_FILENAME}",
    f"{HEALTH_BUNDLE_DIRNAME}/{HEALTH_MODEL_FILENAME}",
    QUERY_FILENAME,
    DECISION_FILENAME,
    INTERVENTION_FILENAME,
    MEMORY_FILENAME,
    REFERENCE_FILENAME,
    WINDOW_FILENAME,
    HOLDOUT_FILENAME,
    NATIVE_FILENAME,
    RESOURCE_FILENAME,
    SUMMARY_FILENAME,
)


class Study4ArtifactError(ValueError):
    """Raised when persisted E4 evidence violates its scientific contract."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_digests(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != MANIFEST_FILENAME
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Study4ArtifactError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise Study4ArtifactError(f"{path.name} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as handle:
        if not columns:
            handle.write("\n")
            return
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _records(path: Path) -> list[dict[str, Any]]:
    if path.stat().st_size <= 2:
        return []
    frame = pd.read_csv(path)
    raw = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
    return [{str(key): value for key, value in row.items()} for row in raw]


def _as_bool(value: object, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    raise Study4ArtifactError(f"{name} must be boolean")


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise Study4ArtifactError(f"{name} must be an integer")
    try:
        if not isinstance(value, (str, int, float, np.integer, np.floating)):
            raise TypeError
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise Study4ArtifactError(f"{name} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise Study4ArtifactError(f"{name} must be an integer")
    return int(number)


def _as_float(value: object, name: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    try:
        if not isinstance(value, (str, int, float, np.integer, np.floating)):
            raise TypeError
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise Study4ArtifactError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        if optional and math.isnan(number):
            return None
        raise Study4ArtifactError(f"{name} must be finite")
    return number


def _close(left: object, right: float | None, *, atol: float = 1e-12) -> bool:
    if right is None:
        return left is None or (isinstance(left, float) and math.isnan(left))
    if not isinstance(left, (str, int, float, np.integer, np.floating)):
        return False
    try:
        return math.isclose(float(left), right, rel_tol=1e-12, abs_tol=atol)
    except ValueError:
        return False


def _validate_fingerprints(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"U", "T", "C", "B"}:
        raise Study4ArtifactError("Study-4 requires exactly U/T/C/B dataset fingerprints")
    result = {str(key): str(item) for key, item in value.items()}
    if any(len(item) != 64 for item in result.values()):
        raise Study4ArtifactError("Study-4 dataset fingerprints must be SHA-256 strings")
    try:
        for item in result.values():
            int(item, 16)
    except ValueError as exc:
        raise Study4ArtifactError("Study-4 dataset fingerprints must be SHA-256 strings") from exc
    if len(set(result.values())) != 4:
        raise Study4ArtifactError("Study-4 dataset fingerprints must be distinct")
    return result


def _partition_ranges(value: object, sequence: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != set(sequence):
        raise Study4ArtifactError("Study-4 partition provenance differs from its sequence")
    result: dict[str, dict[str, Any]] = {}
    for stage, domain in enumerate(sequence):
        raw = value[domain]
        if not isinstance(raw, dict):
            raise Study4ArtifactError("Study-4 partition range must be a mapping")
        required = {"initial_train", "validation", "online_stream", "permanent_holdout"}
        if set(raw) != required:
            raise Study4ArtifactError("Study-4 partition range fields differ")
        online = raw["online_stream"]
        holdout = raw["permanent_holdout"]
        if stage == 0:
            if online is not None or raw["initial_train"] is None or raw["validation"] is None:
                raise Study4ArtifactError("Study-4 source partition roles are invalid")
        elif online is None or raw["initial_train"] is not None or raw["validation"] is not None:
            raise Study4ArtifactError("Study-4 later partition roles are invalid")
        if not isinstance(holdout, list) or len(holdout) != 2:
            raise Study4ArtifactError("Study-4 permanent holdout range is invalid")
        if online is not None:
            if not isinstance(online, list) or len(online) != 2:
                raise Study4ArtifactError("Study-4 online stream range is invalid")
            if _as_int(online[1], "online stop") != _as_int(holdout[0], "holdout start"):
                raise Study4ArtifactError("Study-4 online stream overlaps its permanent holdout")
        result[domain] = raw
    return result


def _window_expected_health(
    row: Mapping[str, Any],
) -> tuple[HealthState, float | None, float | None]:
    assessment = classify_health(
        false_positives=_as_int(row["fp"], "window fp"),
        benign_support=_as_int(row["benign_count"], "window benign_count"),
        true_positives=_as_int(row["tp"], "window tp"),
        attack_support=_as_int(row["attack_count"], "window attack_count"),
        alpha=0.001,
        recall_floor=float(row["eval_recall_floor"]),
        confidence=0.95,
    )
    return assessment.state, assessment.fpr_low, assessment.tpr_high


def _validate_metric_counts(row: Mapping[str, Any], context: str) -> None:
    count = _as_int(row["row_count"], f"{context} row_count")
    attack = _as_int(row["attack_count"], f"{context} attack_count")
    benign = _as_int(row["benign_count"], f"{context} benign_count")
    tp = _as_int(row["tp"], f"{context} tp")
    fp = _as_int(row["fp"], f"{context} fp")
    tn = _as_int(row["tn"], f"{context} tn")
    fn = _as_int(row["fn"], f"{context} fn")
    if min(count, attack, benign, tp, fp, tn, fn) < 0:
        raise Study4ArtifactError(f"{context} contains negative counts")
    if attack + benign != count or tp + fn != attack or fp + tn != benign:
        raise Study4ArtifactError(f"{context} confusion counts are inconsistent")
    expected_tpr = None if attack == 0 else tp / attack
    expected_fpr = None if benign == 0 else fp / benign
    if not _close(row.get("tpr"), expected_tpr) or not _close(row.get("fpr"), expected_fpr):
        raise Study4ArtifactError(f"{context} operating metrics differ from confusion counts")
    if expected_fpr is not None and not _close(row.get("fpr_budget_ratio"), expected_fpr / 0.001):
        raise Study4ArtifactError(f"{context} FPR budget ratio is inconsistent")
    for name in ("pr_auc", "roc_auc"):
        value = _as_float(row.get(name), f"{context} {name}", optional=True)
        if value is not None and not 0.0 <= value <= 1.0:
            raise Study4ArtifactError(f"{context} {name} lies outside [0, 1]")


def _dummy_prediction(start: int, stop: int) -> PredictionView:
    positions = np.arange(start, stop, dtype=np.int64)
    return PredictionView(
        np.zeros((len(positions), 1), dtype=np.float32),
        pd.DataFrame({"timestamp": positions}),
        positions,
        ("placeholder",),
    )


def _validate_queries(
    payload: dict[str, Any],
    windows: list[dict[str, Any]],
    partitions: dict[str, dict[str, Any]],
    method: Study4Method,
) -> tuple[int, int, int]:
    if set(payload) != {"version", "scopes"} or payload["version"] != STUDY4_RUN_ARTIFACT_VERSION:
        raise Study4ArtifactError("Study-4 query log schema/version is invalid")
    scopes = payload["scopes"]
    if not isinstance(scopes, list):
        raise Study4ArtifactError("Study-4 query scopes must be a list")
    if method is Study4Method.STATIC and scopes:
        raise Study4ArtifactError("STATIC must not contain supervision scopes")
    window_by_index = {_as_int(row["prediction_index"], "prediction index"): row for row in windows}
    seen_scope: set[str] = set()
    selected_count = 0
    released_count = 0
    infeasible_count = 0
    for scope in scopes:
        if not isinstance(scope, dict) or set(scope) != {
            "stage",
            "current_domain",
            "opaque_scope_token",
            "supervision",
            "allocation",
            "query_events",
        }:
            raise Study4ArtifactError("Study-4 query scope schema is invalid")
        token = str(scope["opaque_scope_token"])
        if token in seen_scope or len(token) != 64:
            raise Study4ArtifactError("Study-4 query scope identity is duplicate or invalid")
        seen_scope.add(token)
        domain = str(scope["current_domain"])
        stage = _as_int(scope["stage"], "query stage")
        if stage <= 0 or domain not in partitions:
            raise Study4ArtifactError("Study-4 query scope route is invalid")
        events = scope["query_events"]
        if not isinstance(events, list):
            raise Study4ArtifactError("Study-4 query events must be a list")
        successful_ordinals: list[int] = []
        for event in events:
            if not isinstance(event, dict) or set(event) != {"prediction_index", "selection"}:
                raise Study4ArtifactError("Study-4 query event schema is invalid")
            index = _as_int(event["prediction_index"], "query prediction index")
            row = window_by_index.get(index)
            if row is None or str(row["current_domain"]) != domain:
                raise Study4ArtifactError("Study-4 query does not belong to its current window")
            raw = event["selection"]
            if not isinstance(raw, dict):
                raise Study4ArtifactError("Study-4 query selection must be a mapping")
            try:
                selection = CoreQuerySelection(
                    selector_version=str(raw["selector_version"]),
                    seed=_as_int(raw["seed"], "query seed"),
                    opaque_scope_token=str(raw["opaque_scope_token"]),
                    window_id=_as_int(raw["window_id"], "query window"),
                    query_ordinal=_as_int(raw["query_ordinal"], "query ordinal"),
                    current_row_positions_digest=str(raw["current_row_positions_digest"]),
                    candidate_count=_as_int(raw["candidate_count"], "query candidates"),
                    status=QuerySelectionStatus(str(raw["status"])),
                    selected_positions=tuple(int(value) for value in raw["selected_positions"]),
                    selected_positions_digest=raw["selected_positions_digest"],
                    reason=str(raw["reason"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise Study4ArtifactError(f"Study-4 query selection is invalid: {exc}") from exc
            if selection.opaque_scope_token != token or selection.seed != int(row["seed"]):
                raise Study4ArtifactError("Study-4 query selection identity differs from its run")
            expected = select_core_query(
                _dummy_prediction(int(row["row_start"]), int(row["row_stop"])),
                seed=selection.seed,
                opaque_scope_token=token,
                window_id=int(row["window_id"]),
                query_ordinal=selection.query_ordinal,
            )
            if selection != expected:
                raise Study4ArtifactError("Study-4 query differs from deterministic recomputation")
            if selection.status is QuerySelectionStatus.SELECTED:
                successful_ordinals.append(selection.query_ordinal)
                selected_count += CORE_QUERY_BATCH_SIZE
                online = partitions[domain]["online_stream"]
                assert isinstance(online, list)
                if any(
                    not int(online[0]) <= pos < int(online[1])
                    for pos in selection.selected_positions
                ):
                    raise Study4ArtifactError("Study-4 query escaped the ONLINE_STREAM partition")
                holdout = partitions[domain]["permanent_holdout"]
                if any(
                    int(holdout[0]) <= pos < int(holdout[1]) for pos in selection.selected_positions
                ):
                    raise Study4ArtifactError("Study-4 query contains permanent-holdout rows")
            else:
                infeasible_count += 1
        if successful_ordinals != list(range(len(successful_ordinals))):
            raise Study4ArtifactError("Study-4 successful query ordinals are not contiguous")
        if len(successful_ordinals) > 4:
            raise Study4ArtifactError("Study-4 query budget exceeds four events")
        allocation = scope["allocation"]
        if not isinstance(allocation, dict):
            raise Study4ArtifactError("Study-4 allocation manifest must be a mapping")
        validate_allocation_manifest(allocation)
        if allocation["scope_id"] != token:
            raise Study4ArtifactError("Study-4 allocation belongs to another supervision scope")
        if int(allocation["release_count"]) > len(successful_ordinals):
            raise Study4ArtifactError("Study-4 releases exceed successful delayed queries")
        released_count += int(allocation["used_label_budget"])
        supervision = scope["supervision"]
        if not isinstance(supervision, dict) or supervision.get("opaque_scope_token") != token:
            raise Study4ArtifactError("Study-4 supervision manifest identity is invalid")
        delayed = supervision.get("delayed_queue")
        if not isinstance(delayed, dict):
            raise Study4ArtifactError("Study-4 delayed queue manifest is absent")
        delayed_queries = delayed.get("queries", [])
        if not isinstance(delayed_queries, list):
            raise Study4ArtifactError("Study-4 delayed query provenance is invalid")
        selected_events = [
            (int(event["prediction_index"]), event["selection"])
            for event in events
            if event["selection"]["status"] == QuerySelectionStatus.SELECTED.value
        ]
        if len(delayed_queries) != len(selected_events):
            raise Study4ArtifactError(
                "Study-4 delayed queue differs from successful query selections"
            )
        for query, (prediction_index, selection) in zip(
            delayed_queries, selected_events, strict=True
        ):
            if not isinstance(query, dict):
                raise Study4ArtifactError("Study-4 delayed query provenance is invalid")
            if int(query["release_window"]) != int(query["query_window"]) + 1:
                raise Study4ArtifactError("Study-4 delayed labels violate one-prediction delay")
            if (
                query.get("scope_token") != token
                or int(query["query_window"]) != prediction_index
                or tuple(int(value) for value in query["row_positions"])
                != tuple(int(value) for value in selection["selected_positions"])
                or query.get("row_positions_digest") != selection.get("selected_positions_digest")
            ):
                raise Study4ArtifactError(
                    "Study-4 delayed queue differs from its deterministic query"
                )
        releases = allocation.get("releases")
        if not isinstance(releases, list) or len(releases) != int(allocation["release_count"]):
            raise Study4ArtifactError("Study-4 allocation release provenance is invalid")
        for release, query in zip(releases, delayed_queries, strict=False):
            if (
                tuple(int(value) for value in release["queried_positions"])
                != tuple(int(value) for value in query["row_positions"])
                or release.get("queried_positions_digest") != query.get("row_positions_digest")
                or int(release["query_window"]) != int(query["query_window"])
                or int(release["release_window"]) != int(query["release_window"])
            ):
                raise Study4ArtifactError(
                    "Study-4 allocation used future, unreleased, or different query rows"
                )
    return selected_count, released_count, infeasible_count


def _validate_windows(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    provenance: dict[str, Any],
    frozen: FrozenHealthModel,
    partitions: dict[str, dict[str, Any]],
) -> tuple[int, int, int]:
    if not rows:
        raise Study4ArtifactError("Study-4 run has no chronological online windows")
    sequence = tuple(str(value) for value in provenance["sequence"])
    expected_index = 0
    per_domain_next: dict[str, int] = {domain: 0 for domain in sequence[1:]}
    unsafe = missed = false_alarm = 0
    for row in rows:
        index = _as_int(row["prediction_index"], "prediction_index")
        if index != expected_index:
            raise Study4ArtifactError("Study-4 windows are missing, duplicate, or nonchronological")
        expected_index += 1
        domain = str(row["current_domain"])
        stage = _as_int(row["stage"], "window stage")
        if stage <= 0 or stage >= len(sequence) or sequence[stage] != domain:
            raise Study4ArtifactError("Study-4 administrative stage/domain route is invalid")
        window_id = _as_int(row["window_id"], "window_id")
        if per_domain_next[domain] != window_id:
            raise Study4ArtifactError("Study-4 domain windows are missing or out of order")
        per_domain_next[domain] += 1
        start = _as_int(row["row_start"], "window row_start")
        stop = _as_int(row["row_stop"], "window row_stop")
        online = partitions[domain]["online_stream"]
        holdout = partitions[domain]["permanent_holdout"]
        assert isinstance(online, list) and isinstance(holdout, list)
        expected_start = int(online[0]) + window_id * int(config["window_size"])
        if start != expected_start or not start < stop <= int(online[1]):
            raise Study4ArtifactError("Study-4 window row range is not the canonical chronology")
        expected_stop = min(start + int(config["window_size"]), int(online[1]))
        if stop != expected_stop or stop > int(holdout[0]):
            raise Study4ArtifactError("Study-4 window is missing rows or enters permanent holdout")
        _validate_metric_counts(row, f"window {index}")
        vector = PolicyHealthVector.from_mapping(
            {name: row.get(name) for name in POLICY_HEALTH_FEATURES}
        )
        health = frozen.decide(vector)
        if not _close(row["harm_probability"], health.harm_probability, atol=1e-15):
            raise Study4ArtifactError("Study-4 health probability differs from frozen model")
        if str(row["predicted_health_state"]) != health.predicted_state.value:
            raise Study4ArtifactError(
                "Study-4 predicted health state differs from frozen thresholds"
            )
        evaluator_state, fpr_low, tpr_high = _window_expected_health(row)
        if str(row["evaluator_health_state"]) != evaluator_state.value:
            raise Study4ArtifactError("Study-4 evaluator health differs from Wilson recomputation")
        if not _close(row.get("eval_fpr_low"), fpr_low) or not _close(
            row.get("eval_tpr_high"), tpr_high
        ):
            raise Study4ArtifactError("Study-4 evaluator confidence bounds differ")
        expected_unsafe = evaluator_state is HealthState.HARMFUL
        expected_missed = expected_unsafe and health.predicted_state.value != "PREDICTED_HARMFUL"
        expected_false = evaluator_state is HealthState.SAFE and health.predicted_state.value == (
            "PREDICTED_HARMFUL"
        )
        if _as_bool(row["unsafe_exposure"], "unsafe exposure") != expected_unsafe:
            raise Study4ArtifactError("Study-4 unsafe-exposure flag is inconsistent")
        if _as_bool(row["missed_harmful_window"], "missed harm") != expected_missed:
            raise Study4ArtifactError("Study-4 missed-harm flag is inconsistent")
        if _as_bool(row["false_health_alarm"], "false health alarm") != expected_false:
            raise Study4ArtifactError("Study-4 false-health-alarm flag is inconsistent")
        unsafe += int(expected_unsafe)
        missed += int(expected_missed)
        false_alarm += int(expected_false)
    smoke = _as_bool(config["smoke"], "smoke")
    limits = config.get("smoke_limits")
    for stage, domain in enumerate(sequence[1:], start=1):
        online = partitions[domain]["online_stream"]
        assert isinstance(online, list)
        full_count = math.ceil((int(online[1]) - int(online[0])) / int(config["window_size"]))
        expected = full_count
        if smoke:
            if not isinstance(limits, dict):
                raise Study4ArtifactError("Study-4 smoke run lacks smoke limits")
            expected = (
                min(full_count, int(limits["later_windows"]))
                if stage <= int(limits["later_stages"])
                else 0
            )
        if per_domain_next[domain] != expected:
            raise Study4ArtifactError("Study-4 run has incomplete or excess domain windows")
    return unsafe, missed, false_alarm


def _validate_interventions(
    rows: list[dict[str, Any]], windows: list[dict[str, Any]], method: Study4Method
) -> tuple[Counter[str], int, int, int, float]:
    by_window = {_as_int(row["prediction_index"], "prediction index"): row for row in windows}
    action_counts: Counter[str] = Counter()
    accepted = rejected = optimizer_steps = 0
    elapsed = 0.0
    previous_after: str | None = None
    seen: set[tuple[int, str, str]] = set()
    for row in rows:
        prediction_index = _as_int(row["prediction_index"], "intervention prediction index")
        window = by_window.get(prediction_index)
        if window is None:
            raise Study4ArtifactError("Study-4 intervention has no matching prediction")
        action = str(row["action_attempted"])
        decision_id = str(row["decision_id"])
        key = (prediction_index, action, decision_id)
        if key in seen:
            raise Study4ArtifactError("Study-4 intervention log contains a duplicate action")
        seen.add(key)
        action_counts[action] += 1
        before = str(row["model_digest_before"])
        after = str(row["model_digest_after"])
        candidate = str(row["model_digest_candidate"])
        if previous_after is not None and before != previous_after:
            raise Study4ArtifactError("Study-4 intervention model state chain is inconsistent")
        if before != str(window["model_digest_at_prediction"]):
            # A same-window fallback starts from the same incoming state after rollback.
            same_window_prior = [
                prior
                for prior in rows
                if prior is not row
                and _as_int(prior["prediction_index"], "prediction index") == prediction_index
            ]
            if not same_window_prior or before != str(same_window_prior[0]["model_digest_before"]):
                raise Study4ArtifactError("Study-4 intervention did not start from deployed state")
        is_accepted = str(row["accepted"]).casefold() == "true"
        rolled_back = str(row["rolled_back"]).casefold() == "true"
        if is_accepted == rolled_back:
            raise Study4ArtifactError("Study-4 accept/rollback flags are inconsistent")
        if is_accepted:
            if after != candidate:
                raise Study4ArtifactError("Study-4 accepted intervention did not promote candidate")
            accepted += 1
            previous_after = after
        else:
            if after != before:
                raise Study4ArtifactError("Study-4 rejected intervention did not roll back exactly")
            rejected += 1
            previous_after = before
        if str(row["preprocessor_digest_before"]) != str(row["preprocessor_digest_after"]):
            raise Study4ArtifactError("Study-4 intervention changed frozen preprocessing")
        if action in {"A2_HEAD_UPDATE", "A4_REPLAY_UPDATE"} and str(
            row["threshold_digest_before"]
        ) != str(row["threshold_digest_after"]):
            raise Study4ArtifactError("Study-4 A2/A4 changed the frozen threshold")
        target_rows = _as_int(row["target_rows"], "intervention target rows")
        if target_rows <= 0 or target_rows > 80 or target_rows % 20:
            raise Study4ArtifactError("Study-4 intervention target is not released 20-row evidence")
        if action == "A4_REPLAY_UPDATE" and _as_int(row["replay_rows"], "replay rows") <= 0:
            raise Study4ArtifactError("Study-4 A4 lacks historical replay")
        if action == "A2_HEAD_UPDATE" and _as_int(row["replay_rows"], "replay rows") != 0:
            raise Study4ArtifactError("Study-4 A2 consumed replay")
        try:
            visible = json.loads(str(row["policy_health_information"]))
            if not isinstance(visible, dict):
                raise TypeError
            PolicyObservation.from_mapping(
                visible,
                remaining_label_budget=_as_int(
                    row["remaining_label_budget"], "remaining label budget"
                ),
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise Study4ArtifactError(
                "Study-4 policy observation contains evaluator-only information"
            ) from exc
        optimizer_steps += _as_int(row["optimizer_steps"], "optimizer steps")
        elapsed += float(row["wall_clock_update_seconds"])
    if method is Study4Method.STATIC and rows:
        raise Study4ArtifactError("STATIC must not contain interventions")
    if method is Study4Method.ALWAYS_ADAPT and any(
        action != "A4_REPLAY_UPDATE" for action in action_counts
    ):
        raise Study4ArtifactError("ALWAYS_ADAPT may attempt only the frozen A4 action")
    return action_counts, accepted, rejected, optimizer_steps, elapsed


def _validate_holdouts(
    rows: list[dict[str, Any]],
    *,
    sequence: tuple[str, ...],
    windows: list[dict[str, Any]],
    accepted_updates: int,
) -> None:
    if not rows:
        raise Study4ArtifactError("Study-4 run lacks permanent-holdout evaluations")
    events: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        _validate_metric_counts(row, "Study-4 holdout")
        domain = str(row["holdout_dataset_id"])
        if domain not in sequence:
            raise Study4ArtifactError("Study-4 holdout uses an unknown domain")
        if str(row["partition_kind"]) != "permanent_holdout":
            raise Study4ArtifactError("Study-4 holdout artifact is not evaluator-only")
        event_index = _as_int(row["event_index"], "holdout event index")
        events.setdefault((event_index, str(row["event"])), []).append(row)
        compliance = str(row["operating_envelope_state"])
        if compliance not in {state.value for state in HealthState}:
            raise Study4ArtifactError("Study-4 holdout operating-envelope state is invalid")
        learned_tpr = _as_float(row.get("learned_reference_tpr"), "learned tpr", optional=True)
        forgetting_tpr = _as_float(
            row.get("operational_tpr_forgetting"), "TPR forgetting", optional=True
        )
        if learned_tpr is None:
            if forgetting_tpr is not None:
                raise Study4ArtifactError("Study-4 holdout forgetting lacks a learned reference")
        else:
            current = _as_float(row.get("tpr"), "holdout tpr", optional=True)
            expected = None if current is None else learned_tpr - current
            if not _close(row.get("operational_tpr_forgetting"), expected):
                raise Study4ArtifactError("Study-4 operational forgetting is inconsistent")
    ordered = sorted(events)
    if not ordered or ordered[0][1] != "source_initial" or ordered[-1][1] != "final":
        raise Study4ArtifactError("Study-4 holdout timing lacks initial or final evaluation")
    update_events = sum(event == "post_accept" for _, event in ordered)
    if update_events != accepted_updates:
        raise Study4ArtifactError("Study-4 holdout timing differs from accepted updates")
    domain_end = {
        int(rows_[0]["stage"]) for (_, event), rows_ in events.items() if event == "domain_end"
    }
    expected_stages = {int(row["stage"]) for row in windows}
    if domain_end != expected_stages:
        raise Study4ArtifactError("Study-4 holdouts were not evaluated after every domain")


def _derived_resource(
    *,
    windows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    query_counts: tuple[int, int, int],
    intervention_result: tuple[Counter[str], int, int, int, float],
    memory: dict[str, Any],
) -> dict[str, Any]:
    requested, released, infeasible = query_counts
    actions, accepted, rejected, optimizer_steps, elapsed = intervention_result
    unsafe = sum(str(row["unsafe_exposure"]).casefold() == "true" for row in windows)
    missed = sum(str(row["missed_harmful_window"]).casefold() == "true" for row in windows)
    false_alarm = sum(str(row["false_health_alarm"]).casefold() == "true" for row in windows)
    decision_actions = Counter(str(row["action"]) for row in decisions)
    final_memory = memory.get("final_memory")
    if not isinstance(final_memory, dict):
        raise Study4ArtifactError("Study-4 final memory manifest is absent")
    return {
        "labels_requested": requested,
        "labels_released": released,
        "labels_consumed": released,
        "query_events": requested // CORE_QUERY_BATCH_SIZE,
        "query_infeasible_events": infeasible,
        "a0_decisions": decision_actions["A0_NO_OP"],
        "a1_attempts": actions["A1_RECALIBRATE"],
        "a2_attempts": actions["A2_HEAD_UPDATE"],
        "a3_attempts": actions["A3_FULL_FINE_TUNE"],
        "a4_attempts": actions["A4_REPLAY_UPDATE"],
        "accepted_updates": accepted,
        "rejected_updates": rejected,
        "rollbacks": rejected,
        "head_updates": actions["A2_HEAD_UPDATE"],
        "replay_updates": actions["A4_REPLAY_UPDATE"],
        "optimizer_steps": optimizer_steps,
        "wall_clock_adaptation_seconds": elapsed,
        "replay_bytes": _as_int(final_memory["replay_bytes"], "replay bytes"),
        "audit_bytes": _as_int(final_memory["audit_bytes"], "audit bytes"),
        "unsafe_exposure_windows": unsafe,
        "missed_harmful_windows": missed,
        "false_health_alarms": false_alarm,
        "unresolved_unsafe_windows": sum(
            str(row["unresolved_unsafe"]).casefold() == "true" for row in windows
        ),
        "operating_envelope_violations": unsafe,
    }


@dataclass(frozen=True, slots=True)
class ValidatedStudy4Run:
    path: Path
    experiment_id: str
    method: Study4Method
    sequence: tuple[str, ...]
    seed: int
    smoke: bool
    contract_digest: str
    source_checkpoint_sha256: str
    summary: dict[str, Any]
    windows: list[dict[str, Any]]
    holdouts: list[dict[str, Any]]
    interventions: list[dict[str, Any]]


def validate_study4_run(path: str | Path, *, allow_smoke: bool = False) -> ValidatedStudy4Run:
    """Validate a complete E4 run using only its immutable artifact bundle."""

    root = Path(path).resolve()
    manifest = _load_json(root / MANIFEST_FILENAME)
    if set(manifest) != {"version", "files", "bundle_digest"}:
        raise Study4ArtifactError("Study-4 artifact manifest schema is invalid")
    if manifest["version"] != STUDY4_RUN_ARTIFACT_VERSION:
        raise Study4ArtifactError("unsupported Study-4 run artifact version")
    actual_files = _tree_digests(root)
    if manifest["files"] != actual_files or manifest["bundle_digest"] != _digest(actual_files):
        raise Study4ArtifactError("Study-4 artifact digest manifest differs from files")
    if not set(_RUN_FILES).issubset(actual_files):
        raise Study4ArtifactError("Study-4 run artifact file set is incomplete")

    config_raw = yaml.safe_load((root / CONFIG_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(config_raw, dict):
        raise Study4ArtifactError("Study-4 resolved config must be a mapping")
    config: dict[str, Any] = config_raw
    provenance = _load_json(root / PROVENANCE_FILENAME)
    initial = _load_json(root / INITIAL_STATE_FILENAME)
    summary = _load_json(root / SUMMARY_FILENAME)
    resources = _load_json(root / RESOURCE_FILENAME)
    queries = _load_json(root / QUERY_FILENAME)
    decisions_payload = _load_json(root / DECISION_FILENAME)
    memory = _load_json(root / MEMORY_FILENAME)
    references = _load_json(root / REFERENCE_FILENAME)
    windows = _records(root / WINDOW_FILENAME)
    holdouts = _records(root / HOLDOUT_FILENAME)
    interventions = _records(root / INTERVENTION_FILENAME)

    try:
        method = Study4Method(str(provenance["method"]))
    except (KeyError, ValueError) as exc:
        raise Study4ArtifactError("Study-4 method is invalid") from exc
    if method not in {Study4Method.STATIC, Study4Method.ALWAYS_ADAPT, Study4Method.DANIDS_CORE}:
        raise Study4ArtifactError("Study-4 run uses an unimplemented method")
    sequence = tuple(str(value) for value in provenance["sequence"])
    seed = _as_int(provenance["seed"], "Study-4 seed")
    smoke = _as_bool(config.get("smoke"), "smoke")
    if smoke and not allow_smoke:
        raise Study4ArtifactError("smoke Study-4 runs cannot enter confirmatory validation")
    if sequence not in FROZEN_ROTATIONS or seed not in STUDY4_CONFIRMATORY_SEEDS:
        raise Study4ArtifactError("Study-4 rotation/seed is outside the frozen matrix")
    expected_id = f"E4_{method.value}_{'-'.join(sequence)}_s{seed}"
    for item in (config, provenance, summary):
        if item.get("experiment_id") != expected_id:
            raise Study4ArtifactError("Study-4 experiment identity is inconsistent")
    if config.get("method") != method.value or tuple(config.get("sequence", ())) != sequence:
        raise Study4ArtifactError("Study-4 resolved method/sequence differs from provenance")
    if _as_int(config.get("seed"), "config seed") != seed:
        raise Study4ArtifactError("Study-4 resolved seed differs from provenance")
    fingerprints = _validate_fingerprints(provenance.get("dataset_fingerprints"))
    partitions = _partition_ranges(provenance.get("manifest_partition_ranges"), sequence)
    required_contract = {
        "dataset_fingerprints": fingerprints,
        "feature_contract_version": provenance.get("feature_contract_version"),
        "feature_columns": provenance.get("feature_columns"),
        "split_version": provenance.get("split_version"),
        "materializer_version": provenance.get("materializer_version"),
        "preprocessor_version": provenance.get("preprocessor_version"),
        "window_size": provenance.get("window_size"),
        "boundary_mode": provenance.get("boundary_mode"),
        "health_artifact_identity": provenance.get("health_artifact_identity"),
        "health_model_sha256": provenance.get("health_model_sha256"),
        "health_thresholds_digest": provenance.get("health_thresholds_digest"),
        "health_feature_contract_digest": provenance.get("health_feature_contract_digest"),
        "always_adapt_semantics": config.get("always_adapt"),
    }
    contract_digest = _digest(required_contract)
    if provenance.get("scientific_contract_digest") != contract_digest:
        raise Study4ArtifactError("Study-4 scientific contract digest is invalid")

    frozen = validate_health_model_artifact(root / HEALTH_BUNDLE_DIRNAME)
    if frozen.artifact_identity != provenance.get(
        "health_artifact_identity"
    ) or frozen.serialized_model_sha256 != provenance.get("health_model_sha256"):
        raise Study4ArtifactError("Study-4 frozen health artifact identity differs")
    if _digest(frozen.manifest["calibration"]) != provenance.get("health_thresholds_digest"):
        raise Study4ArtifactError("Study-4 frozen health thresholds differ from provenance")
    checkpoint = str(initial.get("source_checkpoint_sha256"))
    if checkpoint != provenance.get("source_checkpoint_sha256"):
        raise Study4ArtifactError("Study-4 imported source checkpoint identity differs")
    if (
        initial.get("source_checkpoint_sha256_before") != checkpoint
        or initial.get("source_checkpoint_sha256_after") != checkpoint
    ):
        raise Study4ArtifactError("Study-4 source checkpoint changed during execution")
    if initial.get("preprocessor_digest_before") != initial.get("preprocessor_digest_after"):
        raise Study4ArtifactError("Study-4 source preprocessor changed during execution")
    if initial.get("threshold_digest_before") != initial.get("threshold_digest_after"):
        raise Study4ArtifactError("Study-4 source threshold changed during execution")
    if method is Study4Method.STATIC and initial.get("model_digest_before") != initial.get(
        "model_digest_after"
    ):
        raise Study4ArtifactError("STATIC changed the imported detector model")

    unsafe, missed, false_alarm = _validate_windows(
        windows,
        config=config,
        provenance=provenance,
        frozen=frozen,
        partitions=partitions,
    )
    if set(decisions_payload) != {"version", "records"} or not isinstance(
        decisions_payload["records"], list
    ):
        raise Study4ArtifactError("Study-4 decision log schema is invalid")
    decisions = decisions_payload["records"]
    if method is Study4Method.DANIDS_CORE:
        if not (root / CORE_BUNDLE_DIRNAME).is_dir():
            raise Study4ArtifactError("DANIDS_CORE run lacks its strict Core artifact bundle")
        core_summary = validate_core_run_artifacts(root / CORE_BUNDLE_DIRNAME)
        if int(core_summary["policy_window_count"]) != len(windows):
            raise Study4ArtifactError("Core artifact window count differs from E4 execution")
    elif (root / CORE_BUNDLE_DIRNAME).exists():
        raise Study4ArtifactError("non-Core Study-4 method contains a Core policy bundle")
    query_counts = _validate_queries(queries, windows, partitions, method)
    intervention_result = _validate_interventions(interventions, windows, method)
    _validate_holdouts(
        holdouts,
        sequence=sequence,
        windows=windows,
        accepted_updates=intervention_result[1],
    )
    if set(memory) != {"version", "source_selection", "scopes", "final_memory"}:
        raise Study4ArtifactError("Study-4 replay/audit memory artifact schema is invalid")
    final_memory = memory["final_memory"]
    if not isinstance(final_memory, dict):
        raise Study4ArtifactError("Study-4 final memory is invalid")
    replay_scopes = final_memory.get("replay_domains")
    audit_scopes = final_memory.get("audit_domains")
    if not isinstance(replay_scopes, list) or not isinstance(audit_scopes, list):
        raise Study4ArtifactError("Study-4 final replay/audit manifests are invalid")
    scope_domains = {
        str(scope["opaque_scope_token"]): str(scope["current_domain"])
        for scope in queries["scopes"]
    }
    for replay in replay_scopes:
        matching = [item for item in audit_scopes if item["domain_id"] == replay["domain_id"]]
        if len(matching) != 1:
            raise Study4ArtifactError("Study-4 replay/audit scopes are not paired")
        if set(replay["row_positions"]).intersection(matching[0]["row_positions"]):
            raise Study4ArtifactError("Study-4 replay and audit rows overlap")
    for item in [*replay_scopes, *audit_scopes]:
        scope_id = str(item["domain_id"])
        domain = sequence[0] if scope_id == sequence[0] else scope_domains.get(scope_id)
        if domain is None:
            raise Study4ArtifactError("Study-4 memory belongs to an unknown supervision scope")
        range_name = "initial_train" if domain == sequence[0] else "online_stream"
        allowed = partitions[domain][range_name]
        holdout = partitions[domain]["permanent_holdout"]
        assert isinstance(allowed, list) and isinstance(holdout, list)
        positions = tuple(int(value) for value in item["row_positions"])
        if (
            len(positions) != len(set(positions))
            or any(not int(allowed[0]) <= value < int(allowed[1]) for value in positions)
            or any(int(holdout[0]) <= value < int(holdout[1]) for value in positions)
            or item.get("row_positions_digest") != row_positions_digest(positions)
        ):
            raise Study4ArtifactError("Study-4 replay/audit memory escaped its eligible partition")
    if not isinstance(references.get("states"), list) or not references["states"]:
        raise Study4ArtifactError("Study-4 R1 reference history is absent")

    expected_resource = _derived_resource(
        windows=windows,
        decisions=decisions,
        query_counts=query_counts,
        intervention_result=intervention_result,
        memory=memory,
    )
    for key, expected in expected_resource.items():
        actual = resources.get(key)
        if isinstance(expected, float):
            if not _close(actual, expected, atol=1e-9):
                raise Study4ArtifactError(f"Study-4 resource {key} differs from recomputation")
        elif actual != expected:
            raise Study4ArtifactError(f"Study-4 resource {key} differs from recomputation")
    expected_summary = {
        "artifact_version": STUDY4_RUN_ARTIFACT_VERSION,
        "status": "complete",
        "experiment_id": expected_id,
        "method": method.value,
        "sequence": list(sequence),
        "seed": seed,
        "smoke": smoke,
        "window_count": len(windows),
        "holdout_evaluation_rows": len(holdouts),
        "unsafe_exposure_windows": unsafe,
        "missed_harmful_windows": missed,
        "false_health_alarms": false_alarm,
        "labels_requested": expected_resource["labels_requested"],
        "labels_released": expected_resource["labels_released"],
        "accepted_updates": expected_resource["accepted_updates"],
        "rejected_updates": expected_resource["rejected_updates"],
        "final_model_digest": initial["model_digest_after"],
        "source_state_reused": True,
        "preprocessor_unchanged": True,
        "threshold_unchanged": True,
    }
    if summary != expected_summary:
        raise Study4ArtifactError("Study-4 persisted summary differs from recomputation")
    return ValidatedStudy4Run(
        root,
        expected_id,
        method,
        sequence,
        seed,
        smoke,
        contract_digest,
        checkpoint,
        summary,
        windows,
        holdouts,
        interventions,
    )


def write_study4_run_manifest(output_dir: str | Path) -> Path:
    """Seal an already-written run directory and independently validate it."""

    root = Path(output_dir)
    if (root / MANIFEST_FILENAME).exists():
        raise FileExistsError(f"refusing to overwrite Study-4 artifact manifest: {root}")
    files = _tree_digests(root)
    if not set(_RUN_FILES).issubset(files):
        raise Study4ArtifactError("cannot seal an incomplete Study-4 run")
    payload = {
        "version": STUDY4_RUN_ARTIFACT_VERSION,
        "files": files,
        "bundle_digest": _digest(files),
    }
    _write_json(root / MANIFEST_FILENAME, payload)
    return root / MANIFEST_FILENAME


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(np.mean(values))


def _study4_outputs(
    runs: Sequence[ValidatedStudy4Run],
) -> dict[str, list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    holdouts: list[dict[str, Any]] = []
    interventions: list[dict[str, Any]] = []
    for run in runs:
        context = {
            "experiment_id": run.experiment_id,
            "method": run.method.value,
            "sequence": "-".join(run.sequence),
            "seed": run.seed,
        }
        run_rows.append({**run.summary, **context})
        windows.extend({**context, **row} for row in run.windows)
        holdouts.extend({**context, **row} for row in run.holdouts)
        interventions.extend({**context, **row} for row in run.interventions)

    method_summary: list[dict[str, Any]] = []
    safety: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    rollbacks: list[dict[str, Any]] = []
    retention: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in run_rows})
    for method in methods:
        selected = [row for row in run_rows if row["method"] == method]
        selected_windows = [row for row in windows if row["method"] == method]
        selected_interventions = [row for row in interventions if row["method"] == method]
        final_holdouts = [
            row for row in holdouts if row["method"] == method and row["event"] == "final"
        ]
        action_counts = Counter(str(row["action_attempted"]) for row in selected_interventions)
        accepted = sum(str(row["accepted"]).casefold() == "true" for row in selected_interventions)
        rejected = len(selected_interventions) - accepted
        method_summary.append(
            {
                "method": method,
                "run_count": len(selected),
                "mean_labels_requested": _mean(selected, "labels_requested"),
                "mean_accepted_updates": _mean(selected, "accepted_updates"),
                "mean_unsafe_exposure_windows": _mean(selected, "unsafe_exposure_windows"),
                "mean_missed_harmful_windows": _mean(selected, "missed_harmful_windows"),
                "mean_final_pr_auc": _mean(final_holdouts, "pr_auc"),
                "mean_final_fpr_budget_ratio": _mean(final_holdouts, "fpr_budget_ratio"),
                "mean_final_tpr_forgetting": _mean(final_holdouts, "operational_tpr_forgetting"),
            }
        )
        safety.append(
            {
                "method": method,
                "window_count": len(selected_windows),
                "unsafe_exposure_windows": sum(
                    _as_bool(row["unsafe_exposure"], "unsafe exposure") for row in selected_windows
                ),
                "missed_harmful_windows": sum(
                    bool(row["missed_harmful_window"]) for row in selected_windows
                ),
                "false_health_alarms": sum(
                    bool(row["false_health_alarm"]) for row in selected_windows
                ),
                "operating_envelope_compliance_rate": 1.0
                - sum(
                    _as_bool(row["unsafe_exposure"], "unsafe exposure") for row in selected_windows
                )
                / len(selected_windows),
            }
        )
        labels.append(
            {
                "method": method,
                "labels_requested": sum(int(row["labels_requested"]) for row in selected),
                "labels_released": sum(int(row["labels_released"]) for row in selected),
                "query_events": sum(int(row["labels_requested"]) for row in selected) // 25,
            }
        )
        for action in (
            "A0_NO_OP",
            "A1_RECALIBRATE",
            "A2_HEAD_UPDATE",
            "A3_FULL_FINE_TUNE",
            "A4_REPLAY_UPDATE",
        ):
            actions.append(
                {"method": method, "action": action, "attempt_count": action_counts[action]}
            )
        rollbacks.append(
            {
                "method": method,
                "attempted_updates": len(selected_interventions),
                "accepted_updates": accepted,
                "rejected_updates": rejected,
                "rollback_rate": None
                if not selected_interventions
                else rejected / len(selected_interventions),
            }
        )
        for metric in (
            "pr_auc",
            "roc_auc",
            "tpr",
            "fpr_budget_ratio",
            "operational_tpr_forgetting",
        ):
            retention.append(
                {"method": method, "metric": metric, "mean_final": _mean(final_holdouts, metric)}
            )

    paired: list[dict[str, Any]] = []
    by_key = {(str(row["sequence"]), int(row["seed"]), str(row["method"])): row for row in run_rows}
    for sequence in sorted({str(row["sequence"]) for row in run_rows}):
        for seed in sorted({int(row["seed"]) for row in run_rows if row["sequence"] == sequence}):
            core = by_key.get((sequence, seed, Study4Method.DANIDS_CORE.value))
            always = by_key.get((sequence, seed, Study4Method.ALWAYS_ADAPT.value))
            if core is None or always is None:
                continue
            paired.append(
                {
                    "sequence": sequence,
                    "seed": seed,
                    "core_minus_always_labels": int(core["labels_requested"])
                    - int(always["labels_requested"]),
                    "core_minus_always_updates": int(core["accepted_updates"])
                    - int(always["accepted_updates"]),
                    "core_minus_always_unsafe_exposure": int(core["unsafe_exposure_windows"])
                    - int(always["unsafe_exposure_windows"]),
                    "core_minus_always_missed_harm": int(core["missed_harmful_windows"])
                    - int(always["missed_harmful_windows"]),
                }
            )
    return {
        "study4_runs.csv": run_rows,
        "study4_windows.csv": windows,
        "study4_holdouts.csv": holdouts,
        "study4_interventions.csv": interventions,
        "study4_method_summary.csv": method_summary,
        "study4_safety_compliance.csv": safety,
        "study4_label_query_usage.csv": labels,
        "study4_action_distribution.csv": actions,
        "study4_rollback_summary.csv": rollbacks,
        "study4_retention_summary.csv": retention,
        "study4_core_vs_always_paired.csv": paired,
    }


def evaluate_study4(
    run_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    allow_smoke: bool = False,
    allow_incomplete: bool = False,
) -> Path:
    """Validate and aggregate completed E4 runs without opening raw flow data."""

    runs = [validate_study4_run(path, allow_smoke=allow_smoke) for path in run_dirs]
    if not runs:
        raise Study4ArtifactError("Study-4 aggregation requires at least one run")
    keys = [(run.sequence, run.seed, run.method) for run in runs]
    if len(keys) != len(set(keys)):
        raise Study4ArtifactError(
            "Study-4 aggregation contains duplicate method/rotation/seed runs"
        )
    contracts = {run.contract_digest for run in runs}
    if len(contracts) != 1:
        raise Study4ArtifactError("Study-4 runs have mixed scientific/pipeline contracts")
    paired_sources: dict[tuple[tuple[str, ...], int], str] = {}
    for run in runs:
        pair = (run.sequence, run.seed)
        previous = paired_sources.setdefault(pair, run.source_checkpoint_sha256)
        if previous != run.source_checkpoint_sha256:
            raise Study4ArtifactError("paired Study-4 methods use different source checkpoints")
    expected = {
        (rotation, seed, method)
        for rotation in FROZEN_ROTATIONS
        for seed in STUDY4_CONFIRMATORY_SEEDS
        for method in (
            Study4Method.STATIC,
            Study4Method.ALWAYS_ADAPT,
            Study4Method.DANIDS_CORE,
        )
    }
    observed = set(keys)
    complete = observed == expected and not any(run.smoke for run in runs)
    if not allow_incomplete and not complete:
        raise Study4ArtifactError("confirmatory Study-4 aggregation requires exactly 36 full runs")
    if any(run.smoke for run in runs) and not allow_smoke:
        raise Study4ArtifactError("smoke runs cannot enter confirmatory Study-4 aggregation")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Study-4 evaluation: {output}")
    output.mkdir(parents=True)
    outputs = _study4_outputs(runs)
    for name, rows in outputs.items():
        _write_csv(output / name, rows)
    source_runs = [
        {
            "experiment_id": run.experiment_id,
            "path": str(run.path),
            "artifact_manifest_sha256": _file_sha256(run.path / MANIFEST_FILENAME),
        }
        for run in sorted(runs, key=lambda item: (item.sequence, item.seed, item.method.value))
    ]
    contract = {
        "version": STUDY4_EVALUATION_VERSION,
        "allow_smoke": allow_smoke,
        "allow_incomplete": allow_incomplete,
        "scientific_contract_digest": runs[0].contract_digest,
        "source_runs": source_runs,
        "observed_pairs": [
            {"sequence": list(sequence), "seed": seed, "method": method.value}
            for sequence, seed, method in sorted(
                keys, key=lambda item: (item[0], item[1], item[2].value)
            )
        ],
    }
    _write_json(output / "evaluation_contract.json", contract)
    summary = {
        "version": STUDY4_EVALUATION_VERSION,
        "status": "complete" if complete else "incomplete",
        "run_count": len(runs),
        "expected_confirmatory_run_count": len(expected),
        "smoke_run_count": sum(run.smoke for run in runs),
        "methods_present": sorted({run.method.value for run in runs}),
        "rotations_present": sorted({"-".join(run.sequence) for run in runs}),
        "seeds_present": sorted({run.seed for run in runs}),
        "paired_core_vs_always_count": len(outputs["study4_core_vs_always_paired.csv"]),
    }
    _write_json(output / "study4_summary.json", summary)
    digests = _tree_digests(output)
    _write_json(
        output / MANIFEST_FILENAME,
        {
            "version": STUDY4_EVALUATION_VERSION,
            "files": digests,
            "bundle_digest": _digest(digests),
        },
    )
    validate_study4_evaluation(output)
    return output


def validate_study4_evaluation(output_dir: str | Path) -> None:
    """Recompute every derived Study-4 table from its canonical artifact datasets."""

    root = Path(output_dir)
    manifest = _load_json(root / MANIFEST_FILENAME)
    files = _tree_digests(root)
    if manifest.get("version") != STUDY4_EVALUATION_VERSION or manifest.get("files") != files:
        raise Study4ArtifactError("Study-4 evaluation artifact manifest differs")
    if manifest.get("bundle_digest") != _digest(files):
        raise Study4ArtifactError("Study-4 evaluation bundle digest differs")
    contract = _load_json(root / "evaluation_contract.json")
    summary = _load_json(root / "study4_summary.json")
    run_rows = _records(root / "study4_runs.csv")
    windows = _records(root / "study4_windows.csv")
    holdouts = _records(root / "study4_holdouts.csv")
    interventions = _records(root / "study4_interventions.csv")
    represented = {(str(row["experiment_id"])) for row in run_rows}
    source_runs = contract.get("source_runs")
    if (
        not isinstance(source_runs, list)
        or {str(item.get("experiment_id")) for item in source_runs if isinstance(item, dict)}
        != represented
    ):
        raise Study4ArtifactError("Study-4 evaluation source-run metadata differs from dataset")
    keys = {(str(row["sequence"]), int(row["seed"]), str(row["method"])) for row in run_rows}
    if len(keys) != len(run_rows):
        raise Study4ArtifactError("Study-4 evaluation canonical runs contain duplicates")
    expected = {
        ("-".join(rotation), seed, method.value)
        for rotation in FROZEN_ROTATIONS
        for seed in STUDY4_CONFIRMATORY_SEEDS
        for method in (
            Study4Method.STATIC,
            Study4Method.ALWAYS_ADAPT,
            Study4Method.DANIDS_CORE,
        )
    }
    complete = keys == expected and int(summary["smoke_run_count"]) == 0
    if summary["status"] != ("complete" if complete else "incomplete"):
        raise Study4ArtifactError("Study-4 evaluation completion status differs from recomputation")
    if (
        int(summary["run_count"]) != len(run_rows)
        or int(summary["expected_confirmatory_run_count"]) != 36
    ):
        raise Study4ArtifactError("Study-4 evaluation run counts differ from recomputation")
    if summary["methods_present"] != sorted({key[2] for key in keys}):
        raise Study4ArtifactError("Study-4 evaluation method metadata differs")
    if summary["rotations_present"] != sorted({key[0] for key in keys}):
        raise Study4ArtifactError("Study-4 evaluation rotation metadata differs")
    if summary["seeds_present"] != sorted({key[1] for key in keys}):
        raise Study4ArtifactError("Study-4 evaluation seed metadata differs")

    synthetic_runs: list[ValidatedStudy4Run] = []
    for row in run_rows:
        experiment_id = str(row["experiment_id"])
        synthetic_runs.append(
            ValidatedStudy4Run(
                path=Path("artifact-only") / experiment_id,
                experiment_id=experiment_id,
                method=Study4Method(str(row["method"])),
                sequence=tuple(str(row["sequence"]).split("-")),
                seed=int(row["seed"]),
                smoke=_as_bool(row["smoke"], "smoke"),
                contract_digest=str(contract["scientific_contract_digest"]),
                source_checkpoint_sha256="artifact-only",
                summary={key: value for key, value in row.items() if key not in {"sequence"}},
                windows=[item for item in windows if item["experiment_id"] == experiment_id],
                holdouts=[item for item in holdouts if item["experiment_id"] == experiment_id],
                interventions=[
                    item for item in interventions if item["experiment_id"] == experiment_id
                ],
            )
        )
    expected_outputs = _study4_outputs(synthetic_runs)
    for name, expected_rows in expected_outputs.items():
        if name in {
            "study4_runs.csv",
            "study4_windows.csv",
            "study4_holdouts.csv",
            "study4_interventions.csv",
        }:
            continue
        if not expected_rows:
            if _records(root / name):
                raise Study4ArtifactError(f"persisted {name} differs from recomputation")
            continue
        actual = pd.read_csv(root / name)
        expected_frame = pd.DataFrame(expected_rows, columns=actual.columns)
        expected_frame = expected_frame.mask(expected_frame.isna(), np.nan)
        try:
            pd.testing.assert_frame_equal(
                actual.reset_index(drop=True),
                expected_frame.reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise Study4ArtifactError(f"persisted {name} differs from recomputation") from exc
    paired_count = len(expected_outputs["study4_core_vs_always_paired.csv"])
    if int(summary["paired_core_vs_always_count"]) != paired_count:
        raise Study4ArtifactError("Study-4 paired-comparison count differs from recomputation")


__all__ = [
    "CORE_BUNDLE_DIRNAME",
    "MANIFEST_FILENAME",
    "STUDY4_EVALUATION_VERSION",
    "STUDY4_RUN_ARTIFACT_VERSION",
    "Study4ArtifactError",
    "ValidatedStudy4Run",
    "evaluate_study4",
    "validate_study4_evaluation",
    "validate_study4_run",
    "write_study4_run_manifest",
]
