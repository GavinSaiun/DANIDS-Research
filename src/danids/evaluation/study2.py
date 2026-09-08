"""Strict artifact-only aggregation for Study-2 continual baselines."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from danids.config.continual import ContinualMethod, load_continual_experiment_config
from danids.continual.initial_state import file_sha256
from danids.continual.metrics import (
    adaptation_gains,
    backward_transfer,
    final_forgetting,
    stage_seen_domain_metrics,
)
from danids.continual.supervision import SupervisionSchedule, row_positions_digest
from danids.data.materialized import MATERIALIZER_VERSION
from danids.evaluation.study1 import CANONICAL_DOMAINS, validate_static_study1_run

EXPECTED_METHODS = ("naive_ft", "ewc", "er", "ft_mem")
REQUIRED_ADAPTIVE_FILES = (
    "summary.json",
    "config.resolved.yaml",
    "provenance.json",
    "supervision_schedule.json",
    "window_metrics.csv",
    "holdout_metrics.csv",
    "retention_matrix.csv",
    "native_attack_metrics.csv",
    "adaptation_log.csv",
    "adaptation_gain.csv",
    "forgetting.csv",
    "bwt.csv",
    "stage_metrics.csv",
    "resource_metrics.json",
    "memory_state_summary.json",
    "final_model.pt",
)


class Study2AggregationError(ValueError):
    """Raised when an adaptive artifact violates the Study-2 contract."""


@dataclass(frozen=True, slots=True)
class StaticReference:
    path: Path
    seed: int
    sequence: tuple[str, ...]
    fingerprints: tuple[tuple[str, str], ...]
    checkpoint_sha256: str
    model_digest: str
    preprocessor_digest: str
    threshold_digest: str


@dataclass(frozen=True, slots=True)
class AdaptiveArtifacts:
    path: Path
    method: ContinualMethod
    seed: int
    sequence: tuple[str, ...]
    schedule_digest: str
    source_identity: tuple[str, str, str, str]
    fingerprints: tuple[tuple[str, str], ...]
    final_holdouts: tuple[dict[str, str], ...]
    final_native: tuple[dict[str, str], ...]
    gains: tuple[dict[str, str], ...]
    forgetting: tuple[dict[str, str], ...]
    bwt: tuple[dict[str, str], ...]
    resource: Mapping[str, Any]
    scientific_signature: str


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Study2AggregationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Study2AggregationError(f"{path} must contain an object")
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Study2AggregationError(f"{path} must contain a mapping")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise Study2AggregationError(f"{path} has no header")
        return [{key: value or "" for key, value in row.items()} for row in reader]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _int(value: object, context: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise Study2AggregationError(f"{context} must be an integer") from exc


def _fingerprints(value: object, context: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict) or set(value) != set(CANONICAL_DOMAINS):
        raise Study2AggregationError(f"{context} must contain exact U/T/C/B fingerprints")
    result = []
    for domain in CANONICAL_DOMAINS:
        fingerprint = value[domain]
        if not isinstance(fingerprint, str) or not fingerprint:
            raise Study2AggregationError(f"{context} fingerprint for {domain} is invalid")
        result.append((domain, fingerprint))
    return tuple(result)


def _static_reference(path: str | Path) -> StaticReference:
    root = Path(path).resolve()
    validated = validate_static_study1_run(root)
    summary = _load_json(root / "summary.json")
    frozen = summary["frozen_state_evidence"]
    if not isinstance(frozen, dict):
        raise Study2AggregationError("static frozen-state evidence is invalid")
    return StaticReference(
        path=root,
        seed=validated.seed,
        sequence=validated.sequence,
        fingerprints=validated.dataset_fingerprints,
        checkpoint_sha256=file_sha256(root / "best_model.pt"),
        model_digest=str(frozen["model_after"]),
        preprocessor_digest=str(frozen["preprocessor_after"]),
        threshold_digest=str(frozen["threshold_after"]),
    )


def _partition_range(provenance: Mapping[str, Any], domain: str, partition: str) -> tuple[int, int]:
    ranges = provenance.get("manifest_partition_ranges")
    domain_ranges = ranges.get(domain) if isinstance(ranges, dict) else None
    value = domain_ranges.get(partition) if isinstance(domain_ranges, dict) else None
    if not isinstance(value, dict):
        raise Study2AggregationError(
            f"adaptive provenance lacks {domain} {partition} partition range"
        )
    start = _int(value.get("start"), f"{domain} {partition} start")
    stop = _int(value.get("stop"), f"{domain} {partition} stop")
    if start < 0 or stop <= start:
        raise Study2AggregationError(f"adaptive provenance has invalid {domain} {partition} range")
    return start, stop


def _positions_within_partition(
    positions: Sequence[int], domain: str, partition: str, provenance: Mapping[str, Any]
) -> bool:
    start, stop = _partition_range(provenance, domain, partition)
    return all(start <= position < stop for position in positions)


def _positions_outside_holdouts(
    positions: Sequence[int], domain: str, provenance: Mapping[str, Any]
) -> bool:
    start, stop = _partition_range(provenance, domain, "permanent_holdout")
    return all(not (start <= position < stop) for position in positions)


def _validate_partition_provenance(
    provenance: Mapping[str, Any], sequence: tuple[str, ...]
) -> None:
    ranges = provenance.get("manifest_partition_ranges")
    if not isinstance(ranges, dict) or set(ranges) != set(CANONICAL_DOMAINS):
        raise Study2AggregationError(
            "adaptive provenance must contain all U/T/C/B partition ranges"
        )
    for index, domain in enumerate(sequence):
        raw = ranges.get(domain)
        if not isinstance(raw, dict) or set(raw) != {
            "initial_train",
            "validation",
            "online_stream",
            "permanent_holdout",
        }:
            raise Study2AggregationError(
                f"adaptive provenance partition schema differs for {domain}"
            )
        holdout_start, row_count = _partition_range(provenance, domain, "permanent_holdout")
        if index == 0:
            train_start, train_stop = _partition_range(provenance, domain, "initial_train")
            validation_start, validation_stop = _partition_range(provenance, domain, "validation")
            if raw.get("online_stream") is not None:
                raise Study2AggregationError(f"source domain {domain} declares an online stream")
            if (
                train_start != 0
                or train_stop != validation_start
                or validation_stop != holdout_start
                or train_stop != int(row_count * 0.60)
                or validation_stop != int(row_count * 0.80)
            ):
                raise Study2AggregationError(
                    f"source domain {domain} partitions are not contiguous"
                )
        else:
            online_start, online_stop = _partition_range(provenance, domain, "online_stream")
            if raw.get("initial_train") is not None or raw.get("validation") is not None:
                raise Study2AggregationError(f"later domain {domain} declares source partitions")
            if (
                online_start != 0
                or online_stop != holdout_start
                or online_stop != int(row_count * 0.80)
            ):
                raise Study2AggregationError(f"later domain {domain} partitions are not contiguous")


def _validate_memory(
    root: Path,
    method: ContinualMethod,
    provenance: Mapping[str, Any],
    schedule: SupervisionSchedule,
    sequence: tuple[str, ...],
) -> None:
    expected_later = {
        entry.dataset_id: tuple(entry.chronological_positions) for entry in schedule.entries
    }
    state = _load_json(root / "memory_state_summary.json")
    if state.get("audit_per_domain") != 0:
        raise Study2AggregationError(f"{root}: audit memory is forbidden in TASK-004")
    final_memory = state.get("final_memory")
    fisher_count = _int(state.get("fisher_consolidations", 0), "Fisher count")
    if method in {"naive_ft", "ewc"} and final_memory is not None:
        raise Study2AggregationError(f"{root}: method must not have replay memory")
    if method == "naive_ft" and fisher_count != 0:
        raise Study2AggregationError(f"{root}: NaiveFT must not have Fisher state")
    if method == "ewc" and fisher_count != 4:
        raise Study2AggregationError(f"{root}: EWC must contain four stage consolidations")
    if method == "ewc":
        checkpoint_path = root / "ewc_state.pt"
        if not checkpoint_path.is_file():
            raise Study2AggregationError(f"{root}: EWC Fisher checkpoint is missing")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        consolidations = checkpoint.get("consolidations") if isinstance(checkpoint, dict) else None
        if not isinstance(consolidations, list) or len(consolidations) != 4:
            raise Study2AggregationError(f"{root}: EWC Fisher checkpoint is incomplete")
        by_domain: dict[str, list[int]] = {}
        for item in consolidations:
            if not isinstance(item, dict) or not isinstance(item.get("row_positions"), list):
                raise Study2AggregationError(f"{root}: EWC Fisher provenance is invalid")
            positions = [_int(value, "Fisher position") for value in item["row_positions"]]
            domain = str(item.get("domain_id"))
            if domain in by_domain or len(positions) != len(set(positions)):
                raise Study2AggregationError(f"{root}: EWC Fisher domain/positions are duplicated")
            by_domain[domain] = positions
        if set(by_domain) != set(sequence):
            raise Study2AggregationError(f"{root}: EWC Fisher domains differ from the sequence")
        for index, domain in enumerate(sequence):
            positions = by_domain[domain]
            if index == 0:
                valid = 0 < len(positions) <= 400 and _positions_within_partition(
                    positions, domain, "initial_train", provenance
                )
            else:
                valid = tuple(sorted(positions)) == expected_later[domain] and (
                    _positions_within_partition(positions, domain, "online_stream", provenance)
                )
            if not valid or not _positions_outside_holdouts(positions, domain, provenance):
                raise Study2AggregationError(f"{root}: EWC Fisher used disallowed rows")
    if method in {"er", "ft_mem"}:
        if not isinstance(final_memory, dict):
            raise Study2AggregationError(f"{root}: replay method lacks final memory")
        manifest_path = root / "memory_manifest.json"
        if not manifest_path.is_file() or _load_json(manifest_path) != final_memory:
            raise Study2AggregationError(f"{root}: replay memory manifest differs from state")
        if _int(final_memory.get("capacity_per_domain"), "memory capacity") != 400:
            raise Study2AggregationError(f"{root}: memory capacity differs from 400/domain")
        if _int(final_memory.get("audit_memory_size"), "audit memory") != 0:
            raise Study2AggregationError(f"{root}: replay memory includes audit examples")
        domains = final_memory.get("domains")
        if not isinstance(domains, list) or len(domains) != 4:
            raise Study2AggregationError(f"{root}: final replay memory must cover four domains")
        expected_selection = "uniform_random" if method == "er" else "embedding_mean_herding"
        by_domain = {}
        for item in domains:
            if not isinstance(item, dict) or item.get("selection") != expected_selection:
                raise Study2AggregationError(f"{root}: method-specific memory policy differs")
            if item.get("partition_kind") not in {"initial_train", "online_stream"}:
                raise Study2AggregationError(f"{root}: non-learning data entered memory")
            if _int(item.get("size"), "domain memory size") > 400:
                raise Study2AggregationError(f"{root}: per-domain memory cap exceeded")
            memory_positions = item.get("row_positions")
            domain = str(item.get("domain_id"))
            if not isinstance(memory_positions, list) or domain in by_domain:
                raise Study2AggregationError(f"{root}: replay memory provenance is invalid")
            positions = [_int(value, "memory position") for value in memory_positions]
            if len(positions) != len(set(positions)) or len(positions) != _int(
                item.get("size"), "domain memory size"
            ):
                raise Study2AggregationError(f"{root}: replay memory positions are inconsistent")
            by_domain[domain] = positions
        if set(by_domain) != set(sequence):
            raise Study2AggregationError(f"{root}: replay memory domains differ from sequence")
        for index, domain in enumerate(sequence):
            positions = by_domain[domain]
            if index == 0:
                valid = 0 < len(positions) <= 400 and _positions_within_partition(
                    positions, domain, "initial_train", provenance
                )
            else:
                valid = tuple(sorted(positions)) == expected_later[domain] and (
                    _positions_within_partition(positions, domain, "online_stream", provenance)
                )
            if not valid or not _positions_outside_holdouts(positions, domain, provenance):
                raise Study2AggregationError(f"{root}: permanent-holdout position entered memory")


def _scientific_signature(config: Mapping[str, Any], provenance: Mapping[str, Any]) -> str:
    adaptation = dict(config["adaptation"])
    adaptation.pop("method", None)
    signature = {
        "sequence": config["datasets"]["sequence"],
        "split_version": config["datasets"]["split_version"],
        "stream": config["stream"],
        "splits": config["splits"],
        "supervision": config["supervision"],
        "operating_envelope": config["operating_envelope"],
        "adaptation": adaptation,
        "memory": config["memory"],
        "feature_contract_version": provenance.get("feature_contract_version"),
        "feature_columns": provenance.get("feature_columns"),
        "materializer_version": provenance.get("materializer_version"),
    }
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def _validate_derived_rows(
    path: Path,
    persisted: Sequence[Mapping[str, str]],
    expected: Sequence[Mapping[str, Any]],
) -> None:
    if len(persisted) != len(expected):
        raise Study2AggregationError(
            f"{path}: derived metric row count differs from holdout recomputation"
        )
    for row_index, (actual, recomputed) in enumerate(zip(persisted, expected, strict=True)):
        for key, expected_value in recomputed.items():
            actual_value = actual.get(key)
            differs = False
            if expected_value is None:
                differs = actual_value not in {None, ""}
            elif isinstance(expected_value, float):
                try:
                    parsed = float(str(actual_value))
                except (TypeError, ValueError):
                    differs = True
                else:
                    differs = not math.isclose(parsed, expected_value, rel_tol=1e-12, abs_tol=1e-12)
            elif isinstance(expected_value, int):
                try:
                    differs = int(str(actual_value)) != expected_value
                except (TypeError, ValueError):
                    differs = True
            else:
                differs = str(actual_value) != str(expected_value)
            if differs:
                raise Study2AggregationError(
                    f"{path}: derived metric differs from holdout recomputation "
                    f"at row {row_index + 1}, field {key}"
                )


def validate_continual_run(path: str | Path, static: StaticReference) -> AdaptiveArtifacts:
    root = Path(path).resolve()
    missing = [name for name in REQUIRED_ADAPTIVE_FILES if not (root / name).is_file()]
    if missing:
        raise Study2AggregationError(f"{root}: missing required files: {', '.join(missing)}")
    summary = _load_json(root / "summary.json")
    config = _load_yaml(root / "config.resolved.yaml")
    provenance = _load_json(root / "provenance.json")
    if summary.get("smoke") is not False:
        raise Study2AggregationError(f"{root}: smoke runs cannot be aggregated")
    parsed = load_continual_experiment_config(root / "config.resolved.yaml")
    if config.get("smoke") not in (None, False) or parsed.adaptation.epochs != 20:
        raise Study2AggregationError(f"{root}: full run must use 20 adaptation epochs")
    method = parsed.adaptation.method
    seed = parsed.experiment.seed
    sequence = parsed.experiment.sequence
    if seed != static.seed or sequence != static.sequence:
        raise Study2AggregationError(f"{root}: seed/sequence differs from paired static run")
    _validate_partition_provenance(provenance, sequence)
    if summary.get("method") != method or summary.get("sequence") != list(sequence):
        raise Study2AggregationError(f"{root}: summary method/sequence differs from config")
    if _int(summary.get("adaptation_count"), "adaptation count") != 3:
        raise Study2AggregationError(f"{root}: full run must contain three adaptations")
    if summary.get("preprocessor_unchanged") is not True:
        raise Study2AggregationError(f"{root}: source preprocessor changed")
    if summary.get("threshold_unchanged") is not True:
        raise Study2AggregationError(f"{root}: source threshold changed")
    if _int(summary.get("target_threshold_recalibrations"), "recalibration count") != 0:
        raise Study2AggregationError(f"{root}: target threshold was recalibrated")
    if _int(summary.get("target_validation_uses"), "target validation uses") != 0:
        raise Study2AggregationError(f"{root}: target validation was used")
    if summary.get("source_state_reused") is not True:
        raise Study2AggregationError(f"{root}: source state was not recorded as reused")
    if summary.get("source_checkpoint_unchanged") is not True:
        raise Study2AggregationError(f"{root}: imported Study-1 checkpoint changed")
    fingerprints = _fingerprints(provenance.get("dataset_fingerprints"), str(root))
    if fingerprints != static.fingerprints:
        raise Study2AggregationError(f"{root}: dataset fingerprints differ from static run")
    if provenance.get("materializer_version") != MATERIALIZER_VERSION:
        raise Study2AggregationError(f"{root}: incompatible chronological materializer version")
    source_identity = (
        str(provenance.get("initial_checkpoint_sha256")),
        str(provenance.get("initial_model_digest")),
        str(provenance.get("initial_preprocessor_digest")),
        str(provenance.get("initial_threshold_digest")),
    )
    expected_identity = (
        static.checkpoint_sha256,
        static.model_digest,
        static.preprocessor_digest,
        static.threshold_digest,
    )
    if source_identity != expected_identity:
        raise Study2AggregationError(f"{root}: imported Study-1 source state differs")
    schedule_path = root / "supervision_schedule.json"
    schedule = SupervisionSchedule.from_json(schedule_path)
    if schedule.seed != seed or schedule.sequence != sequence:
        raise Study2AggregationError(f"{root}: supervision schedule seed/sequence differs")
    for entry in schedule.entries:
        if not _positions_within_partition(
            entry.chronological_positions, entry.dataset_id, "online_stream", provenance
        ) or not _positions_outside_holdouts(
            entry.chronological_positions, entry.dataset_id, provenance
        ):
            raise Study2AggregationError(
                f"{root}: supervision positions fall outside the allowed online stream"
            )
    schedule_digest = schedule.digest()
    if provenance.get("supervision_schedule_digest") != schedule_digest:
        raise Study2AggregationError(f"{root}: supervision schedule digest differs")
    _validate_memory(root, method, provenance, schedule, sequence)

    adaptations = _read_csv(root / "adaptation_log.csv")
    if len(adaptations) != 3:
        raise Study2AggregationError(f"{root}: adaptation log must contain three rows")
    adaptations_by_stage = {row.get("stage"): row for row in adaptations}
    if set(adaptations_by_stage) != {"2", "3", "4"}:
        raise Study2AggregationError(f"{root}: adaptation stages must be exactly 2/3/4")
    for row in adaptations:
        if _int(row.get("queried_labels_available"), "queried labels") != 100:
            raise Study2AggregationError(f"{root}: adaptation did not receive exactly 100 labels")
        if _int(row.get("target_rows"), "target rows") != 100:
            raise Study2AggregationError(f"{root}: adaptation used the wrong target count")
        if row.get("preprocessor_digest") != static.preprocessor_digest:
            raise Study2AggregationError(f"{root}: adaptation changed preprocessor")
        if row.get("threshold_digest") != static.threshold_digest:
            raise Study2AggregationError(f"{root}: adaptation changed threshold")
        stage = _int(row.get("stage"), "adaptation stage")
        entry = schedule.entries[stage - 2]
        if row.get("domain_id") != entry.dataset_id:
            raise Study2AggregationError(f"{root}: adaptation domain differs from schedule")
        expected_target_digest = row_positions_digest(entry.chronological_positions)
        if row.get("target_row_positions_sha256") != expected_target_digest:
            raise Study2AggregationError(
                f"{root}: actual adaptation rows differ from the supervision schedule"
            )
        replay_rows = _int(row.get("replay_rows_available"), "available replay rows")
        rows_processed = _int(row.get("rows_processed"), "processed adaptation rows")
        optimizer_steps = _int(row.get("optimizer_steps"), "adaptation optimizer steps")
        if method in {"naive_ft", "ewc"}:
            if replay_rows != 0 or rows_processed != 2_000 or optimizer_steps != 40:
                raise Study2AggregationError(f"{root}: target-only adaptation contract differs")
        elif method == "er":
            if replay_rows <= 0 or optimizer_steps != 80:
                raise Study2AggregationError(f"{root}: ER minibatch replay contract differs")
        elif replay_rows <= 0 or rows_processed != (100 + replay_rows) * 20:
            raise Study2AggregationError(f"{root}: FT-Mem joint-dataset contract differs")

    windows = _read_csv(root / "window_metrics.csv")
    for row in adaptations:
        adaptation_stage = row.get("stage")
        prediction_digests = {
            item.get("model_digest_at_prediction")
            for item in windows
            if item.get("stage") == adaptation_stage and item.get("window_id") in {"0", "1"}
        }
        if prediction_digests != {row.get("model_digest_before")}:
            raise Study2AggregationError(
                f"{root}: first two windows were not both predicted before adaptation"
            )

    holdouts = _read_csv(root / "holdout_metrics.csv")
    source_events = [row for row in holdouts if row.get("event") == "source_initial"]
    if not source_events or {
        _int(row.get("event_index"), "event index") for row in source_events
    } != {1}:
        raise Study2AggregationError(f"{root}: source-initial holdout event is invalid")
    for stage in (2, 3, 4):
        pre = {
            _int(row.get("event_index"), "event index")
            for row in holdouts
            if row.get("stage") == str(stage) and row.get("event") == "pre_adapt"
        }
        post = {
            _int(row.get("event_index"), "event index")
            for row in holdouts
            if row.get("stage") == str(stage) and row.get("event") == "post_adapt"
        }
        domain_end = {
            _int(row.get("event_index"), "event index")
            for row in holdouts
            if row.get("stage") == str(stage) and row.get("event") == "domain_end"
        }
        if len(pre) != 1 or len(post) != 1 or len(domain_end) != 1:
            raise Study2AggregationError(f"{root}: stage {stage} holdout event set is incomplete")
        if not (next(iter(pre)) < next(iter(post)) < next(iter(domain_end))):
            raise Study2AggregationError(f"{root}: stage {stage} holdout events are out of order")
    final = tuple(row for row in holdouts if row.get("event") == "final")
    if len(final) != 4 or {row.get("holdout_dataset_id") for row in final} != set(
        CANONICAL_DOMAINS
    ):
        raise Study2AggregationError(f"{root}: final event must contain all four holdouts")
    final_event_indices = {_int(row.get("event_index"), "final event index") for row in final}
    domain_end_indices = {
        _int(row.get("event_index"), "domain-end event index")
        for row in holdouts
        if row.get("event") == "domain_end"
    }
    if (
        len(final_event_indices) != 1
        or not domain_end_indices
        or next(iter(final_event_indices)) <= max(domain_end_indices)
    ):
        raise Study2AggregationError(f"{root}: final holdout event is out of order")
    native = _read_csv(root / "native_attack_metrics.csv")
    final_native = tuple(row for row in native if row.get("event") == "final")
    if not final_native:
        raise Study2AggregationError(f"{root}: final native attack metrics are missing")
    expected_gains = adaptation_gains(holdouts)
    expected_forgetting = final_forgetting(holdouts, sequence)
    expected_bwt = backward_transfer(holdouts, sequence)
    expected_stage_metrics = stage_seen_domain_metrics(holdouts, sequence)
    gains = _read_csv(root / "adaptation_gain.csv")
    forgetting = _read_csv(root / "forgetting.csv")
    bwt = _read_csv(root / "bwt.csv")
    _validate_derived_rows(root / "adaptation_gain.csv", gains, expected_gains)
    _validate_derived_rows(root / "forgetting.csv", forgetting, expected_forgetting)
    _validate_derived_rows(root / "bwt.csv", bwt, expected_bwt)
    _validate_derived_rows(
        root / "stage_metrics.csv",
        _read_csv(root / "stage_metrics.csv"),
        expected_stage_metrics,
    )
    return AdaptiveArtifacts(
        path=root,
        method=method,
        seed=seed,
        sequence=sequence,
        schedule_digest=schedule_digest,
        source_identity=source_identity,
        fingerprints=fingerprints,
        final_holdouts=final,
        final_native=final_native,
        gains=tuple(gains),
        forgetting=tuple(forgetting),
        bwt=tuple(bwt),
        resource=_load_json(root / "resource_metrics.json"),
        scientific_signature=_scientific_signature(config, provenance),
    )


def _context_rows(
    run: AdaptiveArtifacts, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "seed": run.seed,
            "sequence": "-".join(run.sequence),
            "method": run.method,
            **row,
        }
        for row in rows
    ]


def _static_final_rows(reference: StaticReference) -> list[dict[str, Any]]:
    rows = _read_csv(reference.path / "holdout_metrics.csv")
    result: list[dict[str, Any]] = []
    for row in rows:
        if row.get("stage") != "4":
            continue
        enriched: dict[str, Any] = {
            "seed": reference.seed,
            "sequence": "-".join(reference.sequence),
            "method": "static",
            **row,
        }
        pr_auc = row.get("pr_auc")
        prevalence = row.get("attack_prevalence")
        if pr_auc is not None and pr_auc != "" and prevalence is not None and prevalence != "":
            enriched["pr_auc_minus_prevalence"] = float(pr_auc) - float(prevalence)
        result.append(enriched)
    return result


def _static_final_native_rows(reference: StaticReference) -> list[dict[str, Any]]:
    rows = _read_csv(reference.path / "native_attack_metrics.csv")
    return [
        {
            "seed": reference.seed,
            "sequence": "-".join(reference.sequence),
            "method": "static",
            "event": "final",
            **row,
        }
        for row in rows
        if row.get("scope") == "holdout" and row.get("stage") == "4"
    ]


def _metric_summary(final_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("pr_auc", "roc_auc", "tpr", "fpr", "pr_auc_minus_prevalence")
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in final_rows:
        for metric in metrics:
            raw = row.get(metric)
            if raw is not None and raw != "":
                value = float(raw)
                if math.isfinite(value):
                    grouped[
                        (
                            str(row["sequence"]),
                            str(row["method"]),
                            str(row["holdout_dataset_id"]),
                            metric,
                        )
                    ].append(value)
    result = []
    for key in sorted(grouped):
        values = grouped[key]
        result.append(
            {
                "sequence": key[0],
                "method": key[1],
                "target_domain": key[2],
                "metric": key[3],
                "n_seeds": len(values),
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values) if len(values) >= 2 else None,
                "minimum": min(values),
                "maximum": max(values),
            }
        )
    return result


def aggregate_continual_study2(
    static_runs: Sequence[str | Path],
    run_dirs: Sequence[str | Path],
    output_dir: str | Path,
) -> Path:
    """Validate and aggregate paired E1/E2 artifacts without raw-data access."""

    if not static_runs or not run_dirs:
        raise Study2AggregationError("at least one static and one adaptive run are required")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite aggregation directory: {output}")
    references = sorted(
        (_static_reference(path) for path in static_runs),
        key=lambda reference: (reference.seed, reference.sequence),
    )
    reference_map: dict[tuple[int, tuple[str, ...]], StaticReference] = {}
    for reference in references:
        key = (reference.seed, reference.sequence)
        if key in reference_map:
            raise Study2AggregationError(f"duplicate static reference for {key}")
        reference_map[key] = reference

    runs: list[AdaptiveArtifacts] = []
    for path in run_dirs:
        config = _load_yaml(Path(path) / "config.resolved.yaml")
        key = (_int(config.get("seed"), "adaptive seed"), tuple(config["datasets"]["sequence"]))
        matched_reference = reference_map.get(key)
        if matched_reference is None:
            raise Study2AggregationError(f"adaptive run {path} has no paired static reference")
        runs.append(validate_continual_run(path, matched_reference))
    identities = [(run.seed, run.sequence, run.method) for run in runs]
    if len(identities) != len(set(identities)):
        raise Study2AggregationError("duplicate adaptive method in a paired set")
    grouped: dict[tuple[int, tuple[str, ...]], list[AdaptiveArtifacts]] = defaultdict(list)
    for run in runs:
        grouped[(run.seed, run.sequence)].append(run)
    for key, paired in grouped.items():
        if len({run.schedule_digest for run in paired}) != 1:
            raise Study2AggregationError(f"paired set {key} uses mixed supervision schedules")
        if len({run.source_identity for run in paired}) != 1:
            raise Study2AggregationError(f"paired set {key} uses mixed source checkpoints")
        if len({run.scientific_signature for run in paired}) != 1:
            raise Study2AggregationError(f"paired set {key} uses mixed scientific defaults")

    runs.sort(key=lambda run: (run.seed, run.sequence, EXPECTED_METHODS.index(run.method)))
    static_final = [row for reference in references for row in _static_final_rows(reference)]
    adaptive_final = [row for run in runs for row in _context_rows(run, run.final_holdouts)]
    final_rows = static_final + adaptive_final
    gains = [row for run in runs for row in _context_rows(run, run.gains)]
    forgetting = [row for run in runs for row in _context_rows(run, run.forgetting)]
    bwt = [row for run in runs for row in _context_rows(run, run.bwt)]
    resources = [
        {
            "seed": run.seed,
            "sequence": "-".join(run.sequence),
            "method": run.method,
            **run.resource,
        }
        for run in runs
    ]
    static_native = [
        row for reference in references for row in _static_final_native_rows(reference)
    ]
    adaptive_native = [row for run in runs for row in _context_rows(run, run.final_native)]
    native = static_native + adaptive_native
    complete_groups = []
    incomplete_groups: dict[str, list[str]] = {}
    for key, paired in sorted(grouped.items()):
        methods = {run.method for run in paired}
        label = f"s{key[0]}:{'-'.join(key[1])}"
        missing = [method for method in EXPECTED_METHODS if method not in methods]
        if missing:
            incomplete_groups[label] = missing
        else:
            complete_groups.append(label)
    summary = {
        "complete": not incomplete_groups,
        "status": "complete" if not incomplete_groups else "incomplete",
        "adaptive_run_count": len(runs),
        "static_reference_count": len(references),
        "expected_methods": list(EXPECTED_METHODS),
        "complete_paired_sets": complete_groups,
        "missing_methods_by_paired_set": incomplete_groups,
        "artifact_only": True,
    }
    output.mkdir(parents=True)
    _write_csv(output / "study2_method_summary.csv", _metric_summary(final_rows))
    _write_csv(output / "study2_adaptation_gain.csv", gains)
    _write_csv(output / "study2_forgetting.csv", forgetting)
    _write_csv(output / "study2_bwt.csv", bwt)
    _write_csv(output / "study2_final_holdouts.csv", final_rows)
    _write_csv(output / "study2_resource_summary.csv", resources)
    _write_csv(output / "study2_native_attack_long.csv", native)
    (output / "study2_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


__all__ = [
    "AdaptiveArtifacts",
    "StaticReference",
    "Study2AggregationError",
    "aggregate_continual_study2",
    "validate_continual_run",
]
