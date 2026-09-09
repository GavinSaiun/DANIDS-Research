"""Strict artifact-only validation for extracted Study-3 health episodes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from danids.config.health import load_health_experiment_config
from danids.continual.initial_state import file_sha256
from danids.continual.supervision import (
    SupervisionSchedule,
    deterministic_query_positions,
    row_positions_digest,
)
from danids.data.materialized import MATERIALIZER_VERSION
from danids.data.preprocessing import PREPROCESSOR_VERSION
from danids.evaluation.study1 import CANONICAL_DOMAINS, FROZEN_ROTATIONS, validate_static_study1_run
from danids.health.cache import REFERENCE_CACHE_VERSION, SIGNAL_CACHE_VERSION
from danids.health.states import classify_health
from danids.shift.signals import deterministic_reference_positions, positions_digest

HEALTH_ARTIFACT_VERSION = "task005-health-artifacts-v1"
REQUIRED_HEALTH_FILES = (
    "config.resolved.yaml",
    "provenance.json",
    "reference_summary.json",
    "reference_positions.json",
    "conformal_calibration.json",
    "supervision_schedule.json",
    "health_windows.csv",
    "distribution_feature_long.csv",
    "native_attack_metrics.csv",
    "health_run_summary.json",
)


@dataclass(frozen=True, slots=True)
class ValidatedHealthRun:
    path: Path
    seed: int
    source_domain: str
    sequence: tuple[str, ...]
    source_identity: str
    contract_digest: str
    windows: pd.DataFrame


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _range(provenance: Mapping[str, Any], domain: str, kind: str) -> tuple[int, int]:
    ranges = _mapping(provenance.get("manifest_partition_ranges"), "partition ranges")
    domain_ranges = _mapping(ranges.get(domain), f"{domain} partition ranges")
    value = domain_ranges.get(kind)
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"missing {domain} {kind} partition range")
    start, stop = int(value[0]), int(value[1])
    if start < 0 or stop <= start:
        raise ValueError(f"invalid {domain} {kind} partition range")
    return start, stop


def _truth(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean artifact value: {value!r}")


def _same_optional_float(actual: object, expected: float | None) -> bool:
    if actual is None:
        return expected is None
    if not isinstance(actual, (str, int, float, np.integer, np.floating)):
        return False
    value = float(actual)
    if expected is None:
        return math.isnan(value)
    return math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)


def _contract_digest(config: Mapping[str, Any], provenance: Mapping[str, Any]) -> str:
    payload = {
        "alpha": config.get("operating_envelope"),
        "sampling": config.get("sampling"),
        "signals": config.get("signals"),
        "predictors": config.get("predictors"),
        "supervision": config.get("supervision"),
        "feature_sets": config.get("feature_sets"),
        "cross_validation": config.get("cross_validation"),
        "cache_version": config.get("cache_version"),
        "feature_columns": provenance.get("feature_columns"),
        "feature_contract_version": provenance.get("feature_contract_version"),
        "materializer_version": provenance.get("materializer_version"),
        "preprocessor_version": provenance.get("preprocessor_version"),
        "distribution_signal_cache_version": provenance.get("distribution_signal_cache_version"),
        "source_reference_cache_version": provenance.get("source_reference_cache_version"),
        "dataset_fingerprints": provenance.get("dataset_fingerprints"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_health_run(run_dir: str | Path, *, allow_smoke: bool = False) -> ValidatedHealthRun:
    root = Path(run_dir).resolve()
    missing = [name for name in REQUIRED_HEALTH_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"{root}: missing Study-3 artifacts: {', '.join(missing)}")
    config_raw = yaml.safe_load((root / "config.resolved.yaml").read_text(encoding="utf-8"))
    config = _mapping(config_raw, "Study-3 config")
    resolved_config = load_health_experiment_config(root / "config.resolved.yaml")
    provenance = _load_json(root / "provenance.json")
    summary = _load_json(root / "health_run_summary.json")
    reference_summary = _load_json(root / "reference_summary.json")
    smoke = bool(summary.get("smoke"))
    configured_smoke = config.get("smoke") is not None
    if smoke != configured_smoke:
        raise ValueError(f"{root}: health run smoke metadata differs from resolved config")
    if smoke and not allow_smoke:
        raise ValueError(f"{root}: smoke health runs cannot enter confirmatory aggregation")
    if provenance.get("artifact_version") != HEALTH_ARTIFACT_VERSION:
        raise ValueError(f"{root}: unsupported health artifact version")
    sequence_raw = config.get("datasets")
    sequence_value = _mapping(sequence_raw, "datasets").get("sequence")
    if not isinstance(sequence_value, list):
        raise ValueError(f"{root}: invalid sequence")
    sequence = tuple(str(item) for item in sequence_value)
    if sequence not in FROZEN_ROTATIONS:
        raise ValueError(f"{root}: sequence is not a frozen rotation")
    seed_raw = config.get("seed")
    if seed_raw is None:
        raise ValueError(f"{root}: config lacks seed")
    seed = int(seed_raw)
    if resolved_config.experiment.seed != seed:
        raise ValueError(f"{root}: parsed/configured seeds disagree")
    source = sequence[0]
    reference_recall = float(reference_summary.get("reference_recall", -1.0))
    recall_floor = float(reference_summary.get("recall_floor", -1.0))
    if not math.isclose(
        recall_floor,
        max(0.0, reference_recall - resolved_config.delta_recall),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{root}: source validation recall floor is inconsistent")
    fingerprints = _mapping(provenance.get("dataset_fingerprints"), "dataset fingerprints")
    if set(fingerprints) != set(CANONICAL_DOMAINS) or any(
        not isinstance(fingerprints.get(domain), str) or not str(fingerprints[domain]).strip()
        for domain in CANONICAL_DOMAINS
    ):
        raise ValueError(f"{root}: dataset fingerprints must contain non-empty U/T/C/B values")
    if provenance.get("materializer_version") != MATERIALIZER_VERSION:
        raise ValueError(f"{root}: materializer version differs")
    if provenance.get("preprocessor_version") != PREPROCESSOR_VERSION:
        raise ValueError(f"{root}: preprocessor version differs")
    if provenance.get("distribution_signal_cache_version") != SIGNAL_CACHE_VERSION:
        raise ValueError(f"{root}: distribution-signal cache version differs")
    if provenance.get("source_reference_cache_version") != REFERENCE_CACHE_VERSION:
        raise ValueError(f"{root}: source-reference cache version differs")
    if provenance.get("source_artifact_kind") != "study1_static":
        raise ValueError(f"{root}: source state is not a Study-1 static run")
    source_run_raw = provenance.get("source_run")
    if not isinstance(source_run_raw, str) or not source_run_raw:
        raise ValueError(f"{root}: source Study-1 run path is missing")
    static = validate_static_study1_run(source_run_raw)
    if static.seed != seed or static.source != source or static.sequence != sequence:
        raise ValueError(f"{root}: imported Study-1 identity differs from health config")
    if dict(static.dataset_fingerprints) != dict(fingerprints):
        raise ValueError(f"{root}: imported Study-1 dataset provenance differs")
    checkpoint_sha = file_sha256(static.path / "best_model.pt")
    if provenance.get("source_checkpoint_sha256") != checkpoint_sha:
        raise ValueError(f"{root}: imported Study-1 checkpoint digest differs")
    if provenance.get("source_run_identity") != checkpoint_sha:
        raise ValueError(f"{root}: static source identity differs from checkpoint digest")
    static_summary = _load_json(static.path / "summary.json")
    static_frozen = _mapping(static_summary.get("frozen_state_evidence"), "Study-1 frozen state")
    expected_source_digests = {
        "source_model_digest": static_frozen.get("model_after"),
        "source_preprocessor_digest": static_frozen.get("preprocessor_after"),
        "source_threshold_digest": static_frozen.get("threshold_after"),
    }
    for key, expected in expected_source_digests.items():
        if provenance.get(key) != expected:
            raise ValueError(f"{root}: imported Study-1 {key} differs")
    frozen = _mapping(provenance.get("frozen_state_evidence"), "frozen state evidence")
    for state in ("model", "preprocessor", "threshold"):
        if frozen.get(f"{state}_before") != frozen.get(f"{state}_after"):
            raise ValueError(f"{root}: frozen {state} state changed")
    if int(frozen.get("optimizer_steps", -1)) != 0:
        raise ValueError(f"{root}: health extraction performed optimizer steps")
    positions = _load_json(root / "reference_positions.json")
    reference_positions = positions.get("source_initial_training_positions")
    if not isinstance(reference_positions, list) or not reference_positions:
        raise ValueError(f"{root}: source reference positions are missing")
    train_start, train_stop = _range(provenance, source, "initial_train")
    if any(int(value) < train_start or int(value) >= train_stop for value in reference_positions):
        raise ValueError(f"{root}: source reference includes rows outside initial training")
    canonical_reference = np.asarray(reference_positions, dtype=np.int64)
    if positions_digest(canonical_reference) != positions.get(
        "source_initial_training_positions_digest"
    ):
        raise ValueError(f"{root}: source reference position digest differs")
    expected_reference = deterministic_reference_positions(
        train_start,
        train_stop,
        resolved_config.sampling.source_reference_rows,
        identity=f"{resolved_config.cache_version}|{fingerprints[source]}|{seed}|source-reference",
    )
    if not np.array_equal(canonical_reference, expected_reference):
        raise ValueError(f"{root}: source reference differs from deterministic sample")
    if provenance.get("reference_positions_digest") != positions_digest(canonical_reference):
        raise ValueError(f"{root}: provenance source-reference digest differs")
    validation_start, validation_stop = _range(provenance, source, "validation")
    if positions.get("source_validation_range") != [validation_start, validation_stop]:
        raise ValueError(f"{root}: source validation reference range differs")
    if positions.get("source_domain") != source or reference_summary.get("source_domain") != source:
        raise ValueError(f"{root}: source-reference domain differs")
    if int(reference_summary.get("reference_sample_count", -1)) != len(canonical_reference):
        raise ValueError(f"{root}: source-reference sample count differs")
    validation_metrics = _mapping(
        reference_summary.get("validation_metrics"), "source validation metrics"
    )
    if int(validation_metrics.get("row_count", -1)) != validation_stop - validation_start:
        raise ValueError(f"{root}: source validation metric row count differs")
    if not _same_optional_float(validation_metrics.get("tpr"), reference_recall):
        raise ValueError(f"{root}: source validation recall differs from R_ref")
    threshold_artifact = _load_json(static.path / "threshold.json")
    if not _same_optional_float(
        validation_metrics.get("threshold"), float(threshold_artifact["threshold"])
    ):
        raise ValueError(f"{root}: deployment threshold differs from Study-1 source")
    conformal = _load_json(root / "conformal_calibration.json")
    if conformal.get("version") != "task005-binary-split-conformal-v1":
        raise ValueError(f"{root}: unsupported conformal calibration version")
    if not _same_optional_float(conformal.get("alpha"), resolved_config.signals.conformal_alpha):
        raise ValueError(f"{root}: conformal alpha differs from frozen config")
    if int(conformal.get("calibration_count", -1)) != validation_stop - validation_start:
        raise ValueError(f"{root}: conformal calibration count differs from validation range")
    quantile = float(conformal.get("quantile", float("nan")))
    if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError(f"{root}: invalid conformal calibration quantile")
    holdout_ranges = {
        domain: _range(provenance, domain, "permanent_holdout") for domain in sequence
    }
    schedule = SupervisionSchedule.from_json(root / "supervision_schedule.json")
    schedule_digest = schedule.digest()
    if provenance.get("supervision_schedule_digest") != schedule_digest:
        raise ValueError(f"{root}: supervision schedule digest differs")
    for entry in schedule.entries:
        if entry.manifest_source_sha256 != fingerprints.get(entry.dataset_id):
            raise ValueError(f"{root}: schedule dataset fingerprint differs")
        online_start, online_stop = _range(provenance, entry.dataset_id, "online_stream")
        if any(
            position < online_start or position >= online_stop
            for position in entry.chronological_positions
        ):
            raise ValueError(f"{root}: schedule positions fall outside online stream")
        expected_query = deterministic_query_positions(
            first_start=online_start,
            first_stop=min(online_start + resolved_config.experiment.window_size, online_stop),
            seed=seed,
            stage=entry.stage,
            dataset_id=entry.dataset_id,
            source_sha256=str(fingerprints[entry.dataset_id]),
        )
        if entry.chronological_positions != expected_query:
            raise ValueError(f"{root}: supervision schedule differs from deterministic query")
    windows = pd.read_csv(root / "health_windows.csv")
    required_columns = {
        "source_domain",
        "current_domain",
        "transition",
        "seed",
        "window_id",
        "row_start",
        "row_stop",
        "partition_kind",
        "health_state",
        "label_free_target_labels_used",
        "delayed_labels_available",
        "model_digest_at_prediction",
    }
    missing_columns = required_columns.difference(windows.columns)
    if missing_columns or windows.empty:
        raise ValueError(f"{root}: invalid health_windows schema: {sorted(missing_columns)}")
    if windows.duplicated(["current_domain", "partition_kind", "window_id"]).any():
        raise ValueError(f"{root}: duplicated health windows")
    for row in windows.to_dict("records"):
        domain = str(row["current_domain"])
        start, stop = int(row["row_start"]), int(row["row_stop"])
        hold_start, hold_stop = holdout_ranges[domain]
        if start < 0 or stop <= start or not (stop <= hold_start or start >= hold_stop):
            raise ValueError(f"{root}: health window touches permanent holdout")
        sample_count = int(row.get("sample_count", 0))
        identity = (
            f"{config.get('cache_version')}|{fingerprints[source]}|{fingerprints[domain]}|"
            f"{seed}|{domain}|{int(row['window_id'])}"
        )
        local = deterministic_reference_positions(0, stop - start, sample_count, identity=identity)
        expected_sample = np.asarray(local + start, dtype=np.int64)
        if positions_digest(expected_sample) != str(row.get("sample_positions_digest")):
            raise ValueError(f"{root}: deterministic window sample digest differs")
        if _truth(row["label_free_target_labels_used"]):
            raise ValueError(f"{root}: target labels entered label-free features")
        recomputed = classify_health(
            false_positives=int(row["fp"]),
            benign_support=int(row["benign_count"]),
            true_positives=int(row["tp"]),
            attack_support=int(row["attack_count"]),
            alpha=resolved_config.alpha,
            recall_floor=recall_floor,
            confidence=resolved_config.confidence_level,
        )
        if str(row["health_state"]) != recomputed.state.value:
            raise ValueError(f"{root}: persisted health state differs from Wilson recomputation")
        for field in ("fpr_low", "fpr_high", "tpr_low", "tpr_high"):
            if not _same_optional_float(row.get(field), getattr(recomputed, field)):
                raise ValueError(f"{root}: persisted Wilson interval differs")
        if any(
            int(row.get(field, -1)) != expected
            for field, expected in (
                ("false_positives", recomputed.false_positives),
                ("benign_support", recomputed.benign_support),
                ("true_positives", recomputed.true_positives),
                ("attack_support", recomputed.attack_support),
            )
        ):
            raise ValueError(f"{root}: duplicated evaluator counts disagree")
        tp, fp = int(row["tp"]), int(row["fp"])
        tn, fn = int(row["tn"]), int(row["fn"])
        benign_count, attack_count = int(row["benign_count"]), int(row["attack_count"])
        if benign_count != tn + fp or attack_count != tp + fn:
            raise ValueError(f"{root}: persisted confusion counts are inconsistent")
        if benign_count + attack_count != stop - start:
            raise ValueError(f"{root}: persisted support differs from window range")
        expected_rates = {
            "fpr": None if benign_count == 0 else fp / benign_count,
            "tpr": None if attack_count == 0 else tp / attack_count,
            "precision": None if tp + fp == 0 else tp / (tp + fp),
            "eval_fpr_budget_ratio": (
                None if benign_count == 0 else (fp / benign_count) / resolved_config.alpha
            ),
            "eval_tpr_loss_from_reference": (
                None if attack_count == 0 else reference_recall - tp / attack_count
            ),
        }
        for name, expected in expected_rates.items():
            if not _same_optional_float(row.get(name), expected):
                raise ValueError(f"{root}: persisted {name} differs from confusion counts")
        available = int(row["delayed_labels_available"])
        delayed_numeric = (
            "delayed_attack_prevalence",
            "delayed_attack_recall",
            "delayed_benign_fpr",
            "delayed_brier_score",
            "delayed_sample_count",
        )
        if available == 0 and any(pd.notna(row.get(name)) for name in delayed_numeric):
            raise ValueError(f"{root}: delayed values are populated before label release")
        if str(row["model_digest_at_prediction"]) != str(frozen.get("model_before")):
            raise ValueError(f"{root}: health window was generated after model adaptation")
        if available == 1:
            delayed_entry = next(
                (item for item in schedule.entries if item.dataset_id == domain), None
            )
            if delayed_entry is None or int(row["window_id"]) <= delayed_entry.label_return_window:
                raise ValueError(f"{root}: delayed labels exposed before release")
            expected = row_positions_digest(delayed_entry.chronological_positions)
            if str(row.get("delayed_positions_digest")) != expected:
                raise ValueError(f"{root}: delayed-supervision positions differ from schedule")
            if int(row["delayed_sample_count"]) != 100:
                raise ValueError(f"{root}: delayed supervision does not contain 100 rows")
        elif available != 0:
            raise ValueError(f"{root}: invalid delayed-label availability indicator")
    for (grouped_domain, partition), group in windows.groupby(
        ["current_domain", "partition_kind"], sort=True
    ):
        ordered = group.sort_values("window_id")
        ids = ordered["window_id"].astype(int).tolist()
        if ids != list(range(len(ids))):
            raise ValueError(f"{root}: missing chronological window for {grouped_domain}")
        starts = ordered["row_start"].astype(int).tolist()
        stops = ordered["row_stop"].astype(int).tolist()
        expected_kind = "validation" if str(grouped_domain) == source else "online_stream"
        if partition != expected_kind:
            raise ValueError(f"{root}: health window partition role differs for {grouped_domain}")
        allowed_start, allowed_stop = _range(provenance, str(grouped_domain), expected_kind)
        if starts[0] != allowed_start or any(
            left != right for left, right in zip(stops[:-1], starts[1:], strict=True)
        ):
            raise ValueError(f"{root}: health windows are not chronologically contiguous")
        sizes = [stop - start for start, stop in zip(starts, stops, strict=True)]
        if any(size != resolved_config.experiment.window_size for size in sizes[:-1]) or any(
            size <= 0 or size > resolved_config.experiment.window_size for size in sizes
        ):
            raise ValueError(f"{root}: health window sizes violate the frozen contract")
        if not smoke and stops[-1] != allowed_stop:
            raise ValueError(
                f"{root}: non-smoke run omits chronological windows for {grouped_domain}"
            )
    expected_count = int(summary.get("health_window_count", -1))
    if expected_count != len(windows):
        raise ValueError(f"{root}: health window count differs from summary")
    if summary.get("status") != "complete":
        raise ValueError(f"{root}: health run summary is not complete")
    if summary.get("source_domain") != source:
        raise ValueError(f"{root}: health run summary source domain differs")
    if summary.get("sequence") != list(sequence):
        raise ValueError(f"{root}: health run summary sequence differs")
    if int(summary.get("seed", -1)) != seed:
        raise ValueError(f"{root}: health run summary seed differs")
    expected_state_counts = {
        state: int((windows["health_state"].astype(str) == state).sum())
        for state in ("SAFE", "UNCERTAIN", "HARMFUL")
    }
    if dict(_mapping(summary.get("state_counts"), "health run state counts")) != (
        expected_state_counts
    ):
        raise ValueError(f"{root}: health run summary state counts differ")
    if int(provenance.get("seed", -1)) != seed or provenance.get("sequence") != list(sequence):
        raise ValueError(f"{root}: health run provenance identity differs")
    source_identity = checkpoint_sha
    return ValidatedHealthRun(
        root,
        seed,
        source,
        sequence,
        source_identity,
        _contract_digest(config, provenance),
        windows,
    )


__all__ = [
    "HEALTH_ARTIFACT_VERSION",
    "REQUIRED_HEALTH_FILES",
    "ValidatedHealthRun",
    "validate_health_run",
]
