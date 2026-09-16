"""Artifact-only full-matrix preflight for RDX-004 training evidence.

This module deliberately stops before experiment execution.  It resolves the
canonical Study-4 B100 Oracle runs from their frozen evaluation contract,
reconstructs each label-blind query population from chronological split
manifests, proves the nested B100/B400/B1600 selection and fixed-audit
contracts, and writes one deterministic, independently re-derived preflight
bundle.

No function in this module imports or calls the Study-4 experiment runner.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from danids.adaptation.memory import deterministic_replay_exemplars
from danids.config.rdx_training_evidence import (
    RDX004_ALL_BUDGETS,
    RDX004_BASE_SELECTOR_VERSION,
    RDX004_EXPECTED_NEW_RUNS,
    RDX004_EXPECTED_QUERY_EVENTS,
    RDX004_HISTORICAL_REPLAY_CAPACITY,
    RDX004_METHOD,
    RDX004_PROJECTION_SEED_OFFSET,
    RDX004_PROTOCOL_SHA256,
    RDX004_PROTOCOL_VERSION,
    RDX004_REPLAY_PROJECTION_VERSION,
    RDX004_SELECTOR_VERSION,
    RDX004Budget,
    RDX004RunIdentity,
    RDX004TrainingEvidenceConfig,
    build_rdx004_run_roster,
    load_rdx004_training_evidence_config,
)
from danids.config.study4 import Study4Method
from danids.continual.supervision import row_positions_digest
from danids.data.manifests import SplitManifest
from danids.data.materialized import open_existing_materialized_dataset
from danids.evaluation.study1 import validate_static_study1_run
from danids.evaluation.study4 import validate_study4_evaluation, validate_study4_run
from danids.policy.rdx_training_evidence import FixedAuditPlan, select_rdx_nested_query
from danids.streaming.prequential import derive_supervision_scope_token

RDX004_PREFLIGHT_VERSION: Final = "rdx004-training-evidence-preflight-v1"
RDX004_PREFLIGHT_VERDICT_GO: Final = "RDX-004 PRE-FLIGHT GO"
RDX004_PREFLIGHT_VERDICT_NO_GO: Final = "RDX-004 PRE-FLIGHT NO-GO"
RDX004_NEXT_AUTHORISED_TASK: Final = (
    "independent review of the RDX-005 implementation and 24-run preflight."
)

SUMMARY_FILENAME: Final = "rdx004_preflight_summary.json"
CONTRACT_FILENAME: Final = "rdx004_preflight_contract.json"
RUN_MATRIX_FILENAME: Final = "rdx004_run_matrix.csv"
QUERY_MATRIX_FILENAME: Final = "rdx004_query_matrix.csv"
NESTED_SELECTION_FILENAME: Final = "rdx004_nested_selection.csv"
AUDIT_IDENTITY_FILENAME: Final = "rdx004_audit_identity.csv"
TRAINING_ALLOCATION_FILENAME: Final = "rdx004_training_allocation.csv"
RELEASE_CHRONOLOGY_FILENAME: Final = "rdx004_release_chronology.csv"
STARTING_STATE_FILENAME: Final = "rdx004_starting_state_pairing.csv"
ACTION_CAPACITY_FILENAME: Final = "rdx004_action_capacity.csv"
MANIFEST_FILENAME: Final = "artifact_manifest.json"

RUN_MATRIX_SCHEMA: Final = (
    "run_id",
    "budget",
    "sequence",
    "seed",
    "method",
    "paired_b100_experiment_id",
    "paired_b100_run_path",
    "paired_b100_artifact_manifest_sha256",
    "paired_b100_manifest_bundle_digest",
    "paired_b100_scientific_contract_digest",
    "source_domain",
    "study1_run_path",
    "source_checkpoint_sha256",
    "initial_model_digest",
    "preprocessor_digest",
    "threshold_digest",
    "source_split_manifest_set_digest",
    "oracle_controls_digest",
    "historical_replay_contract_digest",
    "protocol_sha256",
    "config_digest",
    "status",
)

QUERY_MATRIX_SCHEMA: Final = (
    "run_id",
    "budget",
    "sequence",
    "seed",
    "paired_b100_experiment_id",
    "stage",
    "current_domain",
    "dataset_fingerprint",
    "opaque_scope_token",
    "query_ordinal",
    "selection_window_id",
    "prediction_index",
    "release_prediction_index",
    "candidate_population_size",
    "candidate_population_digest",
    "requested_row_count",
    "training_row_count",
    "audit_row_count",
    "query_set_digest",
    "audit_set_digest",
    "training_set_digest",
    "b100_reference_query_digest",
    "selector_version",
    "inherited_selector_version",
    "status",
)

NESTED_SELECTION_SCHEMA: Final = (
    "sequence",
    "seed",
    "paired_b100_experiment_id",
    "stage",
    "current_domain",
    "dataset_fingerprint",
    "opaque_scope_token",
    "query_ordinal",
    "selection_window_id",
    "prediction_index",
    "candidate_population_size",
    "candidate_population_digest",
    "q100_size",
    "q100_positions_json",
    "q100_digest",
    "persisted_b100_positions_json",
    "persisted_b100_digest",
    "q400_size",
    "q400_positions_json",
    "q400_digest",
    "q1600_size",
    "q1600_positions_json",
    "q1600_digest",
    "reconstructed_q100_equals_persisted",
    "q100_subset_q400",
    "q400_subset_q1600",
    "no_duplicate_with_prior_ordinals",
    "status",
)

AUDIT_IDENTITY_SCHEMA: Final = (
    "sequence",
    "seed",
    "paired_b100_experiment_id",
    "stage",
    "current_domain",
    "opaque_scope_token",
    "query_ordinal",
    "audit_count",
    "fixed_audit_positions_json",
    "fixed_audit_digest",
    "audit_subset_q100",
    "audit_subset_q400",
    "audit_subset_q1600",
    "audit_b100_equals_b400_equals_b1600",
    "audit_excluded_from_all_training",
    "status",
)

TRAINING_ALLOCATION_SCHEMA: Final = (
    "sequence",
    "seed",
    "paired_b100_experiment_id",
    "stage",
    "current_domain",
    "opaque_scope_token",
    "query_ordinal",
    "budget",
    "query_count",
    "query_set_digest",
    "audit_count",
    "audit_set_digest",
    "training_count",
    "expected_training_count",
    "training_positions_json",
    "training_set_digest",
    "exact_query_minus_fixed_audit",
    "train_audit_disjoint",
    "b100_training_positions_retained",
    "status",
)

RELEASE_CHRONOLOGY_SCHEMA: Final = (
    "run_id",
    "budget",
    "sequence",
    "seed",
    "paired_b100_experiment_id",
    "query_ordinal",
    "query_stage",
    "query_domain",
    "query_local_window_id",
    "query_prediction_index",
    "release_stage",
    "release_domain",
    "release_local_window_id",
    "release_prediction_index",
    "prediction_before_query",
    "available_on_query_prediction",
    "delay_prediction_count",
    "cross_boundary_release",
    "terminal_pending",
    "matches_b100_timing",
    "status",
)

STARTING_STATE_SCHEMA: Final = (
    "run_id",
    "budget",
    "sequence",
    "seed",
    "paired_b100_experiment_id",
    "source_domain",
    "study1_run_path",
    "study1_experiment_id",
    "source_checkpoint_sha256",
    "initial_model_digest",
    "preprocessor_digest",
    "threshold_digest",
    "dataset_fingerprints_digest",
    "source_split_manifest_set_digest",
    "study1_checkpoint_file_matches",
    "study1_frozen_state_matches",
    "seed_matches",
    "sequence_matches",
    "source_detector_retrained",
    "status",
)

ACTION_CAPACITY_SCHEMA: Final = (
    "action",
    "budget",
    "optimizer_code_path",
    "cumulative_available_target_rows_json",
    "maximum_available_current_target_rows",
    "maximum_rows_action_can_consume",
    "expected_consumed_current_target_rows",
    "expected_optimizer_steps_at_maximum",
    "historical_replay_capacity_per_domain",
    "historical_replay_rows_by_later_stage_json",
    "a1_minimum_benign_support",
    "a1_structurally_feasible",
    "evidence_truncated",
    "treatment_contrast_preserved",
    "status",
)

OUTPUT_SCHEMAS: Final[dict[str, tuple[str, ...]]] = {
    RUN_MATRIX_FILENAME: RUN_MATRIX_SCHEMA,
    QUERY_MATRIX_FILENAME: QUERY_MATRIX_SCHEMA,
    NESTED_SELECTION_FILENAME: NESTED_SELECTION_SCHEMA,
    AUDIT_IDENTITY_FILENAME: AUDIT_IDENTITY_SCHEMA,
    TRAINING_ALLOCATION_FILENAME: TRAINING_ALLOCATION_SCHEMA,
    RELEASE_CHRONOLOGY_FILENAME: RELEASE_CHRONOLOGY_SCHEMA,
    STARTING_STATE_FILENAME: STARTING_STATE_SCHEMA,
    ACTION_CAPACITY_FILENAME: ACTION_CAPACITY_SCHEMA,
}
ALL_OUTPUT_FILES: Final = frozenset(
    {SUMMARY_FILENAME, CONTRACT_FILENAME, MANIFEST_FILENAME, *OUTPUT_SCHEMAS}
)


class RdxTrainingEvidencePreflightError(RuntimeError):
    """Raised when any RDX-004 preflight invariant fails closed."""


@dataclass(frozen=True, slots=True)
class _CanonicalRun:
    experiment_id: str
    path: Path
    sequence: tuple[str, ...]
    seed: int
    artifact_manifest_sha256: str
    manifest_bundle_digest: str
    scientific_contract_digest: str
    source_checkpoint_sha256: str
    initial_model_digest: str
    preprocessor_digest: str
    threshold_digest: str
    source_domain: str
    study1_run_path: Path
    study1_experiment_id: str
    dataset_fingerprints: dict[str, str]
    partition_ranges: dict[str, Any]
    split_manifest_paths: tuple[Path, ...]
    split_manifest_sha256: dict[str, str]
    split_manifest_set_digest: str
    manifests_by_domain: dict[str, SplitManifest]
    materialization_root: Path
    query_log: dict[str, Any]
    windows_by_index: dict[int, dict[str, Any]]
    oracle_controls_digest: str
    historical_replay_contract_digest: str
    source_files_sha256: dict[str, str]


@dataclass(frozen=True, slots=True)
class _MatchedEvent:
    source: _CanonicalRun
    stage: int
    current_domain: str
    dataset_fingerprint: str
    opaque_scope_token: str
    query_ordinal: int
    selection_window_id: int
    prediction_index: int
    release_prediction_index: int
    candidate_positions: tuple[int, ...]
    candidate_digest: str
    q100: tuple[int, ...]
    q400: tuple[int, ...]
    q1600: tuple[int, ...]
    q100_digest: str
    q400_digest: str
    q1600_digest: str
    audit: tuple[int, ...]
    audit_digest: str
    b100_training: tuple[int, ...]
    b100_training_digest: str


@dataclass(frozen=True, slots=True)
class RdxTrainingEvidencePreflight:
    """Complete in-memory RDX-004 preflight ready for write-once sealing."""

    contract: dict[str, Any]
    summary: dict[str, Any]
    tables: dict[str, list[dict[str, object]]]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digests(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != MANIFEST_FILENAME
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RdxTrainingEvidencePreflightError(f"cannot load JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RdxTrainingEvidencePreflightError(f"JSON artifact root must be an object: {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RdxTrainingEvidencePreflightError(f"cannot load YAML artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RdxTrainingEvidencePreflightError(f"YAML artifact root must be a mapping: {path}")
    return payload


def _csv_cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return _canonical_json(value)
    return value


def _csv_text(columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(columns):
            raise RdxTrainingEvidencePreflightError(
                "preflight table row differs from its fixed schema: "
                f"missing={sorted(set(columns).difference(row))}, "
                f"extra={sorted(set(row).difference(columns))}"
            )
        writer.writerow({column: _csv_cell(row[column]) for column in columns})
    return stream.getvalue()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RdxTrainingEvidencePreflightError(f"{name} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RdxTrainingEvidencePreflightError(f"{name} is not a SHA-256 digest") from exc
    return value.lower()


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise RdxTrainingEvidencePreflightError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise RdxTrainingEvidencePreflightError(f"{name} must be an integer")
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise RdxTrainingEvidencePreflightError(f"{name} must be an integer") from exc
    raise RdxTrainingEvidencePreflightError(f"{name} must be an integer")


def reconstruct_candidate_positions(
    manifest: SplitManifest, *, window_id: int, window_size: int
) -> tuple[int, ...]:
    """Reconstruct one exact ONLINE_STREAM candidate population without labels."""

    if manifest.online_stream is None or manifest.domain_role != "later":
        raise RdxTrainingEvidencePreflightError("query candidates require a later ONLINE_STREAM")
    if type(window_id) is not int or window_id < 0 or window_size <= 0:
        raise RdxTrainingEvidencePreflightError("query window identity is invalid")
    start = manifest.online_stream.start + window_id * window_size
    if start >= manifest.online_stream.stop:
        raise RdxTrainingEvidencePreflightError("query window lies beyond the ONLINE_STREAM")
    stop = min(start + window_size, manifest.online_stream.stop)
    positions = tuple(range(start, stop))
    if len(positions) != len(set(positions)) or positions != tuple(sorted(positions)):
        raise RdxTrainingEvidencePreflightError(
            "candidate population is not chronological/distinct"
        )
    if any(
        manifest.permanent_holdout.start <= value < manifest.permanent_holdout.stop
        for value in positions
    ):
        raise RdxTrainingEvidencePreflightError("candidate population overlaps permanent holdout")
    return positions


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise RdxTrainingEvidencePreflightError(f"{name} must be a string-keyed mapping")
    return {str(key): item for key, item in value.items()}


def _records(value: object, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise RdxTrainingEvidencePreflightError(f"{name} must be a list")
    return [_mapping(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _as_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise RdxTrainingEvidencePreflightError(f"{name} must be a boolean")
    return value


def _manifest_ranges(manifest: SplitManifest) -> dict[str, list[int] | None]:
    def payload(value: object) -> list[int] | None:
        if value is None:
            return None
        start = getattr(value, "start", None)
        stop = getattr(value, "stop", None)
        return [_as_int(start, "manifest range start"), _as_int(stop, "manifest range stop")]

    return {
        "initial_train": payload(manifest.initial_train),
        "validation": payload(manifest.validation),
        "online_stream": payload(manifest.online_stream),
        "permanent_holdout": payload(manifest.permanent_holdout),
    }


def _validate_manifest_bundle(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_FILENAME
    manifest = _load_json(manifest_path)
    files = _tree_digests(root)
    if (
        set(manifest) != {"version", "files", "bundle_digest"}
        or manifest.get("files") != files
        or manifest.get("bundle_digest") != _digest(files)
    ):
        raise RdxTrainingEvidencePreflightError(
            f"artifact manifest differs from the exact files under {root}"
        )
    return manifest


def _validate_b100_contract(
    *,
    config: dict[str, Any],
    provenance: dict[str, Any],
    study1_config: dict[str, Any],
    study1_provenance: dict[str, Any],
    rdx_config: RDX004TrainingEvidenceConfig,
) -> tuple[str, str]:
    core = _mapping(config.get("core_contract"), "B100 core_contract")
    supervision = _mapping(core.get("supervision"), "B100 supervision")
    expected_supervision = {
        "label_budget_per_later_domain": 100,
        "label_delay_windows": 1,
        "query_batch_size": 25,
        "maximum_query_events": 4,
        "maximum_pending_queries": 1,
        "maximum_queries_per_window": 1,
        "selector_version": RDX004_BASE_SELECTOR_VERSION,
        "cancel_pending_on_safe": False,
        "allow_partial_query": False,
    }
    if any(supervision.get(key) != value for key, value in expected_supervision.items()):
        raise RdxTrainingEvidencePreflightError(
            "canonical Oracle source differs from the frozen B100 supervision contract"
        )

    allocation = _mapping(core.get("allocation"), "B100 allocation")
    expected_allocation = {
        "replay_per_release": 20,
        "audit_per_release": 5,
        "source_replay_capacity": 400,
        "source_audit_capacity": 100,
        "later_replay_capacity": 400,
        "later_audit_capacity": 100,
        "selection": "deterministic_binary_stratified",
        "current_scope_audit_eligible": False,
        "activate_when_historical": True,
    }
    if any(allocation.get(key) != value for key, value in expected_allocation.items()):
        raise RdxTrainingEvidencePreflightError(
            "canonical Oracle source differs from the frozen replay/audit contract"
        )

    action_contract = _mapping(
        rdx_config.to_dict().get("frozen_action_contract"),
        "RDX frozen_action_contract",
    )
    actions = _mapping(core.get("actions"), "B100 actions")
    adaptation = _mapping(core.get("adaptation"), "B100 adaptation")
    operating = _mapping(core.get("operating_envelope"), "B100 operating envelope")
    model = _mapping(study1_config.get("model"), "Study-1 model")
    if (
        action_contract.get("action_order") != actions.get("ordering")
        or action_contract.get("architecture") != "compact_mlp_256_128_64_dropout_0.2"
        or model.get("name") != "static_mlp"
        or model.get("hidden_dimensions") != [256, 128, 64]
        or float(model.get("dropout", -1.0)) != 0.2
        or action_contract.get("preprocessor") != study1_provenance.get("preprocessor_version")
        or action_contract.get("optimizer") != adaptation.get("optimizer")
        or action_contract.get("learning_rate") != adaptation.get("learning_rate")
        or action_contract.get("weight_decay") != adaptation.get("weight_decay")
        or action_contract.get("epochs") != adaptation.get("epochs")
        or action_contract.get("batch_size") != adaptation.get("batch_size")
        or action_contract.get("a1_minimum_benign_support")
        != actions.get("a1_minimum_benign_support")
        or action_contract.get("target_fpr") != operating.get("target_fpr")
        or action_contract.get("permanent_holdout_used_for_choice") is not False
    ):
        raise RdxTrainingEvidencePreflightError(
            "RDX action controls differ from the paired frozen Oracle controls"
        )
    oracle_semantics = _mapping(provenance.get("offline_oracle_semantics"), "B100 Oracle semantics")
    if (
        oracle_semantics.get("action_order") != actions.get("ordering")
        or oracle_semantics.get("horizon") != "first_fresh_same_domain_successor"
        or oracle_semantics.get("permanent_holdout_used_for_choice") is not False
        or provenance.get("permanent_holdout_policy_visible") is not False
    ):
        raise RdxTrainingEvidencePreflightError("B100 Oracle information boundary changed")

    controls = {
        "architecture": {
            "name": model.get("name"),
            "hidden_dimensions": model.get("hidden_dimensions"),
            "dropout": model.get("dropout"),
        },
        "preprocessor_version": study1_provenance.get("preprocessor_version"),
        "actions": actions,
        "adaptation": adaptation,
        "operating_envelope": operating,
        "references": core.get("references"),
        "oracle_semantics": oracle_semantics,
        "rdx_frozen_action_contract": action_contract,
    }
    historical = {
        "source_replay_capacity": allocation.get("source_replay_capacity"),
        "source_audit_capacity": allocation.get("source_audit_capacity"),
        "later_replay_capacity": allocation.get("later_replay_capacity"),
        "later_audit_capacity": allocation.get("later_audit_capacity"),
        "selection": allocation.get("selection"),
        "current_scope_audit_eligible": allocation.get("current_scope_audit_eligible"),
        "activate_when_historical": allocation.get("activate_when_historical"),
    }
    return _digest(controls), _digest(historical)


def _resolve_canonical_sources(
    config: RDX004TrainingEvidenceConfig,
    *,
    study4_evaluation_root: Path,
    manifest_root: Path,
) -> tuple[tuple[_CanonicalRun, ...], dict[str, Any], dict[str, Any]]:
    evaluation_root = study4_evaluation_root.resolve()
    if (
        config.canonical_b100_authority.resolve()
        != (evaluation_root / "evaluation_contract.json").resolve()
    ):
        raise RdxTrainingEvidencePreflightError(
            "Study-4 evaluation root differs from the configured canonical B100 authority"
        )
    validate_study4_evaluation(evaluation_root)
    aggregate_manifest = _validate_manifest_bundle(evaluation_root)
    evaluation_contract_path = evaluation_root / "evaluation_contract.json"
    evaluation_contract = _load_json(evaluation_contract_path)
    if evaluation_contract.get("allow_smoke") is not False:
        raise RdxTrainingEvidencePreflightError("canonical evaluation permits smoke runs")
    if evaluation_contract.get("allow_incomplete") is not False:
        raise RdxTrainingEvidencePreflightError("canonical evaluation permits incomplete runs")

    roster = build_rdx004_run_roster(config)
    expected_ids = {identity.paired_b100_experiment_id for identity in roster}
    raw_source_runs = _records(evaluation_contract.get("source_runs"), "source_runs")
    source_records: dict[str, dict[str, Any]] = {}
    for record in raw_source_runs:
        experiment_id = str(record.get("experiment_id", ""))
        if experiment_id not in expected_ids:
            continue
        if experiment_id in source_records:
            raise RdxTrainingEvidencePreflightError(
                f"duplicate canonical B100 authority record: {experiment_id}"
            )
        source_records[experiment_id] = record
    if set(source_records) != expected_ids:
        raise RdxTrainingEvidencePreflightError(
            "evaluation contract does not contain the exact 12 paired B100 Oracle runs"
        )

    treatment_digests = _mapping(
        evaluation_contract.get("treatment_scientific_contract_digests"),
        "treatment scientific contract digests",
    )
    expected_scientific_digest = _require_sha256(
        treatment_digests.get(RDX004_METHOD), "OFFLINE_ORACLE scientific contract"
    )
    manifest_root = manifest_root.resolve()
    sources: list[_CanonicalRun] = []
    seen_pairs: set[tuple[tuple[str, ...], int]] = set()
    for rotation in config.rotations:
        for seed in config.seeds:
            pair_id = next(
                identity.paired_b100_experiment_id
                for identity in roster
                if identity.rotation == rotation and identity.seed == seed
            )
            record = source_records[pair_id]
            run_path = Path(str(record.get("path", ""))).resolve()
            manifest_path = run_path / MANIFEST_FILENAME
            declared_manifest_sha = _require_sha256(
                record.get("artifact_manifest_sha256"),
                f"{pair_id} artifact manifest",
            )
            if _file_sha256(manifest_path) != declared_manifest_sha:
                raise RdxTrainingEvidencePreflightError(
                    f"canonical artifact-manifest identity changed for {pair_id}"
                )
            validated = validate_study4_run(run_path, allow_smoke=False)
            if (
                validated.experiment_id != pair_id
                or validated.method is not Study4Method.OFFLINE_ORACLE
                or validated.sequence != rotation
                or validated.seed != seed
                or validated.smoke
                or validated.contract_digest != expected_scientific_digest
            ):
                raise RdxTrainingEvidencePreflightError(
                    f"canonical B100 run identity/contract differs for {pair_id}"
                )
            pair = (validated.sequence, validated.seed)
            if pair in seen_pairs:
                raise RdxTrainingEvidencePreflightError("duplicate B100 rotation/seed pair")
            seen_pairs.add(pair)
            run_manifest = _load_json(manifest_path)

            run_config = _load_yaml(run_path / "config.resolved.yaml")
            run_provenance = _load_json(run_path / "provenance.json")
            initial = _load_json(run_path / "initial_state_identity.json")
            query_log = _load_json(run_path / "query_log.json")
            if (
                run_config.get("experiment_id") != pair_id
                or run_config.get("method") != RDX004_METHOD
                or tuple(run_config.get("sequence", ())) != rotation
                or _as_int(run_config.get("seed"), "B100 seed") != seed
                or run_config.get("smoke") is not False
            ):
                raise RdxTrainingEvidencePreflightError(
                    f"resolved B100 configuration differs for {pair_id}"
                )
            source_checkpoint = _require_sha256(
                initial.get("source_checkpoint_sha256"), "source checkpoint"
            )
            if source_checkpoint != validated.source_checkpoint_sha256:
                raise RdxTrainingEvidencePreflightError("B100 checkpoint identity is inconsistent")
            initial_model = _require_sha256(
                initial.get("model_digest_before"), "initial model digest"
            )
            preprocessor = _require_sha256(
                initial.get("preprocessor_digest_before"), "preprocessor digest"
            )
            threshold = _require_sha256(initial.get("threshold_digest_before"), "threshold digest")
            if (
                initial.get("preprocessor_digest_after") != preprocessor
                or initial.get("threshold_digest_after") != threshold
                or initial.get("source_checkpoint_sha256_before") != source_checkpoint
                or initial.get("source_checkpoint_sha256_after") != source_checkpoint
            ):
                raise RdxTrainingEvidencePreflightError(
                    f"B100 starting state was not reused exactly for {pair_id}"
                )

            study1_path = Path(str(run_config.get("initial_run", ""))).resolve()
            study1_summary = _load_json(study1_path / "summary.json")
            study1_provenance = _load_json(study1_path / "provenance.json")
            study1_config = _load_yaml(study1_path / "config.resolved.yaml")
            validated_study1 = validate_static_study1_run(study1_path)
            expected_study1_id = f"E1_STATIC_MLP_{'-'.join(rotation)}_s{seed}"
            evidence = _mapping(
                study1_summary.get("frozen_state_evidence"), "Study-1 frozen state evidence"
            )
            fingerprints = _mapping(
                run_provenance.get("dataset_fingerprints"), "B100 dataset fingerprints"
            )
            if (
                validated_study1.path != study1_path
                or validated_study1.source != rotation[0]
                or validated_study1.sequence != rotation
                or validated_study1.seed != seed
                or dict(validated_study1.dataset_fingerprints) != fingerprints
                or study1_summary.get("experiment_id") != expected_study1_id
                or study1_summary.get("sequence") != list(rotation)
                or study1_summary.get("smoke") is not False
                or study1_config.get("experiment_id") != expected_study1_id
                or tuple(
                    _mapping(study1_config.get("datasets"), "Study-1 datasets").get("sequence", ())
                )
                != rotation
                or _as_int(study1_config.get("seed"), "Study-1 config seed") != seed
                or _as_int(study1_provenance.get("seed"), "Study-1 provenance seed") != seed
                or _file_sha256(study1_path / "best_model.pt") != source_checkpoint
                or evidence.get("model_before") != initial_model
                or evidence.get("model_after") != initial_model
                or evidence.get("preprocessor_before") != preprocessor
                or evidence.get("preprocessor_after") != preprocessor
                or evidence.get("threshold_before") != threshold
                or evidence.get("threshold_after") != threshold
                or evidence.get("model_unchanged") is not True
                or evidence.get("preprocessor_unchanged") is not True
                or evidence.get("threshold_unchanged") is not True
                or _as_int(
                    evidence.get("target_optimizer_steps"),
                    "Study-1 target optimizer steps",
                )
                != 0
            ):
                raise RdxTrainingEvidencePreflightError(
                    f"paired Study-1 starting state differs for {pair_id}"
                )

            ranges = _mapping(
                run_provenance.get("manifest_partition_ranges"), "B100 manifest ranges"
            )
            health_contract = _mapping(
                run_config.get("health_signal_contract"), "B100 health signal contract"
            )
            materialization = _mapping(
                health_contract.get("materialization"), "B100 materialization"
            )
            materialization_root = Path(str(materialization.get("cache_root", ""))).resolve()
            manifests: dict[str, SplitManifest] = {}
            split_paths: list[Path] = []
            split_hashes: dict[str, str] = {}
            for stage, domain in enumerate(rotation, start=1):
                split_path = (
                    manifest_root
                    / "-".join(rotation)
                    / f"s{seed}"
                    / f"stage-{stage:02d}-{domain}.json"
                ).resolve()
                manifest = SplitManifest.from_json(split_path)
                expected_role = "initial" if stage == 1 else "later"
                if (
                    manifest.dataset_id != domain
                    or manifest.domain_role != expected_role
                    or manifest.generation_seed != seed
                    or manifest.source.sha256 != fingerprints.get(domain)
                    or _manifest_ranges(manifest) != ranges.get(domain)
                ):
                    raise RdxTrainingEvidencePreflightError(
                        f"split manifest differs from B100 provenance: {split_path}"
                    )
                materialized = open_existing_materialized_dataset(materialization_root, manifest)
                if materialized.metadata.source_sha256 != manifest.source.sha256:
                    raise RdxTrainingEvidencePreflightError(
                        f"materialized cache identity differs for {domain}"
                    )
                manifests[domain] = manifest
                split_paths.append(split_path)
                split_hashes[split_path.name] = _file_sha256(split_path)

            controls_digest, historical_digest = _validate_b100_contract(
                config=run_config,
                provenance=run_provenance,
                study1_config=study1_config,
                study1_provenance=study1_provenance,
                rdx_config=config,
            )
            windows_by_index: dict[int, dict[str, Any]] = {}
            for window in validated.windows:
                index = _as_int(window.get("prediction_index"), "window prediction index")
                if index in windows_by_index:
                    raise RdxTrainingEvidencePreflightError("duplicate Study-4 prediction index")
                windows_by_index[index] = window

            source_files = {
                str(path): _file_sha256(path)
                for path in (
                    manifest_path,
                    run_path / "config.resolved.yaml",
                    run_path / "provenance.json",
                    run_path / "initial_state_identity.json",
                    run_path / "query_log.json",
                    study1_path / "best_model.pt",
                    study1_path / "config.resolved.yaml",
                    study1_path / "provenance.json",
                    study1_path / "summary.json",
                    *split_paths,
                )
            }
            sources.append(
                _CanonicalRun(
                    experiment_id=pair_id,
                    path=run_path,
                    sequence=rotation,
                    seed=seed,
                    artifact_manifest_sha256=declared_manifest_sha,
                    manifest_bundle_digest=_require_sha256(
                        run_manifest.get("bundle_digest"), "B100 manifest bundle digest"
                    ),
                    scientific_contract_digest=validated.contract_digest,
                    source_checkpoint_sha256=source_checkpoint,
                    initial_model_digest=initial_model,
                    preprocessor_digest=preprocessor,
                    threshold_digest=threshold,
                    source_domain=rotation[0],
                    study1_run_path=study1_path,
                    study1_experiment_id=expected_study1_id,
                    dataset_fingerprints={
                        domain: _require_sha256(fingerprints.get(domain), domain)
                        for domain in rotation
                    },
                    partition_ranges=ranges,
                    split_manifest_paths=tuple(split_paths),
                    split_manifest_sha256=dict(sorted(split_hashes.items())),
                    split_manifest_set_digest=_digest(dict(sorted(split_hashes.items()))),
                    manifests_by_domain=manifests,
                    materialization_root=materialization_root,
                    query_log=query_log,
                    windows_by_index=windows_by_index,
                    oracle_controls_digest=controls_digest,
                    historical_replay_contract_digest=historical_digest,
                    source_files_sha256=source_files,
                )
            )

    if len(sources) != 12 or len({source.source_checkpoint_sha256 for source in sources}) != 12:
        raise RdxTrainingEvidencePreflightError(
            "canonical B100 roster/starting checkpoints are not 12 unique paired states"
        )
    if len({source.oracle_controls_digest for source in sources}) != 1:
        raise RdxTrainingEvidencePreflightError("paired B100 Oracle controls are inconsistent")
    if len({source.historical_replay_contract_digest for source in sources}) != 1:
        raise RdxTrainingEvidencePreflightError("B100 historical replay controls are inconsistent")
    evaluation_identity = {
        "path": str(evaluation_root),
        "evaluation_contract_path": str(evaluation_contract_path.resolve()),
        "evaluation_contract_sha256": _file_sha256(evaluation_contract_path),
        "artifact_manifest_path": str((evaluation_root / MANIFEST_FILENAME).resolve()),
        "artifact_manifest_sha256": _file_sha256(evaluation_root / MANIFEST_FILENAME),
        "artifact_manifest_bundle_digest": _require_sha256(
            aggregate_manifest.get("bundle_digest"), "evaluation bundle digest"
        ),
        "shared_contract_digest": _require_sha256(
            evaluation_contract.get("shared_contract_digest"), "shared contract digest"
        ),
    }
    source_sentinels = {
        path: digest for source in sources for path, digest in source.source_files_sha256.items()
    }
    return tuple(sources), evaluation_identity, dict(sorted(source_sentinels.items()))


def _derive_matched_events(sources: Sequence[_CanonicalRun]) -> tuple[_MatchedEvent, ...]:
    matched: list[_MatchedEvent] = []
    for source in sources:
        scopes = _records(source.query_log.get("scopes"), "B100 query scopes")
        if len(scopes) != 3:
            raise RdxTrainingEvidencePreflightError(
                f"{source.experiment_id} does not contain exactly three later scopes"
            )
        scopes_by_stage = {
            _as_int(scope.get("stage"), "query scope stage"): scope for scope in scopes
        }
        if set(scopes_by_stage) != {1, 2, 3} or len(scopes_by_stage) != len(scopes):
            raise RdxTrainingEvidencePreflightError("B100 query scope stages are not 1/2/3")
        for stage in range(1, 4):
            scope = scopes_by_stage[stage]
            current_domain = source.sequence[stage]
            if scope.get("current_domain") != current_domain:
                raise RdxTrainingEvidencePreflightError("B100 query scope/domain mismatch")
            manifest = source.manifests_by_domain[current_domain]
            assert manifest.online_stream is not None
            scope_token = _require_sha256(
                scope.get("opaque_scope_token"), "B100 opaque scope token"
            )
            expected_scope_token = derive_supervision_scope_token(
                source_sha256=manifest.source.sha256,
                split_version=manifest.split_version,
                row_start=manifest.online_stream.start,
                row_stop=manifest.online_stream.stop,
            )
            if scope_token != expected_scope_token:
                raise RdxTrainingEvidencePreflightError(
                    "B100 opaque scope token differs from the exact split manifest"
                )
            events = _records(scope.get("query_events"), "B100 query events")
            allocation = _mapping(scope.get("allocation"), "B100 allocation")
            releases = _records(allocation.get("releases"), "B100 releases")
            supervision = _mapping(scope.get("supervision"), "B100 supervision state")
            delayed = _mapping(supervision.get("delayed_queue"), "B100 delayed queue")
            if (
                len(events) != 4
                or len(releases) != 4
                or _as_int(allocation.get("release_count"), "B100 release count") != 4
                or _as_int(allocation.get("label_budget"), "B100 allocation budget") != 100
                or _as_int(allocation.get("labels_per_release"), "B100 release size") != 25
                or _as_int(allocation.get("replay_per_release"), "B100 replay size") != 20
                or _as_int(allocation.get("audit_per_release"), "B100 audit size") != 5
                or _as_int(supervision.get("query_count"), "B100 query count") != 4
                or _as_int(delayed.get("delay_windows"), "B100 delay") != 1
            ):
                raise RdxTrainingEvidencePreflightError(
                    "B100 scope differs from the exact 4-query 25/20/5 D1 contract"
                )
            releases_by_query = {
                _as_int(item.get("query_window"), "release query window"): item for item in releases
            }
            if len(releases_by_query) != 4:
                raise RdxTrainingEvidencePreflightError("duplicate B100 release query window")
            events_by_ordinal: dict[int, dict[str, Any]] = {}
            for event in events:
                selection_raw = _mapping(event.get("selection"), "B100 selection")
                ordinal = _as_int(selection_raw.get("query_ordinal"), "query ordinal")
                if ordinal in events_by_ordinal:
                    raise RdxTrainingEvidencePreflightError("duplicate B100 query ordinal")
                events_by_ordinal[ordinal] = event
            if set(events_by_ordinal) != {0, 1, 2, 3}:
                raise RdxTrainingEvidencePreflightError("B100 query ordinals are not 0/1/2/3")

            prior_queried: set[int] = set()
            scope_events: list[_MatchedEvent] = []
            for ordinal in range(4):
                event = events_by_ordinal[ordinal]
                selection_raw = _mapping(event.get("selection"), "B100 selection")
                prediction_index = _as_int(
                    event.get("prediction_index"), "B100 query prediction index"
                )
                window_id = _as_int(selection_raw.get("window_id"), "selection window ID")
                if (
                    window_id != ordinal
                    or selection_raw.get("opaque_scope_token") != scope_token
                    or _as_int(selection_raw.get("seed"), "selection seed") != source.seed
                    or selection_raw.get("selector_version") != RDX004_BASE_SELECTOR_VERSION
                    or selection_raw.get("status") != "QUERY_SELECTED"
                ):
                    raise RdxTrainingEvidencePreflightError(
                        "canonical B100 selection identity differs from the matched event"
                    )
                window = source.windows_by_index.get(prediction_index)
                if window is None:
                    raise RdxTrainingEvidencePreflightError("B100 query lacks prediction window")
                candidate_positions = reconstruct_candidate_positions(
                    manifest,
                    window_id=window_id,
                    window_size=_as_int(window.get("row_count"), "query window row count"),
                )
                if (
                    _as_int(window.get("stage"), "window stage") != stage
                    or window.get("current_domain") != current_domain
                    or _as_int(window.get("window_id"), "window ID") != window_id
                    or _as_int(window.get("row_start"), "window row start")
                    != candidate_positions[0]
                    or _as_int(window.get("row_stop"), "window row stop")
                    != candidate_positions[-1] + 1
                    or set(candidate_positions).intersection(prior_queried)
                    or len(candidate_positions) < 400
                ):
                    raise RdxTrainingEvidencePreflightError(
                        "query candidate population is not an exact legitimate unqueried window"
                    )
                candidate_digest = row_positions_digest(candidate_positions)
                if (
                    _as_int(selection_raw.get("candidate_count"), "candidate count")
                    != len(candidate_positions)
                    or selection_raw.get("current_row_positions_digest") != candidate_digest
                ):
                    raise RdxTrainingEvidencePreflightError(
                        "B100 candidate population differs from manifest reconstruction"
                    )
                nested = select_rdx_nested_query(
                    candidate_positions,
                    seed=source.seed,
                    opaque_scope_token=scope_token,
                    window_id=window_id,
                    query_ordinal=ordinal,
                    current_digest=candidate_digest,
                )
                persisted_q100 = tuple(
                    _as_int(value, "persisted B100 query position")
                    for value in selection_raw.get("selected_positions", ())
                )
                if nested.b100_positions != persisted_q100 or selection_raw.get(
                    "selected_positions_digest"
                ) != nested.digest_for(RDX004Budget.B100):
                    raise RdxTrainingEvidencePreflightError(
                        "RDX nested selector does not reconstruct the canonical B100 prefix"
                    )
                release = releases_by_query.get(prediction_index)
                if release is None:
                    raise RdxTrainingEvidencePreflightError("B100 query lacks matched release")
                release_index = _as_int(release.get("release_index"), "release index")
                release_prediction_index = _as_int(
                    release.get("release_window"), "release prediction index"
                )
                q100 = tuple(
                    _as_int(value, "release queried position")
                    for value in release.get("queried_positions", ())
                )
                training = tuple(
                    _as_int(value, "release training position")
                    for value in release.get("replay_positions", ())
                )
                audit = tuple(
                    _as_int(value, "release audit position")
                    for value in release.get("audit_positions", ())
                )
                if (
                    release_index != ordinal
                    or release_prediction_index != prediction_index + 1
                    or q100 != nested.b100_positions
                    or release.get("queried_positions_digest") != row_positions_digest(q100)
                    or release.get("replay_positions_digest") != row_positions_digest(training)
                    or release.get("audit_positions_digest") != row_positions_digest(audit)
                ):
                    raise RdxTrainingEvidencePreflightError(
                        "canonical allocation/release does not match the B100 query"
                    )
                release_window = source.windows_by_index.get(release_prediction_index)
                if (
                    release_window is None
                    or _as_int(release_window.get("stage"), "release stage") != stage
                    or release_window.get("current_domain") != current_domain
                    or _as_int(release_window.get("window_id"), "release local window")
                    != window_id + 1
                ):
                    raise RdxTrainingEvidencePreflightError(
                        "B100 release does not occur on the next same-domain prediction"
                    )
                plan = FixedAuditPlan(
                    dataset_fingerprint=manifest.source.sha256,
                    opaque_scope_token=scope_token,
                    selection_window_id=window_id,
                    query_ordinal=ordinal,
                    prediction_index=prediction_index,
                    release_prediction_index=release_prediction_index,
                    b100_query_positions=q100,
                    b100_training_positions=training,
                    audit_positions=audit,
                )
                plan.validate_selection(nested)
                if any(
                    value >= manifest.permanent_holdout.start for value in nested.b1600_positions
                ) or set(nested.b1600_positions).intersection(prior_queried):
                    raise RdxTrainingEvidencePreflightError(
                        "RDX nested query reuses a row or crosses permanent holdout"
                    )
                current = _MatchedEvent(
                    source=source,
                    stage=stage,
                    current_domain=current_domain,
                    dataset_fingerprint=manifest.source.sha256,
                    opaque_scope_token=scope_token,
                    query_ordinal=ordinal,
                    selection_window_id=window_id,
                    prediction_index=prediction_index,
                    release_prediction_index=release_prediction_index,
                    candidate_positions=candidate_positions,
                    candidate_digest=candidate_digest,
                    q100=nested.b100_positions,
                    q400=nested.b400_positions,
                    q1600=nested.b1600_positions,
                    q100_digest=nested.digest_for(RDX004Budget.B100),
                    q400_digest=nested.digest_for(RDX004Budget.B400),
                    q1600_digest=nested.digest_for(RDX004Budget.B1600),
                    audit=plan.audit_positions,
                    audit_digest=plan.audit_digest,
                    b100_training=plan.b100_training_positions,
                    b100_training_digest=plan.b100_training_digest,
                )
                scope_events.append(current)
                prior_queried.update(nested.b1600_positions)

            all_audit = {value for event in scope_events for value in event.audit}
            if len(all_audit) != 20:
                raise RdxTrainingEvidencePreflightError(
                    "canonical fixed audit does not contain 20 distinct rows per scope"
                )
            for budget in RDX004_ALL_BUDGETS:
                all_training = {
                    value
                    for event in scope_events
                    for value in set(
                        event.q100
                        if budget is RDX004Budget.B100
                        else event.q400
                        if budget is RDX004Budget.B400
                        else event.q1600
                    ).difference(event.audit)
                }
                if all_training.intersection(all_audit):
                    raise RdxTrainingEvidencePreflightError(
                        "a fixed B100 audit row enters current-scope training"
                    )
            matched.extend(scope_events)
    if len(matched) != 144:
        raise RdxTrainingEvidencePreflightError(
            f"matched B100 event count is {len(matched)}, expected 144"
        )
    return tuple(matched)


def _query_for_budget(event: _MatchedEvent, budget: RDX004Budget) -> tuple[int, ...]:
    if budget is RDX004Budget.B100:
        return event.q100
    if budget is RDX004Budget.B400:
        return event.q400
    if budget is RDX004Budget.B1600:
        return event.q1600
    raise RdxTrainingEvidencePreflightError(f"unsupported RDX budget: {budget}")


def _training_for_budget(event: _MatchedEvent, budget: RDX004Budget) -> tuple[int, ...]:
    query = _query_for_budget(event, budget)
    training = tuple(sorted(set(query).difference(event.audit)))
    expected = budget.training_per_query
    if (
        len(training) != expected
        or set(training).intersection(event.audit)
        or set(training).union(event.audit) != set(query)
        or not set(event.b100_training).issubset(training)
    ):
        raise RdxTrainingEvidencePreflightError(
            f"{budget.value} training rows are not exact query-minus-fixed-audit evidence"
        )
    return training


def _derive_tables(
    config: RDX004TrainingEvidenceConfig,
    roster: Sequence[RDX004RunIdentity],
    sources: Sequence[_CanonicalRun],
    events: Sequence[_MatchedEvent],
) -> dict[str, list[dict[str, object]]]:
    source_by_id = {source.experiment_id: source for source in sources}
    events_by_source: dict[str, list[_MatchedEvent]] = {
        source.experiment_id: [] for source in sources
    }
    for event in events:
        events_by_source[event.source.experiment_id].append(event)
    for values in events_by_source.values():
        values.sort(key=lambda item: (item.stage, item.query_ordinal))

    run_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    chronology_rows: list[dict[str, object]] = []
    starting_rows: list[dict[str, object]] = []
    for identity in roster:
        source = source_by_id.get(identity.paired_b100_experiment_id)
        if source is None:
            raise RdxTrainingEvidencePreflightError(
                f"prospective run lacks canonical pair: {identity.run_id}"
            )
        treatment = config.treatment(identity.budget)
        run_rows.append(
            {
                "run_id": identity.run_id,
                "budget": identity.budget.value,
                "sequence": identity.sequence_token,
                "seed": identity.seed,
                "method": config.method,
                "paired_b100_experiment_id": source.experiment_id,
                "paired_b100_run_path": str(source.path),
                "paired_b100_artifact_manifest_sha256": source.artifact_manifest_sha256,
                "paired_b100_manifest_bundle_digest": source.manifest_bundle_digest,
                "paired_b100_scientific_contract_digest": source.scientific_contract_digest,
                "source_domain": source.source_domain,
                "study1_run_path": str(source.study1_run_path),
                "source_checkpoint_sha256": source.source_checkpoint_sha256,
                "initial_model_digest": source.initial_model_digest,
                "preprocessor_digest": source.preprocessor_digest,
                "threshold_digest": source.threshold_digest,
                "source_split_manifest_set_digest": source.split_manifest_set_digest,
                "oracle_controls_digest": source.oracle_controls_digest,
                "historical_replay_contract_digest": (source.historical_replay_contract_digest),
                "protocol_sha256": config.protocol_sha256,
                "config_digest": config.run_config_sha256(identity),
                "status": "PREFLIGHT_READY",
            }
        )
        starting_rows.append(
            {
                "run_id": identity.run_id,
                "budget": identity.budget.value,
                "sequence": identity.sequence_token,
                "seed": identity.seed,
                "paired_b100_experiment_id": source.experiment_id,
                "source_domain": source.source_domain,
                "study1_run_path": str(source.study1_run_path),
                "study1_experiment_id": source.study1_experiment_id,
                "source_checkpoint_sha256": source.source_checkpoint_sha256,
                "initial_model_digest": source.initial_model_digest,
                "preprocessor_digest": source.preprocessor_digest,
                "threshold_digest": source.threshold_digest,
                "dataset_fingerprints_digest": _digest(source.dataset_fingerprints),
                "source_split_manifest_set_digest": source.split_manifest_set_digest,
                "study1_checkpoint_file_matches": True,
                "study1_frozen_state_matches": True,
                "seed_matches": True,
                "sequence_matches": True,
                "source_detector_retrained": False,
                "status": "STARTING_STATE_PAIRED",
            }
        )
        for event in events_by_source[source.experiment_id]:
            query = _query_for_budget(event, identity.budget)
            training = _training_for_budget(event, identity.budget)
            query_rows.append(
                {
                    "run_id": identity.run_id,
                    "budget": identity.budget.value,
                    "sequence": identity.sequence_token,
                    "seed": identity.seed,
                    "paired_b100_experiment_id": source.experiment_id,
                    "stage": event.stage,
                    "current_domain": event.current_domain,
                    "dataset_fingerprint": event.dataset_fingerprint,
                    "opaque_scope_token": event.opaque_scope_token,
                    "query_ordinal": event.query_ordinal,
                    "selection_window_id": event.selection_window_id,
                    "prediction_index": event.prediction_index,
                    "release_prediction_index": event.release_prediction_index,
                    "candidate_population_size": len(event.candidate_positions),
                    "candidate_population_digest": event.candidate_digest,
                    "requested_row_count": treatment.selected_per_query,
                    "training_row_count": len(training),
                    "audit_row_count": len(event.audit),
                    "query_set_digest": row_positions_digest(query),
                    "audit_set_digest": event.audit_digest,
                    "training_set_digest": row_positions_digest(training),
                    "b100_reference_query_digest": event.q100_digest,
                    "selector_version": RDX004_SELECTOR_VERSION,
                    "inherited_selector_version": RDX004_BASE_SELECTOR_VERSION,
                    "status": "QUERY_PREFLIGHT_READY",
                }
            )
            chronology_rows.append(
                {
                    "run_id": identity.run_id,
                    "budget": identity.budget.value,
                    "sequence": identity.sequence_token,
                    "seed": identity.seed,
                    "paired_b100_experiment_id": source.experiment_id,
                    "query_ordinal": event.query_ordinal,
                    "query_stage": event.stage,
                    "query_domain": event.current_domain,
                    "query_local_window_id": event.selection_window_id,
                    "query_prediction_index": event.prediction_index,
                    "release_stage": event.stage,
                    "release_domain": event.current_domain,
                    "release_local_window_id": event.selection_window_id + 1,
                    "release_prediction_index": event.release_prediction_index,
                    "prediction_before_query": True,
                    "available_on_query_prediction": False,
                    "delay_prediction_count": treatment.delay_windows,
                    "cross_boundary_release": False,
                    "terminal_pending": False,
                    "matches_b100_timing": True,
                    "status": "D1_CHRONOLOGY_VERIFIED",
                }
            )

    nested_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    allocation_rows: list[dict[str, object]] = []
    for event in events:
        context: dict[str, object] = {
            "sequence": "-".join(event.source.sequence),
            "seed": event.source.seed,
            "paired_b100_experiment_id": event.source.experiment_id,
            "stage": event.stage,
            "current_domain": event.current_domain,
            "dataset_fingerprint": event.dataset_fingerprint,
            "opaque_scope_token": event.opaque_scope_token,
            "query_ordinal": event.query_ordinal,
        }
        nested_rows.append(
            {
                **context,
                "selection_window_id": event.selection_window_id,
                "prediction_index": event.prediction_index,
                "candidate_population_size": len(event.candidate_positions),
                "candidate_population_digest": event.candidate_digest,
                "q100_size": len(event.q100),
                "q100_positions_json": list(event.q100),
                "q100_digest": event.q100_digest,
                "persisted_b100_positions_json": list(event.q100),
                "persisted_b100_digest": event.q100_digest,
                "q400_size": len(event.q400),
                "q400_positions_json": list(event.q400),
                "q400_digest": event.q400_digest,
                "q1600_size": len(event.q1600),
                "q1600_positions_json": list(event.q1600),
                "q1600_digest": event.q1600_digest,
                "reconstructed_q100_equals_persisted": True,
                "q100_subset_q400": set(event.q100) < set(event.q400),
                "q400_subset_q1600": set(event.q400) < set(event.q1600),
                "no_duplicate_with_prior_ordinals": True,
                "status": "NESTING_VERIFIED",
            }
        )
        audit_rows.append(
            {
                key: context[key]
                for key in (
                    "sequence",
                    "seed",
                    "paired_b100_experiment_id",
                    "stage",
                    "current_domain",
                    "opaque_scope_token",
                    "query_ordinal",
                )
            }
            | {
                "audit_count": len(event.audit),
                "fixed_audit_positions_json": list(event.audit),
                "fixed_audit_digest": event.audit_digest,
                "audit_subset_q100": set(event.audit).issubset(event.q100),
                "audit_subset_q400": set(event.audit).issubset(event.q400),
                "audit_subset_q1600": set(event.audit).issubset(event.q1600),
                "audit_b100_equals_b400_equals_b1600": True,
                "audit_excluded_from_all_training": True,
                "status": "FIXED_AUDIT_VERIFIED",
            }
        )
        for budget in RDX004_ALL_BUDGETS:
            query = _query_for_budget(event, budget)
            training = _training_for_budget(event, budget)
            allocation_rows.append(
                {
                    key: context[key]
                    for key in (
                        "sequence",
                        "seed",
                        "paired_b100_experiment_id",
                        "stage",
                        "current_domain",
                        "opaque_scope_token",
                        "query_ordinal",
                    )
                }
                | {
                    "budget": budget.value,
                    "query_count": len(query),
                    "query_set_digest": row_positions_digest(query),
                    "audit_count": len(event.audit),
                    "audit_set_digest": event.audit_digest,
                    "training_count": len(training),
                    "expected_training_count": budget.training_per_query,
                    "training_positions_json": list(training),
                    "training_set_digest": row_positions_digest(training),
                    "exact_query_minus_fixed_audit": True,
                    "train_audit_disjoint": True,
                    "b100_training_positions_retained": set(event.b100_training).issubset(training),
                    "status": "TRAINING_ALLOCATION_VERIFIED",
                }
            )

    action_paths = {
        "A1_RECALIBRATE": (
            "src/danids/adaptation/actions.py::InterventionExecutor._attempt_with_limits"
            "[A1_RECALIBRATE]"
        ),
        "A2_HEAD_UPDATE": "src/danids/adaptation/actions.py::adapt_head_only",
        "A3_FULL_FINE_TUNE": "src/danids/adaptation/actions.py::adapt_naive_ft",
        "A4_REPLAY_UPDATE": "src/danids/adaptation/actions.py::adapt_er",
    }
    optimizer_steps = {
        "A1_RECALIBRATE": {"B100": 0, "B400": 0, "B1600": 0},
        "A2_HEAD_UPDATE": {"B100": 40, "B400": 120, "B1600": 500},
        "A3_FULL_FINE_TUNE": {"B100": 40, "B400": 120, "B1600": 500},
        "A4_REPLAY_UPDATE": {"B100": 60, "B400": 240, "B1600": 1000},
    }
    action_rows: list[dict[str, object]] = []
    for action in action_paths:
        for budget in RDX004_ALL_BUDGETS:
            treatment = config.treatment(budget)
            closure_rows = treatment.historical_replay_rows_at_closure
            historical_by_stage = (
                [400, 400 + closure_rows, 400 + 2 * closure_rows]
                if action == "A4_REPLAY_UPDATE"
                else [0, 0, 0]
            )
            a1_feasible = action == "A1_RECALIBRATE" and treatment.maximum_training_rows >= 3838
            optimizer_capable = action in {
                "A2_HEAD_UPDATE",
                "A3_FULL_FINE_TUNE",
                "A4_REPLAY_UPDATE",
            }
            action_rows.append(
                {
                    "action": action,
                    "budget": budget.value,
                    "optimizer_code_path": action_paths[action],
                    "cumulative_available_target_rows_json": list(
                        treatment.cumulative_training_rows
                    ),
                    "maximum_available_current_target_rows": (treatment.maximum_training_rows),
                    "maximum_rows_action_can_consume": treatment.maximum_training_rows,
                    "expected_consumed_current_target_rows": (
                        treatment.maximum_training_rows if optimizer_capable else 0
                    ),
                    "expected_optimizer_steps_at_maximum": optimizer_steps[action][budget.value],
                    "historical_replay_capacity_per_domain": (RDX004_HISTORICAL_REPLAY_CAPACITY),
                    "historical_replay_rows_by_later_stage_json": historical_by_stage,
                    "a1_minimum_benign_support": 3838,
                    "a1_structurally_feasible": a1_feasible,
                    "evidence_truncated": False,
                    "treatment_contrast_preserved": optimizer_capable,
                    "status": (
                        "UNCHANGED_A1_INFEASIBLE"
                        if action == "A1_RECALIBRATE"
                        else "FULL_TARGET_EVIDENCE_REACHES_ACTION"
                    ),
                }
            )

    tables = {
        RUN_MATRIX_FILENAME: run_rows,
        QUERY_MATRIX_FILENAME: query_rows,
        NESTED_SELECTION_FILENAME: nested_rows,
        AUDIT_IDENTITY_FILENAME: audit_rows,
        TRAINING_ALLOCATION_FILENAME: allocation_rows,
        RELEASE_CHRONOLOGY_FILENAME: chronology_rows,
        STARTING_STATE_FILENAME: starting_rows,
        ACTION_CAPACITY_FILENAME: action_rows,
    }
    expected_counts = {
        RUN_MATRIX_FILENAME: 24,
        QUERY_MATRIX_FILENAME: 288,
        NESTED_SELECTION_FILENAME: 144,
        AUDIT_IDENTITY_FILENAME: 144,
        TRAINING_ALLOCATION_FILENAME: 432,
        RELEASE_CHRONOLOGY_FILENAME: 288,
        STARTING_STATE_FILENAME: 24,
        ACTION_CAPACITY_FILENAME: 12,
    }
    for name, count in expected_counts.items():
        if len(tables[name]) != count:
            raise RdxTrainingEvidencePreflightError(
                f"{name} contains {len(tables[name])} rows, expected {count}"
            )
    return tables


def _build_contract(
    config: RDX004TrainingEvidenceConfig,
    *,
    study4_evaluation_root: Path,
    manifest_root: Path,
    evaluation_identity: Mapping[str, Any],
    sources: Sequence[_CanonicalRun],
    roster: Sequence[RDX004RunIdentity],
    source_sentinels: Mapping[str, str],
) -> dict[str, Any]:
    source_records = [
        {
            "experiment_id": source.experiment_id,
            "method": RDX004_METHOD,
            "sequence": list(source.sequence),
            "seed": source.seed,
            "source_run_path": str(source.path),
            "source_artifact_manifest_sha256": source.artifact_manifest_sha256,
            "source_manifest_bundle_digest": source.manifest_bundle_digest,
            "scientific_contract_digest": source.scientific_contract_digest,
            "source_checkpoint_sha256": source.source_checkpoint_sha256,
            "initial_model_digest": source.initial_model_digest,
            "preprocessor_digest": source.preprocessor_digest,
            "threshold_digest": source.threshold_digest,
            "study1_run_path": str(source.study1_run_path),
            "study1_experiment_id": source.study1_experiment_id,
            "dataset_fingerprints": source.dataset_fingerprints,
            "split_manifest_paths": [str(path) for path in source.split_manifest_paths],
            "split_manifest_sha256": source.split_manifest_sha256,
            "split_manifest_set_digest": source.split_manifest_set_digest,
            "oracle_controls_digest": source.oracle_controls_digest,
            "historical_replay_contract_digest": (source.historical_replay_contract_digest),
        }
        for source in sources
    ]
    prospective_records = [
        {
            **identity.to_dict(),
            "paired_b100_experiment_id": identity.paired_b100_experiment_id,
            "resolved_run_config_sha256": config.run_config_sha256(identity),
        }
        for identity in roster
    ]
    return {
        "version": RDX004_PREFLIGHT_VERSION,
        "protocol": {
            "version": RDX004_PROTOCOL_VERSION,
            "path": str(config.protocol_path),
            "sha256": RDX004_PROTOCOL_SHA256,
        },
        "implementation": {
            "version": config.implementation_version,
            "status": config.implementation_status,
            "execution_authorized": False,
            "b400_b1600_runs_executed": False,
        },
        "inputs": {
            "configuration_path": str(config.path),
            "configuration_file_sha256": _file_sha256(config.path),
            "configuration_contract_version": config.contract_version,
            "configuration_contract_sha256": config.contract_sha256,
            "repository_root": str(config.repository_root),
            "study4_evaluation_root": str(study4_evaluation_root),
            "manifest_root": str(manifest_root),
            "canonical_b100_authority": str(config.canonical_b100_authority),
        },
        "source_study4_evaluation": dict(evaluation_identity),
        "canonical_b100_sources": source_records,
        "prospective_run_roster": prospective_records,
        "matrix_contract": {
            "method": config.method,
            "historical_comparator": config.historical_comparator.value,
            "prospective_budgets": [item.value for item in config.prospective_budgets],
            "rotations": [list(rotation) for rotation in config.rotations],
            "seeds": list(config.seeds),
            "expected_new_runs": config.expected_new_runs,
            "expected_query_events": config.expected_query_events,
        },
        "selection_contract": {
            "selector_version": RDX004_SELECTOR_VERSION,
            "inherited_selector_version": RDX004_BASE_SELECTOR_VERSION,
            "prefix_sizes": [25, 100, 400],
            "label_blind": True,
            "ranking_window_identity": "LOCAL_SELECTION_WINDOW_ID",
            "candidate_source": "EXACT_SPLIT_MANIFEST_ONLINE_STREAM_RANGE",
            "minimum_candidate_rows": 400,
            "prior_query_rows_ineligible": True,
        },
        "fixed_audit_contract": {
            "source": "CANONICAL_B100_PERSISTED_ALLOCATION",
            "audit_rows_per_query": 5,
            "audit_rows_per_scope": 20,
            "allocation_recomputed": False,
            "audit_training_eligible": False,
        },
        "delay_contract": {
            "delay_predictions": 1,
            "prediction_before_query": True,
            "available_on_query_prediction": False,
            "release_timing_source": "CANONICAL_B100_QUERY_LOG",
        },
        "historical_replay_contract": {
            "capacity_per_previous_domain": RDX004_HISTORICAL_REPLAY_CAPACITY,
            "projection_primitive": "deterministic_replay_exemplars",
            "projection_code_path": (
                f"{deterministic_replay_exemplars.__module__}."
                f"{deterministic_replay_exemplars.__name__}"
            ),
            "projection_version": RDX004_REPLAY_PROJECTION_VERSION,
            "projection_seed_offset": RDX004_PROJECTION_SEED_OFFSET,
            "project_only_above_capacity": True,
            "under_capacity_policy": "RETAIN_ALL_WITH_EXACT_B100_POSITION_PROVENANCE",
            "canonical_b100_provenance_preserved": True,
            "canonical_b100_digest": sources[0].historical_replay_contract_digest,
            "all_treatments_identical": True,
        },
        "action_control_contract": {
            "canonical_b100_digest": sources[0].oracle_controls_digest,
            "all_treatments_identical": True,
            "a1_minimum_benign_support": 3838,
            "a1_definition_changed": False,
            "optimizer_capable_actions": [
                "A2_HEAD_UPDATE",
                "A3_FULL_FINE_TUNE",
                "A4_REPLAY_UPDATE",
            ],
        },
        "source_immutability_sentinels": dict(sorted(source_sentinels.items())),
        "output_contract": {
            "file_count": 11,
            "files": sorted(ALL_OUTPUT_FILES),
            "csv_schemas": {
                name: list(columns) for name, columns in sorted(OUTPUT_SCHEMAS.items())
            },
            "write_once": True,
            "manifest_excluded_from_own_digest": True,
        },
    }


def _build_summary(
    config: RDX004TrainingEvidenceConfig,
    tables: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, Any]:
    candidate_sizes = [
        _as_int(row.get("candidate_population_size"), "candidate population size")
        for row in tables[NESTED_SELECTION_FILENAME]
    ]
    return {
        "version": RDX004_PREFLIGHT_VERSION,
        "status": "complete",
        "verdict": RDX004_PREFLIGHT_VERDICT_GO,
        "protocol_sha256": config.protocol_sha256,
        "implementation_version": config.implementation_version,
        "prospective_runs_expected": RDX004_EXPECTED_NEW_RUNS,
        "prospective_runs_constructed": len(tables[RUN_MATRIX_FILENAME]),
        "prospective_query_events_expected": RDX004_EXPECTED_QUERY_EVENTS,
        "prospective_query_events_constructed": len(tables[QUERY_MATRIX_FILENAME]),
        "canonical_b100_runs": 12,
        "matched_b100_query_events": len(tables[NESTED_SELECTION_FILENAME]),
        "minimum_candidate_population": min(candidate_sizes),
        "fixed_audit_event_count": len(tables[AUDIT_IDENTITY_FILENAME]),
        "training_allocation_row_count": len(tables[TRAINING_ALLOCATION_FILENAME]),
        "action_capacity_row_count": len(tables[ACTION_CAPACITY_FILENAME]),
        "checks": {
            "exact_24_run_roster": True,
            "canonical_b100_pairing": True,
            "starting_state_identity": True,
            "query_opportunity_identity": True,
            "b1600_candidate_capacity": True,
            "nested_query_selection": True,
            "fixed_b100_audit_identity": True,
            "training_audit_disjoint": True,
            "d1_chronology": True,
            "historical_replay_unchanged": True,
            "target_evidence_reaches_optimizer_actions": True,
            "permanent_holdout_excluded": True,
            "deterministic_configs_sealed": True,
            "frozen_sources_unchanged": True,
        },
        "blockers": [],
        "execution_authorized": False,
        "b400_b1600_runs_executed": False,
        "next_authorised_task": RDX004_NEXT_AUTHORISED_TASK,
    }


def _assert_sentinels_unchanged(source_sentinels: Mapping[str, str]) -> None:
    for raw_path, expected in source_sentinels.items():
        path = Path(raw_path)
        if not path.is_file() or _file_sha256(path) != expected:
            raise RdxTrainingEvidencePreflightError(
                f"frozen source changed during preflight derivation: {path}"
            )


def derive_rdx004_training_evidence_preflight(
    config_path: str | Path,
    *,
    study4_evaluation_root: str | Path | None = None,
    manifest_root: str | Path | None = None,
) -> RdxTrainingEvidencePreflight:
    """Independently derive the complete 24-run artifact-only preflight."""

    config = load_rdx004_training_evidence_config(config_path)
    if config.execution_authorized:
        raise RdxTrainingEvidencePreflightError(
            "preflight refuses a configuration that authorises execution"
        )
    evaluation_root = (
        config.canonical_b100_authority.parent
        if study4_evaluation_root is None
        else Path(study4_evaluation_root)
    ).resolve()
    resolved_manifest_root = (
        config.repository_root / "manifests" / "study4-confirmatory"
        if manifest_root is None
        else Path(manifest_root)
    ).resolve()
    roster = build_rdx004_run_roster(config)
    if (
        len(roster) != RDX004_EXPECTED_NEW_RUNS
        or len({identity.run_id for identity in roster}) != RDX004_EXPECTED_NEW_RUNS
    ):
        raise RdxTrainingEvidencePreflightError("prospective roster is not 24 unique runs")
    sources, evaluation_identity, source_sentinels = _resolve_canonical_sources(
        config,
        study4_evaluation_root=evaluation_root,
        manifest_root=resolved_manifest_root,
    )
    source_sentinels = {
        **source_sentinels,
        str(evaluation_root / "evaluation_contract.json"): _file_sha256(
            evaluation_root / "evaluation_contract.json"
        ),
        str(evaluation_root / MANIFEST_FILENAME): _file_sha256(evaluation_root / MANIFEST_FILENAME),
        str(config.path): _file_sha256(config.path),
        str(config.protocol_path): _file_sha256(config.protocol_path),
    }
    events = _derive_matched_events(sources)
    tables = _derive_tables(config, roster, sources, events)
    contract = _build_contract(
        config,
        study4_evaluation_root=evaluation_root,
        manifest_root=resolved_manifest_root,
        evaluation_identity=evaluation_identity,
        sources=sources,
        roster=roster,
        source_sentinels=source_sentinels,
    )
    summary = _build_summary(config, tables)
    _assert_sentinels_unchanged(source_sentinels)
    return RdxTrainingEvidencePreflight(contract=contract, summary=summary, tables=tables)


def _protected_output_roots(preflight: RdxTrainingEvidencePreflight) -> set[Path]:
    inputs = _mapping(preflight.contract.get("inputs"), "preflight inputs")
    protected = {
        Path(str(inputs["study4_evaluation_root"])).resolve(),
        Path(str(inputs["manifest_root"])).resolve(),
    }
    canonical = _records(preflight.contract.get("canonical_b100_sources"), "canonical B100 sources")
    protected.update(Path(str(record["source_run_path"])).resolve() for record in canonical)
    protected.update(Path(str(record["study1_run_path"])).resolve() for record in canonical)
    evaluation_root = Path(str(inputs["study4_evaluation_root"])).resolve()
    protected.update(
        candidate
        for candidate in (evaluation_root, *evaluation_root.parents)
        if candidate.name.casefold() == "study4"
    )
    return protected


def _write_unsealed_bundle(preflight: RdxTrainingEvidencePreflight, output: Path) -> None:
    if set(preflight.tables) != set(OUTPUT_SCHEMAS):
        raise RdxTrainingEvidencePreflightError(
            "derived table set differs from the fixed eight-table contract"
        )
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"RDX-004 staging directory is not empty: {output}")
    else:
        output.mkdir(parents=False, exist_ok=False)
    for name in sorted(OUTPUT_SCHEMAS):
        (output / name).write_text(
            _csv_text(OUTPUT_SCHEMAS[name], preflight.tables[name]),
            encoding="utf-8",
            newline="",
        )
    (output / CONTRACT_FILENAME).write_text(
        _json_text(preflight.contract), encoding="utf-8", newline=""
    )
    (output / SUMMARY_FILENAME).write_text(
        _json_text(preflight.summary), encoding="utf-8", newline=""
    )
    files = _tree_digests(output)
    if set(files) != ALL_OUTPUT_FILES.difference({MANIFEST_FILENAME}):
        raise RdxTrainingEvidencePreflightError(
            "pre-manifest bundle differs from the exact 10-file payload contract"
        )
    manifest = {
        "version": RDX004_PREFLIGHT_VERSION,
        "files": files,
        "bundle_digest": _digest(files),
    }
    (output / MANIFEST_FILENAME).write_text(_json_text(manifest), encoding="utf-8", newline="")


def write_rdx004_training_evidence_preflight(
    preflight: RdxTrainingEvidencePreflight, output_dir: str | Path
) -> Path:
    """Atomically publish one independently validated, write-once 11-file bundle."""

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite RDX-004 preflight: {output}")
    for protected in _protected_output_roots(preflight):
        if output == protected or protected in output.parents:
            raise RdxTrainingEvidencePreflightError(
                "preflight output must not be inside a frozen source directory"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    ).resolve()
    try:
        _write_unsealed_bundle(preflight, temporary)
        validate_rdx004_training_evidence_preflight(temporary)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite RDX-004 preflight: {output}")
        temporary.rename(output)
    except Exception:
        if temporary.is_dir() and temporary.parent == output.parent:
            shutil.rmtree(temporary)
        raise
    return output


def build_rdx004_training_evidence_preflight(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    study4_evaluation_root: str | Path | None = None,
    manifest_root: str | Path | None = None,
) -> Path:
    """Derive, atomically seal, and independently validate the full preflight."""

    preflight = derive_rdx004_training_evidence_preflight(
        config_path,
        study4_evaluation_root=study4_evaluation_root,
        manifest_root=manifest_root,
    )
    return write_rdx004_training_evidence_preflight(preflight, output_dir)


def validate_rdx004_training_evidence_preflight(output_dir: str | Path) -> None:
    """Source-rederive and byte-compare every file in an RDX-004 preflight."""

    root = Path(output_dir).resolve()
    if not root.is_dir():
        raise RdxTrainingEvidencePreflightError(
            f"RDX-004 preflight directory does not exist: {root}"
        )
    entries = list(root.iterdir())
    if (
        any(not entry.is_file() for entry in entries)
        or {entry.name for entry in entries} != ALL_OUTPUT_FILES
    ):
        raise RdxTrainingEvidencePreflightError(
            "RDX-004 preflight differs from the exact 11-file contract"
        )
    manifest = _load_json(root / MANIFEST_FILENAME)
    if (root / MANIFEST_FILENAME).read_bytes() != _json_text(manifest).encode("utf-8"):
        raise RdxTrainingEvidencePreflightError(
            "RDX-004 artifact manifest is not canonically serialized"
        )
    files = _tree_digests(root)
    if (
        set(manifest) != {"version", "files", "bundle_digest"}
        or manifest.get("version") != RDX004_PREFLIGHT_VERSION
        or manifest.get("files") != files
        or manifest.get("bundle_digest") != _digest(files)
        or set(files) != ALL_OUTPUT_FILES.difference({MANIFEST_FILENAME})
    ):
        raise RdxTrainingEvidencePreflightError(
            "RDX-004 artifact manifest differs from its exact files"
        )

    contract = _load_json(root / CONTRACT_FILENAME)
    summary = _load_json(root / SUMMARY_FILENAME)
    if (root / CONTRACT_FILENAME).read_bytes() != _json_text(contract).encode("utf-8"):
        raise RdxTrainingEvidencePreflightError(
            "RDX-004 contract JSON is not canonically serialized"
        )
    if (root / SUMMARY_FILENAME).read_bytes() != _json_text(summary).encode("utf-8"):
        raise RdxTrainingEvidencePreflightError(
            "RDX-004 summary JSON is not canonically serialized"
        )
    inputs = _mapping(contract.get("inputs"), "persisted preflight inputs")
    expected = derive_rdx004_training_evidence_preflight(
        Path(str(inputs.get("configuration_path", ""))),
        study4_evaluation_root=Path(str(inputs.get("study4_evaluation_root", ""))),
        manifest_root=Path(str(inputs.get("manifest_root", ""))),
    )
    if contract != expected.contract:
        raise RdxTrainingEvidencePreflightError(
            "persisted preflight contract differs from immutable-source rederivation"
        )
    if summary != expected.summary:
        raise RdxTrainingEvidencePreflightError(
            "persisted preflight summary differs from immutable-source rederivation"
        )
    for name, rows in expected.tables.items():
        expected_bytes = _csv_text(OUTPUT_SCHEMAS[name], rows).encode("utf-8")
        if (root / name).read_bytes() != expected_bytes:
            raise RdxTrainingEvidencePreflightError(
                f"persisted {name} differs from immutable-source rederivation"
            )


__all__ = [
    "ACTION_CAPACITY_FILENAME",
    "ALL_OUTPUT_FILES",
    "AUDIT_IDENTITY_FILENAME",
    "CONTRACT_FILENAME",
    "MANIFEST_FILENAME",
    "NESTED_SELECTION_FILENAME",
    "QUERY_MATRIX_FILENAME",
    "RDX004_NEXT_AUTHORISED_TASK",
    "RDX004_PREFLIGHT_VERSION",
    "RELEASE_CHRONOLOGY_FILENAME",
    "RUN_MATRIX_FILENAME",
    "STARTING_STATE_FILENAME",
    "SUMMARY_FILENAME",
    "TRAINING_ALLOCATION_FILENAME",
    "RdxTrainingEvidencePreflight",
    "RdxTrainingEvidencePreflightError",
    "build_rdx004_training_evidence_preflight",
    "derive_rdx004_training_evidence_preflight",
    "reconstruct_candidate_positions",
    "validate_rdx004_training_evidence_preflight",
    "write_rdx004_training_evidence_preflight",
]
