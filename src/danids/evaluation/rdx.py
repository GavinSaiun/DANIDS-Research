"""Artifact-only Recoverability Diagnostic Extension (RDX) derivations.

RDX is deliberately downstream of the immutable Study-4 confirmatory artifacts.  This
module never executes a model, reads raw data, or changes a Study-4 result.  It resolves
the exact 48-run corpus sealed by the canonical Study-4 evaluation contract, validates
every source bundle, derives the frozen RDX-A--RDX-D diagnostic tables, and writes a
deterministic self-validating bundle.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from danids.adaptation.actions import InterventionAction
from danids.config.continual import FROZEN_ROTATIONS
from danids.config.study4 import STUDY4_CONFIRMATORY_SEEDS, Study4Method
from danids.evaluation.study4 import (
    CORE_BUNDLE_DIRNAME,
    ORACLE_FILENAME,
    Study4ArtifactError,
    ValidatedStudy4Run,
    validate_study4_evaluation,
    validate_study4_run,
)
from danids.evaluation.study4 import (
    MANIFEST_FILENAME as STUDY4_MANIFEST_FILENAME,
)

RDX_EVALUATION_VERSION = "rdx-002-artifact-evaluation-v1"
RDX_PROTOCOL_VERSION = "RDX-001 v1.0"
RDX_SOURCE_RESEARCH_TAG = "v2.0-thesis-freeze"
RDX_PROTOCOL_SHA256 = "728d791e6f841c33b4953b329548c05549706f1c5656679d7360d90565c30e3d"

CONTRACT_FILENAME = "rdx_contract.json"
SUMMARY_FILENAME = "rdx_summary.json"
MANIFEST_FILENAME = "artifact_manifest.json"

HARM_FILENAME = "rdx_harm_windows.csv"
CORE_PATHWAY_FILENAME = "rdx_core_pathways.csv"
CORE_ATTEMPT_FILENAME = "rdx_core_attempts.csv"
CORE_AUDIT_FILENAME = "rdx_core_audits.csv"
EVIDENCE_FILENAME = "rdx_evidence_snapshots.csv"
ORACLE_CANDIDATE_FILENAME = "rdx_oracle_candidates.csv"
ORACLE_DECISION_FILENAME = "rdx_oracle_decisions.csv"
RECOVERY_FILENAME = "rdx_deployed_recovery.csv"
INCIDENT_FILENAME = "rdx_incident_links.csv"

_COMMON_ID_COLUMNS = (
    "experiment_id",
    "method",
    "sequence",
    "seed",
)
_COMMON_SOURCE_COLUMNS = (
    "source_run_path",
    "source_manifest_sha256",
    "source_artifact_identity",
    "source_artifact_sha256",
)

OUTPUT_SCHEMAS: dict[str, tuple[str, ...]] = {
    HARM_FILENAME: (
        *_COMMON_ID_COLUMNS,
        "prediction_index",
        "stage",
        "current_domain",
        "window_id",
        "row_start",
        "row_stop",
        "predicted_health_state",
        "evaluator_health_state",
        "recognised_harm",
        "not_recognised_as_harm",
        "predicted_safe_on_harm",
        "predicted_uncertain_on_harm",
        "harm_probability",
        "unsafe_exposure",
        "eval_recall_floor",
        "attack_count",
        "benign_count",
        "tp",
        "fp",
        "tn",
        "fn",
        "tpr",
        "fpr",
        "false_positives_per_million",
        "fpr_budget_ratio",
        "eval_tpr_loss_from_reference",
        *_COMMON_SOURCE_COLUMNS,
    ),
    CORE_PATHWAY_FILENAME: (
        *_COMMON_ID_COLUMNS,
        "prediction_index",
        "stage",
        "current_domain",
        "window_id",
        "predicted_health_state",
        "recognised_harm",
        "decision_count",
        "intervention_attempt_count",
        "model_changing_action_attempted",
        "a2_attempted",
        "a4_attempted",
        "explicit_execution_failure_count",
        "candidate_rejected_after_audit_count",
        "candidate_accepted_count",
        "rollback_count",
        "ordered_decision_ids_json",
        "ordered_actions_json",
        "ordered_decision_reasons_json",
        "ordered_skipped_actions_json",
        "ordered_attempt_decision_ids_json",
        "ordered_audit_states_json",
        "a3_core_availability",
        *_COMMON_SOURCE_COLUMNS,
    ),
    CORE_ATTEMPT_FILENAME: (
        *_COMMON_ID_COLUMNS,
        "prediction_index",
        "stage",
        "current_domain",
        "window_id",
        "decision_ordinal",
        "decision_id",
        "action",
        "decision_reason",
        "requires_feedback",
        "evidence_signature",
        "skipped_actions_json",
        "a1_infeasible",
        "a1_infeasibility_reason",
        "a1_minimum_benign_support",
        "intervention_attempt_recorded",
        "attempt_availability",
        "labels_requested",
        "labels_available",
        "optimizer_steps",
        "target_rows",
        "replay_rows",
        "audit_result",
        "accepted",
        "rolled_back",
        "rejection_reason",
        "explicit_execution_failure",
        "model_digest_before",
        "model_digest_candidate",
        "model_digest_after",
        "threshold_digest_before",
        "threshold_digest_candidate",
        "threshold_digest_after",
        *_COMMON_SOURCE_COLUMNS,
    ),
    CORE_AUDIT_FILENAME: (
        *_COMMON_ID_COLUMNS,
        "prediction_index",
        "decision_id",
        "purpose",
        "panel_identity",
        "panel_evaluated",
        "evaluation_skipped",
        "availability_state",
        "skip_reason",
        "audit_state",
        "benign_support",
        "attack_support",
        "fp",
        "tp",
        "fpr_low",
        "fpr_high",
        "tpr_low",
        "tpr_high",
        "recall_floor",
        "expected_panel_count",
        "evaluated_panel_count",
        "missing_panel_count",
        "candidate_action",
        "candidate_audit_result",
        "candidate_accepted",
        "candidate_rolled_back",
        "candidate_rejection_reason",
        *_COMMON_SOURCE_COLUMNS,
    ),
    EVIDENCE_FILENAME: (
        *_COMMON_ID_COLUMNS,
        "prediction_index",
        "stage",
        "current_domain",
        "window_id",
        "evaluator_health_state",
        "predicted_health_state",
        "released_label_count",
        "released_labels_digest",
        "optimizer_eligible_released_label_count",
        "all_current_scope_released_label_count",
        "query_pending",
        "remaining_label_budget",
        "query_count",
        "audit_escrow_row_count",
        "audit_escrow_digest",
        "current_scope_audit_escrow_row_count",
        "replay_row_count",
        "replay_digest",
        "active_historical_audit_panels",
        "active_historical_audit_panel_count",
        "audit_memory_digest",
        "evidence_availability",
        *_COMMON_SOURCE_COLUMNS,
    ),
    ORACLE_CANDIDATE_FILENAME: (
        *_COMMON_ID_COLUMNS,
        "prediction_index",
        "stage",
        "current_domain",
        "window_id",
        "decision_id",
        "action",
        "action_rank",
        "current_evaluator_state",
        "feasible",
        "feasibility_reason",
        "execution_succeeded",
        "audit_admissible",
        "audit_state",
        "successor_evaluator_state",
        "confirmed_success",
        "selected",
        "selection_reason",
        "optimizer_steps",
        "target_rows",
        "replay_rows",
        "rows_consumed",
        "incoming_model_digest",
        "deployed_model_digest",
        "deployed_threshold_digest",
        "deployed_r1_digest",
        "numeric_successor_availability",
        "successor_prediction_index",
        "successor_window_id",
        "successor_row_start",
        "successor_row_stop",
        "successor_attack_count",
        "successor_benign_count",
        "successor_tp",
        "successor_fp",
        "successor_tn",
        "successor_fn",
        "successor_tpr",
        "successor_fpr",
        "successor_false_positives_per_million",
        "successor_fpr_budget_ratio",
        "permanent_holdout_used",
        *_COMMON_SOURCE_COLUMNS,
    ),
    ORACLE_DECISION_FILENAME: (
        *_COMMON_ID_COLUMNS,
        "prediction_index",
        "stage",
        "current_domain",
        "window_id",
        "decision_id",
        "current_evaluator_state",
        "diagnostic_class",
        "confirmed_success_count",
        "a0_confirmed_success_count",
        "core_accessible_confirmed_success_count",
        "core_update_confirmed_success_count",
        "a3_confirmed_success_count",
        "model_change_feasible_count",
        "model_change_execution_success_count",
        "selected_action",
        "selected_confirmed_success",
        "selection_reason",
        "candidate_count",
        "candidate_actions_json",
        "evaluator_truth_visible",
        "non_deployable_upper_bound",
        "permanent_holdout_used_for_choice",
        *_COMMON_SOURCE_COLUMNS,
    ),
    RECOVERY_FILENAME: (
        *_COMMON_ID_COLUMNS,
        "anchor_prediction_index",
        "anchor_decision_id",
        "stage",
        "current_domain",
        "anchor_window_id",
        "action",
        "audit_result",
        "audit_admissibility_state",
        "model_digest_after",
        "next_accepted_model_change_prediction_index",
        "immediate_prediction_index",
        "immediate_window_id",
        "immediate_evaluator_state",
        "immediate_safe",
        "immediate_outcome",
        "immediate_recovery_state",
        "sustained_first_prediction_index",
        "sustained_first_evaluator_state",
        "sustained_second_prediction_index",
        "sustained_second_window_id",
        "sustained_second_evaluator_state",
        "sustained_safe",
        "sustained_outcome",
        "sustained_two_window_status",
        "eligible_successor_count_before_censoring",
        "censoring_boundary",
        "censor_reason",
        "later_same_domain_trajectory_json",
        "post_accept_holdout_references_json",
        "domain_end_holdout_references_json",
        "final_holdout_references_json",
        *_COMMON_SOURCE_COLUMNS,
    ),
    INCIDENT_FILENAME: (
        *_COMMON_ID_COLUMNS,
        "prediction_index",
        "stage",
        "current_domain",
        "window_id",
        "predicted_health_state",
        "active_incident_after_decision",
        "incident_index_after_decision",
        "unresolved_harmful_windows_after_decision",
        "lifecycle_events_json",
        "unresolved_exposure_recorded",
        "unresolved_exposure_incident_index",
        "unresolved_counter_after",
        "unresolved_actions_json",
        "unresolved_reasons_json",
        "incident_link_availability",
        *_COMMON_SOURCE_COLUMNS,
    ),
}

ALL_OUTPUT_FILES = frozenset(
    {*OUTPUT_SCHEMAS, CONTRACT_FILENAME, SUMMARY_FILENAME, MANIFEST_FILENAME}
)

_MODEL_CHANGING_ACTIONS = frozenset(
    {
        InterventionAction.HEAD_UPDATE.value,
        InterventionAction.FULL_FINE_TUNE.value,
        InterventionAction.REPLAY_UPDATE.value,
    }
)
_CORE_ACCESSIBLE_ACTIONS = frozenset(
    {
        InterventionAction.RECALIBRATE.value,
        InterventionAction.HEAD_UPDATE.value,
        InterventionAction.REPLAY_UPDATE.value,
    }
)
_ORACLE_CLASSES = (
    "A0_ONE_STEP_SUCCESS",
    "CORE_ACCESSIBLE_ONE_STEP_SUCCESS",
    "ORACLE_ONLY_A3_ONE_STEP_SUCCESS",
    "EXECUTED_MODEL_CHANGE_NO_ONE_STEP_SUCCESS",
    "MODEL_CHANGE_FEASIBLE_EXECUTION_FAILED",
    "NO_FEASIBLE_MODEL_CHANGE",
)


class RdxArtifactError(RuntimeError):
    """Raised when source or derived RDX artifacts violate the frozen contract."""


@dataclass(frozen=True, slots=True)
class RdxSourceRun:
    """One exact, validated Study-4 source and its RDX-consumed artifacts."""

    run: ValidatedStudy4Run
    manifest_sha256: str
    manifest_bundle_digest: str
    source_files_sha256: dict[str, str]
    core_policy_windows: dict[str, Any] | None = None
    core_incident_lifecycle: dict[str, Any] | None = None
    core_unresolved_exposures: dict[str, Any] | None = None
    core_administrative_audits: dict[str, Any] | None = None
    oracle_decisions: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RdxCorpus:
    """Exact canonical Study-4 corpus used for one RDX derivation."""

    evaluation_root: Path
    evaluation_contract: dict[str, Any]
    evaluation_contract_sha256: str
    evaluation_manifest_sha256: str
    evaluation_manifest_bundle_digest: str
    runs: tuple[RdxSourceRun, ...]


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                hasher.update(chunk)
    except OSError as exc:
        raise RdxArtifactError(f"cannot hash {path}: {exc}") from exc
    return hasher.hexdigest()


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _json_cell(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RdxArtifactError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RdxArtifactError(f"{path} must contain a JSON object")
    return value


def _csv_text(columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        extras = set(row).difference(columns)
        if extras:
            raise RdxArtifactError(
                "derived row contains unexpected columns: " + ", ".join(sorted(extras))
            )
        writer.writerow(
            {column: "" if row.get(column) is None else row.get(column, "") for column in columns}
        )
    return handle.getvalue()


def _tree_digests(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == MANIFEST_FILENAME:
            continue
        result[path.relative_to(root).as_posix()] = _file_sha256(path)
    return result


def _as_bool(value: object, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise RdxArtifactError(f"{context} is not Boolean")


def _as_int(value: object, context: str) -> int:
    if isinstance(value, bool):
        raise RdxArtifactError(f"{context} is not an integer")
    raw: Any = value
    try:
        number: int = int(raw)
    except (TypeError, ValueError) as exc:
        raise RdxArtifactError(f"{context} is not an integer") from exc
    try:
        if float(raw) != float(number):
            raise RdxArtifactError(f"{context} is not an exact integer")
    except (TypeError, ValueError) as exc:
        raise RdxArtifactError(f"{context} is not an exact integer") from exc
    return int(number)


def _optional_int(value: object, context: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return _as_int(value, context)


def _optional_bool(value: object, context: str) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    return _as_bool(value, context)


def _optional_text(value: object) -> str:
    """Canonicalise a legitimately nullable text field without creating ``"None"``."""

    return "" if value is None else str(value)


def _require_list(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise RdxArtifactError(f"{context} must be a list")
    return value


def _common_context(source: RdxSourceRun) -> dict[str, object]:
    run = source.run
    return {
        "experiment_id": run.experiment_id,
        "method": run.method.value,
        "sequence": "-".join(run.sequence),
        "seed": run.seed,
    }


def _source_context(source: RdxSourceRun, artifact: str) -> dict[str, object]:
    try:
        artifact_sha = source.source_files_sha256[artifact]
    except KeyError as exc:
        raise RdxArtifactError(
            f"{source.run.experiment_id} does not seal required artifact {artifact}"
        ) from exc
    return {
        "source_run_path": str(source.run.path),
        "source_manifest_sha256": source.manifest_sha256,
        "source_artifact_identity": artifact,
        "source_artifact_sha256": artifact_sha,
    }


def _window_index(source: RdxSourceRun) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in source.run.windows:
        index = _as_int(row.get("prediction_index"), "window prediction_index")
        if index in result:
            raise RdxArtifactError(f"{source.run.experiment_id} has duplicate prediction_index")
        result[index] = row
    return result


def _intervention_index(source: RdxSourceRun) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in source.run.interventions:
        decision_id = str(row.get("decision_id", ""))
        if not decision_id:
            raise RdxArtifactError(f"{source.run.experiment_id} has blank intervention decision_id")
        result.setdefault(decision_id, []).append(row)
    return result


def _expected_roster() -> set[tuple[Study4Method, tuple[str, ...], int]]:
    methods = (
        Study4Method.STATIC,
        Study4Method.ALWAYS_ADAPT,
        Study4Method.DANIDS_CORE,
        Study4Method.OFFLINE_ORACLE,
    )
    return {
        (method, tuple(rotation), seed)
        for method in methods
        for rotation in FROZEN_ROTATIONS
        for seed in STUDY4_CONFIRMATORY_SEEDS
    }


def _load_consumed_json(
    root: Path, manifest_files: Mapping[str, object], relative: str
) -> dict[str, Any]:
    if relative not in manifest_files:
        raise RdxArtifactError(f"source manifest lacks required artifact {relative}")
    path = root / Path(relative)
    expected = str(manifest_files[relative])
    if _file_sha256(path) != expected:
        raise RdxArtifactError(f"source artifact digest differs for {path}")
    return _load_json(path)


def _resolve_canonical_sources(study4_evaluation_root: str | Path) -> RdxCorpus:
    """Resolve and validate exactly the contracted 48-run Study-4 corpus."""

    root = Path(study4_evaluation_root).resolve()
    try:
        validate_study4_evaluation(root)
    except (Study4ArtifactError, FileNotFoundError, OSError, ValueError) as exc:
        raise RdxArtifactError(f"canonical Study-4 evaluation is invalid: {exc}") from exc
    contract_path = root / "evaluation_contract.json"
    aggregate_manifest_path = root / MANIFEST_FILENAME
    contract = _load_json(contract_path)
    aggregate_manifest = _load_json(aggregate_manifest_path)
    if contract.get("allow_incomplete") is not False or contract.get("allow_smoke") is not False:
        raise RdxArtifactError("RDX requires the complete non-smoke Study-4 contract")
    sources_raw = contract.get("source_runs")
    observed_raw = contract.get("observed_pairs")
    if not isinstance(sources_raw, list) or len(sources_raw) != 48:
        raise RdxArtifactError("Study-4 evaluation contract must enumerate exactly 48 source runs")
    if not isinstance(observed_raw, list) or len(observed_raw) != 48:
        raise RdxArtifactError("Study-4 evaluation contract must enumerate exactly 48 pairs")

    expected = _expected_roster()
    observed_contract: set[tuple[Study4Method, tuple[str, ...], int]] = set()
    for raw in observed_raw:
        if not isinstance(raw, dict):
            raise RdxArtifactError("Study-4 observed pair must be an object")
        try:
            key = (
                Study4Method(str(raw["method"])),
                tuple(str(value) for value in raw["sequence"]),
                _as_int(raw["seed"], "observed seed"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RdxArtifactError("Study-4 observed pair is invalid") from exc
        if key in observed_contract:
            raise RdxArtifactError("Study-4 evaluation contract has a duplicate observed pair")
        observed_contract.add(key)
    if observed_contract != expected:
        raise RdxArtifactError("Study-4 evaluation contract is not the exact frozen 48-run roster")

    runs: list[RdxSourceRun] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    seen_roster: set[tuple[Study4Method, tuple[str, ...], int]] = set()
    for raw in sources_raw:
        if not isinstance(raw, dict) or set(raw) != {
            "artifact_manifest_sha256",
            "experiment_id",
            "path",
        }:
            raise RdxArtifactError("Study-4 source-run contract entry is invalid")
        experiment_id = str(raw["experiment_id"])
        run_path = Path(str(raw["path"])).resolve()
        if experiment_id in seen_ids or run_path in seen_paths:
            raise RdxArtifactError("Study-4 source-run contract contains a duplicate source")
        seen_ids.add(experiment_id)
        seen_paths.add(run_path)
        manifest_path = run_path / STUDY4_MANIFEST_FILENAME
        manifest_sha = _file_sha256(manifest_path)
        if manifest_sha != str(raw["artifact_manifest_sha256"]):
            raise RdxArtifactError(f"contracted source manifest digest differs for {experiment_id}")
        try:
            validated = validate_study4_run(run_path, allow_smoke=False)
        except (Study4ArtifactError, FileNotFoundError, OSError, ValueError) as exc:
            raise RdxArtifactError(
                f"contracted source run {experiment_id} is invalid: {exc}"
            ) from exc
        if validated.experiment_id != experiment_id or validated.smoke:
            raise RdxArtifactError("contracted source identity/smoke status differs")
        roster_key = (validated.method, tuple(validated.sequence), validated.seed)
        if roster_key in seen_roster:
            raise RdxArtifactError("contracted source roster contains a duplicate run")
        seen_roster.add(roster_key)

        manifest = _load_json(manifest_path)
        manifest_files_raw = manifest.get("files")
        if not isinstance(manifest_files_raw, dict):
            raise RdxArtifactError("Study-4 source manifest files must be an object")
        manifest_files = {str(key): str(value) for key, value in manifest_files_raw.items()}
        consumed = {
            name: manifest_files[name]
            for name in (
                "window_metrics.csv",
                "holdout_metrics.csv",
                "intervention_log.csv",
            )
            if name in manifest_files
        }
        if set(consumed) != {
            "window_metrics.csv",
            "holdout_metrics.csv",
            "intervention_log.csv",
        }:
            raise RdxArtifactError(f"{experiment_id} lacks common RDX source artifacts")

        core_policy = core_lifecycle = core_unresolved = core_audits = oracle = None
        if validated.method is Study4Method.DANIDS_CORE:
            core_names = {
                "policy": f"{CORE_BUNDLE_DIRNAME}/core_policy_windows.json",
                "lifecycle": f"{CORE_BUNDLE_DIRNAME}/core_incident_lifecycle.json",
                "unresolved": f"{CORE_BUNDLE_DIRNAME}/core_unresolved_exposures.json",
                "audits": f"{CORE_BUNDLE_DIRNAME}/core_administrative_audit_evidence.json",
            }
            for relative in core_names.values():
                if relative not in manifest_files:
                    raise RdxArtifactError(f"{experiment_id} lacks {relative}")
                consumed[relative] = manifest_files[relative]
            core_policy = _load_consumed_json(run_path, manifest_files, core_names["policy"])
            core_lifecycle = _load_consumed_json(run_path, manifest_files, core_names["lifecycle"])
            core_unresolved = _load_consumed_json(
                run_path, manifest_files, core_names["unresolved"]
            )
            core_audits = _load_consumed_json(run_path, manifest_files, core_names["audits"])
        elif validated.method is Study4Method.OFFLINE_ORACLE:
            if ORACLE_FILENAME not in manifest_files:
                raise RdxArtifactError(f"{experiment_id} lacks {ORACLE_FILENAME}")
            consumed[ORACLE_FILENAME] = manifest_files[ORACLE_FILENAME]
            oracle = _load_consumed_json(run_path, manifest_files, ORACLE_FILENAME)

        runs.append(
            RdxSourceRun(
                run=validated,
                manifest_sha256=manifest_sha,
                manifest_bundle_digest=str(manifest.get("bundle_digest", "")),
                source_files_sha256=dict(sorted(consumed.items())),
                core_policy_windows=core_policy,
                core_incident_lifecycle=core_lifecycle,
                core_unresolved_exposures=core_unresolved,
                core_administrative_audits=core_audits,
                oracle_decisions=oracle,
            )
        )
    if seen_roster != expected:
        raise RdxArtifactError("resolved source runs are not the exact frozen 48-run roster")
    runs.sort(
        key=lambda source: (
            "-".join(source.run.sequence),
            source.run.seed,
            source.run.method.value,
        )
    )
    return RdxCorpus(
        evaluation_root=root,
        evaluation_contract=contract,
        evaluation_contract_sha256=_file_sha256(contract_path),
        evaluation_manifest_sha256=_file_sha256(aggregate_manifest_path),
        evaluation_manifest_bundle_digest=str(aggregate_manifest.get("bundle_digest", "")),
        runs=tuple(runs),
    )


def _derive_harm_windows(corpus: RdxCorpus) -> list[dict[str, object]]:
    """Create the RDX-A primary population: every evaluator-HARMFUL window."""

    rows: list[dict[str, object]] = []
    allowed_predictions = {"PREDICTED_HARMFUL", "PREDICTED_SAFE", "PREDICTED_UNCERTAIN"}
    for source in corpus.runs:
        context = _common_context(source)
        provenance = _source_context(source, "window_metrics.csv")
        for window in source.run.windows:
            if str(window.get("evaluator_health_state")) != "HARMFUL":
                continue
            predicted = str(window.get("predicted_health_state"))
            if predicted not in allowed_predictions:
                raise RdxArtifactError(
                    f"{source.run.experiment_id} has invalid predicted health state {predicted}"
                )
            recognised = predicted == "PREDICTED_HARMFUL"
            predicted_safe = predicted == "PREDICTED_SAFE"
            predicted_uncertain = predicted == "PREDICTED_UNCERTAIN"
            if sum((recognised, predicted_safe, predicted_uncertain)) != 1:
                raise RdxArtifactError("harm recognition indicators are not mutually exclusive")
            rows.append(
                {
                    **context,
                    "prediction_index": _as_int(
                        window.get("prediction_index"), "harm prediction_index"
                    ),
                    "stage": _as_int(window.get("stage"), "harm stage"),
                    "current_domain": str(window.get("current_domain")),
                    "window_id": _as_int(window.get("window_id"), "harm window_id"),
                    "row_start": _as_int(window.get("row_start"), "harm row_start"),
                    "row_stop": _as_int(window.get("row_stop"), "harm row_stop"),
                    "predicted_health_state": predicted,
                    "evaluator_health_state": "HARMFUL",
                    "recognised_harm": recognised,
                    "not_recognised_as_harm": not recognised,
                    "predicted_safe_on_harm": predicted_safe,
                    "predicted_uncertain_on_harm": predicted_uncertain,
                    "harm_probability": window.get("harm_probability"),
                    "unsafe_exposure": window.get("unsafe_exposure"),
                    "eval_recall_floor": window.get("eval_recall_floor"),
                    "attack_count": window.get("attack_count"),
                    "benign_count": window.get("benign_count"),
                    "tp": window.get("tp"),
                    "fp": window.get("fp"),
                    "tn": window.get("tn"),
                    "fn": window.get("fn"),
                    "tpr": window.get("tpr"),
                    "fpr": window.get("fpr"),
                    "false_positives_per_million": window.get("false_positives_per_million"),
                    "fpr_budget_ratio": window.get("fpr_budget_ratio"),
                    "eval_tpr_loss_from_reference": window.get("eval_tpr_loss_from_reference"),
                    **provenance,
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["experiment_id"]),
            _as_int(row["prediction_index"], "harm sort index"),
        )
    )
    return rows


def _core_trace_index(source: RdxSourceRun) -> dict[int, dict[str, Any]]:
    payload = source.core_policy_windows
    if payload is None:
        raise RdxArtifactError(f"{source.run.experiment_id} lacks Core policy windows")
    traces = _require_list(payload.get("windows"), "Core policy windows")
    result: dict[int, dict[str, Any]] = {}
    for raw in traces:
        if not isinstance(raw, dict):
            raise RdxArtifactError("Core policy-window entry must be an object")
        index = _as_int(raw.get("prediction_index"), "Core policy prediction_index")
        if index in result:
            raise RdxArtifactError("Core policy artifact has duplicate prediction_index")
        result[index] = raw
    if set(result) != set(_window_index(source)):
        raise RdxArtifactError("Core policy-window coverage differs from window metrics")
    return result


def _harmful_core_windows(
    source: RdxSourceRun,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    traces = _core_trace_index(source)
    result: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for window in source.run.windows:
        if str(window.get("evaluator_health_state")) == "HARMFUL":
            index = _as_int(window.get("prediction_index"), "Core harmful prediction_index")
            result.append((window, traces[index]))
    return result


def _feedback_by_decision(trace: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    decisions = _require_list(trace.get("decisions"), "Core decisions")
    feedback = _require_list(trace.get("intervention_feedback"), "Core intervention feedback")
    requiring: list[str] = []
    for raw in decisions:
        if not isinstance(raw, dict):
            raise RdxArtifactError("Core decision entry must be an object")
        if _as_bool(raw.get("requires_feedback"), "Core requires_feedback"):
            requiring.append(str(raw.get("decision_id", "")))
    if len(requiring) != len(feedback):
        raise RdxArtifactError("Core feedback count differs from feedback-requiring decisions")
    result: dict[str, dict[str, Any]] = {}
    for decision_id, raw in zip(requiring, feedback, strict=True):
        if not decision_id or not isinstance(raw, dict) or decision_id in result:
            raise RdxArtifactError("Core feedback cannot be mapped uniquely to a decision")
        result[decision_id] = raw
    return result


def _explicit_execution_failure(rejection_reason: object) -> bool:
    """Return true only for a source reason that explicitly records execution failure."""

    text = str(rejection_reason or "").strip().casefold()
    if not text:
        return False
    return "execution" in text and any(token in text for token in ("fail", "error", "exception"))


def _derive_evidence_snapshots(corpus: RdxCorpus) -> list[dict[str, object]]:
    """Derive decision-time RDX-D evidence at every harmful Core window."""

    rows: list[dict[str, object]] = []
    artifact = f"{CORE_BUNDLE_DIRNAME}/core_policy_windows.json"
    for source in corpus.runs:
        if source.run.method is not Study4Method.DANIDS_CORE:
            continue
        context = _common_context(source)
        provenance = _source_context(source, artifact)
        for window, trace in _harmful_core_windows(source):
            observation = trace.get("observation")
            if not isinstance(observation, dict):
                raise RdxArtifactError("Core trace lacks a pre-decision observation")
            active_panels = _as_int(
                observation.get("active_historical_audit_panels"),
                "active historical audit panels",
            )
            if active_panels < 0:
                raise RdxArtifactError("Core active historical audit panel count is negative")
            released = _as_int(
                observation.get("released_label_count"), "optimizer-eligible released labels"
            )
            escrow = _as_int(
                observation.get("audit_escrow_row_count"), "current-scope audit escrow"
            )
            if released < 0 or escrow < 0:
                raise RdxArtifactError("Core released/audit evidence counts cannot be negative")
            rows.append(
                {
                    **context,
                    "prediction_index": _as_int(
                        window.get("prediction_index"), "evidence prediction_index"
                    ),
                    "stage": _as_int(window.get("stage"), "evidence stage"),
                    "current_domain": str(window.get("current_domain")),
                    "window_id": _as_int(window.get("window_id"), "evidence window_id"),
                    "evaluator_health_state": "HARMFUL",
                    "predicted_health_state": str(window.get("predicted_health_state")),
                    "released_label_count": released,
                    "released_labels_digest": _optional_text(
                        observation.get("released_labels_digest")
                    ),
                    "optimizer_eligible_released_label_count": released,
                    "all_current_scope_released_label_count": released + escrow,
                    "query_pending": _as_bool(
                        observation.get("query_pending"), "decision-time query_pending"
                    ),
                    "remaining_label_budget": _as_int(
                        observation.get("remaining_label_budget"),
                        "decision-time remaining_label_budget",
                    ),
                    "query_count": _as_int(
                        observation.get("query_count"), "decision-time query_count"
                    ),
                    "audit_escrow_row_count": escrow,
                    "audit_escrow_digest": _optional_text(observation.get("audit_escrow_digest")),
                    "current_scope_audit_escrow_row_count": escrow,
                    "replay_row_count": _as_int(
                        observation.get("replay_row_count"), "decision-time replay rows"
                    ),
                    "replay_digest": _optional_text(observation.get("replay_digest")),
                    "active_historical_audit_panels": active_panels,
                    "active_historical_audit_panel_count": active_panels,
                    "audit_memory_digest": _optional_text(observation.get("audit_memory_digest")),
                    "evidence_availability": "AVAILABLE_PRE_DECISION_OBSERVATION",
                    **provenance,
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["experiment_id"]),
            _as_int(row["prediction_index"], "evidence sort prediction_index"),
        )
    )
    return rows


def _single_intervention_for_decision(
    index: Mapping[str, list[dict[str, Any]]], decision_id: str
) -> dict[str, Any] | None:
    records = index.get(decision_id, [])
    if len(records) > 1:
        raise RdxArtifactError(f"decision {decision_id} has duplicate intervention attempts")
    return None if not records else records[0]


def _derive_core_pathways(
    corpus: RdxCorpus,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Derive RDX-A Core window, ordered-decision/attempt, and panel audit views."""

    pathways: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    policy_artifact = f"{CORE_BUNDLE_DIRNAME}/core_policy_windows.json"
    audit_artifact = f"{CORE_BUNDLE_DIRNAME}/core_administrative_audit_evidence.json"
    for source in corpus.runs:
        if source.run.method is not Study4Method.DANIDS_CORE:
            continue
        context = _common_context(source)
        policy_provenance = _source_context(source, policy_artifact)
        intervention_index = _intervention_index(source)
        harmful_prediction_indices: set[int] = set()
        harmful_decision_ids: set[str] = set()
        for window, trace in _harmful_core_windows(source):
            harmful_prediction_indices.add(
                _as_int(window.get("prediction_index"), "Core harmful prediction_index")
            )
            raw_decisions = _require_list(trace.get("decisions"), "Core trace decisions")
            feedback_index = _feedback_by_decision(trace)
            ordered_actions: list[str] = []
            ordered_decision_ids: list[str] = []
            ordered_reasons: list[str] = []
            ordered_skipped: list[object] = []
            attempt_decision_ids: list[str] = []
            audit_states: list[str] = []
            window_attempts: list[dict[str, object]] = []
            for ordinal, raw in enumerate(raw_decisions):
                if not isinstance(raw, dict):
                    raise RdxArtifactError("Core decision entry must be an object")
                decision_id = str(raw.get("decision_id", ""))
                action = str(raw.get("action", ""))
                if not decision_id or not action:
                    raise RdxArtifactError("Core decision lacks decision_id/action")
                ordered_decision_ids.append(decision_id)
                harmful_decision_ids.add(decision_id)
                ordered_actions.append(action)
                ordered_reasons.append(str(raw.get("reason") or ""))
                ordered_skipped.append(raw.get("skipped_actions") or [])
                intervention = _single_intervention_for_decision(intervention_index, decision_id)
                feedback = feedback_index.get(decision_id)
                if intervention is not None:
                    attempt_decision_ids.append(decision_id)
                    audit_state = str(intervention.get("audit_result") or "")
                    if audit_state:
                        audit_states.append(audit_state)
                a1_reason = str(raw.get("a1_infeasible_reason") or "")
                rejection = (
                    "" if intervention is None else str(intervention.get("rejection_reason") or "")
                )
                attempt_row: dict[str, object] = {
                    **context,
                    "prediction_index": _as_int(
                        window.get("prediction_index"), "Core attempt prediction_index"
                    ),
                    "stage": _as_int(window.get("stage"), "Core attempt stage"),
                    "current_domain": str(window.get("current_domain")),
                    "window_id": _as_int(window.get("window_id"), "Core attempt window_id"),
                    "decision_ordinal": ordinal,
                    "decision_id": decision_id,
                    "action": action,
                    "decision_reason": str(raw.get("reason") or ""),
                    "requires_feedback": _as_bool(
                        raw.get("requires_feedback"), "Core decision requires_feedback"
                    ),
                    "evidence_signature": str(raw.get("evidence_signature") or ""),
                    "skipped_actions_json": _json_cell(raw.get("skipped_actions") or []),
                    "a1_infeasible": bool(a1_reason),
                    "a1_infeasibility_reason": a1_reason,
                    "a1_minimum_benign_support": raw.get("a1_minimum_benign_support"),
                    "intervention_attempt_recorded": intervention is not None,
                    "attempt_availability": (
                        "AVAILABLE_INTERVENTION_ATTEMPT"
                        if intervention is not None
                        else "NOT_APPLICABLE_POLICY_DECISION_ONLY"
                    ),
                    "labels_requested": (
                        None if intervention is None else intervention.get("labels_requested")
                    ),
                    "labels_available": (
                        None if intervention is None else intervention.get("labels_available")
                    ),
                    "optimizer_steps": (
                        None if intervention is None else intervention.get("optimizer_steps")
                    ),
                    "target_rows": (
                        None if intervention is None else intervention.get("target_rows")
                    ),
                    "replay_rows": (
                        None if intervention is None else intervention.get("replay_rows")
                    ),
                    "audit_result": (
                        None if intervention is None else intervention.get("audit_result")
                    ),
                    "accepted": (
                        None
                        if intervention is None
                        else _as_bool(intervention.get("accepted"), "Core attempt accepted")
                    ),
                    "rolled_back": (
                        None
                        if intervention is None
                        else _as_bool(intervention.get("rolled_back"), "Core attempt rollback")
                    ),
                    "rejection_reason": None if intervention is None else rejection,
                    "explicit_execution_failure": (
                        None if intervention is None else _explicit_execution_failure(rejection)
                    ),
                    "model_digest_before": (
                        None if intervention is None else intervention.get("model_digest_before")
                    ),
                    "model_digest_candidate": (
                        None if intervention is None else intervention.get("model_digest_candidate")
                    ),
                    "model_digest_after": (
                        None if intervention is None else intervention.get("model_digest_after")
                    ),
                    "threshold_digest_before": (
                        None
                        if intervention is None
                        else intervention.get("threshold_digest_before")
                    ),
                    "threshold_digest_candidate": (
                        None
                        if intervention is None
                        else intervention.get("threshold_digest_candidate")
                    ),
                    "threshold_digest_after": (
                        None if intervention is None else intervention.get("threshold_digest_after")
                    ),
                    **policy_provenance,
                }
                if feedback is not None and intervention is None:
                    raise RdxArtifactError("Core feedback exists without an intervention attempt")
                window_attempts.append(attempt_row)
                attempts.append(attempt_row)

            model_attempts = [
                row
                for row in window_attempts
                if row["intervention_attempt_recorded"]
                and str(row["action"]) in _MODEL_CHANGING_ACTIONS
            ]
            candidate_rejections = [
                row
                for row in model_attempts
                if row["accepted"] is False and bool(str(row["audit_result"] or ""))
            ]
            pathways.append(
                {
                    **context,
                    "prediction_index": _as_int(
                        window.get("prediction_index"), "Core pathway prediction_index"
                    ),
                    "stage": _as_int(window.get("stage"), "Core pathway stage"),
                    "current_domain": str(window.get("current_domain")),
                    "window_id": _as_int(window.get("window_id"), "Core pathway window_id"),
                    "predicted_health_state": str(window.get("predicted_health_state")),
                    "recognised_harm": str(window.get("predicted_health_state"))
                    == "PREDICTED_HARMFUL",
                    "decision_count": len(raw_decisions),
                    "intervention_attempt_count": len(attempt_decision_ids),
                    "model_changing_action_attempted": bool(model_attempts),
                    "a2_attempted": any(
                        row["intervention_attempt_recorded"]
                        and row["action"] == InterventionAction.HEAD_UPDATE.value
                        for row in window_attempts
                    ),
                    "a4_attempted": any(
                        row["intervention_attempt_recorded"]
                        and row["action"] == InterventionAction.REPLAY_UPDATE.value
                        for row in window_attempts
                    ),
                    "explicit_execution_failure_count": sum(
                        row["explicit_execution_failure"] is True for row in model_attempts
                    ),
                    "candidate_rejected_after_audit_count": len(candidate_rejections),
                    "candidate_accepted_count": sum(
                        row["accepted"] is True for row in model_attempts
                    ),
                    "rollback_count": sum(row["rolled_back"] is True for row in model_attempts),
                    "ordered_decision_ids_json": _json_cell(ordered_decision_ids),
                    "ordered_actions_json": _json_cell(ordered_actions),
                    "ordered_decision_reasons_json": _json_cell(ordered_reasons),
                    "ordered_skipped_actions_json": _json_cell(ordered_skipped),
                    "ordered_attempt_decision_ids_json": _json_cell(attempt_decision_ids),
                    "ordered_audit_states_json": _json_cell(audit_states),
                    "a3_core_availability": "NOT_APPLICABLE_ORACLE_ONLY_ACTION",
                    **policy_provenance,
                }
            )

        audit_payload = source.core_administrative_audits
        if audit_payload is None:
            raise RdxArtifactError("Core administrative audit artifact is absent")
        evidence = _require_list(
            audit_payload.get("evidence"), "Core administrative audit evidence"
        )
        audit_provenance = _source_context(source, audit_artifact)
        for raw in evidence:
            if not isinstance(raw, dict):
                raise RdxArtifactError("Core administrative audit evidence must be an object")
            decision_id = str(raw.get("decision_id", ""))
            prediction_index = _as_int(
                raw.get("prediction_index"), "Core administrative audit prediction_index"
            )
            if prediction_index not in harmful_prediction_indices or decision_id not in (
                harmful_decision_ids
            ):
                continue
            expected = [
                str(value)
                for value in _require_list(
                    raw.get("expected_scope_tokens"), "expected audit panels"
                )
            ]
            missing = [
                str(value)
                for value in _require_list(raw.get("missing_scope_tokens"), "missing audit panels")
            ]
            panel_decisions = _require_list(raw.get("decisions"), "audit panel decisions")
            skipped = _as_bool(raw.get("evaluation_skipped"), "audit evaluation_skipped")
            intervention = _single_intervention_for_decision(intervention_index, decision_id)
            base = {
                **context,
                "prediction_index": prediction_index,
                "decision_id": decision_id,
                "purpose": str(raw.get("purpose", "")),
                "evaluation_skipped": skipped,
                "skip_reason": _optional_text(raw.get("skip_reason")),
                "expected_panel_count": len(expected),
                "evaluated_panel_count": len(panel_decisions),
                "missing_panel_count": len(missing),
                "candidate_action": (
                    None if intervention is None else intervention.get("action_attempted")
                ),
                "candidate_audit_result": (
                    None if intervention is None else intervention.get("audit_result")
                ),
                "candidate_accepted": (
                    None
                    if intervention is None
                    else _as_bool(intervention.get("accepted"), "audit candidate accepted")
                ),
                "candidate_rolled_back": (
                    None
                    if intervention is None
                    else _as_bool(intervention.get("rolled_back"), "audit candidate rollback")
                ),
                "candidate_rejection_reason": (
                    None
                    if intervention is None
                    else _optional_text(intervention.get("rejection_reason"))
                ),
                **audit_provenance,
            }
            represented: set[str] = set()
            for panel in panel_decisions:
                if not isinstance(panel, dict):
                    raise RdxArtifactError("Core audit panel decision must be an object")
                identity = str(panel.get("domain_id", ""))
                if not identity or identity in represented:
                    raise RdxArtifactError("Core audit panel identity is blank or duplicate")
                represented.add(identity)
                state = str(panel.get("state", ""))
                if state not in {"SAFE", "UNCERTAIN", "HARMFUL"}:
                    raise RdxArtifactError("Core audit panel state is invalid")
                audits.append(
                    {
                        **base,
                        "panel_identity": identity,
                        "panel_evaluated": True,
                        "availability_state": "AVAILABLE",
                        "audit_state": state,
                        "benign_support": panel.get("benign_support"),
                        "attack_support": panel.get("attack_support"),
                        "fp": panel.get("false_positives"),
                        "tp": panel.get("true_positives"),
                        "fpr_low": panel.get("fpr_low"),
                        "fpr_high": panel.get("fpr_high"),
                        "tpr_low": panel.get("tpr_low"),
                        "tpr_high": panel.get("tpr_high"),
                        "recall_floor": panel.get("recall_floor"),
                    }
                )
            for identity in missing:
                if identity in represented:
                    raise RdxArtifactError("Core audit panel is both evaluated and missing")
                represented.add(identity)
                audits.append(
                    {
                        **base,
                        "panel_identity": identity,
                        "panel_evaluated": False,
                        "availability_state": "NOT_EVALUATED",
                        "audit_state": None,
                        "benign_support": None,
                        "attack_support": None,
                        "fp": None,
                        "tp": None,
                        "fpr_low": None,
                        "fpr_high": None,
                        "tpr_low": None,
                        "tpr_high": None,
                        "recall_floor": None,
                    }
                )
            if not panel_decisions and not missing:
                audits.append(
                    {
                        **base,
                        "panel_identity": None,
                        "panel_evaluated": False,
                        "availability_state": "NOT_EVALUATED" if skipped else "NOT_AVAILABLE",
                        "audit_state": None,
                        "benign_support": None,
                        "attack_support": None,
                        "fp": None,
                        "tp": None,
                        "fpr_low": None,
                        "fpr_high": None,
                        "tpr_low": None,
                        "tpr_high": None,
                        "recall_floor": None,
                    }
                )
    pathways.sort(
        key=lambda row: (
            str(row["experiment_id"]),
            _as_int(row["prediction_index"], "pathway sort prediction_index"),
        )
    )
    attempts.sort(
        key=lambda row: (
            str(row["experiment_id"]),
            _as_int(row["prediction_index"], "attempt sort prediction_index"),
            _as_int(row["decision_ordinal"], "attempt sort decision_ordinal"),
        )
    )
    audits.sort(
        key=lambda row: (
            str(row["experiment_id"]),
            _as_int(row["prediction_index"], "audit sort prediction_index"),
            str(row["decision_id"]),
            str(row["purpose"]),
            str(row["panel_identity"] or ""),
        )
    )
    return pathways, attempts, audits


def _classify_oracle_decision(candidates: Sequence[Mapping[str, object]]) -> str:
    """Assign the frozen mutually exclusive RDX-B class in precedence order."""

    by_action: dict[str, Mapping[str, object]] = {}
    for candidate in candidates:
        action = str(candidate.get("action", ""))
        if action in by_action:
            raise RdxArtifactError("Oracle decision has a duplicate candidate action")
        by_action[action] = candidate
    expected = {action.value for action in InterventionAction}
    if set(by_action) != expected:
        raise RdxArtifactError("Oracle decision must retain exactly A0--A4")

    def succeeded(action: str) -> bool:
        return _as_bool(by_action[action].get("confirmed_success"), "Oracle confirmed success")

    if succeeded(InterventionAction.NO_OP.value):
        return _ORACLE_CLASSES[0]
    if any(succeeded(action) for action in _CORE_ACCESSIBLE_ACTIONS):
        return _ORACLE_CLASSES[1]
    if succeeded(InterventionAction.FULL_FINE_TUNE.value):
        return _ORACLE_CLASSES[2]
    model_changes = [by_action[action] for action in _MODEL_CHANGING_ACTIONS]
    feasible = [
        candidate
        for candidate in model_changes
        if _as_bool(candidate.get("feasible"), "Oracle model-change feasibility")
    ]
    if any(
        _as_bool(candidate.get("execution_succeeded"), "Oracle candidate execution")
        for candidate in feasible
    ):
        return _ORACLE_CLASSES[3]
    if feasible:
        return _ORACLE_CLASSES[4]
    return _ORACLE_CLASSES[5]


def _oracle_successor_window(
    source: RdxSourceRun, record: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    expected_id = _as_int(record.get("successor_window_id"), "Oracle successor window_id")
    expected_start = _as_int(record.get("successor_row_start"), "Oracle successor row_start")
    expected_stop = _as_int(record.get("successor_row_stop"), "Oracle successor row_stop")
    expected_stage = _as_int(record.get("stage"), "Oracle stage")
    expected_domain = str(record.get("current_domain"))
    matches: list[tuple[int, dict[str, Any]]] = []
    for window in source.run.windows:
        if (
            _as_int(window.get("stage"), "Oracle window stage") == expected_stage
            and str(window.get("current_domain")) == expected_domain
            and _as_int(window.get("window_id"), "Oracle window_id") == expected_id
            and _as_int(window.get("row_start"), "Oracle row_start") == expected_start
            and _as_int(window.get("row_stop"), "Oracle row_stop") == expected_stop
        ):
            matches.append(
                (
                    _as_int(window.get("prediction_index"), "Oracle successor prediction_index"),
                    window,
                )
            )
    if len(matches) != 1:
        raise RdxArtifactError("Oracle successor window does not resolve uniquely")
    return matches[0]


def _derive_oracle_candidates(
    corpus: RdxCorpus,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Derive the complete nested RDX-B Oracle map and decision classes."""

    candidate_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    for source in corpus.runs:
        if source.run.method is not Study4Method.OFFLINE_ORACLE:
            continue
        payload = source.oracle_decisions
        if payload is None:
            raise RdxArtifactError(f"{source.run.experiment_id} lacks Oracle decisions")
        if (
            _as_bool(payload.get("permanent_holdout_used_for_choice"), "Oracle holdout use")
            or not _as_bool(payload.get("evaluator_truth_visible"), "Oracle truth visibility")
            or not _as_bool(payload.get("non_deployable_upper_bound"), "Oracle deployment flag")
        ):
            raise RdxArtifactError("Oracle information-policy flags violate the frozen contract")
        records = _require_list(payload.get("records"), "Oracle decision records")
        context = _common_context(source)
        provenance = _source_context(source, ORACLE_FILENAME)
        current_windows = _window_index(source)
        for raw in records:
            if not isinstance(raw, dict):
                raise RdxArtifactError("Oracle decision record must be an object")
            prediction_index = _as_int(
                raw.get("prediction_index"), "Oracle decision prediction_index"
            )
            try:
                current_window = current_windows[prediction_index]
            except KeyError as exc:
                raise RdxArtifactError("Oracle decision lacks its current window") from exc
            successor_index, successor_window = _oracle_successor_window(source, raw)
            decision_id = str(raw.get("decision_id", ""))
            selected_action = str(raw.get("selected_action", ""))
            selection_reason = str(raw.get("tie_break_reason", ""))
            current_state = str(raw.get("current_evaluator_state", ""))
            if current_state != str(current_window.get("evaluator_health_state")):
                raise RdxArtifactError("Oracle current evaluator state differs from window metrics")
            raw_candidates = _require_list(raw.get("candidates"), "Oracle candidates")
            local_rows: list[dict[str, object]] = []
            selected_count = 0
            for candidate in raw_candidates:
                if not isinstance(candidate, dict):
                    raise RdxArtifactError("Oracle candidate must be an object")
                action = str(candidate.get("action", ""))
                try:
                    rank = InterventionAction(action).rank
                except ValueError as exc:
                    raise RdxArtifactError(f"Oracle candidate action is invalid: {action}") from exc
                is_selected = action == selected_action
                selected_count += int(is_selected)
                candidate_successor_state = str(candidate.get("successor_evaluator_state", ""))
                if is_selected and candidate_successor_state != str(
                    successor_window.get("evaluator_health_state")
                ):
                    raise RdxArtifactError(
                        "selected Oracle successor state differs from deployed window metrics"
                    )
                numeric = successor_window if is_selected else None
                row: dict[str, object] = {
                    **context,
                    "prediction_index": prediction_index,
                    "stage": _as_int(raw.get("stage"), "Oracle decision stage"),
                    "current_domain": str(raw.get("current_domain")),
                    "window_id": _as_int(raw.get("window_id"), "Oracle decision window_id"),
                    "decision_id": decision_id,
                    "action": action,
                    "action_rank": rank,
                    "current_evaluator_state": current_state,
                    "feasible": _as_bool(candidate.get("feasible"), "Oracle feasibility"),
                    "feasibility_reason": str(candidate.get("feasibility_reason") or ""),
                    "execution_succeeded": _as_bool(
                        candidate.get("execution_succeeded"), "Oracle execution success"
                    ),
                    "audit_admissible": _as_bool(
                        candidate.get("audit_admissible"), "Oracle audit admissibility"
                    ),
                    "audit_state": str(candidate.get("audit_state") or ""),
                    "successor_evaluator_state": candidate_successor_state,
                    "confirmed_success": _as_bool(
                        candidate.get("confirmed_success"), "Oracle confirmed success"
                    ),
                    "selected": is_selected,
                    "selection_reason": selection_reason if is_selected else "NOT_SELECTED",
                    "optimizer_steps": candidate.get("optimizer_steps"),
                    "target_rows": candidate.get("target_rows"),
                    "replay_rows": candidate.get("replay_rows"),
                    "rows_consumed": candidate.get("rows_consumed"),
                    "incoming_model_digest": str(candidate.get("incoming_model_digest") or ""),
                    "deployed_model_digest": str(candidate.get("deployed_model_digest") or ""),
                    "deployed_threshold_digest": str(
                        candidate.get("deployed_threshold_digest") or ""
                    ),
                    "deployed_r1_digest": str(candidate.get("deployed_r1_digest") or ""),
                    "numeric_successor_availability": (
                        "AVAILABLE_SELECTED_DEPLOYED_BRANCH"
                        if is_selected
                        else "NOT_AVAILABLE_UNSELECTED_COUNTERFACTUAL"
                    ),
                    "successor_prediction_index": successor_index,
                    "successor_window_id": candidate.get("successor_window_id"),
                    "successor_row_start": candidate.get("successor_row_start"),
                    "successor_row_stop": candidate.get("successor_row_stop"),
                    "successor_attack_count": (
                        None if numeric is None else numeric.get("attack_count")
                    ),
                    "successor_benign_count": (
                        None if numeric is None else numeric.get("benign_count")
                    ),
                    "successor_tp": None if numeric is None else numeric.get("tp"),
                    "successor_fp": None if numeric is None else numeric.get("fp"),
                    "successor_tn": None if numeric is None else numeric.get("tn"),
                    "successor_fn": None if numeric is None else numeric.get("fn"),
                    "successor_tpr": None if numeric is None else numeric.get("tpr"),
                    "successor_fpr": None if numeric is None else numeric.get("fpr"),
                    "successor_false_positives_per_million": (
                        None if numeric is None else numeric.get("false_positives_per_million")
                    ),
                    "successor_fpr_budget_ratio": (
                        None if numeric is None else numeric.get("fpr_budget_ratio")
                    ),
                    "permanent_holdout_used": _as_bool(
                        candidate.get("permanent_holdout_used"), "Oracle candidate holdout use"
                    ),
                    **provenance,
                }
                if row["permanent_holdout_used"]:
                    raise RdxArtifactError("Oracle candidate used a permanent holdout")
                local_rows.append(row)
                candidate_rows.append(row)
            if selected_count != 1:
                raise RdxArtifactError("Oracle selected action does not identify one candidate")
            local_rows.sort(key=lambda row: _as_int(row["action_rank"], "Oracle action rank"))
            diagnostic_class = _classify_oracle_decision(local_rows)
            successes = [row for row in local_rows if row["confirmed_success"] is True]
            model_feasible = [
                row
                for row in local_rows
                if str(row["action"]) in _MODEL_CHANGING_ACTIONS and row["feasible"] is True
            ]
            selected_candidate = next(row for row in local_rows if row["selected"] is True)
            decision_rows.append(
                {
                    **context,
                    "prediction_index": prediction_index,
                    "stage": _as_int(raw.get("stage"), "Oracle summary stage"),
                    "current_domain": str(raw.get("current_domain")),
                    "window_id": _as_int(raw.get("window_id"), "Oracle summary window_id"),
                    "decision_id": decision_id,
                    "current_evaluator_state": current_state,
                    "diagnostic_class": diagnostic_class,
                    "confirmed_success_count": len(successes),
                    "a0_confirmed_success_count": sum(
                        row["action"] == InterventionAction.NO_OP.value for row in successes
                    ),
                    "core_accessible_confirmed_success_count": sum(
                        str(row["action"]) in _CORE_ACCESSIBLE_ACTIONS for row in successes
                    ),
                    "core_update_confirmed_success_count": sum(
                        str(row["action"])
                        in {
                            InterventionAction.HEAD_UPDATE.value,
                            InterventionAction.REPLAY_UPDATE.value,
                        }
                        for row in successes
                    ),
                    "a3_confirmed_success_count": sum(
                        row["action"] == InterventionAction.FULL_FINE_TUNE.value
                        for row in successes
                    ),
                    "model_change_feasible_count": len(model_feasible),
                    "model_change_execution_success_count": sum(
                        row["execution_succeeded"] is True for row in model_feasible
                    ),
                    "selected_action": selected_action,
                    "selected_confirmed_success": selected_candidate["confirmed_success"],
                    "selection_reason": selection_reason,
                    "candidate_count": len(local_rows),
                    "candidate_actions_json": _json_cell(
                        [str(row["action"]) for row in local_rows]
                    ),
                    "evaluator_truth_visible": _as_bool(
                        raw.get("evaluator_truth_visible"), "Oracle record truth visibility"
                    ),
                    "non_deployable_upper_bound": _as_bool(
                        raw.get("non_deployable_upper_bound"), "Oracle record deployment flag"
                    ),
                    "permanent_holdout_used_for_choice": _as_bool(
                        raw.get("permanent_holdout_used_for_choice"),
                        "Oracle record holdout choice",
                    ),
                    **provenance,
                }
            )
    candidate_rows.sort(
        key=lambda row: (
            str(row["experiment_id"]),
            _as_int(row["prediction_index"], "Oracle candidate sort prediction_index"),
            _as_int(row["action_rank"], "Oracle candidate sort action_rank"),
        )
    )
    decision_rows.sort(
        key=lambda row: (
            str(row["experiment_id"]),
            _as_int(row["prediction_index"], "Oracle decision sort prediction_index"),
        )
    )
    return candidate_rows, decision_rows


def _holdout_reference(
    row: Mapping[str, Any], *, anchor_stage: int, sequence: Sequence[str]
) -> dict[str, object]:
    stage = _optional_int(row.get("stage"), "holdout stage")
    holdout_dataset_id = str(row.get("holdout_dataset_id", ""))
    try:
        holdout_position = tuple(sequence).index(holdout_dataset_id)
    except ValueError as exc:
        raise RdxArtifactError(
            f"holdout dataset {holdout_dataset_id!r} is absent from the deployment sequence"
        ) from exc
    return {
        "event_index": _as_int(row.get("event_index"), "holdout event_index"),
        "event": str(row.get("event", "")),
        "stage": stage,
        "prediction_index": _optional_int(row.get("prediction_index"), "holdout prediction_index"),
        "holdout_dataset_id": holdout_dataset_id,
        "operating_envelope_state": str(row.get("operating_envelope_state", "")),
        "learned_reference_tpr": row.get("learned_reference_tpr"),
        "operational_tpr_forgetting": row.get("operational_tpr_forgetting"),
        "model_digest": _optional_text(row.get("model_digest")),
        "was_learned_by_anchor_stage": holdout_position <= anchor_stage,
    }


def _retention_references(
    source: RdxSourceRun,
    *,
    event: str,
    anchor_stage: int,
    anchor_prediction_index: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, Any]] = []
    for row in source.run.holdouts:
        if str(row.get("event")) != event:
            continue
        if event == "post_accept":
            if _optional_int(row.get("prediction_index"), "post_accept prediction_index") != (
                anchor_prediction_index
            ):
                continue
        elif event == "domain_end":
            if _optional_int(row.get("stage"), "domain_end stage") != anchor_stage:
                continue
        selected.append(row)
    selected.sort(
        key=lambda row: (
            _as_int(row.get("event_index"), "retention event_index"),
            str(row.get("holdout_dataset_id", "")),
        )
    )
    return [
        _holdout_reference(
            row,
            anchor_stage=anchor_stage,
            sequence=source.run.sequence,
        )
        for row in selected
    ]


def _derive_deployed_recovery(corpus: RdxCorpus) -> list[dict[str, object]]:
    """Apply the frozen immediate/two-window horizon to actual accepted updates."""

    rows: list[dict[str, object]] = []
    for source in corpus.runs:
        context = _common_context(source)
        provenance = _source_context(source, "intervention_log.csv")
        windows = sorted(
            source.run.windows,
            key=lambda row: _as_int(row.get("prediction_index"), "recovery window index"),
        )
        accepted: list[dict[str, Any]] = []
        for intervention in source.run.interventions:
            action = str(intervention.get("action_attempted", ""))
            if action not in _MODEL_CHANGING_ACTIONS:
                continue
            if _as_bool(intervention.get("accepted"), "recovery intervention accepted"):
                accepted.append(intervention)
        accepted.sort(
            key=lambda row: (
                _as_int(row.get("prediction_index"), "accepted intervention index"),
                str(row.get("decision_id", "")),
            )
        )
        for anchor in accepted:
            anchor_index = _as_int(
                anchor.get("prediction_index"), "recovery anchor prediction_index"
            )
            stage = _as_int(anchor.get("domain_stage"), "recovery anchor stage")
            domain = str(anchor.get("current_domain", ""))
            next_updates = [
                item
                for item in accepted
                if _as_int(item.get("prediction_index"), "next update prediction_index")
                > anchor_index
                and _as_int(item.get("domain_stage"), "next update stage") == stage
                and str(item.get("current_domain", "")) == domain
            ]
            next_update = (
                None
                if not next_updates
                else min(
                    _as_int(item.get("prediction_index"), "next accepted update")
                    for item in next_updates
                )
            )
            eligible = [
                window
                for window in windows
                if _as_int(window.get("prediction_index"), "eligible recovery index") > anchor_index
                and _as_int(window.get("stage"), "eligible recovery stage") == stage
                and str(window.get("current_domain", "")) == domain
                and (
                    next_update is None
                    or _as_int(window.get("prediction_index"), "eligible recovery boundary")
                    <= next_update
                )
            ]
            eligible.sort(
                key=lambda window: _as_int(
                    window.get("prediction_index"), "eligible recovery sort index"
                )
            )
            immediate = eligible[0] if eligible else None
            first = eligible[0] if eligible else None
            second = eligible[1] if len(eligible) >= 2 else None
            immediate_state = (
                None if immediate is None else str(immediate.get("evaluator_health_state"))
            )
            if immediate_state is not None and immediate_state not in {
                "SAFE",
                "UNCERTAIN",
                "HARMFUL",
            }:
                raise RdxArtifactError("recovery immediate evaluator state is invalid")
            sustained_states = [] if first is None else [str(first.get("evaluator_health_state"))]
            if second is not None:
                sustained_states.append(str(second.get("evaluator_health_state")))
            if any(state not in {"SAFE", "UNCERTAIN", "HARMFUL"} for state in sustained_states):
                raise RdxArtifactError("recovery sustained evaluator state is invalid")
            enough = len(eligible) >= 2
            sustained_safe = enough and sustained_states == ["SAFE", "SAFE"]
            censoring = (
                "NOT_APPLICABLE"
                if enough
                else ("NEXT_ACCEPTED_INTERVENTION" if next_update is not None else "DOMAIN_END")
            )
            immediate_outcome = "NOT_ASSESSED_CENSORED" if immediate is None else immediate_state
            sustained_outcome = (
                ("SAFE" if sustained_safe else "NOT_SAFE_OBSERVED")
                if enough
                else "NOT_ASSESSED_CENSORED"
            )
            trajectory = [
                {
                    "prediction_index": _as_int(
                        window.get("prediction_index"), "trajectory prediction_index"
                    ),
                    "window_id": _as_int(window.get("window_id"), "trajectory window_id"),
                    "evaluator_health_state": str(window.get("evaluator_health_state")),
                }
                for window in eligible
            ]
            rows.append(
                {
                    **context,
                    "anchor_prediction_index": anchor_index,
                    "anchor_decision_id": str(anchor.get("decision_id", "")),
                    "stage": stage,
                    "current_domain": domain,
                    "anchor_window_id": _as_int(
                        anchor.get("window_id"), "recovery anchor window_id"
                    ),
                    "action": str(anchor.get("action_attempted", "")),
                    "audit_result": _optional_text(anchor.get("audit_result")),
                    "audit_admissibility_state": "ADMISSIBLE_ACCEPTED",
                    "model_digest_after": _optional_text(anchor.get("model_digest_after")),
                    "next_accepted_model_change_prediction_index": next_update,
                    "immediate_prediction_index": (
                        None
                        if immediate is None
                        else _as_int(
                            immediate.get("prediction_index"), "immediate prediction_index"
                        )
                    ),
                    "immediate_window_id": (
                        None
                        if immediate is None
                        else _as_int(immediate.get("window_id"), "immediate window_id")
                    ),
                    "immediate_evaluator_state": immediate_state,
                    "immediate_safe": None if immediate is None else immediate_state == "SAFE",
                    "immediate_outcome": immediate_outcome,
                    "immediate_recovery_state": immediate_outcome,
                    "sustained_first_prediction_index": (
                        None
                        if first is None
                        else _as_int(first.get("prediction_index"), "sustained first index")
                    ),
                    "sustained_first_evaluator_state": (
                        None if first is None else str(first.get("evaluator_health_state"))
                    ),
                    "sustained_second_prediction_index": (
                        None
                        if second is None
                        else _as_int(second.get("prediction_index"), "sustained second index")
                    ),
                    "sustained_second_window_id": (
                        None
                        if second is None
                        else _as_int(second.get("window_id"), "sustained second window_id")
                    ),
                    "sustained_second_evaluator_state": (
                        None if second is None else str(second.get("evaluator_health_state"))
                    ),
                    "sustained_safe": sustained_safe if enough else None,
                    "sustained_outcome": sustained_outcome,
                    "sustained_two_window_status": sustained_outcome,
                    "eligible_successor_count_before_censoring": len(eligible),
                    "censoring_boundary": censoring,
                    "censor_reason": censoring,
                    "later_same_domain_trajectory_json": _json_cell(trajectory),
                    "post_accept_holdout_references_json": _json_cell(
                        _retention_references(
                            source,
                            event="post_accept",
                            anchor_stage=stage,
                            anchor_prediction_index=anchor_index,
                        )
                    ),
                    "domain_end_holdout_references_json": _json_cell(
                        _retention_references(
                            source,
                            event="domain_end",
                            anchor_stage=stage,
                            anchor_prediction_index=anchor_index,
                        )
                    ),
                    "final_holdout_references_json": _json_cell(
                        _retention_references(
                            source,
                            event="final",
                            anchor_stage=stage,
                            anchor_prediction_index=anchor_index,
                        )
                    ),
                    **provenance,
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["experiment_id"]),
            _as_int(row["anchor_prediction_index"], "recovery sort anchor index"),
            str(row["anchor_decision_id"]),
        )
    )
    return rows


def _derive_incident_links(corpus: RdxCorpus) -> list[dict[str, object]]:
    """Left-join every harmful Core window to secondary controller lifecycle facts."""

    rows: list[dict[str, object]] = []
    policy_artifact = f"{CORE_BUNDLE_DIRNAME}/core_policy_windows.json"
    for source in corpus.runs:
        if source.run.method is not Study4Method.DANIDS_CORE:
            continue
        lifecycle_payload = source.core_incident_lifecycle
        unresolved_payload = source.core_unresolved_exposures
        if lifecycle_payload is None or unresolved_payload is None:
            raise RdxArtifactError("Core lifecycle/unresolved exposure artifacts are absent")
        lifecycle = _require_list(lifecycle_payload.get("events"), "Core incident events")
        unresolved = _require_list(unresolved_payload.get("exposures"), "Core unresolved exposures")
        by_lifecycle: dict[int, list[dict[str, Any]]] = {}
        for raw in lifecycle:
            if not isinstance(raw, dict):
                raise RdxArtifactError("Core incident event must be an object")
            index = _as_int(raw.get("prediction_index"), "incident event prediction_index")
            by_lifecycle.setdefault(index, []).append(raw)
        by_unresolved: dict[int, list[dict[str, Any]]] = {}
        for raw in unresolved:
            if not isinstance(raw, dict):
                raise RdxArtifactError("Core unresolved exposure must be an object")
            index = _as_int(raw.get("prediction_index"), "unresolved prediction_index")
            by_unresolved.setdefault(index, []).append(raw)
        context = _common_context(source)
        provenance = _source_context(source, policy_artifact)
        for window, trace in _harmful_core_windows(source):
            index = _as_int(window.get("prediction_index"), "incident link prediction_index")
            state = trace.get("controller_state_after")
            if not isinstance(state, dict):
                raise RdxArtifactError("Core trace lacks controller_state_after")
            counters = state.get("counters")
            if not isinstance(counters, dict):
                raise RdxArtifactError("Core controller state lacks counters")
            events = sorted(
                by_lifecycle.get(index, []),
                key=lambda event: (
                    str(event.get("event", "")),
                    _optional_int(event.get("incident_index"), "incident event index") or -1,
                ),
            )
            exposures = by_unresolved.get(index, [])
            if len(exposures) > 1:
                raise RdxArtifactError("harmful Core window has duplicate unresolved exposures")
            exposure = None if not exposures else exposures[0]
            rows.append(
                {
                    **context,
                    "prediction_index": index,
                    "stage": _as_int(window.get("stage"), "incident link stage"),
                    "current_domain": str(window.get("current_domain", "")),
                    "window_id": _as_int(window.get("window_id"), "incident link window_id"),
                    "predicted_health_state": str(window.get("predicted_health_state", "")),
                    "active_incident_after_decision": _as_bool(
                        state.get("active_incident"), "active incident after decision"
                    ),
                    "incident_index_after_decision": _optional_int(
                        state.get("incident_index"), "incident index after decision"
                    ),
                    "unresolved_harmful_windows_after_decision": _as_int(
                        counters.get("unresolved_harmful_windows"),
                        "unresolved harmful windows after decision",
                    ),
                    "lifecycle_events_json": _json_cell(events),
                    "unresolved_exposure_recorded": exposure is not None,
                    "unresolved_exposure_incident_index": (
                        None
                        if exposure is None
                        else _optional_int(
                            exposure.get("incident_index"), "unresolved incident index"
                        )
                    ),
                    "unresolved_counter_after": (
                        None
                        if exposure is None
                        else _as_int(
                            exposure.get("unresolved_counter_after"),
                            "unresolved exposure counter",
                        )
                    ),
                    "unresolved_actions_json": (
                        _json_cell([])
                        if exposure is None
                        else _json_cell(exposure.get("actions") or [])
                    ),
                    "unresolved_reasons_json": (
                        _json_cell([])
                        if exposure is None
                        else _json_cell(exposure.get("reasons") or [])
                    ),
                    "incident_link_availability": "AVAILABLE_PERSISTED_CONTROLLER_STATE",
                    **provenance,
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["experiment_id"]),
            _as_int(row["prediction_index"], "incident sort prediction_index"),
        )
    )
    return rows


def _derive_all(corpus: RdxCorpus) -> dict[str, list[dict[str, object]]]:
    """Derive every frozen RDX-A--RDX-D table and enforce cross-table coverage."""

    harm = _derive_harm_windows(corpus)
    pathways, attempts, audits = _derive_core_pathways(corpus)
    evidence = _derive_evidence_snapshots(corpus)
    oracle_candidates, oracle_decisions = _derive_oracle_candidates(corpus)
    recovery = _derive_deployed_recovery(corpus)
    incidents = _derive_incident_links(corpus)

    core_harm_keys = {
        (
            str(row["experiment_id"]),
            _as_int(row["prediction_index"], "Core harm coverage prediction_index"),
        )
        for row in harm
        if row["method"] == Study4Method.DANIDS_CORE.value
    }
    for name, rows in {
        CORE_PATHWAY_FILENAME: pathways,
        EVIDENCE_FILENAME: evidence,
        INCIDENT_FILENAME: incidents,
    }.items():
        keys = {
            (
                str(row["experiment_id"]),
                _as_int(row["prediction_index"], "Core table coverage prediction_index"),
            )
            for row in rows
        }
        if keys != core_harm_keys:
            raise RdxArtifactError(f"{name} coverage differs from harmful Core windows")
    attempt_keys = {
        (
            str(row["experiment_id"]),
            _as_int(row["prediction_index"], "attempt coverage prediction_index"),
        )
        for row in attempts
    }
    if not attempt_keys.issubset(core_harm_keys):
        raise RdxArtifactError("Core attempts contain a non-harmful primary window")
    audit_keys = {
        (
            str(row["experiment_id"]),
            _as_int(row["prediction_index"], "audit coverage prediction_index"),
        )
        for row in audits
    }
    if not audit_keys.issubset(core_harm_keys):
        raise RdxArtifactError("Core audits contain a non-harmful primary window")
    if len(oracle_candidates) != 5 * len(oracle_decisions):
        raise RdxArtifactError("Oracle candidate/decision cardinality is not exactly five-to-one")
    return {
        HARM_FILENAME: harm,
        CORE_PATHWAY_FILENAME: pathways,
        CORE_ATTEMPT_FILENAME: attempts,
        CORE_AUDIT_FILENAME: audits,
        EVIDENCE_FILENAME: evidence,
        ORACLE_CANDIDATE_FILENAME: oracle_candidates,
        ORACLE_DECISION_FILENAME: oracle_decisions,
        RECOVERY_FILENAME: recovery,
        INCIDENT_FILENAME: incidents,
    }


def _protocol_path() -> Path:
    return Path(__file__).resolve().parents[3] / "docs" / "rdx_protocol.md"


def _validate_protocol_identity() -> Path:
    path = _protocol_path()
    if _file_sha256(path) != RDX_PROTOCOL_SHA256:
        raise RdxArtifactError("docs/rdx_protocol.md differs from frozen RDX-001 v1.0")
    return path


def _build_contract(corpus: RdxCorpus) -> dict[str, object]:
    protocol = _validate_protocol_identity()
    run_records: list[dict[str, object]] = []
    for source in corpus.runs:
        run = source.run
        run_records.append(
            {
                "experiment_id": run.experiment_id,
                "method": run.method.value,
                "sequence": list(run.sequence),
                "seed": run.seed,
                "source_run_path": str(run.path),
                "source_manifest_sha256": source.manifest_sha256,
                "source_manifest_bundle_digest": source.manifest_bundle_digest,
                "consumed_source_files_sha256": source.source_files_sha256,
            }
        )
    return {
        "version": RDX_EVALUATION_VERSION,
        "protocol": {
            "version": RDX_PROTOCOL_VERSION,
            "path": str(protocol),
            "sha256": RDX_PROTOCOL_SHA256,
            "source_research_tag": RDX_SOURCE_RESEARCH_TAG,
            "post_freeze_status": "POST_FREEZE_DIAGNOSTIC_EXTENSION_NOT_ORIGINAL_STUDY4",
        },
        "source_study4_evaluation": {
            "path": str(corpus.evaluation_root),
            "evaluation_contract_path": str(corpus.evaluation_root / "evaluation_contract.json"),
            "evaluation_contract_sha256": corpus.evaluation_contract_sha256,
            "artifact_manifest_path": str(corpus.evaluation_root / MANIFEST_FILENAME),
            "artifact_manifest_sha256": corpus.evaluation_manifest_sha256,
            "artifact_manifest_bundle_digest": corpus.evaluation_manifest_bundle_digest,
            "shared_contract_digest": str(
                corpus.evaluation_contract.get("shared_contract_digest", "")
            ),
        },
        "source_runs": run_records,
        "source_roster": {
            "run_count": 48,
            "methods": sorted(
                method.value
                for method in (
                    Study4Method.STATIC,
                    Study4Method.ALWAYS_ADAPT,
                    Study4Method.DANIDS_CORE,
                    Study4Method.OFFLINE_ORACLE,
                )
            ),
            "rotations": sorted("-".join(rotation) for rotation in FROZEN_ROTATIONS),
            "seeds": list(STUDY4_CONFIRMATORY_SEEDS),
            "allow_smoke": False,
            "allow_incomplete": False,
        },
        "derivation_contract": {
            "primary_unit": "evaluator-HARMFUL decision window (experiment_id, prediction_index)",
            "recognition_timing": "same_window_harm_screening",
            "decision_time_evidence_source": "core_policy_windows.observation_pre_decision",
            "oracle_candidate_actions": [action.value for action in InterventionAction],
            "oracle_class_precedence": list(_ORACLE_CLASSES),
            "recovery_actions": sorted(_MODEL_CHANGING_ACTIONS),
            "immediate_horizon": "first later same-stage same-domain deployed prediction",
            "sustained_horizon": (
                "first two consecutive later same-stage same-domain deployed predictions"
            ),
            "later_update_boundary": (
                "include later update window pre-action; exclude every subsequent window"
            ),
            "censoring_state": "NOT_ASSESSED_CENSORED",
            "unselected_oracle_numeric_successor_state": "NOT_AVAILABLE_UNSELECTED_COUNTERFACTUAL",
            "permanent_holdout_action_choice": "forbidden",
            "scientific_verdicts_emitted": False,
        },
        "outputs": {
            "files": sorted(OUTPUT_SCHEMAS),
            "schemas": {name: list(columns) for name, columns in sorted(OUTPUT_SCHEMAS.items())},
            "json_files": [CONTRACT_FILENAME, SUMMARY_FILENAME, MANIFEST_FILENAME],
            "canonical_json": "sorted_keys_indent_2_lf_no_nan",
            "canonical_csv": "fixed_columns_lf_blank_for_null",
        },
    }


def _build_summary(
    corpus: RdxCorpus, tables: Mapping[str, Sequence[Mapping[str, object]]]
) -> dict[str, object]:
    table_records: dict[str, dict[str, object]] = {}
    for name, rows in sorted(tables.items()):
        text = _csv_text(OUTPUT_SCHEMAS[name], rows)
        table_records[name] = {
            "row_count": len(rows),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    method_counts = {
        method.value: sum(source.run.method is method for source in corpus.runs)
        for method in (
            Study4Method.STATIC,
            Study4Method.ALWAYS_ADAPT,
            Study4Method.DANIDS_CORE,
            Study4Method.OFFLINE_ORACLE,
        )
    }
    return {
        "version": RDX_EVALUATION_VERSION,
        "status": "VALIDATED_ARTIFACT_ONLY_DERIVATION",
        "protocol_version": RDX_PROTOCOL_VERSION,
        "source_research_tag": RDX_SOURCE_RESEARCH_TAG,
        "source_run_count": len(corpus.runs),
        "source_method_counts": method_counts,
        "source_rotations": sorted({"-".join(source.run.sequence) for source in corpus.runs}),
        "source_seeds": sorted({source.run.seed for source in corpus.runs}),
        "source_study4_evaluation_contract_sha256": corpus.evaluation_contract_sha256,
        "tables": table_records,
        "scientific_verdicts_emitted": False,
        "new_experiments_run": False,
        "source_artifacts_mutated": False,
    }


def _write_evaluation_bundle(
    corpus: RdxCorpus,
    output_dir: str | Path,
    *,
    tables: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
) -> Path:
    """Write one new deterministic bundle; existing output paths are never reused."""

    output = Path(output_dir).resolve()
    frozen_study4_root = next(
        (
            candidate
            for candidate in (corpus.evaluation_root, *corpus.evaluation_root.parents)
            if candidate.name.casefold() == "study4"
        ),
        corpus.evaluation_root,
    )
    protected_roots = {
        frozen_study4_root,
        corpus.evaluation_root,
        *(source.run.path for source in corpus.runs),
    }
    if any(output == protected or protected in output.parents for protected in protected_roots):
        raise RdxArtifactError("RDX output must not be written inside a frozen Study-4 directory")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite RDX output directory: {output}")
    derived = dict(_derive_all(corpus) if tables is None else tables)
    if set(derived) != set(OUTPUT_SCHEMAS):
        raise RdxArtifactError("derived RDX table set differs from the fixed output contract")
    contract = _build_contract(corpus)
    summary = _build_summary(corpus, derived)

    output.mkdir(parents=True, exist_ok=False)
    for name in sorted(OUTPUT_SCHEMAS):
        (output / name).write_text(
            _csv_text(OUTPUT_SCHEMAS[name], derived[name]),
            encoding="utf-8",
            newline="",
        )
    (output / CONTRACT_FILENAME).write_text(_json_text(contract), encoding="utf-8", newline="")
    (output / SUMMARY_FILENAME).write_text(_json_text(summary), encoding="utf-8", newline="")
    files = _tree_digests(output)
    if set(files) != ALL_OUTPUT_FILES.difference({MANIFEST_FILENAME}):
        raise RdxArtifactError("RDX pre-manifest file set differs from the output contract")
    manifest = {
        "version": RDX_EVALUATION_VERSION,
        "files": files,
        "bundle_digest": _digest(files),
    }
    (output / MANIFEST_FILENAME).write_text(_json_text(manifest), encoding="utf-8", newline="")
    return output


def evaluate_rdx_recoverability(study4_evaluation_root: str | Path, output_dir: str | Path) -> Path:
    """Derive, seal, and independently validate the frozen RDX diagnostic bundle."""

    corpus = _resolve_canonical_sources(study4_evaluation_root)
    output = _write_evaluation_bundle(corpus, output_dir)
    validate_rdx_evaluation(output)
    return output


def validate_rdx_evaluation(output_dir: str | Path) -> None:
    """Source-rederive and byte-compare an RDX bundle against immutable Study 4."""

    root = Path(output_dir).resolve()
    if not root.is_dir():
        raise RdxArtifactError(f"RDX output directory does not exist: {root}")
    entries = list(root.iterdir())
    if any(not entry.is_file() for entry in entries) or {entry.name for entry in entries} != (
        ALL_OUTPUT_FILES
    ):
        raise RdxArtifactError("RDX output file set differs from the exact 12-file contract")
    manifest = _load_json(root / MANIFEST_FILENAME)
    if (root / MANIFEST_FILENAME).read_bytes() != _json_text(manifest).encode("utf-8"):
        raise RdxArtifactError("RDX manifest JSON is not canonically serialized")
    files = _tree_digests(root)
    if (
        set(manifest) != {"version", "files", "bundle_digest"}
        or manifest.get("version") != RDX_EVALUATION_VERSION
        or manifest.get("files") != files
        or manifest.get("bundle_digest") != _digest(files)
        or set(files) != ALL_OUTPUT_FILES.difference({MANIFEST_FILENAME})
    ):
        raise RdxArtifactError("RDX artifact manifest differs from its files")

    contract = _load_json(root / CONTRACT_FILENAME)
    summary = _load_json(root / SUMMARY_FILENAME)
    if (root / CONTRACT_FILENAME).read_bytes() != _json_text(contract).encode("utf-8"):
        raise RdxArtifactError("RDX contract JSON is not canonically serialized")
    if (root / SUMMARY_FILENAME).read_bytes() != _json_text(summary).encode("utf-8"):
        raise RdxArtifactError("RDX summary JSON is not canonically serialized")
    source_record = contract.get("source_study4_evaluation")
    if not isinstance(source_record, dict):
        raise RdxArtifactError("RDX contract lacks Study-4 source provenance")
    source_root = Path(str(source_record.get("path", "")))
    corpus = _resolve_canonical_sources(source_root)
    expected_contract = _build_contract(corpus)
    if contract != expected_contract:
        raise RdxArtifactError("RDX contract differs from current immutable source identities")
    tables = _derive_all(corpus)
    for name, rows in tables.items():
        expected_text = _csv_text(OUTPUT_SCHEMAS[name], rows)
        if (root / name).read_bytes() != expected_text.encode("utf-8"):
            raise RdxArtifactError(f"persisted {name} differs from source-backed recomputation")
    expected_summary = _build_summary(corpus, tables)
    if summary != expected_summary:
        raise RdxArtifactError("RDX summary differs from source-backed recomputation")


__all__ = [
    "ALL_OUTPUT_FILES",
    "MANIFEST_FILENAME",
    "OUTPUT_SCHEMAS",
    "RDX_EVALUATION_VERSION",
    "RDX_PROTOCOL_VERSION",
    "RdxArtifactError",
    "evaluate_rdx_recoverability",
    "validate_rdx_evaluation",
]
