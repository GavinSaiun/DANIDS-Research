"""Artifact-only RDX-007 analysis of the frozen RDX-004 experiment.

The module validates all historical and prospective source bundles before it
loads outcome records.  It then derives the prospectively frozen P1/P2
estimands at the rotation-by-seed unit, applies the registered gate, and writes
one deterministic, write-once analysis package.  It never opens raw NetFlow
data, trains a model, or mutates a source artifact.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Any, Final

import numpy as np

from danids.adaptation.actions import ACTION_ORDER
from danids.config.rdx_training_evidence import (
    RDX004_ALL_BUDGETS,
    RDX004_PROSPECTIVE_BUDGETS,
    RDX004_PROTOCOL_SHA256,
    RDX004_PROTOCOL_VERSION,
    RDX004_ROTATIONS,
    RDX004_SEEDS,
    RDX004Budget,
    RDX004RunIdentity,
    rdx004_treatment,
)
from danids.config.rdx_training_execution import (
    RDX006_EXECUTION_IMPLEMENTATION_VERSION,
    RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST,
)
from danids.config.study4 import Study4Method
from danids.evaluation import study4 as study4_artifacts
from danids.evaluation.rdx_training_execution import (
    EXECUTION_CONTRACT_FILENAME,
    HOLDOUT_FILENAME,
    MEMORY_FILENAME,
    ORACLE_COUNTERFACTUALS_FILENAME,
    QUERY_EVIDENCE_FILENAME,
    RESOURCE_FILENAME,
    WINDOW_FILENAME,
    validate_rdx004_runs,
)
from danids.experiments import rdx_training_evidence as preflight_artifacts
from danids.policy.oracle import OracleCandidate

RDX007_ANALYSIS_VERSION: Final = "rdx007-training-evidence-analysis-v1"
RDX006_EXECUTION_COMMIT: Final = "0d6a0b228f13957e88acd2db6cf51fa62985f26d"
BOOTSTRAP_PAYLOAD: Final = f"{RDX004_PROTOCOL_VERSION}|paired-bootstrap-v1"
BOOTSTRAP_SHA256: Final = hashlib.sha256(BOOTSTRAP_PAYLOAD.encode("utf-8")).hexdigest()
BOOTSTRAP_SEED: Final = int.from_bytes(
    hashlib.sha256(BOOTSTRAP_PAYLOAD.encode("utf-8")).digest()[:8], "big"
)
BOOTSTRAP_REPLICATES: Final = 10_000
MATERIAL_IMPROVEMENT: Final = Fraction(1, 50)

CONTRACT_FILENAME: Final = "rdx007_analysis_contract.json"
MATRIX_FILENAME: Final = "rdx007_matrix_validation.csv"
PRIMARY_FILENAME: Final = "rdx007_primary_unit_outcomes.csv"
TREATMENT_FILENAME: Final = "rdx007_treatment_summary.csv"
ROTATION_FILENAME: Final = "rdx007_rotation_summary.csv"
PAIRED_FILENAME: Final = "rdx007_paired_contrasts.csv"
BOOTSTRAP_FILENAME: Final = "rdx007_bootstrap_intervals.csv"
MECHANISM_FILENAME: Final = "rdx007_action_mechanisms.csv"
DOSE_FILENAME: Final = "rdx007_evidence_dose.csv"
CONTEXT_FILENAME: Final = "rdx007_contextual_outcomes.csv"
GATE_FILENAME: Final = "rdx007_decision_gate.json"
SUMMARY_FILENAME: Final = "rdx007_analysis_summary.json"
REPORT_FILENAME: Final = "rdx007_analysis_report.md"
MANIFEST_FILENAME: Final = "artifact_manifest.json"

OUTPUT_SCHEMAS: Final[dict[str, tuple[str, ...]]] = {
    MATRIX_FILENAME: (
        "budget",
        "evidence_status",
        "sequence",
        "seed",
        "run_id",
        "run_path",
        "artifact_manifest_sha256",
        "artifact_bundle_digest",
        "paired_b100_experiment_id",
        "paired_study1_experiment_id",
        "protocol_sha256",
        "preflight_bundle_digest",
        "execution_implementation_version",
        "execution_commit",
        "query_event_count",
        "scope_closure_count",
        "oracle_decision_count",
        "smoke",
        "validation_status",
    ),
    PRIMARY_FILENAME: (
        "budget",
        "evidence_status",
        "sequence",
        "seed",
        "run_id",
        "harmful_denominator",
        "p1_numerator",
        "p1",
        "p1_exact",
        "p2_numerator",
        "p2",
        "p2_exact",
        "a3_exclusive_numerator",
        "a3_exclusive",
        "a3_exclusive_exact",
        "availability",
        "analysis_unit",
    ),
    TREATMENT_FILENAME: (
        "budget",
        "outcome",
        "n_units",
        "unavailable_units",
        "eligible_harmful_total",
        "mean_unit_rate",
        "mean_unit_rate_exact",
        "median_unit_rate",
        "median_unit_rate_exact",
        "minimum_unit_rate",
        "maximum_unit_rate",
        "aggregation_unit",
    ),
    ROTATION_FILENAME: (
        "budget",
        "outcome",
        "sequence",
        "n_seed_units",
        "median_unit_rate",
        "median_unit_rate_exact",
    ),
    PAIRED_FILENAME: (
        "contrast",
        "outcome",
        "sequence",
        "seed",
        "baseline_budget",
        "comparison_budget",
        "baseline_value",
        "baseline_value_exact",
        "comparison_value",
        "comparison_value_exact",
        "paired_difference",
        "paired_difference_exact",
        "direction",
    ),
    BOOTSTRAP_FILENAME: (
        "contrast",
        "outcome",
        "n_units",
        "mean_difference",
        "mean_difference_exact",
        "median_difference",
        "median_difference_exact",
        "mean_ci_lower",
        "mean_ci_upper",
        "median_ci_lower",
        "median_ci_upper",
        "positive_unit_count",
        "zero_unit_count",
        "negative_unit_count",
        "positive_rotation_medians",
        "rotation_medians_json",
        "bootstrap_replicates",
        "bootstrap_seed",
        "generator",
        "percentile_method",
        "interval_role",
    ),
    MECHANISM_FILENAME: (
        "budget",
        "sequence",
        "seed",
        "action",
        "eligible_harmful_decisions",
        "feasible_count",
        "feasible_rate",
        "explicit_execution_failure_count",
        "execution_success_count",
        "execution_denominator_feasible",
        "execution_success_rate_given_feasible",
        "audit_admissible_count",
        "audit_rejected_count",
        "audit_denominator_executed_defined",
        "audit_admissible_rate_given_executed",
        "successor_safe_count",
        "successor_uncertain_count",
        "successor_harmful_count",
        "successor_unavailable_count",
        "confirmed_success_count",
        "confirmed_success_rate",
        "selected_count",
    ),
    DOSE_FILENAME: (
        "budget",
        "sequence",
        "seed",
        "expected_maximum_optimizer_eligible_target_rows",
        "expected_cumulative_target_doses_json",
        "scheduled_cumulative_target_doses_by_scope_json",
        "target_availability_provenance",
        "observed_maximum_target_rows_available",
        "observed_maximum_target_rows_consumed",
        "observed_cumulative_target_doses_available_json",
        "observed_cumulative_target_doses_consumed_json",
        "target_doses_available_by_action_json",
        "target_doses_consumed_by_action_json",
        "scheduled_cumulative_target_doses_verified",
        "consumed_target_doses_within_schedule",
        "labels_requested",
        "labels_released",
        "training_rows_allocated",
        "later_domain_scope_count",
        "later_domain_audit_rows_json",
        "audit_rows_per_later_domain_exactly_20",
        "later_domain_replay_rows_json",
        "maximum_later_domain_replay_rows",
        "historical_replay_capacity_per_domain",
        "b1600_projection_identity",
        "contrast_note",
    ),
    CONTEXT_FILENAME: (
        "budget",
        "sequence",
        "seed",
        "run_id",
        "window_count",
        "operating_envelope_compliance",
        "unsafe_exposure_windows",
        "accepted_model_changing_updates",
        "requested_labels",
        "released_labels",
        "selected_optimizer_steps",
        "final_replay_rows",
        "final_audit_rows",
        "final_holdout_rows",
        "final_holdout_metrics_json",
        "final_mean_operational_tpr_forgetting",
        "wall_clock_selected_update_seconds",
        "wall_clock_comparable",
        "interpretation_scope",
    ),
}

ALL_OUTPUT_FILES: Final = frozenset(
    {
        *OUTPUT_SCHEMAS,
        CONTRACT_FILENAME,
        GATE_FILENAME,
        SUMMARY_FILENAME,
        REPORT_FILENAME,
        MANIFEST_FILENAME,
    }
)

_CANDIDATE_FIELDS: Final = tuple(field.name for field in fields(OracleCandidate))
_ACTION_VALUES: Final = tuple(action.value for action in ACTION_ORDER)
_PRIMARY_OUTCOMES: Final = ("P1", "P2", "A3_EXCLUSIVE")
_CONTRASTS: Final = (
    ("B400_MINUS_B100", RDX004Budget.B100, RDX004Budget.B400),
    ("B1600_MINUS_B100", RDX004Budget.B100, RDX004Budget.B1600),
    ("B1600_MINUS_B400", RDX004Budget.B400, RDX004Budget.B1600),
)


class RdxTrainingAnalysisError(RuntimeError):
    """Raised when RDX-007 must fail closed."""


@dataclass(frozen=True, slots=True)
class PrimaryOutcome:
    """One frozen rotation-by-seed capability outcome."""

    harmful_denominator: int
    p1_numerator: int
    p2_numerator: int
    a3_exclusive_numerator: int

    @property
    def p1(self) -> Fraction | None:
        return _optional_fraction(self.p1_numerator, self.harmful_denominator)

    @property
    def p2(self) -> Fraction | None:
        return _optional_fraction(self.p2_numerator, self.harmful_denominator)

    @property
    def a3_exclusive(self) -> Fraction | None:
        return _optional_fraction(self.a3_exclusive_numerator, self.harmful_denominator)


@dataclass(frozen=True, slots=True)
class _CorpusRun:
    budget: RDX004Budget
    evidence_status: str
    sequence: tuple[str, str, str, str]
    seed: int
    run_id: str
    path: Path
    artifact_manifest_sha256: str
    artifact_bundle_digest: str
    paired_b100_experiment_id: str
    paired_study1_experiment_id: str
    protocol_sha256: str
    preflight_bundle_digest: str
    execution_implementation_version: str
    execution_commit: str
    query_event_count: int
    scope_closure_count: int
    oracle_decision_count: int
    smoke: bool


@dataclass(frozen=True, slots=True)
class _AnalysisInputs:
    study4_evaluation_dir: Path
    preflight_dir: Path
    run_root: Path
    runs: tuple[_CorpusRun, ...]
    matrix_rows: tuple[dict[str, object], ...]
    contract: dict[str, Any]
    source_sentinels: dict[str, str]


@dataclass(frozen=True, slots=True)
class _RunEvidence:
    run: _CorpusRun
    oracle_records: tuple[dict[str, Any], ...]
    resources: dict[str, Any]
    memory: dict[str, Any]
    query: dict[str, Any]
    windows: tuple[dict[str, str], ...]
    holdouts: tuple[dict[str, str], ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RdxTrainingAnalysisError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RdxTrainingAnalysisError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RdxTrainingAnalysisError(f"JSON artifact must be an object: {path}")
    return value


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RdxTrainingAnalysisError(f"cannot read CSV artifact {path}: {exc}") from exc
    if "\r" in text or (text and not text.endswith("\n")):
        raise RdxTrainingAnalysisError(f"CSV artifact is not LF-canonical: {path}")
    if not text:
        return ()
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise RdxTrainingAnalysisError(f"CSV header is absent or duplicated: {path}")
    rows: list[dict[str, str]] = []
    try:
        for raw in reader:
            if None in raw or any(value is None for value in raw.values()):
                raise RdxTrainingAnalysisError(f"CSV row differs from header: {path}")
            rows.append({str(key): str(value) for key, value in raw.items()})
    except csv.Error as exc:
        raise RdxTrainingAnalysisError(f"invalid CSV artifact {path}: {exc}") from exc
    return tuple(rows)


def _csv_text(columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        extra = set(row).difference(columns)
        if extra:
            raise RdxTrainingAnalysisError(
                "analysis row contains unexpected columns: " + ", ".join(sorted(extra))
            )
        writer.writerow(
            {column: "" if row.get(column) is None else row.get(column, "") for column in columns}
        )
    return handle.getvalue()


def _tree_digests(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name != MANIFEST_FILENAME
    }


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise RdxTrainingAnalysisError(f"{name} must be an integer")
    try:
        result = int(str(value))
    except (TypeError, ValueError) as exc:
        raise RdxTrainingAnalysisError(f"{name} must be an integer") from exc
    return result


def _as_bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    raise RdxTrainingAnalysisError(f"{name} must be a boolean")


def _optional_fraction(numerator: int, denominator: int) -> Fraction | None:
    return None if denominator == 0 else Fraction(numerator, denominator)


def _fraction_text(value: Fraction | None) -> str:
    return "" if value is None else f"{value.numerator}/{value.denominator}"


def _fraction_float(value: Fraction | None) -> float | None:
    return None if value is None else float(value)


def _parse_fraction(value: object) -> Fraction | None:
    if value is None or value == "":
        return None
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction(str(value))
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise RdxTrainingAnalysisError(f"invalid exact fraction: {value!r}") from exc


def _mean_fraction(values: Sequence[Fraction]) -> Fraction:
    if not values:
        raise RdxTrainingAnalysisError("cannot average an empty exact sequence")
    return sum(values, Fraction(0, 1)) / len(values)


def _median_fraction(values: Sequence[Fraction]) -> Fraction:
    if not values:
        raise RdxTrainingAnalysisError("cannot take median of an empty exact sequence")
    return median(values)


def _typed_candidate(raw: Mapping[str, object]) -> OracleCandidate:
    try:
        return OracleCandidate.from_mapping({name: raw.get(name) for name in _CANDIDATE_FIELDS})
    except (TypeError, ValueError) as exc:
        raise RdxTrainingAnalysisError(
            f"Oracle candidate violates frozen semantics: {exc}"
        ) from exc


def _validated_candidates(record: Mapping[str, object]) -> dict[str, OracleCandidate]:
    raw_candidates = record.get("candidates")
    if not isinstance(raw_candidates, list):
        raise RdxTrainingAnalysisError("Oracle decision candidates must be a list")
    candidates: dict[str, OracleCandidate] = {}
    for raw in raw_candidates:
        if not isinstance(raw, Mapping):
            raise RdxTrainingAnalysisError("Oracle candidate must be a mapping")
        candidate = _typed_candidate(raw)
        if candidate.action.value in candidates:
            raise RdxTrainingAnalysisError("Oracle decision has a duplicate action")
        candidates[candidate.action.value] = candidate
    if tuple(candidates) != _ACTION_VALUES and set(candidates) != set(_ACTION_VALUES):
        raise RdxTrainingAnalysisError("eligible Oracle decision requires complete A0--A4")
    return candidates


def compute_primary_outcome(records: Sequence[Mapping[str, object]]) -> PrimaryOutcome:
    """Compute P1, P2, and A3-exclusive capability for one run."""

    denominator = 0
    p1 = 0
    p2 = 0
    a3_exclusive = 0
    for record in records:
        candidates = _validated_candidates(record)
        if str(record.get("current_evaluator_state")) != "HARMFUL":
            continue
        denominator += 1
        successful = {
            action for action, candidate in candidates.items() if candidate.confirmed_success
        }
        p1 += bool(successful)
        p2 += bool(
            successful.intersection({"A1_RECALIBRATE", "A2_HEAD_UPDATE", "A4_REPLAY_UPDATE"})
        )
        a3_exclusive += successful == {"A3_FULL_FINE_TUNE"}
    return PrimaryOutcome(
        harmful_denominator=denominator,
        p1_numerator=p1,
        p2_numerator=p2,
        a3_exclusive_numerator=a3_exclusive,
    )


def stratified_bootstrap_interval(
    rotation_differences: Mapping[str, Sequence[Fraction | float]],
) -> dict[str, float | int | str]:
    """Apply the frozen, independently reinitialised stratified paired bootstrap."""

    expected = {"-".join(rotation) for rotation in RDX004_ROTATIONS}
    if set(rotation_differences) != expected:
        raise RdxTrainingAnalysisError("bootstrap requires exactly the four frozen rotations")
    arrays: list[np.ndarray] = []
    for rotation in ("-".join(item) for item in RDX004_ROTATIONS):
        values = np.asarray([float(value) for value in rotation_differences[rotation]], dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise RdxTrainingAnalysisError(
                "bootstrap requires three finite seed differences/rotation"
            )
        arrays.append(values)
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    samples = np.empty((BOOTSTRAP_REPLICATES, 12), dtype=float)
    offset = 0
    for values in arrays:
        indexes = generator.integers(0, 3, size=(BOOTSTRAP_REPLICATES, 3))
        samples[:, offset : offset + 3] = values[indexes]
        offset += 3
    means = samples.mean(axis=1)
    medians = np.median(samples, axis=1)
    mean_bounds = np.percentile(means, (2.5, 97.5), method="linear")
    median_bounds = np.percentile(medians, (2.5, 97.5), method="linear")
    return {
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "generator": "NumPy PCG64",
        "percentile_method": "linear",
        "mean_lower": float(mean_bounds[0]),
        "mean_upper": float(mean_bounds[1]),
        "median_lower": float(median_bounds[0]),
        "median_upper": float(median_bounds[1]),
    }


def _manifest_identity(root: Path) -> tuple[str, str]:
    manifest_path = root / MANIFEST_FILENAME
    manifest = _load_json(manifest_path)
    bundle = manifest.get("bundle_digest")
    if not isinstance(bundle, str) or len(bundle) != 64:
        raise RdxTrainingAnalysisError(f"invalid artifact bundle digest: {manifest_path}")
    return _file_sha256(manifest_path), bundle


def _matrix_row(run: _CorpusRun) -> dict[str, object]:
    return {
        "budget": run.budget.value,
        "evidence_status": run.evidence_status,
        "sequence": "-".join(run.sequence),
        "seed": run.seed,
        "run_id": run.run_id,
        "run_path": str(run.path),
        "artifact_manifest_sha256": run.artifact_manifest_sha256,
        "artifact_bundle_digest": run.artifact_bundle_digest,
        "paired_b100_experiment_id": run.paired_b100_experiment_id,
        "paired_study1_experiment_id": run.paired_study1_experiment_id,
        "protocol_sha256": run.protocol_sha256,
        "preflight_bundle_digest": run.preflight_bundle_digest,
        "execution_implementation_version": run.execution_implementation_version,
        "execution_commit": run.execution_commit,
        "query_event_count": run.query_event_count,
        "scope_closure_count": run.scope_closure_count,
        "oracle_decision_count": run.oracle_decision_count,
        "smoke": run.smoke,
        "validation_status": "VALID",
    }


def _preflight_contract(preflight_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        preflight_artifacts.validate_rdx004_training_evidence_preflight(preflight_dir)
    except Exception as exc:
        raise RdxTrainingAnalysisError(
            f"RDX-007 NO-GO: frozen RDX-005 preflight validation failed: {exc}"
        ) from exc
    manifest = _load_json(preflight_dir / preflight_artifacts.MANIFEST_FILENAME)
    if manifest.get("bundle_digest") != RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST:
        raise RdxTrainingAnalysisError("RDX-007 NO-GO: preflight bundle digest differs")
    contract = _load_json(preflight_dir / preflight_artifacts.CONTRACT_FILENAME)
    return contract, manifest


def _validate_b100_sources(
    study4_evaluation_dir: Path,
    preflight_contract: Mapping[str, Any],
) -> tuple[_CorpusRun, ...]:
    try:
        study4_artifacts.validate_study4_evaluation(study4_evaluation_dir)
    except Exception as exc:
        raise RdxTrainingAnalysisError(
            f"RDX-007 NO-GO: canonical Study-4 evaluation validation failed: {exc}"
        ) from exc
    evaluation = _load_json(study4_evaluation_dir / "evaluation_contract.json")
    source_runs = evaluation.get("source_runs")
    if not isinstance(source_runs, list):
        raise RdxTrainingAnalysisError("RDX-007 NO-GO: Study-4 source roster is absent")
    oracle_sources = {
        str(item.get("experiment_id")): item
        for item in source_runs
        if isinstance(item, Mapping)
        and str(item.get("experiment_id", "")).startswith("E4_OFFLINE_ORACLE_")
    }
    preflight_sources = preflight_contract.get("canonical_b100_sources")
    if not isinstance(preflight_sources, list):
        raise RdxTrainingAnalysisError("RDX-007 NO-GO: preflight B100 roster is absent")
    preflight_by_id = {
        str(item.get("experiment_id")): item
        for item in preflight_sources
        if isinstance(item, Mapping)
    }
    expected_ids = {
        f"E4_OFFLINE_ORACLE_{'-'.join(rotation)}_s{seed}"
        for rotation in RDX004_ROTATIONS
        for seed in RDX004_SEEDS
    }
    if set(oracle_sources) != expected_ids or set(preflight_by_id) != expected_ids:
        raise RdxTrainingAnalysisError("RDX-007 NO-GO: canonical B100 roster is not exact")

    runs: list[_CorpusRun] = []
    for rotation in RDX004_ROTATIONS:
        for seed in RDX004_SEEDS:
            run_id = f"E4_OFFLINE_ORACLE_{'-'.join(rotation)}_s{seed}"
            evaluation_row = oracle_sources[run_id]
            preflight_row = preflight_by_id[run_id]
            path = Path(str(evaluation_row.get("path", ""))).resolve()
            if path != Path(str(preflight_row.get("source_run_path", ""))).resolve():
                raise RdxTrainingAnalysisError(
                    f"RDX-007 NO-GO: B100 source path differs for {run_id}"
                )
            try:
                validated = study4_artifacts.validate_study4_run(path, allow_smoke=False)
            except Exception as exc:
                raise RdxTrainingAnalysisError(
                    f"RDX-007 NO-GO: B100 source validation failed for {run_id}: {exc}"
                ) from exc
            manifest_sha, bundle_digest = _manifest_identity(path)
            expected_manifest_sha = str(evaluation_row.get("artifact_manifest_sha256", ""))
            if (
                validated.experiment_id != run_id
                or validated.method is not Study4Method.OFFLINE_ORACLE
                or tuple(validated.sequence) != rotation
                or validated.seed != seed
                or validated.smoke
                or manifest_sha != expected_manifest_sha
                or manifest_sha != str(preflight_row.get("source_artifact_manifest_sha256", ""))
                or bundle_digest != str(preflight_row.get("source_manifest_bundle_digest", ""))
                or validated.contract_digest
                != str(preflight_row.get("scientific_contract_digest", ""))
                or validated.source_checkpoint_sha256
                != str(preflight_row.get("source_checkpoint_sha256", ""))
            ):
                raise RdxTrainingAnalysisError(
                    f"RDX-007 NO-GO: B100 identity/provenance differs for {run_id}"
                )
            oracle = _load_json(path / study4_artifacts.ORACLE_FILENAME)
            records = oracle.get("records")
            query = _load_json(path / study4_artifacts.QUERY_FILENAME)
            scopes = query.get("scopes")
            if not isinstance(records, list) or not isinstance(scopes, list):
                raise RdxTrainingAnalysisError(
                    f"RDX-007 NO-GO: B100 Oracle/query evidence is malformed for {run_id}"
                )
            query_count = sum(
                len(scope.get("query_events", [])) for scope in scopes if isinstance(scope, Mapping)
            )
            if query_count != 12 or len(scopes) != 3:
                raise RdxTrainingAnalysisError(
                    f"RDX-007 NO-GO: B100 query/closure count differs for {run_id}"
                )
            runs.append(
                _CorpusRun(
                    budget=RDX004Budget.B100,
                    evidence_status=rdx004_treatment(RDX004Budget.B100).evidence_status,
                    sequence=rotation,
                    seed=seed,
                    run_id=run_id,
                    path=path,
                    artifact_manifest_sha256=manifest_sha,
                    artifact_bundle_digest=bundle_digest,
                    paired_b100_experiment_id=run_id,
                    paired_study1_experiment_id=str(preflight_row.get("study1_experiment_id", "")),
                    protocol_sha256=RDX004_PROTOCOL_SHA256,
                    preflight_bundle_digest=RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST,
                    execution_implementation_version=study4_artifacts.STUDY4_RUN_ARTIFACT_VERSION,
                    execution_commit="HISTORICAL_CANONICAL_COMPARATOR",
                    query_event_count=query_count,
                    scope_closure_count=len(scopes),
                    oracle_decision_count=len(records),
                    smoke=False,
                )
            )
    return tuple(runs)


def _validate_prospective_sources(
    run_root: Path,
    preflight_dir: Path,
) -> tuple[_CorpusRun, ...]:
    if not run_root.is_dir():
        raise RdxTrainingAnalysisError("RDX-007 NO-GO: prospective run root is absent")
    identities = tuple(
        RDX004RunIdentity(budget=budget, rotation=rotation, seed=seed)
        for budget in RDX004_PROSPECTIVE_BUDGETS
        for rotation in RDX004_ROTATIONS
        for seed in RDX004_SEEDS
    )
    expected_names = {identity.run_id for identity in identities}
    entries = tuple(run_root.iterdir())
    observed_names = {entry.name for entry in entries}
    if (
        observed_names != expected_names
        or len(entries) != len(expected_names)
        or any(not entry.is_dir() for entry in entries)
    ):
        missing = sorted(expected_names.difference(observed_names))
        extra = sorted(observed_names.difference(expected_names))
        raise RdxTrainingAnalysisError(
            f"RDX-007 NO-GO: prospective roster differs; missing={missing}, extra={extra}"
        )

    paths = tuple((run_root / identity.run_id).resolve() for identity in identities)
    try:
        validated_runs = validate_rdx004_runs(
            paths,
            expected_smoke=False,
            preflight_dir=preflight_dir,
        )
    except Exception as exc:
        raise RdxTrainingAnalysisError(
            f"RDX-007 NO-GO: prospective batch validation failed: {exc}"
        ) from exc
    if len(validated_runs) != len(identities):
        raise RdxTrainingAnalysisError(
            "RDX-007 NO-GO: prospective batch validator returned an incomplete roster"
        )

    runs: list[_CorpusRun] = []
    for identity, path, validated in zip(identities, paths, validated_runs, strict=True):
        contract = _load_json(path / EXECUTION_CONTRACT_FILENAME)
        code = contract.get("code")
        execution_config = contract.get("execution_config")
        protocol = contract.get("protocol")
        if not all(isinstance(item, Mapping) for item in (code, execution_config, protocol)):
            raise RdxTrainingAnalysisError(
                f"RDX-007 NO-GO: execution provenance is malformed for {identity.run_id}"
            )
        assert isinstance(code, Mapping)
        assert isinstance(execution_config, Mapping)
        assert isinstance(protocol, Mapping)
        manifest_sha, bundle_digest = _manifest_identity(path)
        if (
            validated.run_id != identity.run_id
            or validated.expected_confirmatory_run_id != identity.run_id
            or validated.budget is not identity.budget
            or tuple(validated.rotation) != identity.rotation
            or validated.seed != identity.seed
            or validated.smoke
            or validated.paired_b100_experiment_id != identity.paired_b100_experiment_id
            or validated.preflight_bundle_digest != RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST
            or validated.artifact_bundle_digest != bundle_digest
            or validated.query_event_count != 12
            or validated.scope_closure_count != 3
            or protocol.get("version") != RDX004_PROTOCOL_VERSION
            or protocol.get("sha256") != RDX004_PROTOCOL_SHA256
            or execution_config.get("implementation_version")
            != RDX006_EXECUTION_IMPLEMENTATION_VERSION
            or code.get("commit") != RDX006_EXECUTION_COMMIT
            or code.get("working_tree_dirty") is not False
        ):
            raise RdxTrainingAnalysisError(
                f"RDX-007 NO-GO: prospective scientific identity differs for {identity.run_id}"
            )
        runs.append(
            _CorpusRun(
                budget=identity.budget,
                evidence_status=rdx004_treatment(identity.budget).evidence_status,
                sequence=identity.rotation,
                seed=identity.seed,
                run_id=identity.run_id,
                path=path,
                artifact_manifest_sha256=manifest_sha,
                artifact_bundle_digest=bundle_digest,
                paired_b100_experiment_id=validated.paired_b100_experiment_id,
                paired_study1_experiment_id=validated.paired_study1_experiment_id,
                protocol_sha256=RDX004_PROTOCOL_SHA256,
                preflight_bundle_digest=validated.preflight_bundle_digest,
                execution_implementation_version=RDX006_EXECUTION_IMPLEMENTATION_VERSION,
                execution_commit=RDX006_EXECUTION_COMMIT,
                query_event_count=validated.query_event_count,
                scope_closure_count=validated.scope_closure_count,
                oracle_decision_count=validated.oracle_decision_count,
                smoke=False,
            )
        )
    return tuple(runs)


def _source_sentinels(
    study4_evaluation_dir: Path,
    preflight_dir: Path,
    runs: Sequence[_CorpusRun],
) -> dict[str, str]:
    roots = {study4_evaluation_dir.resolve(), preflight_dir.resolve(), *(run.path for run in runs)}
    paths = {path.resolve() for root in roots for path in root.rglob("*") if path.is_file()}
    return {str(path): _file_sha256(path) for path in sorted(paths)}


def _build_analysis_contract(
    *,
    study4_evaluation_dir: Path,
    preflight_dir: Path,
    run_root: Path,
    runs: Sequence[_CorpusRun],
    preflight_manifest: Mapping[str, Any],
    source_sentinels: Mapping[str, str],
) -> dict[str, Any]:
    source_rows = [
        {
            "budget": run.budget.value,
            "evidence_status": run.evidence_status,
            "sequence": "-".join(run.sequence),
            "seed": run.seed,
            "run_id": run.run_id,
            "path": str(run.path),
            "artifact_manifest_sha256": run.artifact_manifest_sha256,
            "artifact_bundle_digest": run.artifact_bundle_digest,
            "paired_b100_experiment_id": run.paired_b100_experiment_id,
            "paired_study1_experiment_id": run.paired_study1_experiment_id,
            "execution_implementation_version": run.execution_implementation_version,
            "execution_commit": run.execution_commit,
        }
        for run in runs
    ]
    return {
        "version": RDX007_ANALYSIS_VERSION,
        "status": "FROZEN_BEFORE_OUTCOME_DERIVATION",
        "input_paths": {
            "study4_evaluation_dir": str(study4_evaluation_dir),
            "preflight_dir": str(preflight_dir),
            "prospective_run_root": str(run_root),
        },
        "protocol": {
            "version": RDX004_PROTOCOL_VERSION,
            "sha256": RDX004_PROTOCOL_SHA256,
            "preflight_bundle_digest": RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST,
            "observed_preflight_bundle_digest": preflight_manifest.get("bundle_digest"),
            "prospective_execution_implementation": RDX006_EXECUTION_IMPLEMENTATION_VERSION,
            "prospective_execution_commit": RDX006_EXECUTION_COMMIT,
        },
        "corpus": {
            "required_counts": {"B100": 12, "B400": 12, "B1600": 12},
            "analysis_unit": "rotation x seed",
            "nested_observations_are_not_replicates": [
                "flows",
                "windows",
                "decisions",
                "Oracle candidates",
            ],
            "sources": source_rows,
            "validated_before_outcome_loading": True,
            "source_artifact_sentinels": dict(source_sentinels),
        },
        "primary_outcomes": {
            "denominator": "eligible Oracle decisions with current evaluator state HARMFUL",
            "P1": "any confirmed-success candidate among A0,A1,A2,A3,A4",
            "P2": "any confirmed-success candidate among A1,A2,A4 only",
            "A3_EXCLUSIVE": ("A3 confirmed success with no confirmed success among A0,A1,A2,A4"),
            "confirmed_success": (
                "feasible and execution succeeded and audit admissible and first fresh "
                "same-domain successor SAFE"
            ),
            "uncertain_is_success": False,
            "zero_denominator": "UNAVAILABLE",
            "selected_action_success_rate": False,
        },
        "pairing": {
            "unit": "rotation x seed",
            "contrasts": [item[0] for item in _CONTRASTS],
            "decision_level_pairing": False,
        },
        "bootstrap": {
            "payload": BOOTSTRAP_PAYLOAD,
            "payload_sha256": BOOTSTRAP_SHA256,
            "outcomes": ["P1", "P2"],
            "seed": BOOTSTRAP_SEED,
            "generator": "NumPy PCG64",
            "replicates": BOOTSTRAP_REPLICATES,
            "within_rotation_sample_size": 3,
            "rotations": 4,
            "values_per_replicate": 12,
            "percentiles": [2.5, 97.5],
            "percentile_method": "linear",
            "generator_reinitialized_per_contrast_outcome": True,
            "role": "descriptive protocol-defined uncertainty interval; not significance test",
            "numpy_version": np.__version__,
        },
        "decision_gate": {
            "material_improvement_exact": "1/50",
            "material_improvement_percentage_points": 2.0,
            "rotation_rule": "strictly positive median in at least 3 of 4 rotations",
            "precedence": [
                "NOT_ASSESSED_INCOMPLETE",
                "CORE_FOLLOWUP_JUSTIFIED",
                "ACTION_SPACE_FOLLOWUP_JUSTIFIED",
                "TRAINING_EVIDENCE_INCREASE_NOT_MATERIALLY_RECOVERABILITY_EXPANDING",
                "NO_FROZEN_FOLLOWUP_GATE_REACHED_MIXED_OR_ORDER_DEPENDENT",
            ],
            "all_qualifying_budgets_reported": True,
        },
        "mechanistic_rules": {
            "feasibility_denominator": "all eligible HARMFUL decisions",
            "execution_denominator": "feasible candidates",
            "explicit_execution_failure": "feasible and execution_succeeded is false",
            "audit_denominator": "feasible successfully executed candidates with defined audit",
            "successor_denominator": "all eligible HARMFUL decisions",
            "stages_kept_separate": True,
            "zero_execution_failures_implies_recovery": False,
            "case_c_is_descriptive_only": True,
        },
        "evidence_dose": {
            "maximum_optimizer_eligible_target_rows": {
                "B100": 80,
                "B400": 380,
                "B1600": 1580,
            },
            "audit_rows_per_later_domain": 20,
            "historical_replay_capacity_per_domain": 400,
            "b1600_later_stage_note": (
                "increased target training evidence under a fixed 400-row historical replay "
                "constraint; not a pure raw-label contrast"
            ),
        },
        "interpretation_templates": {
            "CASE_A": "material P2 improvement",
            "CASE_B": "material P1 but not P2 improvement with A3-exclusive concentration",
            "CASE_C": "descriptive audit-bottleneck pattern; cannot authorize follow-up",
            "CASE_D": "neither P1 nor P2 materially improves",
            "MIXED": "mixed or order-dependent without a frozen follow-up gate",
        },
        "evidence_timing": {
            "B100": "HISTORICAL_CANONICAL_COMPARATOR",
            "B400": "PROSPECTIVE_NEW_TREATMENT",
            "B1600": "PROSPECTIVE_NEW_TREATMENT",
            "three_arm_evidence_is_wholly_concurrent_prospective": False,
        },
        "output_contract": {
            "files": sorted(ALL_OUTPUT_FILES),
            "canonical_json": "sorted keys, two-space indent, LF, no NaN",
            "canonical_csv": "fixed columns, LF, blank nulls",
            "write_once": True,
            "source_backed_recomputation": True,
        },
        "scientific_boundaries": {
            "experiments_run": False,
            "source_artifacts_mutated": False,
            "rdx004_protocol_modified": False,
            "danids_2_0_evidence_freeze_modified": False,
            "new_experiment_authorized": False,
        },
    }


def _validate_corpus_inputs(
    study4_evaluation_dir: str | Path,
    preflight_dir: str | Path,
    run_root: str | Path,
) -> _AnalysisInputs:
    study4_root = Path(study4_evaluation_dir).resolve()
    preflight_root = Path(preflight_dir).resolve()
    prospective_root = Path(run_root).resolve()
    preflight_contract, preflight_manifest = _preflight_contract(preflight_root)
    b100 = _validate_b100_sources(study4_root, preflight_contract)
    prospective = _validate_prospective_sources(prospective_root, preflight_root)
    budget_order = {budget: index for index, budget in enumerate(RDX004_ALL_BUDGETS)}
    rotation_order = {rotation: index for index, rotation in enumerate(RDX004_ROTATIONS)}
    runs = tuple(
        sorted(
            (*b100, *prospective),
            key=lambda item: (
                budget_order[item.budget],
                rotation_order[item.sequence],
                item.seed,
            ),
        )
    )
    if len(runs) != 36:
        raise RdxTrainingAnalysisError("RDX-007 NO-GO: validated corpus is not 36 runs")
    by_key = {(run.budget, run.sequence, run.seed): run for run in runs}
    if len(by_key) != 36:
        raise RdxTrainingAnalysisError("RDX-007 NO-GO: duplicate treatment unit")
    for rotation in RDX004_ROTATIONS:
        for seed in RDX004_SEEDS:
            trio = [by_key.get((budget, rotation, seed)) for budget in RDX004_ALL_BUDGETS]
            if any(item is None for item in trio):
                raise RdxTrainingAnalysisError("RDX-007 NO-GO: incomplete paired treatment trio")
            b100_run, b400_run, b1600_run = trio
            assert b100_run is not None and b400_run is not None and b1600_run is not None
            if (
                b400_run.paired_b100_experiment_id != b100_run.run_id
                or b1600_run.paired_b100_experiment_id != b100_run.run_id
                or len(
                    {
                        b100_run.paired_study1_experiment_id,
                        b400_run.paired_study1_experiment_id,
                        b1600_run.paired_study1_experiment_id,
                    }
                )
                != 1
            ):
                raise RdxTrainingAnalysisError(
                    "RDX-007 NO-GO: source-state or B100 pairing differs within a trio"
                )
    matrix_rows = tuple(_matrix_row(run) for run in runs)
    sentinels = _source_sentinels(study4_root, preflight_root, runs)
    contract = _build_analysis_contract(
        study4_evaluation_dir=study4_root,
        preflight_dir=preflight_root,
        run_root=prospective_root,
        runs=runs,
        preflight_manifest=preflight_manifest,
        source_sentinels=sentinels,
    )
    return _AnalysisInputs(
        study4_evaluation_dir=study4_root,
        preflight_dir=preflight_root,
        run_root=prospective_root,
        runs=runs,
        matrix_rows=matrix_rows,
        contract=contract,
        source_sentinels=sentinels,
    )


def _load_run_evidence(run: _CorpusRun) -> _RunEvidence:
    historical = run.budget is RDX004Budget.B100
    oracle_name = (
        study4_artifacts.ORACLE_FILENAME if historical else ORACLE_COUNTERFACTUALS_FILENAME
    )
    resource_name = study4_artifacts.RESOURCE_FILENAME if historical else RESOURCE_FILENAME
    memory_name = study4_artifacts.MEMORY_FILENAME if historical else MEMORY_FILENAME
    query_name = study4_artifacts.QUERY_FILENAME if historical else QUERY_EVIDENCE_FILENAME
    window_name = study4_artifacts.WINDOW_FILENAME if historical else WINDOW_FILENAME
    holdout_name = study4_artifacts.HOLDOUT_FILENAME if historical else HOLDOUT_FILENAME
    oracle = _load_json(run.path / oracle_name)
    records_raw = oracle.get("records")
    if not isinstance(records_raw, list) or not all(
        isinstance(item, Mapping) for item in records_raw
    ):
        raise RdxTrainingAnalysisError(f"Oracle evidence is malformed for {run.run_id}")
    records = tuple(dict(item) for item in records_raw if isinstance(item, Mapping))
    if len(records) != run.oracle_decision_count:
        raise RdxTrainingAnalysisError(f"Oracle record count differs for {run.run_id}")
    later_domains = set(run.sequence[1:])
    for record in records:
        stage = _as_int(record.get("stage"), "Oracle stage")
        current_domain = str(record.get("current_domain", ""))
        if stage not in {1, 2, 3} or current_domain not in later_domains:
            raise RdxTrainingAnalysisError(
                f"Oracle evidence is outside matched later-domain scope for {run.run_id}"
            )
    return _RunEvidence(
        run=run,
        oracle_records=records,
        resources=_load_json(run.path / resource_name),
        memory=_load_json(run.path / memory_name),
        query=_load_json(run.path / query_name),
        windows=_read_csv(run.path / window_name),
        holdouts=_read_csv(run.path / holdout_name),
    )


def _assert_sources_unchanged(inputs: _AnalysisInputs) -> None:
    current = _source_sentinels(
        inputs.study4_evaluation_dir,
        inputs.preflight_dir,
        inputs.runs,
    )
    if current != inputs.source_sentinels:
        raise RdxTrainingAnalysisError("source artifact tree changed during RDX-007")


def _primary_rows(evidence: Sequence[_RunEvidence]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for item in evidence:
        outcome = compute_primary_outcome(item.oracle_records)
        rows.append(
            {
                "budget": item.run.budget.value,
                "evidence_status": item.run.evidence_status,
                "sequence": "-".join(item.run.sequence),
                "seed": item.run.seed,
                "run_id": item.run.run_id,
                "harmful_denominator": outcome.harmful_denominator,
                "p1_numerator": outcome.p1_numerator,
                "p1": _fraction_float(outcome.p1),
                "p1_exact": _fraction_text(outcome.p1),
                "p2_numerator": outcome.p2_numerator,
                "p2": _fraction_float(outcome.p2),
                "p2_exact": _fraction_text(outcome.p2),
                "a3_exclusive_numerator": outcome.a3_exclusive_numerator,
                "a3_exclusive": _fraction_float(outcome.a3_exclusive),
                "a3_exclusive_exact": _fraction_text(outcome.a3_exclusive),
                "availability": (
                    "AVAILABLE" if outcome.harmful_denominator else "UNAVAILABLE_NO_HARMFUL"
                ),
                "analysis_unit": "rotation x seed",
            }
        )
    return tuple(rows)


def _outcome_fraction(row: Mapping[str, object], outcome: str) -> Fraction | None:
    column = {
        "P1": "p1_exact",
        "P2": "p2_exact",
        "A3_EXCLUSIVE": "a3_exclusive_exact",
    }.get(outcome)
    if column is None:
        raise RdxTrainingAnalysisError(f"unsupported primary outcome {outcome}")
    return _parse_fraction(row.get(column))


def _treatment_summaries(
    primary: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for budget in RDX004_ALL_BUDGETS:
        selected = [row for row in primary if row.get("budget") == budget.value]
        if len(selected) != 12:
            raise RdxTrainingAnalysisError("treatment summary requires 12 units per budget")
        for outcome_name in _PRIMARY_OUTCOMES:
            values = [_outcome_fraction(row, outcome_name) for row in selected]
            available = [value for value in values if value is not None]
            complete = len(available) == len(selected)
            mean_value = _mean_fraction(available) if complete else None
            median_value = _median_fraction(available) if complete else None
            rows.append(
                {
                    "budget": budget.value,
                    "outcome": outcome_name,
                    "n_units": len(selected),
                    "unavailable_units": len(selected) - len(available),
                    "eligible_harmful_total": sum(
                        _as_int(row.get("harmful_denominator"), "harmful denominator")
                        for row in selected
                    ),
                    "mean_unit_rate": _fraction_float(mean_value),
                    "mean_unit_rate_exact": _fraction_text(mean_value),
                    "median_unit_rate": _fraction_float(median_value),
                    "median_unit_rate_exact": _fraction_text(median_value),
                    "minimum_unit_rate": (_fraction_float(min(available)) if complete else None),
                    "maximum_unit_rate": (_fraction_float(max(available)) if complete else None),
                    "aggregation_unit": "unweighted rotation x seed",
                }
            )
    return tuple(rows)


def _rotation_summaries(
    primary: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for budget in RDX004_ALL_BUDGETS:
        for outcome_name in _PRIMARY_OUTCOMES:
            for rotation in RDX004_ROTATIONS:
                sequence = "-".join(rotation)
                selected = [
                    row
                    for row in primary
                    if row.get("budget") == budget.value and row.get("sequence") == sequence
                ]
                if len(selected) != 3:
                    raise RdxTrainingAnalysisError(
                        "rotation summary requires three seeds per treatment"
                    )
                values = [_outcome_fraction(row, outcome_name) for row in selected]
                available = [value for value in values if value is not None]
                median_value = _median_fraction(available) if len(available) == 3 else None
                rows.append(
                    {
                        "budget": budget.value,
                        "outcome": outcome_name,
                        "sequence": sequence,
                        "n_seed_units": len(available),
                        "median_unit_rate": _fraction_float(median_value),
                        "median_unit_rate_exact": _fraction_text(median_value),
                    }
                )
    return tuple(rows)


def _paired_rows(
    primary: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    indexed = {
        (str(row.get("budget")), str(row.get("sequence")), _as_int(row.get("seed"), "seed")): row
        for row in primary
    }
    if len(indexed) != 36:
        raise RdxTrainingAnalysisError("paired analysis requires 36 distinct treatment units")
    rows: list[dict[str, object]] = []
    for contrast, baseline, comparison in _CONTRASTS:
        for outcome_name in _PRIMARY_OUTCOMES:
            for rotation in RDX004_ROTATIONS:
                sequence = "-".join(rotation)
                for seed in RDX004_SEEDS:
                    base = _outcome_fraction(
                        indexed[(baseline.value, sequence, seed)], outcome_name
                    )
                    new = _outcome_fraction(
                        indexed[(comparison.value, sequence, seed)], outcome_name
                    )
                    difference = None if base is None or new is None else new - base
                    rows.append(
                        {
                            "contrast": contrast,
                            "outcome": outcome_name,
                            "sequence": sequence,
                            "seed": seed,
                            "baseline_budget": baseline.value,
                            "comparison_budget": comparison.value,
                            "baseline_value": _fraction_float(base),
                            "baseline_value_exact": _fraction_text(base),
                            "comparison_value": _fraction_float(new),
                            "comparison_value_exact": _fraction_text(new),
                            "paired_difference": _fraction_float(difference),
                            "paired_difference_exact": _fraction_text(difference),
                            "direction": (
                                "UNAVAILABLE"
                                if difference is None
                                else "POSITIVE"
                                if difference > 0
                                else "NEGATIVE"
                                if difference < 0
                                else "ZERO"
                            ),
                        }
                    )
    return tuple(rows)


def _bootstrap_rows(
    paired: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for contrast, _, _ in _CONTRASTS:
        for outcome_name in _PRIMARY_OUTCOMES:
            selected = [
                row
                for row in paired
                if row.get("contrast") == contrast and row.get("outcome") == outcome_name
            ]
            if len(selected) != 12:
                raise RdxTrainingAnalysisError("bootstrap requires 12 paired units")
            values = [_parse_fraction(row.get("paired_difference_exact")) for row in selected]
            available = [value for value in values if value is not None]
            rotation_values: dict[str, list[Fraction]] = {}
            rotation_medians: dict[str, Fraction | None] = {}
            for rotation in RDX004_ROTATIONS:
                sequence = "-".join(rotation)
                group = [
                    _parse_fraction(row.get("paired_difference_exact"))
                    for row in selected
                    if row.get("sequence") == sequence
                ]
                finite = [value for value in group if value is not None]
                rotation_values[sequence] = finite
                rotation_medians[sequence] = _median_fraction(finite) if len(finite) == 3 else None
            complete = len(available) == 12 and all(
                len(value) == 3 for value in rotation_values.values()
            )
            bootstrap_registered = outcome_name in {"P1", "P2"}
            interval = (
                stratified_bootstrap_interval(rotation_values)
                if complete and bootstrap_registered
                else {
                    "mean_lower": None,
                    "mean_upper": None,
                    "median_lower": None,
                    "median_upper": None,
                }
            )
            mean_value = _mean_fraction(available) if complete else None
            median_value = _median_fraction(available) if complete else None
            rows.append(
                {
                    "contrast": contrast,
                    "outcome": outcome_name,
                    "n_units": len(available),
                    "mean_difference": _fraction_float(mean_value),
                    "mean_difference_exact": _fraction_text(mean_value),
                    "median_difference": _fraction_float(median_value),
                    "median_difference_exact": _fraction_text(median_value),
                    "mean_ci_lower": interval["mean_lower"],
                    "mean_ci_upper": interval["mean_upper"],
                    "median_ci_lower": interval["median_lower"],
                    "median_ci_upper": interval["median_upper"],
                    "positive_unit_count": sum(value > 0 for value in available),
                    "zero_unit_count": sum(value == 0 for value in available),
                    "negative_unit_count": sum(value < 0 for value in available),
                    "positive_rotation_medians": sum(
                        value is not None and value > 0 for value in rotation_medians.values()
                    ),
                    "rotation_medians_json": json.dumps(
                        {key: _fraction_text(value) for key, value in rotation_medians.items()},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "bootstrap_replicates": (BOOTSTRAP_REPLICATES if bootstrap_registered else 0),
                    "bootstrap_seed": BOOTSTRAP_SEED if bootstrap_registered else None,
                    "generator": "NumPy PCG64" if bootstrap_registered else "NOT_APPLICABLE",
                    "percentile_method": "linear" if bootstrap_registered else "NOT_APPLICABLE",
                    "interval_role": (
                        "descriptive protocol-defined uncertainty interval; not significance test"
                        if bootstrap_registered
                        else (
                            "deterministic paired summary only; no A3-exclusive interval registered"
                        )
                    ),
                }
            )
    return tuple(rows)


def evaluate_decision_gate(
    primary_rows: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Apply the prospectively frozen RDX-004 gate with exact fractions."""

    indexed = {
        (str(row.get("budget")), str(row.get("sequence")), _as_int(row.get("seed"), "seed")): row
        for row in primary_rows
    }
    expected = {
        (budget.value, "-".join(rotation), seed)
        for budget in RDX004_ALL_BUDGETS
        for rotation in RDX004_ROTATIONS
        for seed in RDX004_SEEDS
    }
    assessments: dict[str, dict[str, Any]] = {}
    incomplete = set(indexed) != expected
    for budget in RDX004_PROSPECTIVE_BUDGETS:
        budget_result: dict[str, Any] = {}
        for outcome_name in _PRIMARY_OUTCOMES:
            differences: list[Fraction] = []
            rotation_medians: dict[str, Fraction | None] = {}
            for rotation in RDX004_ROTATIONS:
                sequence = "-".join(rotation)
                rotation_differences: list[Fraction] = []
                for seed in RDX004_SEEDS:
                    baseline_row = indexed.get((RDX004Budget.B100.value, sequence, seed))
                    comparison_row = indexed.get((budget.value, sequence, seed))
                    baseline = (
                        None
                        if baseline_row is None
                        else _outcome_fraction(baseline_row, outcome_name)
                    )
                    comparison = (
                        None
                        if comparison_row is None
                        else _outcome_fraction(comparison_row, outcome_name)
                    )
                    if baseline is None or comparison is None:
                        incomplete = True
                        continue
                    value = comparison - baseline
                    differences.append(value)
                    rotation_differences.append(value)
                rotation_medians[sequence] = (
                    _median_fraction(rotation_differences)
                    if len(rotation_differences) == 3
                    else None
                )
            overall_median = _median_fraction(differences) if len(differences) == 12 else None
            positive_rotations = sum(
                value is not None and value > 0 for value in rotation_medians.values()
            )
            qualifies = (
                overall_median is not None
                and overall_median >= MATERIAL_IMPROVEMENT
                and positive_rotations >= 3
            )
            budget_result[outcome_name] = {
                "median_paired_improvement": _fraction_float(overall_median),
                "median_paired_improvement_exact": _fraction_text(overall_median),
                "positive_rotation_medians": positive_rotations,
                "rotation_medians": {
                    key: _fraction_text(value) for key, value in rotation_medians.items()
                },
                "material_two_part_rule_satisfied": qualifies,
            }
        assessments[budget.value] = budget_result

    core = [
        budget.value
        for budget in RDX004_PROSPECTIVE_BUDGETS
        if assessments[budget.value]["P2"]["material_two_part_rule_satisfied"]
    ]
    action_space = [
        budget.value
        for budget in RDX004_PROSPECTIVE_BUDGETS
        if assessments[budget.value]["P1"]["material_two_part_rule_satisfied"]
        and not assessments[budget.value]["P2"]["material_two_part_rule_satisfied"]
        and assessments[budget.value]["A3_EXCLUSIVE"]["material_two_part_rule_satisfied"]
    ]
    no_material = all(
        (
            _parse_fraction(assessments[budget.value][outcome]["median_paired_improvement_exact"])
            or Fraction(0)
        )
        < MATERIAL_IMPROVEMENT
        for budget in RDX004_PROSPECTIVE_BUDGETS
        for outcome in ("P1", "P2")
    )
    if incomplete:
        status = "NOT_ASSESSED_INCOMPLETE"
        qualifying: list[str] = []
        interpretation_case = "NOT_ASSESSED"
    elif core:
        status = "CORE_FOLLOWUP_JUSTIFIED"
        qualifying = core
        interpretation_case = "CASE_A"
    elif action_space:
        status = "ACTION_SPACE_FOLLOWUP_JUSTIFIED"
        qualifying = action_space
        interpretation_case = "CASE_B"
    elif no_material:
        status = "TRAINING_EVIDENCE_INCREASE_NOT_MATERIALLY_RECOVERABILITY_EXPANDING"
        qualifying = []
        interpretation_case = "CASE_D"
    else:
        status = "NO_FROZEN_FOLLOWUP_GATE_REACHED_MIXED_OR_ORDER_DEPENDENT"
        qualifying = []
        interpretation_case = "MIXED"
    return {
        "version": RDX007_ANALYSIS_VERSION,
        "status": status,
        "qualifying_budgets": qualifying,
        "material_threshold_exact": "1/50",
        "material_threshold_percentage_points": 2.0,
        "rotation_rule": "strictly positive median in at least 3 of 4 rotations",
        "assessments": assessments,
        "interpretation_case": interpretation_case,
        "case_c_can_authorize_followup": False,
        "new_experiment_authorized": False,
    }


def _mechanism_rows(evidence: Sequence[_RunEvidence]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    defined_audit_states = {"SAFE", "UNCERTAIN", "HARMFUL"}
    successor_states = ("SAFE", "UNCERTAIN", "HARMFUL")
    for item in evidence:
        harmful = [
            record
            for record in item.oracle_records
            if str(record.get("current_evaluator_state")) == "HARMFUL"
        ]
        selected_counts = {action: 0 for action in _ACTION_VALUES}
        for record in harmful:
            selected_action = str(record.get("selected_action", ""))
            if selected_action not in selected_counts:
                raise RdxTrainingAnalysisError(
                    f"Oracle selected action is invalid for {item.run.run_id}"
                )
            selected_counts[selected_action] += 1
        by_action: dict[str, list[OracleCandidate]] = {action: [] for action in _ACTION_VALUES}
        for record in harmful:
            for action, candidate in _validated_candidates(record).items():
                by_action[action].append(candidate)
        for action in _ACTION_VALUES:
            candidates = by_action[action]
            if len(candidates) != len(harmful):
                raise RdxTrainingAnalysisError("mechanism candidate denominator differs")
            feasible = [candidate for candidate in candidates if candidate.feasible]
            executed = [
                candidate
                for candidate in feasible
                if candidate.execution_succeeded and candidate.audit_state in defined_audit_states
            ]
            audit_admissible = [candidate for candidate in executed if candidate.audit_admissible]
            successor_counts = {
                state: sum(candidate.successor_evaluator_state == state for candidate in candidates)
                for state in successor_states
            }
            unavailable = len(candidates) - sum(successor_counts.values())
            denominator = len(harmful)
            rows.append(
                {
                    "budget": item.run.budget.value,
                    "sequence": "-".join(item.run.sequence),
                    "seed": item.run.seed,
                    "action": action,
                    "eligible_harmful_decisions": denominator,
                    "feasible_count": len(feasible),
                    "feasible_rate": (
                        float(Fraction(len(feasible), denominator)) if denominator else None
                    ),
                    "explicit_execution_failure_count": sum(
                        not candidate.execution_succeeded for candidate in feasible
                    ),
                    "execution_success_count": sum(
                        candidate.execution_succeeded for candidate in feasible
                    ),
                    "execution_denominator_feasible": len(feasible),
                    "execution_success_rate_given_feasible": (
                        float(
                            Fraction(
                                sum(candidate.execution_succeeded for candidate in feasible),
                                len(feasible),
                            )
                        )
                        if feasible
                        else None
                    ),
                    "audit_admissible_count": len(audit_admissible),
                    "audit_rejected_count": len(executed) - len(audit_admissible),
                    "audit_denominator_executed_defined": len(executed),
                    "audit_admissible_rate_given_executed": (
                        float(Fraction(len(audit_admissible), len(executed))) if executed else None
                    ),
                    "successor_safe_count": successor_counts["SAFE"],
                    "successor_uncertain_count": successor_counts["UNCERTAIN"],
                    "successor_harmful_count": successor_counts["HARMFUL"],
                    "successor_unavailable_count": unavailable,
                    "confirmed_success_count": sum(
                        candidate.confirmed_success for candidate in candidates
                    ),
                    "confirmed_success_rate": (
                        float(
                            Fraction(
                                sum(candidate.confirmed_success for candidate in candidates),
                                denominator,
                            )
                        )
                        if denominator
                        else None
                    ),
                    "selected_count": selected_counts[action],
                }
            )
    return tuple(rows)


def _final_memory_entries(
    memory: Mapping[str, Any],
    key: str,
) -> list[Mapping[str, Any]]:
    final_memory = memory.get("final_memory")
    if not isinstance(final_memory, Mapping):
        raise RdxTrainingAnalysisError("memory artifact lacks final_memory")
    entries = final_memory.get(key)
    if not isinstance(entries, list) or not all(isinstance(item, Mapping) for item in entries):
        raise RdxTrainingAnalysisError(f"memory artifact lacks {key}")
    return [item for item in entries if isinstance(item, Mapping)]


def _scope_token_rows(item: _RunEvidence) -> list[Mapping[str, Any]]:
    required = {"stage", "current_domain", "opaque_scope_token"}
    for container in (item.query, item.memory):
        scopes = container.get("scopes")
        if isinstance(scopes, list) and all(isinstance(scope, Mapping) for scope in scopes):
            rows = [scope for scope in scopes if isinstance(scope, Mapping)]
            if len(rows) == 3 and all(required.issubset(scope) for scope in rows):
                return rows
    raise RdxTrainingAnalysisError(f"scope-token provenance is absent for {item.run.run_id}")


def _later_memory_sizes(
    item: _RunEvidence,
    key: str,
) -> tuple[list[int], list[str]]:
    entries = _final_memory_entries(item.memory, key)
    by_token = {str(entry.get("domain_id", "")): entry for entry in entries}
    scopes = sorted(
        _scope_token_rows(item),
        key=lambda scope: _as_int(scope.get("stage"), "scope stage"),
    )
    later: list[Mapping[str, Any]] = []
    for scope in scopes:
        stage = _as_int(scope.get("stage"), "scope stage")
        token = str(scope.get("opaque_scope_token", ""))
        domain = str(scope.get("current_domain", ""))
        if stage not in {1, 2, 3} or domain != item.run.sequence[stage] or token not in by_token:
            raise RdxTrainingAnalysisError(f"scope-to-memory mapping differs for {item.run.run_id}")
        later.append(by_token[token])
    if len({str(entry.get("domain_id", "")) for entry in later}) != 3:
        raise RdxTrainingAnalysisError(f"later memory scopes are not unique for {item.run.run_id}")
    sizes = [_as_int(entry.get("size"), f"{key} size") for entry in later]
    selections = sorted(
        {str(entry.get("selection")) for entry in later if entry.get("selection") not in {None, ""}}
    )
    return sizes, selections


def _position_count(value: object, name: str) -> int:
    if not isinstance(value, list) or any(
        isinstance(position, bool) or not isinstance(position, int) for position in value
    ):
        raise RdxTrainingAnalysisError(f"{name} must be an integer-position list")
    if len(value) != len(set(value)):
        raise RdxTrainingAnalysisError(f"{name} contains duplicate positions")
    return len(value)


def _scheduled_target_doses(item: _RunEvidence) -> dict[str, list[int]]:
    """Derive cumulative optimizer-eligible doses from release provenance only."""

    expected_scopes = {
        (stage, item.run.sequence[stage]) for stage in range(1, len(item.run.sequence))
    }
    grouped: dict[tuple[int, str], list[tuple[int, int]]] = {}
    if item.run.budget is RDX004Budget.B100:
        scopes = item.query.get("scopes")
        if not isinstance(scopes, list) or not all(isinstance(scope, Mapping) for scope in scopes):
            raise RdxTrainingAnalysisError(f"B100 query scopes are malformed for {item.run.run_id}")
        for raw_scope in scopes:
            assert isinstance(raw_scope, Mapping)
            stage = _as_int(raw_scope.get("stage"), "B100 query scope stage")
            domain = str(raw_scope.get("current_domain", ""))
            allocation = raw_scope.get("allocation")
            if not isinstance(allocation, Mapping):
                raise RdxTrainingAnalysisError(f"B100 allocation is absent for {item.run.run_id}")
            releases = allocation.get("releases")
            if not isinstance(releases, list) or not all(
                isinstance(release, Mapping) for release in releases
            ):
                raise RdxTrainingAnalysisError(
                    f"B100 allocation releases are malformed for {item.run.run_id}"
                )
            key = (stage, domain)
            if key in grouped:
                raise RdxTrainingAnalysisError(
                    f"B100 query scope is duplicated for {item.run.run_id}"
                )
            grouped[key] = [
                (
                    _as_int(release.get("release_index"), "B100 release index"),
                    _position_count(
                        release.get("replay_positions"), "B100 optimizer-eligible positions"
                    ),
                )
                for release in releases
                if isinstance(release, Mapping)
            ]
    else:
        events = item.query.get("events")
        if not isinstance(events, list) or not all(isinstance(event, Mapping) for event in events):
            raise RdxTrainingAnalysisError(
                f"prospective query events are malformed for {item.run.run_id}"
            )
        for raw_event in events:
            assert isinstance(raw_event, Mapping)
            if not _as_bool(raw_event.get("released"), "query released marker"):
                raise RdxTrainingAnalysisError(
                    f"confirmatory query remained unreleased for {item.run.run_id}"
                )
            stage = _as_int(raw_event.get("stage"), "prospective query stage")
            domain = str(raw_event.get("current_domain", ""))
            ordinal = _as_int(raw_event.get("query_ordinal"), "prospective query ordinal")
            grouped.setdefault((stage, domain), []).append(
                (
                    ordinal,
                    _position_count(
                        raw_event.get("training_positions"),
                        "prospective optimizer-eligible positions",
                    ),
                )
            )
    if set(grouped) != expected_scopes:
        raise RdxTrainingAnalysisError(
            f"optimizer-eligible release scopes differ for {item.run.run_id}"
        )
    schedules: dict[str, list[int]] = {}
    for stage, domain in sorted(grouped):
        releases = sorted(grouped[(stage, domain)])
        if [ordinal for ordinal, _ in releases] != list(range(4)):
            raise RdxTrainingAnalysisError(
                f"optimizer-eligible release ordinals differ for {item.run.run_id}"
            )
        running = 0
        cumulative: list[int] = []
        for _, count in releases:
            running += count
            cumulative.append(running)
        schedules[f"{stage}:{domain}"] = cumulative
    return schedules


def _dose_rows(evidence: Sequence[_RunEvidence]) -> tuple[dict[str, object], ...]:
    expected_target = {
        RDX004Budget.B100: 80,
        RDX004Budget.B400: 380,
        RDX004Budget.B1600: 1580,
    }
    expected_cumulative = {
        RDX004Budget.B100: [20, 40, 60, 80],
        RDX004Budget.B400: [95, 190, 285, 380],
        RDX004Budget.B1600: [395, 790, 1185, 1580],
    }
    rows: list[dict[str, object]] = []
    for item in evidence:
        cumulative = expected_cumulative[item.run.budget]
        schedules = _scheduled_target_doses(item)
        if any(schedule != cumulative for schedule in schedules.values()):
            raise RdxTrainingAnalysisError(
                "RDX-007 NO-GO: scheduled cumulative optimizer-eligible evidence differs "
                f"for {item.run.run_id}; observed={schedules}, expected={cumulative}"
            )
        available: list[int] = []
        consumed: list[int] = []
        available_by_action: dict[str, set[int]] = {action: set() for action in _ACTION_VALUES}
        consumed_by_action: dict[str, set[int]] = {action: set() for action in _ACTION_VALUES}
        explicit_availability_markers: list[bool] = []
        for record in item.oracle_records:
            raw_candidates = record.get("candidates")
            if not isinstance(raw_candidates, list):
                raise RdxTrainingAnalysisError("Oracle candidates are absent in dose analysis")
            for raw in raw_candidates:
                if not isinstance(raw, Mapping):
                    raise RdxTrainingAnalysisError("Oracle candidate is malformed in dose analysis")
                candidate = _typed_candidate(raw)
                explicit = "target_rows_available" in raw
                explicit_availability_markers.append(explicit)
                if explicit:
                    target_available = _as_int(
                        raw.get("target_rows_available"), "target rows available"
                    )
                    available.append(target_available)
                    if target_available > 0:
                        available_by_action[candidate.action.value].add(target_available)
                consumed.append(candidate.target_rows)
                if candidate.target_rows > 0:
                    consumed_by_action[candidate.action.value].add(candidate.target_rows)
        if any(explicit_availability_markers) and not all(explicit_availability_markers):
            raise RdxTrainingAnalysisError(
                f"candidate availability provenance is inconsistent for {item.run.run_id}"
            )
        explicit_availability = bool(explicit_availability_markers) and all(
            explicit_availability_markers
        )
        if item.run.budget is not RDX004Budget.B100 and not explicit_availability:
            raise RdxTrainingAnalysisError(
                f"prospective target availability is absent for {item.run.run_id}"
            )
        expected = expected_target[item.run.budget]
        observed_available = (
            sorted({value for value in available if value > 0})
            if explicit_availability
            else list(cumulative)
        )
        if any(value not in cumulative for value in observed_available) or (
            max(observed_available, default=0) != expected
        ):
            raise RdxTrainingAnalysisError(
                "RDX-007 NO-GO: optimizer-eligible candidate availability differs "
                f"for {item.run.run_id}; observed={observed_available}, expected={cumulative}"
            )
        max_available = max(observed_available, default=0)
        max_consumed = max(consumed, default=0)
        observed_consumed = sorted({value for value in consumed if value > 0})
        if any(value not in cumulative for value in observed_consumed) or max_consumed > expected:
            raise RdxTrainingAnalysisError(
                "RDX-007 NO-GO: action-consumed target evidence is outside the scheduled "
                f"cumulative doses for {item.run.run_id}; consumed={observed_consumed}, "
                f"scheduled={cumulative}"
            )
        audit_sizes, _ = _later_memory_sizes(item, "audit_domains")
        replay_sizes, replay_selections = _later_memory_sizes(item, "replay_domains")
        if len(audit_sizes) != 3 or audit_sizes != [20, 20, 20]:
            raise RdxTrainingAnalysisError(
                f"RDX-007 NO-GO: later-domain audit dose differs for {item.run.run_id}"
            )
        expected_replay_max = {
            RDX004Budget.B100: 80,
            RDX004Budget.B400: 380,
            RDX004Budget.B1600: 400,
        }[item.run.budget]
        if len(replay_sizes) != 3 or max(replay_sizes, default=0) != expected_replay_max:
            raise RdxTrainingAnalysisError(
                f"RDX-007 NO-GO: historical replay dose differs for {item.run.run_id}"
            )
        expected_projection = {
            RDX004Budget.B400: "rdx004-retain-all-under-cap400-v1",
            RDX004Budget.B1600: "rdx004-deterministic-binary-stratified-cap400-v1",
        }.get(item.run.budget)
        if expected_projection is not None and replay_selections != [expected_projection]:
            raise RdxTrainingAnalysisError(
                f"RDX-007 NO-GO: replay projection identity differs for {item.run.run_id}"
            )
        final_memory = item.memory.get("final_memory")
        assert isinstance(final_memory, Mapping)
        capacity = _as_int(
            final_memory.get("replay_capacity_per_domain"), "replay capacity per domain"
        )
        if capacity != 400:
            raise RdxTrainingAnalysisError("historical replay capacity is not 400")
        resources = item.resources
        training_rows_allocated = _as_int(
            resources.get(
                "training_rows_allocated",
                _as_int(resources.get("labels_released"), "labels released") - sum(audit_sizes),
            ),
            "training rows allocated",
        )
        if training_rows_allocated != sum(schedule[-1] for schedule in schedules.values()):
            raise RdxTrainingAnalysisError(
                f"allocated training rows differ from release provenance for {item.run.run_id}"
            )
        rows.append(
            {
                "budget": item.run.budget.value,
                "sequence": "-".join(item.run.sequence),
                "seed": item.run.seed,
                "expected_maximum_optimizer_eligible_target_rows": expected,
                "expected_cumulative_target_doses_json": json.dumps(
                    cumulative, separators=(",", ":")
                ),
                "scheduled_cumulative_target_doses_by_scope_json": json.dumps(
                    schedules, sort_keys=True, separators=(",", ":")
                ),
                "target_availability_provenance": (
                    "candidate_target_rows_available"
                    if explicit_availability
                    else "validated_B100_release_allocation"
                ),
                "observed_maximum_target_rows_available": max_available,
                "observed_maximum_target_rows_consumed": max_consumed,
                "observed_cumulative_target_doses_available_json": json.dumps(
                    observed_available, separators=(",", ":")
                ),
                "observed_cumulative_target_doses_consumed_json": json.dumps(
                    observed_consumed, separators=(",", ":")
                ),
                "target_doses_available_by_action_json": json.dumps(
                    {action: sorted(values) for action, values in available_by_action.items()},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "target_doses_consumed_by_action_json": json.dumps(
                    {action: sorted(values) for action, values in consumed_by_action.items()},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "scheduled_cumulative_target_doses_verified": True,
                "consumed_target_doses_within_schedule": True,
                "labels_requested": _as_int(resources.get("labels_requested"), "labels requested"),
                "labels_released": _as_int(resources.get("labels_released"), "labels released"),
                "training_rows_allocated": training_rows_allocated,
                "later_domain_scope_count": len(audit_sizes),
                "later_domain_audit_rows_json": json.dumps(audit_sizes, separators=(",", ":")),
                "audit_rows_per_later_domain_exactly_20": True,
                "later_domain_replay_rows_json": json.dumps(replay_sizes, separators=(",", ":")),
                "maximum_later_domain_replay_rows": max(replay_sizes),
                "historical_replay_capacity_per_domain": capacity,
                "b1600_projection_identity": (
                    "rdx004-deterministic-binary-stratified-cap400-v1"
                    if item.run.budget is RDX004Budget.B1600
                    else "NOT_APPLICABLE"
                ),
                "contrast_note": (
                    "B1600 later stages combine 1,580 target rows with a deterministic "
                    "fixed-cap-400 historical replay projection; not a pure raw-label contrast"
                    if item.run.budget is RDX004Budget.B1600
                    else "target-evidence treatment under frozen audit and replay rules"
                ),
            }
        )
    return tuple(rows)


def _as_float(value: object, name: str) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError) as exc:
        raise RdxTrainingAnalysisError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise RdxTrainingAnalysisError(f"{name} must be finite")
    return result


def _optional_float(value: object, name: str) -> float | None:
    return None if value in {None, ""} else _as_float(value, name)


def _accepted_model_changing_updates(
    records: Sequence[Mapping[str, object]],
) -> int:
    accepted = 0
    for record in records:
        selected_action = str(record.get("selected_action", ""))
        candidates = _validated_candidates(record)
        if selected_action not in candidates:
            raise RdxTrainingAnalysisError("selected Oracle action has no candidate")
        if selected_action not in {
            "A2_HEAD_UPDATE",
            "A3_FULL_FINE_TUNE",
            "A4_REPLAY_UPDATE",
        }:
            continue
        selected_candidate = candidates[selected_action]
        if not selected_candidate.execution_succeeded or not selected_candidate.audit_admissible:
            raise RdxTrainingAnalysisError("selected model-changing action was not accepted")
        accepted += 1
    return accepted


def _context_rows(evidence: Sequence[_RunEvidence]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for item in evidence:
        resources = item.resources
        window_count = len(item.windows)
        unsafe_windows = sum(row.get("evaluator_health_state") == "HARMFUL" for row in item.windows)
        if window_count == 0:
            raise RdxTrainingAnalysisError(f"window evidence is empty for {item.run.run_id}")
        compliance = 1.0 - (unsafe_windows / window_count)
        if _as_int(resources.get("unsafe_exposure_windows"), "unsafe exposure windows") != (
            unsafe_windows
        ):
            raise RdxTrainingAnalysisError(
                f"unsafe exposure count differs from windows for {item.run.run_id}"
            )
        persisted_compliance = resources.get("operating_envelope_compliance")
        if persisted_compliance is not None and not np.isclose(
            _as_float(persisted_compliance, "operating-envelope compliance"),
            compliance,
            rtol=0.0,
            atol=1e-15,
        ):
            raise RdxTrainingAnalysisError(
                f"operating-envelope compliance differs for {item.run.run_id}"
            )
        selected_steps = 0
        for record in item.oracle_records:
            selected_action = str(record.get("selected_action", ""))
            candidates = _validated_candidates(record)
            if selected_action not in candidates:
                raise RdxTrainingAnalysisError("selected Oracle action has no candidate")
            selected_candidate = candidates[selected_action]
            selected_steps += selected_candidate.optimizer_steps
        accepted_model_changing = _accepted_model_changing_updates(item.oracle_records)
        persisted_steps = resources.get(
            "selected_optimizer_steps", resources.get("optimizer_steps")
        )
        if _as_int(persisted_steps, "selected optimizer steps") != selected_steps:
            raise RdxTrainingAnalysisError(f"selected optimizer steps differ for {item.run.run_id}")
        accepted_state_changes = sum(
            _as_bool(row.get("accepted_update_after_prediction"), "window accepted update")
            for row in item.windows
        )
        if (
            _as_int(resources.get("accepted_updates"), "accepted updates")
            != (accepted_state_changes)
            or accepted_model_changing > accepted_state_changes
        ):
            raise RdxTrainingAnalysisError(
                f"accepted update accounting differs for {item.run.run_id}"
            )
        replay_total = sum(
            _as_int(entry.get("size"), "final replay domain size")
            for entry in _final_memory_entries(item.memory, "replay_domains")
        )
        audit_total = sum(
            _as_int(entry.get("size"), "final audit domain size")
            for entry in _final_memory_entries(item.memory, "audit_domains")
        )
        for key, observed in (
            ("final_replay_rows", replay_total),
            ("final_audit_rows", audit_total),
        ):
            persisted = resources.get(key)
            if persisted is not None and _as_int(persisted, key) != observed:
                raise RdxTrainingAnalysisError(f"{key} differs for {item.run.run_id}")
        final_holdouts = [row for row in item.holdouts if row.get("event") == "final"]
        by_domain = {row.get("holdout_dataset_id"): row for row in final_holdouts}
        if len(final_holdouts) != 4 or set(by_domain) != set(item.run.sequence):
            raise RdxTrainingAnalysisError(f"final holdout coverage differs for {item.run.run_id}")
        final_metrics: list[dict[str, object]] = []
        for domain in item.run.sequence:
            holdout = by_domain[domain]
            metric = {
                "domain": domain,
                "row_count": _as_int(holdout.get("row_count"), "holdout row count"),
                "attack_count": _as_int(holdout.get("attack_count"), "holdout attack count"),
                "benign_count": _as_int(holdout.get("benign_count"), "holdout benign count"),
                "pr_auc": _optional_float(holdout.get("pr_auc"), "holdout PR-AUC"),
                "roc_auc": _optional_float(holdout.get("roc_auc"), "holdout ROC-AUC"),
                "tpr": _optional_float(holdout.get("tpr"), "holdout TPR"),
                "fpr": _optional_float(holdout.get("fpr"), "holdout FPR"),
                "precision": _optional_float(holdout.get("precision"), "holdout precision"),
                "macro_f1": _optional_float(holdout.get("macro_f1"), "holdout macro-F1"),
                "false_positives_per_million": _optional_float(
                    holdout.get("false_positives_per_million"),
                    "holdout false positives per million",
                ),
                "operating_envelope_state": holdout.get("operating_envelope_state", ""),
                "learned_reference_tpr": _optional_float(
                    holdout.get("learned_reference_tpr"), "learned-reference TPR"
                ),
                "operational_tpr_forgetting": _optional_float(
                    holdout.get("operational_tpr_forgetting"), "operational TPR forgetting"
                ),
            }
            if "fpr_budget_ratio" in holdout:
                metric["fpr_budget_ratio"] = _optional_float(
                    holdout.get("fpr_budget_ratio"), "holdout FPR-budget ratio"
                )
            final_metrics.append(metric)
        forgetting = [
            _as_float(row["operational_tpr_forgetting"], "operational TPR forgetting")
            for row in final_holdouts
            if row.get("operational_tpr_forgetting", "") != ""
        ]
        rows.append(
            {
                "budget": item.run.budget.value,
                "sequence": "-".join(item.run.sequence),
                "seed": item.run.seed,
                "run_id": item.run.run_id,
                "window_count": window_count,
                "operating_envelope_compliance": compliance,
                "unsafe_exposure_windows": unsafe_windows,
                "accepted_model_changing_updates": accepted_model_changing,
                "requested_labels": _as_int(resources.get("labels_requested"), "requested labels"),
                "released_labels": _as_int(resources.get("labels_released"), "released labels"),
                "selected_optimizer_steps": selected_steps,
                "final_replay_rows": replay_total,
                "final_audit_rows": audit_total,
                "final_holdout_rows": len(final_holdouts),
                "final_holdout_metrics_json": json.dumps(
                    final_metrics, sort_keys=True, separators=(",", ":")
                ),
                "final_mean_operational_tpr_forgetting": (
                    sum(forgetting) / len(forgetting) if forgetting else None
                ),
                "wall_clock_selected_update_seconds": None,
                "wall_clock_comparable": False,
                "interpretation_scope": (
                    "secondary deployed outcome; cannot override P1/P2; wall-clock excluded "
                    "because shared-hardware contention was not frozen as comparable"
                ),
            }
        )
    return tuple(rows)


def _summary_row(
    rows: Sequence[Mapping[str, object]],
    *,
    budget: str,
    outcome: str,
) -> Mapping[str, object]:
    matches = [row for row in rows if row.get("budget") == budget and row.get("outcome") == outcome]
    if len(matches) != 1:
        raise RdxTrainingAnalysisError("summary row lookup is not unique")
    return matches[0]


def _contrast_row(
    rows: Sequence[Mapping[str, object]],
    *,
    contrast: str,
    outcome: str,
) -> Mapping[str, object]:
    matches = [
        row for row in rows if row.get("contrast") == contrast and row.get("outcome") == outcome
    ]
    if len(matches) != 1:
        raise RdxTrainingAnalysisError("contrast summary row lookup is not unique")
    return matches[0]


def _interpretation(gate: Mapping[str, Any]) -> dict[str, object]:
    case = str(gate.get("interpretation_case"))
    text = {
        "CASE_A": (
            "Under at least one increased-evidence condition, B100 training scarcity was a "
            "meaningful limitation on one-step recoverability for Core-accessible actions."
        ),
        "CASE_B": (
            "Additional training evidence expanded one-step recoverability, but the gain was "
            "materially concentrated in A3 outside the normal Core action space."
        ),
        "CASE_D": (
            "B100 training scarcity alone was insufficient to explain the observed recovery "
            "ceiling under the frozen compact model and A0--A4 intervention family."
        ),
        "MIXED": (
            "The increased-evidence effects were mixed or deployment-order dependent and did "
            "not satisfy a prospectively frozen follow-up gate."
        ),
        "NOT_ASSESSED": "The frozen gate was not assessed because the corpus was incomplete.",
    }.get(case)
    if text is None:
        raise RdxTrainingAnalysisError("decision gate has an unknown interpretation case")
    return {
        "case": case,
        "bounded_interpretation": text,
        "case_c": (
            "Audit-stage patterns are descriptive only: no audit-dominance threshold was "
            "registered, so Case C cannot authorize a follow-up."
        ),
        "scope": (
            "Bounded to B400/B1600, the compact MLP, frozen datasets, D1 chronology, and the "
            "frozen A0--A4 family; not general evidence about arbitrary labelled-data amounts."
        ),
    }


def _build_summary(
    inputs: _AnalysisInputs,
    tables: Mapping[str, Sequence[Mapping[str, object]]],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    treatment = tables[TREATMENT_FILENAME]
    bootstrap = tables[BOOTSTRAP_FILENAME]
    return {
        "version": RDX007_ANALYSIS_VERSION,
        "status": "VALIDATED_ARTIFACT_ONLY_SCIENTIFIC_ANALYSIS",
        "source_corpus": {
            "total_runs": len(inputs.runs),
            "counts": {
                budget.value: sum(run.budget is budget for run in inputs.runs)
                for budget in RDX004_ALL_BUDGETS
            },
            "historical_b100_runs": 12,
            "prospective_b400_b1600_runs": 24,
            "full_three_arm_evidence_wholly_prospective": False,
            "all_sources_validated_before_outcome_derivation": True,
        },
        "primary_treatment_summaries": {
            budget.value: {
                outcome: {
                    "eligible_harmful_total": _as_int(
                        _summary_row(treatment, budget=budget.value, outcome=outcome).get(
                            "eligible_harmful_total"
                        ),
                        "eligible harmful total",
                    ),
                    "mean_unit_rate": _summary_row(
                        treatment, budget=budget.value, outcome=outcome
                    ).get("mean_unit_rate"),
                    "mean_unit_rate_exact": _summary_row(
                        treatment, budget=budget.value, outcome=outcome
                    ).get("mean_unit_rate_exact"),
                    "median_unit_rate": _summary_row(
                        treatment, budget=budget.value, outcome=outcome
                    ).get("median_unit_rate"),
                    "median_unit_rate_exact": _summary_row(
                        treatment, budget=budget.value, outcome=outcome
                    ).get("median_unit_rate_exact"),
                }
                for outcome in ("P1", "P2", "A3_EXCLUSIVE")
            }
            for budget in RDX004_ALL_BUDGETS
        },
        "paired_contrasts": {
            contrast: {
                outcome: {
                    "mean_difference": _contrast_row(
                        bootstrap, contrast=contrast, outcome=outcome
                    ).get("mean_difference"),
                    "mean_difference_exact": _contrast_row(
                        bootstrap, contrast=contrast, outcome=outcome
                    ).get("mean_difference_exact"),
                    "median_difference": _contrast_row(
                        bootstrap, contrast=contrast, outcome=outcome
                    ).get("median_difference"),
                    "median_difference_exact": _contrast_row(
                        bootstrap, contrast=contrast, outcome=outcome
                    ).get("median_difference_exact"),
                    "mean_95_percentile_interval": [
                        _contrast_row(bootstrap, contrast=contrast, outcome=outcome).get(
                            "mean_ci_lower"
                        ),
                        _contrast_row(bootstrap, contrast=contrast, outcome=outcome).get(
                            "mean_ci_upper"
                        ),
                    ],
                    "median_95_percentile_interval": [
                        _contrast_row(bootstrap, contrast=contrast, outcome=outcome).get(
                            "median_ci_lower"
                        ),
                        _contrast_row(bootstrap, contrast=contrast, outcome=outcome).get(
                            "median_ci_upper"
                        ),
                    ],
                    "positive_rotation_medians": _as_int(
                        _contrast_row(bootstrap, contrast=contrast, outcome=outcome).get(
                            "positive_rotation_medians"
                        ),
                        "positive rotation medians",
                    ),
                }
                for outcome in _PRIMARY_OUTCOMES
            }
            for contrast, _, _ in _CONTRASTS
        },
        "decision_gate": dict(gate),
        "scientific_interpretation": _interpretation(gate),
        "secondary_outcomes": {
            "kept_separate_from_primary": True,
            "wall_clock_comparisons_made": False,
        },
        "source_artifacts_mutated": False,
        "new_experiment_run": False,
        "new_experiment_authorized": False,
        "danids_2_0_hypotheses_modified": False,
    }


def _percent(value: object) -> str:
    if value in {None, ""}:
        return "NA"
    return f"{100.0 * float(str(value)):.3f}%"


def _report_text(
    tables: Mapping[str, Sequence[Mapping[str, object]]],
    gate: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> str:
    treatment = tables[TREATMENT_FILENAME]
    bootstrap = tables[BOOTSTRAP_FILENAME]
    treatment_lines = [
        "| Budget | P1 mean | P1 median | P2 mean | P2 median | Eligible HARMFUL |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for budget in RDX004_ALL_BUDGETS:
        p1 = _summary_row(treatment, budget=budget.value, outcome="P1")
        p2 = _summary_row(treatment, budget=budget.value, outcome="P2")
        treatment_lines.append(
            "| "
            + " | ".join(
                [
                    budget.value,
                    _percent(p1.get("mean_unit_rate")),
                    _percent(p1.get("median_unit_rate")),
                    _percent(p2.get("mean_unit_rate")),
                    _percent(p2.get("median_unit_rate")),
                    str(p1.get("eligible_harmful_total")),
                ]
            )
            + " |"
        )
    contrast_lines = [
        "| Contrast | Outcome | Mean difference | Median difference | Positive rotations | "
        "Mean 95% interval | Median 95% interval |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for contrast, _, _ in _CONTRASTS:
        for outcome in _PRIMARY_OUTCOMES:
            row = _contrast_row(bootstrap, contrast=contrast, outcome=outcome)
            contrast_lines.append(
                "| "
                + " | ".join(
                    [
                        contrast,
                        outcome,
                        _percent(row.get("mean_difference")),
                        _percent(row.get("median_difference")),
                        str(row.get("positive_rotation_medians")),
                        (
                            f"[{_percent(row.get('mean_ci_lower'))}, "
                            f"{_percent(row.get('mean_ci_upper'))}]"
                        ),
                        (
                            f"[{_percent(row.get('median_ci_lower'))}, "
                            f"{_percent(row.get('median_ci_upper'))}]"
                        ),
                    ]
                )
                + " |"
            )
    interpretation = summary["scientific_interpretation"]
    assert isinstance(interpretation, Mapping)
    qualifying = gate.get("qualifying_budgets", [])
    qualifying_text = ", ".join(str(value) for value in qualifying) or "none"
    return f"""# RDX-007 training-evidence sensitivity analysis

This is a separately versioned post-freeze, artifact-only analysis. It validates 12 historical
B100 Offline-Oracle runs and 24 prospectively executed B400/B1600 runs. Rotation x seed is the
scientific unit; flows, windows, decisions, and candidates are nested descriptive observations.
No experiment was run or regenerated by this analysis.

## Primary outcomes

P1 is any confirmed one-step success among A0--A4 from an eligible currently HARMFUL decision.
P2 uses the same denominator and only A1/A2/A4. UNCERTAIN is not success, and unfavourable
decisions remain in the denominator.

{chr(10).join(treatment_lines)}

All 12 per-unit values and their exact eligible-HARMFUL denominators are in
`{PRIMARY_FILENAME}`. Rotation medians are in `{ROTATION_FILENAME}`.

## Paired treatment contrasts

Differences are paired only by rotation x seed. P1 and P2 95% intervals use the prospectively
frozen 10,000-replicate within-rotation paired bootstrap and are descriptive, not significance
tests. A3-exclusive mean and median differences are reported without an interval because no
A3-exclusive bootstrap interval was registered.

{chr(10).join(contrast_lines)}

## Frozen decision gate

**{gate["status"]}**. Qualifying budget conditions: **{qualifying_text}**.

{interpretation["bounded_interpretation"]}

{interpretation["case_c"]}

## Mechanistic decomposition

`{MECHANISM_FILENAME}` keeps feasibility, explicit execution failure, audit admissibility and
rejection, successor SAFE/UNCERTAIN/HARMFUL, and confirmed success separate for every action.
Zero explicit execution failures must not be interpreted as recovery, and the descriptive stage
counts do not identify a causal bottleneck.

## Evidence dose

`{DOSE_FILENAME}` verifies from immutable release/allocation provenance maximum
optimizer-eligible target evidence of 80, 380, and 1,580 rows for B100, B400, and B1600,
respectively. It reports the scheduled cumulative doses (20/40/60/80, 95/190/285/380, and
395/790/1,185/1,580) separately from the doses actually consumed by actions, with exactly 20
audit rows per later domain. An action need not occur at every scheduled dose. B1600 later
stages use a deterministic projection of 1,580 historical rows to the fixed 400-row replay cap,
so they are not a pure raw-label contrast.

## Contextual deployed outcomes

`{CONTEXT_FILENAME}` reports compliance, unsafe exposure, accepted model-changing updates,
labels, selected optimizer steps, final memory sizes, and the supported per-domain final
retention/forgetting metrics. These secondary outcomes do not overwrite the P1/P2 conclusion.
Wall-clock comparisons are not made because shared-hardware contention was not frozen as
comparable.

## Evidence timing and scope

B100 is historical canonical comparator evidence; B400 and B1600 are prospective new-treatment
evidence. The combined three-arm analysis is therefore not wholly concurrent prospective
evidence. Conclusions are bounded to the frozen compact MLP, datasets, D1 chronology, and A0--A4
family. No new experiment is authorised, and the DANIDS 2.0 evidence freeze is unchanged.
"""


def _derive_analysis(
    inputs: _AnalysisInputs,
) -> tuple[dict[str, tuple[dict[str, object], ...]], dict[str, Any], dict[str, Any], str]:
    evidence = tuple(_load_run_evidence(run) for run in inputs.runs)
    primary = _primary_rows(evidence)
    treatment = _treatment_summaries(primary)
    rotation = _rotation_summaries(primary)
    paired = _paired_rows(primary)
    bootstrap = _bootstrap_rows(paired)
    mechanism = _mechanism_rows(evidence)
    dose = _dose_rows(evidence)
    context = _context_rows(evidence)
    gate = evaluate_decision_gate(primary)
    tables: dict[str, tuple[dict[str, object], ...]] = {
        MATRIX_FILENAME: inputs.matrix_rows,
        PRIMARY_FILENAME: primary,
        TREATMENT_FILENAME: treatment,
        ROTATION_FILENAME: rotation,
        PAIRED_FILENAME: paired,
        BOOTSTRAP_FILENAME: bootstrap,
        MECHANISM_FILENAME: mechanism,
        DOSE_FILENAME: dose,
        CONTEXT_FILENAME: context,
    }
    summary = _build_summary(inputs, tables, gate)
    report = _report_text(tables, gate, summary)
    _assert_sources_unchanged(inputs)
    return tables, gate, summary, report


def _ensure_output_is_separate(inputs: _AnalysisInputs, output: Path) -> None:
    protected = {
        inputs.study4_evaluation_dir,
        inputs.preflight_dir,
        inputs.run_root,
        *(run.path for run in inputs.runs),
    }
    for source in protected:
        if output == source or source in output.parents or output in source.parents:
            raise RdxTrainingAnalysisError(
                "RDX-007 output must be separate from every immutable source tree"
            )


def _manifest_payload(inputs: _AnalysisInputs, root: Path) -> dict[str, Any]:
    files = _tree_digests(root)
    expected = ALL_OUTPUT_FILES.difference({MANIFEST_FILENAME})
    if set(files) != expected:
        raise RdxTrainingAnalysisError("RDX-007 pre-manifest file set differs")
    return {
        "version": RDX007_ANALYSIS_VERSION,
        "protocol_version": RDX004_PROTOCOL_VERSION,
        "protocol_sha256": RDX004_PROTOCOL_SHA256,
        "preflight_bundle_digest": RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST,
        "analysis_contract_sha256": files[CONTRACT_FILENAME],
        "source_bundle_digests": {run.run_id: run.artifact_bundle_digest for run in inputs.runs},
        "files": files,
        "bundle_digest": _digest(files),
    }


def _expected_texts(
    inputs: _AnalysisInputs,
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    tables, gate, summary, report = _derive_analysis(inputs)
    text_by_name = {name: _csv_text(OUTPUT_SCHEMAS[name], tables[name]) for name in OUTPUT_SCHEMAS}
    text_by_name.update(
        {
            CONTRACT_FILENAME: _json_text(inputs.contract),
            GATE_FILENAME: _json_text(gate),
            SUMMARY_FILENAME: _json_text(summary),
            REPORT_FILENAME: report,
        }
    )
    return text_by_name, gate, summary


def _validate_output_with_inputs(
    output: Path,
    inputs: _AnalysisInputs,
    *,
    expected_texts: Mapping[str, str] | None = None,
) -> None:
    if not output.is_dir():
        raise RdxTrainingAnalysisError(f"RDX-007 output directory is absent: {output}")
    entries = tuple(output.iterdir())
    if any(not entry.is_file() for entry in entries) or {entry.name for entry in entries} != (
        ALL_OUTPUT_FILES
    ):
        raise RdxTrainingAnalysisError("RDX-007 output file set differs from the contract")
    manifest = _load_json(output / MANIFEST_FILENAME)
    if (output / MANIFEST_FILENAME).read_bytes() != _json_text(manifest).encode("utf-8"):
        raise RdxTrainingAnalysisError("RDX-007 manifest is not canonical JSON")
    files = _tree_digests(output)
    expected_manifest = {
        "version": RDX007_ANALYSIS_VERSION,
        "protocol_version": RDX004_PROTOCOL_VERSION,
        "protocol_sha256": RDX004_PROTOCOL_SHA256,
        "preflight_bundle_digest": RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST,
        "analysis_contract_sha256": files.get(CONTRACT_FILENAME),
        "source_bundle_digests": {run.run_id: run.artifact_bundle_digest for run in inputs.runs},
        "files": files,
        "bundle_digest": _digest(files),
    }
    if manifest != expected_manifest:
        raise RdxTrainingAnalysisError("RDX-007 manifest differs from persisted files/sources")
    if expected_texts is None:
        expected_texts, _, _ = _expected_texts(inputs)
    for name, expected_text in expected_texts.items():
        if (output / name).read_bytes() != expected_text.encode("utf-8"):
            raise RdxTrainingAnalysisError(
                f"persisted {name} differs from deterministic source-backed derivation"
            )
    _assert_sources_unchanged(inputs)


def _write_bundle(inputs: _AnalysisInputs, output_dir: str | Path) -> Path:
    output = Path(output_dir).resolve()
    _ensure_output_is_separate(inputs, output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite RDX-007 analysis directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=str(output.parent))).resolve()
    try:
        texts, _, _ = _expected_texts(inputs)
        for name, value in texts.items():
            (temporary / name).write_text(value, encoding="utf-8", newline="")
        manifest = _manifest_payload(inputs, temporary)
        (temporary / MANIFEST_FILENAME).write_text(
            _json_text(manifest), encoding="utf-8", newline=""
        )
        _validate_output_with_inputs(temporary, inputs, expected_texts=texts)
        validate_rdx_training_analysis(temporary)
        temporary.replace(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    _assert_sources_unchanged(inputs)
    return output


def validate_rdx_training_analysis(output_dir: str | Path) -> None:
    """Revalidate all 36 sources and byte-rederive the complete RDX-007 package."""

    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise RdxTrainingAnalysisError(f"RDX-007 output directory is absent: {output}")
    entries = tuple(output.iterdir())
    if any(not entry.is_file() for entry in entries) or {entry.name for entry in entries} != (
        ALL_OUTPUT_FILES
    ):
        raise RdxTrainingAnalysisError("RDX-007 output file set differs from the contract")
    contract = _load_json(output / CONTRACT_FILENAME)
    paths = contract.get("input_paths")
    if not isinstance(paths, Mapping):
        raise RdxTrainingAnalysisError("RDX-007 contract lacks input provenance")
    inputs = _validate_corpus_inputs(
        Path(str(paths.get("study4_evaluation_dir", ""))),
        Path(str(paths.get("preflight_dir", ""))),
        Path(str(paths.get("prospective_run_root", ""))),
    )
    _ensure_output_is_separate(inputs, output)
    _validate_output_with_inputs(output, inputs)


def analyze_rdx004_training_evidence(
    study4_evaluation_dir: str | Path,
    preflight_dir: str | Path,
    run_root: str | Path,
    output_dir: str | Path,
) -> Path:
    """Validate, derive, seal, and independently revalidate the frozen RDX-007 analysis."""

    inputs = _validate_corpus_inputs(study4_evaluation_dir, preflight_dir, run_root)
    return _write_bundle(inputs, output_dir)


__all__ = [
    "ALL_OUTPUT_FILES",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CONTRACT_FILENAME",
    "GATE_FILENAME",
    "MATERIAL_IMPROVEMENT",
    "OUTPUT_SCHEMAS",
    "RDX007_ANALYSIS_VERSION",
    "SUMMARY_FILENAME",
    "RdxTrainingAnalysisError",
    "analyze_rdx004_training_evidence",
    "compute_primary_outcome",
    "evaluate_decision_gate",
    "stratified_bootstrap_interval",
    "validate_rdx_training_analysis",
]
