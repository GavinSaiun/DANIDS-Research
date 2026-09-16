"""Deterministic artifact-only scientific analysis of the frozen RDX bundle.

RDX-003 is intentionally downstream of the source-backed RDX-002 derivation.  It
validates that exact bundle, performs only descriptive aggregation, and writes a
new write-once package.  It never reads raw data, executes a model, or changes a
frozen DANIDS hypothesis or RDX-001 definition.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

import pandas as pd

from danids.evaluation.rdx import (
    CORE_ATTEMPT_FILENAME,
    CORE_AUDIT_FILENAME,
    CORE_PATHWAY_FILENAME,
    EVIDENCE_FILENAME,
    HARM_FILENAME,
    INCIDENT_FILENAME,
    ORACLE_CANDIDATE_FILENAME,
    ORACLE_DECISION_FILENAME,
    RDX_EVALUATION_VERSION,
    RDX_PROTOCOL_VERSION,
    RECOVERY_FILENAME,
    RdxArtifactError,
    validate_rdx_evaluation,
)
from danids.evaluation.rdx import (
    MANIFEST_FILENAME as RDX_MANIFEST_FILENAME,
)
from danids.evaluation.rdx import (
    OUTPUT_SCHEMAS as RDX_OUTPUT_SCHEMAS,
)

getcontext().prec = 40

RDX_ANALYSIS_VERSION: Final = "rdx-003-artifact-analysis-v1"
EXPECTED_SOURCE_BUNDLE_DIGEST: Final = (
    "f93e9230ddcd000f0dae2ffe51b7d730bcc90c5444d4fd276768859dae3f7cd5"
)
EXPECTED_ROW_COUNTS: Final = {
    HARM_FILENAME: 32_133,
    CORE_PATHWAY_FILENAME: 8_290,
    CORE_ATTEMPT_FILENAME: 8_296,
    CORE_AUDIT_FILENAME: 114,
    EVIDENCE_FILENAME: 8_290,
    ORACLE_CANDIDATE_FILENAME: 48_060,
    ORACLE_DECISION_FILENAME: 9_612,
    RECOVERY_FILENAME: 130,
    INCIDENT_FILENAME: 8_290,
}

CONTRACT_FILENAME: Final = "rdx_analysis_contract.json"
SUMMARY_FILENAME: Final = "rdx_analysis_summary.json"
REPORT_FILENAME: Final = "rdx_analysis_report.md"
MANIFEST_FILENAME: Final = "artifact_manifest.json"

SOURCE_VALIDATION_FILENAME: Final = "rdx_source_validation.csv"
RECOGNITION_FILENAME: Final = "rdx_a_recognition_summary.csv"
CORE_PATHWAY_SUMMARY_FILENAME: Final = "rdx_ad_core_pathway_summary.csv"
CORE_ATTEMPT_AUDIT_FILENAME: Final = "rdx_ad_core_attempt_audit_summary.csv"
ORACLE_CLASS_FILENAME: Final = "rdx_b_oracle_class_summary.csv"
ORACLE_CANDIDATE_SUMMARY_FILENAME: Final = "rdx_b_oracle_candidate_summary.csv"
ORACLE_SUCCESS_OVERLAP_FILENAME: Final = "rdx_b_oracle_success_overlap.csv"
ORACLE_NUMERIC_AVAILABILITY_FILENAME: Final = "rdx_b_oracle_numeric_availability.csv"
RECOVERY_SUMMARY_FILENAME: Final = "rdx_c_recovery_summary.csv"
EVIDENCE_DISTRIBUTION_FILENAME: Final = "rdx_d_evidence_distribution.csv"
EVIDENCE_VALUE_COUNTS_FILENAME: Final = "rdx_d_evidence_value_counts.csv"

ORACLE_CLASSES: Final = (
    "A0_ONE_STEP_SUCCESS",
    "CORE_ACCESSIBLE_ONE_STEP_SUCCESS",
    "ORACLE_ONLY_A3_ONE_STEP_SUCCESS",
    "EXECUTED_MODEL_CHANGE_NO_ONE_STEP_SUCCESS",
    "MODEL_CHANGE_FEASIBLE_EXECUTION_FAILED",
    "NO_FEASIBLE_MODEL_CHANGE",
)
ORACLE_ACTIONS: Final = (
    "A0_NO_OP",
    "A1_RECALIBRATE",
    "A2_HEAD_UPDATE",
    "A3_FULL_FINE_TUNE",
    "A4_REPLAY_UPDATE",
)
PREDICTED_STATES: Final = (
    "PREDICTED_HARMFUL",
    "PREDICTED_UNCERTAIN",
    "PREDICTED_SAFE",
)
RECOGNITION_LABELS: Final = {
    "PREDICTED_HARMFUL": "RECOGNISED_HARM",
    "PREDICTED_UNCERTAIN": "UNCERTAIN_ON_HARM",
    "PREDICTED_SAFE": "SAFE_ON_HARM",
}
IMMEDIATE_OUTCOMES: Final = (
    "SAFE",
    "UNCERTAIN",
    "HARMFUL",
    "NOT_ASSESSED_CENSORED",
)
SUSTAINED_OUTCOMES: Final = (
    "SAFE",
    "NOT_SAFE_OBSERVED",
    "NOT_ASSESSED_CENSORED",
)

OUTPUT_SCHEMAS: Final[dict[str, tuple[str, ...]]] = {
    SOURCE_VALIDATION_FILENAME: (
        "source_table",
        "observed_row_count",
        "expected_row_count",
        "source_sha256",
        "validation_status",
    ),
    RECOGNITION_FILENAME: (
        "stratum_type",
        "method",
        "sequence",
        "seed",
        "current_domain",
        "predicted_health_state",
        "recognition_category",
        "count",
        "denominator",
        "proportion",
        "proportion_exact",
        "denominator_definition",
    ),
    CORE_PATHWAY_SUMMARY_FILENAME: (
        "indicator_type",
        "indicator",
        "count",
        "denominator",
        "proportion",
        "proportion_exact",
        "population_unit",
        "denominator_definition",
        "source_evidence",
    ),
    CORE_ATTEMPT_AUDIT_FILENAME: (
        "population_unit",
        "metric",
        "count",
        "denominator",
        "proportion",
        "proportion_exact",
        "denominator_definition",
        "source_evidence",
    ),
    ORACLE_CLASS_FILENAME: (
        "stratum_type",
        "sequence",
        "seed",
        "current_domain",
        "current_evaluator_state",
        "diagnostic_class",
        "count",
        "denominator",
        "proportion",
        "proportion_exact",
        "denominator_definition",
    ),
    ORACLE_CANDIDATE_SUMMARY_FILENAME: (
        "stratum_type",
        "sequence",
        "seed",
        "current_domain",
        "current_evaluator_state",
        "action",
        "candidate_count",
        "feasible_count",
        "feasible_proportion_all",
        "execution_success_count",
        "execution_success_proportion_all",
        "execution_success_denominator_feasible",
        "execution_success_proportion_given_feasible",
        "audit_admissible_count",
        "audit_admissible_proportion_all",
        "audit_admissible_denominator_executed",
        "audit_admissible_proportion_given_executed",
        "confirmed_success_count",
        "confirmed_success_proportion_all",
        "confirmed_success_denominator_audit_admissible",
        "confirmed_success_proportion_given_audit_admissible",
        "selected_count",
        "categorical_fields_complete",
        "numeric_successor_metrics_analyzed",
    ),
    ORACLE_SUCCESS_OVERLAP_FILENAME: (
        "successful_action_pattern",
        "decision_count",
        "denominator_decisions",
        "proportion",
        "proportion_exact",
        "note",
    ),
    ORACLE_NUMERIC_AVAILABILITY_FILENAME: (
        "action",
        "selection_state",
        "availability_state",
        "metric",
        "row_count",
        "available_count",
        "null_count",
        "note",
    ),
    RECOVERY_SUMMARY_FILENAME: (
        "stratum_type",
        "method",
        "action",
        "sequence",
        "seed",
        "current_domain",
        "horizon",
        "outcome",
        "count",
        "total_accepted_interventions",
        "assessable_interventions",
        "censored_interventions",
        "category_proportion_of_total",
        "category_proportion_of_total_exact",
        "assessable_proportion",
        "assessable_proportion_exact",
        "denominator_definition",
    ),
    EVIDENCE_DISTRIBUTION_FILENAME: (
        "stratum_type",
        "stratum_value",
        "metric",
        "n_available",
        "n_missing",
        "minimum",
        "q1",
        "median",
        "q3",
        "maximum",
        "mean",
        "zero_count",
        "unique_count",
        "quantile_definition",
        "association_warning",
    ),
    EVIDENCE_VALUE_COUNTS_FILENAME: (
        "metric",
        "value",
        "count",
        "denominator",
        "proportion",
        "proportion_exact",
        "denominator_definition",
    ),
}

ALL_OUTPUT_FILES: Final = frozenset(
    {
        *OUTPUT_SCHEMAS,
        CONTRACT_FILENAME,
        SUMMARY_FILENAME,
        REPORT_FILENAME,
        MANIFEST_FILENAME,
    }
)

EVIDENCE_METRICS: Final = (
    "optimizer_eligible_released_label_count",
    "current_scope_audit_escrow_row_count",
    "replay_row_count",
    "active_historical_audit_panel_count",
    "remaining_label_budget",
    "query_count",
)


class RdxAnalysisError(RuntimeError):
    """Raised when RDX-003 inputs or outputs violate the fixed analysis contract."""


@dataclass(frozen=True, slots=True)
class RdxAnalysisInputs:
    """Validated RDX-002 source bundle and its fixed tables."""

    root: Path
    manifest: dict[str, Any]
    contract: dict[str, Any]
    summary: dict[str, Any]
    tables: dict[str, pd.DataFrame]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RdxAnalysisError(f"cannot hash {path}: {exc}") from exc
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
    except (OSError, json.JSONDecodeError) as exc:
        raise RdxAnalysisError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RdxAnalysisError(f"{path} must contain a JSON object")
    return value


def _decimal_fraction(value: Fraction, places: int = 12) -> str:
    quantizer = Decimal(1).scaleb(-places)
    decimal = Decimal(value.numerator) / Decimal(value.denominator)
    text = format(decimal.quantize(quantizer), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _ratio(count: int, denominator: int) -> tuple[str, str]:
    if denominator <= 0:
        return "", ""
    exact = Fraction(count, denominator)
    return _decimal_fraction(exact), f"{exact.numerator}/{exact.denominator}"


def _optional_ratio(count: int, denominator: int) -> str:
    return _ratio(count, denominator)[0]


def _csv_text(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        extras = set(row).difference(columns)
        if extras:
            raise RdxAnalysisError(
                "analysis row contains unexpected columns: " + ", ".join(sorted(extras))
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


def _context(**values: object) -> dict[str, object]:
    result: dict[str, object] = {
        "method": "",
        "sequence": "",
        "seed": "",
        "current_domain": "",
        "current_evaluator_state": "",
    }
    result.update(values)
    return result


def _iter_groups(
    frame: pd.DataFrame,
    definitions: Sequence[tuple[str, tuple[str, ...]]],
) -> Iterable[tuple[str, dict[str, object], pd.DataFrame]]:
    for stratum_type, columns in definitions:
        if not columns:
            yield stratum_type, {}, frame
            continue
        grouper: str | list[str] = columns[0] if len(columns) == 1 else list(columns)
        for raw_key, group in frame.groupby(grouper, sort=True, dropna=False):
            keys = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            yield stratum_type, dict(zip(columns, keys, strict=True)), group


def load_rdx_analysis_inputs(source_dir: str | Path) -> RdxAnalysisInputs:
    """Source-validate and load exactly the canonical RDX-002 diagnostic bundle."""

    root = Path(source_dir).resolve()
    try:
        validate_rdx_evaluation(root)
    except (RdxArtifactError, FileNotFoundError, OSError, ValueError) as exc:
        raise RdxAnalysisError(f"RDX-002 source bundle is invalid: {exc}") from exc
    manifest = _load_json(root / RDX_MANIFEST_FILENAME)
    if manifest.get("bundle_digest") != EXPECTED_SOURCE_BUNDLE_DIGEST:
        raise RdxAnalysisError("RDX-002 bundle digest differs from the frozen RDX-003 source")
    summary = _load_json(root / "rdx_summary.json")
    contract = _load_json(root / "rdx_contract.json")
    if (
        summary.get("version") != RDX_EVALUATION_VERSION
        or summary.get("protocol_version") != RDX_PROTOCOL_VERSION
        or summary.get("status") != "VALIDATED_ARTIFACT_ONLY_DERIVATION"
    ):
        raise RdxAnalysisError("RDX-002 source identity differs from the frozen contract")

    tables: dict[str, pd.DataFrame] = {}
    for name, expected_count in EXPECTED_ROW_COUNTS.items():
        frame = pd.read_csv(root / name)
        if tuple(frame.columns) != RDX_OUTPUT_SCHEMAS[name]:
            raise RdxAnalysisError(f"{name} schema differs from RDX-002")
        if len(frame) != expected_count:
            raise RdxAnalysisError(
                f"{name} row count is {len(frame):,}, expected {expected_count:,}"
            )
        tables[name] = frame
    return RdxAnalysisInputs(
        root=root,
        manifest=manifest,
        contract=contract,
        summary=summary,
        tables=tables,
    )


def build_recognition_summary(harm: pd.DataFrame) -> list[dict[str, object]]:
    """Aggregate evaluator-HARMFUL windows without treating them as replicates."""

    if set(harm["evaluator_health_state"].astype(str)) != {"HARMFUL"}:
        raise RdxAnalysisError("RDX-A recognition source contains a non-HARMFUL evaluator state")
    observed_states = set(harm["predicted_health_state"].astype(str))
    if not observed_states.issubset(PREDICTED_STATES):
        raise RdxAnalysisError("RDX-A recognition source contains an unknown predicted state")
    definitions = (
        ("OVERALL", ()),
        ("METHOD", ("method",)),
        ("ROTATION", ("sequence",)),
        ("CURRENT_DOMAIN", ("current_domain",)),
        ("ROTATION_SEED", ("sequence", "seed")),
    )
    rows: list[dict[str, object]] = []
    for stratum_type, keys, group in _iter_groups(harm, definitions):
        denominator = len(group)
        for state in PREDICTED_STATES:
            count = int((group["predicted_health_state"] == state).sum())
            proportion, exact = _ratio(count, denominator)
            context = _context(**keys)
            rows.append(
                {
                    "stratum_type": stratum_type,
                    "method": context["method"],
                    "sequence": context["sequence"],
                    "seed": context["seed"],
                    "current_domain": context["current_domain"],
                    "predicted_health_state": state,
                    "recognition_category": RECOGNITION_LABELS[state],
                    "count": count,
                    "denominator": denominator,
                    "proportion": proportion,
                    "proportion_exact": exact,
                    "denominator_definition": "evaluator-HARMFUL decision windows in stratum",
                }
            )
    return rows


def _summary_row(
    *,
    population_unit: str,
    metric: str,
    count: int,
    denominator: int,
    denominator_definition: str,
    source_evidence: str,
) -> dict[str, object]:
    proportion, exact = _ratio(count, denominator)
    return {
        "population_unit": population_unit,
        "metric": metric,
        "count": count,
        "denominator": denominator,
        "proportion": proportion,
        "proportion_exact": exact,
        "denominator_definition": denominator_definition,
        "source_evidence": source_evidence,
    }


def build_core_pathway_summaries(
    pathways: pd.DataFrame,
    attempts: pd.DataFrame,
    audits: pd.DataFrame,
    evidence: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]], pd.DataFrame]:
    """Build overlapping Core pathway indicators from explicit window/attempt evidence."""

    key = ["experiment_id", "prediction_index"]
    if pathways.duplicated(key).any() or evidence.duplicated(key).any():
        raise RdxAnalysisError("Core pathway/evidence tables must be one-to-one by harmful window")
    if set(map(tuple, pathways[key].to_numpy())) != set(map(tuple, evidence[key].to_numpy())):
        raise RdxAnalysisError("Core pathway and evidence harmful-window coverage differs")

    attempt_window = (
        attempts.groupby(key, sort=True)
        .agg(a1_structurally_infeasible=("a1_infeasible", "any"))
        .reset_index()
    )
    recorded_attempts = attempts.loc[attempts["intervention_attempt_recorded"].astype(bool)]
    audit_window = (
        recorded_attempts.assign(
            audit_harmful=recorded_attempts["audit_result"].eq("HARMFUL"),
            audit_uncertain=recorded_attempts["audit_result"].eq("UNCERTAIN"),
        )
        .groupby(key, sort=True)
        .agg(
            audit_harmful_window=("audit_harmful", "any"),
            audit_uncertain_window=("audit_uncertain", "any"),
        )
        .reset_index()
    )
    merged = pathways.merge(evidence, on=key, validate="one_to_one", suffixes=("", "_evidence"))
    merged = merged.merge(attempt_window, on=key, how="left", validate="one_to_one")
    merged = merged.merge(audit_window, on=key, how="left", validate="one_to_one")
    for column in (
        "a1_structurally_infeasible",
        "audit_harmful_window",
        "audit_uncertain_window",
    ):
        merged[column] = merged[column].eq(True)

    denominator = len(merged)
    single_indicators: list[tuple[str, pd.Series, str]] = [
        ("RECOGNISED_HARM", merged["recognised_harm"], CORE_PATHWAY_FILENAME),
        (
            "PREDICTED_UNCERTAIN",
            merged["predicted_health_state"].eq("PREDICTED_UNCERTAIN"),
            CORE_PATHWAY_FILENAME,
        ),
        (
            "PREDICTED_SAFE",
            merged["predicted_health_state"].eq("PREDICTED_SAFE"),
            CORE_PATHWAY_FILENAME,
        ),
        ("QUERY_PENDING", merged["query_pending"], EVIDENCE_FILENAME),
        (
            "ZERO_OPTIMIZER_ELIGIBLE_EVIDENCE",
            merged["optimizer_eligible_released_label_count"].eq(0),
            EVIDENCE_FILENAME,
        ),
        (
            "MODEL_CHANGING_ACTION_ATTEMPTED",
            merged["model_changing_action_attempted"],
            CORE_PATHWAY_FILENAME,
        ),
        (
            "A1_STRUCTURALLY_INFEASIBLE",
            merged["a1_structurally_infeasible"],
            CORE_ATTEMPT_FILENAME,
        ),
        ("A2_ATTEMPTED", merged["a2_attempted"], CORE_PATHWAY_FILENAME),
        ("A4_ATTEMPTED", merged["a4_attempted"], CORE_PATHWAY_FILENAME),
        (
            "EXPLICIT_EXECUTION_FAILURE",
            merged["explicit_execution_failure_count"].gt(0),
            CORE_PATHWAY_FILENAME,
        ),
        ("AUDIT_HARMFUL_WINDOW", merged["audit_harmful_window"], CORE_ATTEMPT_FILENAME),
        (
            "AUDIT_UNCERTAIN_WINDOW",
            merged["audit_uncertain_window"],
            CORE_ATTEMPT_FILENAME,
        ),
        (
            "ACCEPTED_CANDIDATE",
            merged["candidate_accepted_count"].gt(0),
            CORE_PATHWAY_FILENAME,
        ),
        ("ROLLBACK", merged["rollback_count"].gt(0), CORE_PATHWAY_FILENAME),
    ]
    combinations: list[tuple[str, pd.Series, str, pd.Series, str]] = [
        (
            "RECOGNISED_HARM_AND_ZERO_OPTIMIZER_EVIDENCE",
            merged["recognised_harm"] & merged["optimizer_eligible_released_label_count"].eq(0),
            f"{CORE_PATHWAY_FILENAME}+{EVIDENCE_FILENAME}",
            merged["recognised_harm"],
            "recognised Core evaluator-HARMFUL windows",
        ),
        (
            "RECOGNISED_HARM_AND_EVIDENCE_AND_NO_MODEL_CHANGE_ATTEMPT",
            merged["recognised_harm"]
            & merged["optimizer_eligible_released_label_count"].gt(0)
            & ~merged["model_changing_action_attempted"],
            f"{CORE_PATHWAY_FILENAME}+{EVIDENCE_FILENAME}",
            merged["recognised_harm"],
            "recognised Core evaluator-HARMFUL windows",
        ),
        (
            "RECOGNISED_HARM_AND_MODEL_CHANGE_ATTEMPT",
            merged["recognised_harm"] & merged["model_changing_action_attempted"],
            CORE_PATHWAY_FILENAME,
            merged["recognised_harm"],
            "recognised Core evaluator-HARMFUL windows",
        ),
        (
            "MODEL_CHANGE_ATTEMPT_AND_AUDIT_HARMFUL",
            merged["model_changing_action_attempted"] & merged["audit_harmful_window"],
            f"{CORE_PATHWAY_FILENAME}+{CORE_ATTEMPT_FILENAME}",
            merged["model_changing_action_attempted"],
            "Core evaluator-HARMFUL windows with a model-changing attempt",
        ),
        (
            "MODEL_CHANGE_ATTEMPT_AND_ACCEPTED",
            merged["model_changing_action_attempted"] & merged["candidate_accepted_count"].gt(0),
            CORE_PATHWAY_FILENAME,
            merged["model_changing_action_attempted"],
            "Core evaluator-HARMFUL windows with a model-changing attempt",
        ),
    ]
    pathway_rows: list[dict[str, object]] = []
    for indicator, mask, source in single_indicators:
        count = int(mask.astype(bool).sum())
        proportion, exact = _ratio(count, denominator)
        pathway_rows.append(
            {
                "indicator_type": "OVERLAPPING_INDICATOR",
                "indicator": indicator,
                "count": count,
                "denominator": denominator,
                "proportion": proportion,
                "proportion_exact": exact,
                "population_unit": "CORE_EVALUATOR_HARMFUL_WINDOW",
                "denominator_definition": (
                    "all Core evaluator-HARMFUL decision windows; indicators may overlap"
                ),
                "source_evidence": source,
            }
        )
    for indicator, mask, source, denominator_mask, denominator_definition in combinations:
        count = int(mask.astype(bool).sum())
        combination_denominator = int(denominator_mask.astype(bool).sum())
        proportion, exact = _ratio(count, combination_denominator)
        pathway_rows.append(
            {
                "indicator_type": "DIRECTLY_SUPPORTED_COMBINATION",
                "indicator": indicator,
                "count": count,
                "denominator": combination_denominator,
                "proportion": proportion,
                "proportion_exact": exact,
                "population_unit": "CORE_EVALUATOR_HARMFUL_WINDOW",
                "denominator_definition": denominator_definition,
                "source_evidence": source,
            }
        )

    decision_denominator = len(attempts)
    attempt_denominator = len(recorded_attempts)
    attempt_rows = [
        _summary_row(
            population_unit="CORE_DECISION",
            metric="A1_STRUCTURALLY_INFEASIBLE",
            count=int(attempts["a1_infeasible"].astype(bool).sum()),
            denominator=decision_denominator,
            denominator_definition="all persisted Core decisions at evaluator-HARMFUL windows",
            source_evidence=CORE_ATTEMPT_FILENAME,
        ),
        _summary_row(
            population_unit="MODEL_CHANGE_ATTEMPT",
            metric="EXPLICIT_EXECUTION_FAILURE",
            count=int(recorded_attempts["explicit_execution_failure"].eq(True).sum()),
            denominator=attempt_denominator,
            denominator_definition="persisted Core intervention-attempt records only",
            source_evidence=CORE_ATTEMPT_FILENAME,
        ),
        _summary_row(
            population_unit="MODEL_CHANGE_ATTEMPT",
            metric="AUDIT_HARMFUL",
            count=int(recorded_attempts["audit_result"].eq("HARMFUL").sum()),
            denominator=attempt_denominator,
            denominator_definition="persisted Core intervention-attempt records only",
            source_evidence=CORE_ATTEMPT_FILENAME,
        ),
        _summary_row(
            population_unit="MODEL_CHANGE_ATTEMPT",
            metric="AUDIT_UNCERTAIN",
            count=int(recorded_attempts["audit_result"].eq("UNCERTAIN").sum()),
            denominator=attempt_denominator,
            denominator_definition="persisted Core intervention-attempt records only",
            source_evidence=CORE_ATTEMPT_FILENAME,
        ),
        _summary_row(
            population_unit="MODEL_CHANGE_ATTEMPT",
            metric="ACCEPTED",
            count=int(recorded_attempts["accepted"].eq(True).sum()),
            denominator=attempt_denominator,
            denominator_definition="persisted Core intervention-attempt records only",
            source_evidence=CORE_ATTEMPT_FILENAME,
        ),
        _summary_row(
            population_unit="MODEL_CHANGE_ATTEMPT",
            metric="ROLLED_BACK",
            count=int(recorded_attempts["rolled_back"].eq(True).sum()),
            denominator=attempt_denominator,
            denominator_definition="persisted Core intervention-attempt records only",
            source_evidence=CORE_ATTEMPT_FILENAME,
        ),
    ]
    for purpose, population_unit in (
        ("candidate_guard", "CANDIDATE_GUARD_PANEL"),
        ("incident_reset", "INCIDENT_RESET_PANEL"),
    ):
        panels = audits.loc[audits["purpose"].eq(purpose)]
        if panels.empty:
            raise RdxAnalysisError(f"Core audit table lacks explicit {purpose} panels")
        for state in ("HARMFUL", "UNCERTAIN"):
            attempt_rows.append(
                _summary_row(
                    population_unit=population_unit,
                    metric=f"AUDIT_{state}",
                    count=int(panels["audit_state"].eq(state).sum()),
                    denominator=len(panels),
                    denominator_definition=f"explicit persisted {purpose} audit panels only",
                    source_evidence=CORE_AUDIT_FILENAME,
                )
            )
    return pathway_rows, attempt_rows, merged


def build_oracle_class_summary(decisions: pd.DataFrame) -> list[dict[str, object]]:
    """Aggregate the mutually exclusive frozen six-class Oracle map."""

    observed = set(decisions["diagnostic_class"].astype(str))
    if not observed.issubset(ORACLE_CLASSES):
        raise RdxAnalysisError("Oracle decision table contains an unknown diagnostic class")
    definitions = (
        ("OVERALL", ()),
        ("ROTATION", ("sequence",)),
        ("CURRENT_DOMAIN", ("current_domain",)),
        ("ROTATION_SEED", ("sequence", "seed")),
        ("CURRENT_EVALUATOR_STATE", ("current_evaluator_state",)),
    )
    rows: list[dict[str, object]] = []
    for stratum_type, keys, group in _iter_groups(decisions, definitions):
        denominator = len(group)
        context = _context(**keys)
        for diagnostic_class in ORACLE_CLASSES:
            count = int(group["diagnostic_class"].eq(diagnostic_class).sum())
            proportion, exact = _ratio(count, denominator)
            rows.append(
                {
                    "stratum_type": stratum_type,
                    "sequence": context["sequence"],
                    "seed": context["seed"],
                    "current_domain": context["current_domain"],
                    "current_evaluator_state": context["current_evaluator_state"],
                    "diagnostic_class": diagnostic_class,
                    "count": count,
                    "denominator": denominator,
                    "proportion": proportion,
                    "proportion_exact": exact,
                    "denominator_definition": "eligible Offline-Oracle decisions in stratum",
                }
            )
    return rows


def build_oracle_candidate_summaries(
    candidates: pd.DataFrame,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Keep complete categorical branches separate from selected numeric availability."""

    categorical = (
        "feasible",
        "execution_succeeded",
        "audit_admissible",
        "confirmed_success",
        "selected",
    )
    if candidates[list(categorical)].isna().any().any():
        raise RdxAnalysisError("Oracle categorical candidate results contain null values")
    definitions = (
        ("OVERALL", ()),
        ("ROTATION", ("sequence",)),
        ("CURRENT_DOMAIN", ("current_domain",)),
        ("ROTATION_SEED", ("sequence", "seed")),
        ("CURRENT_EVALUATOR_STATE", ("current_evaluator_state",)),
    )
    rows: list[dict[str, object]] = []
    for stratum_type, keys, stratum in _iter_groups(candidates, definitions):
        context = _context(**keys)
        for action in ORACLE_ACTIONS:
            group = stratum.loc[stratum["action"].eq(action)]
            candidate_count = len(group)
            if candidate_count == 0:
                raise RdxAnalysisError(
                    f"Oracle candidate stratum {stratum_type} lacks action {action}"
                )
            feasible = int(group["feasible"].astype(bool).sum())
            executed = int(group["execution_succeeded"].astype(bool).sum())
            admissible = int(group["audit_admissible"].astype(bool).sum())
            successful = int(group["confirmed_success"].astype(bool).sum())
            selected = int(group["selected"].astype(bool).sum())
            rows.append(
                {
                    "stratum_type": stratum_type,
                    "sequence": context["sequence"],
                    "seed": context["seed"],
                    "current_domain": context["current_domain"],
                    "current_evaluator_state": context["current_evaluator_state"],
                    "action": action,
                    "candidate_count": candidate_count,
                    "feasible_count": feasible,
                    "feasible_proportion_all": _optional_ratio(feasible, candidate_count),
                    "execution_success_count": executed,
                    "execution_success_proportion_all": _optional_ratio(executed, candidate_count),
                    "execution_success_denominator_feasible": feasible,
                    "execution_success_proportion_given_feasible": _optional_ratio(
                        executed, feasible
                    ),
                    "audit_admissible_count": admissible,
                    "audit_admissible_proportion_all": _optional_ratio(admissible, candidate_count),
                    "audit_admissible_denominator_executed": executed,
                    "audit_admissible_proportion_given_executed": _optional_ratio(
                        admissible, executed
                    ),
                    "confirmed_success_count": successful,
                    "confirmed_success_proportion_all": _optional_ratio(
                        successful, candidate_count
                    ),
                    "confirmed_success_denominator_audit_admissible": admissible,
                    "confirmed_success_proportion_given_audit_admissible": _optional_ratio(
                        successful, admissible
                    ),
                    "selected_count": selected,
                    "categorical_fields_complete": True,
                    "numeric_successor_metrics_analyzed": False,
                }
            )

    decision_key = ["experiment_id", "prediction_index", "decision_id"]
    overlap: dict[str, int] = {}
    for _, group in candidates.groupby(decision_key, sort=True):
        successful_actions = tuple(
            action
            for action in ORACLE_ACTIONS
            if bool(group.loc[group["action"].eq(action), "confirmed_success"].iloc[0])
        )
        pattern = "+".join(successful_actions) if successful_actions else "NONE"
        overlap[pattern] = overlap.get(pattern, 0) + 1
    decision_count = sum(overlap.values())
    overlap_rows: list[dict[str, object]] = []
    for pattern, count in sorted(overlap.items(), key=lambda item: (-item[1], item[0])):
        proportion, exact = _ratio(count, decision_count)
        overlap_rows.append(
            {
                "successful_action_pattern": pattern,
                "decision_count": count,
                "denominator_decisions": decision_count,
                "proportion": proportion,
                "proportion_exact": exact,
                "note": (
                    "per-action success counts overlap; these mutually exclusive patterns "
                    "expose that overlap"
                ),
            }
        )

    numeric_metrics = (
        "successor_attack_count",
        "successor_benign_count",
        "successor_tpr",
        "successor_fpr",
        "successor_false_positives_per_million",
        "successor_fpr_budget_ratio",
    )
    availability_rows: list[dict[str, object]] = []
    for action in ORACLE_ACTIONS:
        action_rows = candidates.loc[candidates["action"].eq(action)]
        for selected_state, selected_rows in action_rows.groupby("selected", sort=False):
            selection = "SELECTED" if bool(selected_state) else "UNSELECTED"
            availability_counts = selected_rows["numeric_successor_availability"].value_counts(
                dropna=False
            )
            for availability_state, count in sorted(
                availability_counts.items(), key=lambda item: str(item[0])
            ):
                availability_rows.append(
                    {
                        "action": action,
                        "selection_state": selection,
                        "availability_state": str(availability_state),
                        "metric": "ALL_NUMERIC_SUCCESSOR_FIELDS",
                        "row_count": int(count),
                        "available_count": "",
                        "null_count": "",
                        "note": "availability state; unselected numeric metrics remain unavailable",
                    }
                )
            for metric in numeric_metrics:
                available = int(selected_rows[metric].notna().sum())
                null = int(selected_rows[metric].isna().sum())
                availability_rows.append(
                    {
                        "action": action,
                        "selection_state": selection,
                        "availability_state": "FIELD_COMPLETENESS",
                        "metric": metric,
                        "row_count": len(selected_rows),
                        "available_count": available,
                        "null_count": null,
                        "note": "null is preserved and is never converted to zero",
                    }
                )
    return rows, overlap_rows, availability_rows


def build_recovery_summary(recovery: pd.DataFrame) -> list[dict[str, object]]:
    """Summarise later deployed outcomes while excluding censoring from failure rates."""

    definitions = (
        ("OVERALL", ()),
        ("METHOD", ("method",)),
        ("ACTION", ("action",)),
        ("ROTATION", ("sequence",)),
        ("CURRENT_DOMAIN", ("current_domain",)),
        ("ROTATION_SEED", ("sequence", "seed")),
    )
    horizons = (
        ("IMMEDIATE", "immediate_outcome", IMMEDIATE_OUTCOMES),
        ("TWO_WINDOW_SUSTAINED", "sustained_outcome", SUSTAINED_OUTCOMES),
    )
    rows: list[dict[str, object]] = []
    for stratum_type, keys, group in _iter_groups(recovery, definitions):
        context = _context(**keys)
        for horizon, column, outcomes in horizons:
            if group[column].isna().any():
                raise RdxAnalysisError(f"{horizon} recovery contains a null outcome")
            raw = group[column].astype(str)
            observed = set(raw)
            if not observed.issubset(outcomes):
                raise RdxAnalysisError(f"{horizon} recovery contains an unknown outcome")
            total = len(group)
            censored = int(raw.eq("NOT_ASSESSED_CENSORED").sum())
            assessable = total - censored
            for outcome in outcomes:
                count = int(raw.eq(outcome).sum())
                total_proportion, total_exact = _ratio(count, total)
                if outcome == "NOT_ASSESSED_CENSORED":
                    assessable_proportion = ""
                    assessable_exact = ""
                else:
                    assessable_proportion, assessable_exact = _ratio(count, assessable)
                rows.append(
                    {
                        "stratum_type": stratum_type,
                        "method": context["method"],
                        "action": keys.get("action", ""),
                        "sequence": context["sequence"],
                        "seed": context["seed"],
                        "current_domain": context["current_domain"],
                        "horizon": horizon,
                        "outcome": outcome,
                        "count": count,
                        "total_accepted_interventions": total,
                        "assessable_interventions": assessable,
                        "censored_interventions": censored,
                        "category_proportion_of_total": total_proportion,
                        "category_proportion_of_total_exact": total_exact,
                        "assessable_proportion": assessable_proportion,
                        "assessable_proportion_exact": assessable_exact,
                        "denominator_definition": (
                            "category proportion uses all accepted interventions; non-censored "
                            "outcomes additionally use assessable interventions only"
                        ),
                    }
                )
    return rows


def _linear_quantile(values: Sequence[int], quantile: Fraction) -> Fraction:
    if not values:
        raise RdxAnalysisError("cannot calculate a quantile from no values")
    ordered = sorted(values)
    position = Fraction(len(ordered) - 1) * quantile
    lower = position.numerator // position.denominator
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return Fraction(ordered[lower]) * (1 - weight) + Fraction(ordered[upper]) * weight


def _number_text(value: Fraction) -> str:
    return _decimal_fraction(value) if value.denominator != 1 else str(value.numerator)


def _evidence_distribution_row(
    *,
    stratum_type: str,
    stratum_value: str,
    metric: str,
    series: pd.Series,
) -> dict[str, object]:
    available = series.dropna()
    if available.empty:
        return {
            "stratum_type": stratum_type,
            "stratum_value": stratum_value,
            "metric": metric,
            "n_available": 0,
            "n_missing": len(series),
            "minimum": "",
            "q1": "",
            "median": "",
            "q3": "",
            "maximum": "",
            "mean": "",
            "zero_count": 0,
            "unique_count": 0,
            "quantile_definition": "linear interpolation at (n-1)p",
            "association_warning": "descriptive association only; no causal interpretation",
        }
    numbers = [int(value) for value in available.tolist()]
    return {
        "stratum_type": stratum_type,
        "stratum_value": stratum_value,
        "metric": metric,
        "n_available": len(numbers),
        "n_missing": len(series) - len(numbers),
        "minimum": min(numbers),
        "q1": _number_text(_linear_quantile(numbers, Fraction(1, 4))),
        "median": _number_text(_linear_quantile(numbers, Fraction(1, 2))),
        "q3": _number_text(_linear_quantile(numbers, Fraction(3, 4))),
        "maximum": max(numbers),
        "mean": _number_text(Fraction(sum(numbers), len(numbers))),
        "zero_count": sum(value == 0 for value in numbers),
        "unique_count": len(set(numbers)),
        "quantile_definition": "linear interpolation at (n-1)p",
        "association_warning": "descriptive association only; no causal interpretation",
    }


def build_evidence_summaries(
    merged_core: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Describe decision-time evidence without imputing unavailable values."""

    frame = merged_core.copy()
    frame["model_change_pathway"] = frame["model_changing_action_attempted"].map(
        {True: "ATTEMPTED", False: "NOT_ATTEMPTED"}
    )
    frame["candidate_acceptance_pathway"] = (
        frame["candidate_accepted_count"].gt(0).map({True: "ACCEPTED", False: "NOT_ACCEPTED"})
    )
    definitions = (
        ("OVERALL", (), "ALL"),
        ("PREDICTED_STATE", ("predicted_health_state",), None),
        ("MODEL_CHANGE_PATHWAY", ("model_change_pathway",), None),
        ("CANDIDATE_ACCEPTANCE_PATHWAY", ("candidate_acceptance_pathway",), None),
    )
    distribution_rows: list[dict[str, object]] = []
    for stratum_type, columns, fixed_value in definitions:
        groups = _iter_groups(frame, ((stratum_type, columns),))
        for _, keys, group in groups:
            if fixed_value is None:
                stratum_value = str(next(iter(keys.values())))
            else:
                stratum_value = fixed_value
            for metric in EVIDENCE_METRICS:
                distribution_rows.append(
                    _evidence_distribution_row(
                        stratum_type=stratum_type,
                        stratum_value=stratum_value,
                        metric=metric,
                        series=group[metric],
                    )
                )

    value_rows: list[dict[str, object]] = []
    discrete_metrics = (*EVIDENCE_METRICS, "query_pending")
    denominator = len(frame)
    for metric in discrete_metrics:
        available = frame[metric].dropna()
        for value, count_raw in sorted(
            available.value_counts(dropna=False).items(), key=lambda item: str(item[0])
        ):
            count = int(count_raw)
            proportion, exact = _ratio(count, denominator)
            raw_value: Any = value
            if isinstance(raw_value, bool):
                rendered_value = "true" if raw_value else "false"
            elif isinstance(raw_value, float) and raw_value.is_integer():
                rendered_value = str(int(raw_value))
            else:
                rendered_value = str(raw_value)
            value_rows.append(
                {
                    "metric": metric,
                    "value": rendered_value,
                    "count": count,
                    "denominator": denominator,
                    "proportion": proportion,
                    "proportion_exact": exact,
                    "denominator_definition": (
                        "all Core evaluator-HARMFUL evidence snapshots; nulls are not zero"
                    ),
                }
            )
    return distribution_rows, value_rows


def _derive_analysis_tables(
    inputs: RdxAnalysisInputs,
) -> tuple[dict[str, list[dict[str, object]]], pd.DataFrame]:
    manifest_files = inputs.manifest.get("files")
    if not isinstance(manifest_files, dict):
        raise RdxAnalysisError("RDX-002 manifest lacks its source file digest map")
    source_rows: list[dict[str, object]] = []
    for name, expected in sorted(EXPECTED_ROW_COUNTS.items()):
        source_rows.append(
            {
                "source_table": name,
                "observed_row_count": len(inputs.tables[name]),
                "expected_row_count": expected,
                "source_sha256": str(manifest_files.get(name, "")),
                "validation_status": "VALIDATED_EXACT",
            }
        )

    pathway_rows, attempt_rows, merged = build_core_pathway_summaries(
        inputs.tables[CORE_PATHWAY_FILENAME],
        inputs.tables[CORE_ATTEMPT_FILENAME],
        inputs.tables[CORE_AUDIT_FILENAME],
        inputs.tables[EVIDENCE_FILENAME],
    )
    candidate_rows, overlap_rows, numeric_rows = build_oracle_candidate_summaries(
        inputs.tables[ORACLE_CANDIDATE_FILENAME]
    )
    evidence_distribution, evidence_values = build_evidence_summaries(merged)
    tables = {
        SOURCE_VALIDATION_FILENAME: source_rows,
        RECOGNITION_FILENAME: build_recognition_summary(inputs.tables[HARM_FILENAME]),
        CORE_PATHWAY_SUMMARY_FILENAME: pathway_rows,
        CORE_ATTEMPT_AUDIT_FILENAME: attempt_rows,
        ORACLE_CLASS_FILENAME: build_oracle_class_summary(inputs.tables[ORACLE_DECISION_FILENAME]),
        ORACLE_CANDIDATE_SUMMARY_FILENAME: candidate_rows,
        ORACLE_SUCCESS_OVERLAP_FILENAME: overlap_rows,
        ORACLE_NUMERIC_AVAILABILITY_FILENAME: numeric_rows,
        RECOVERY_SUMMARY_FILENAME: build_recovery_summary(inputs.tables[RECOVERY_FILENAME]),
        EVIDENCE_DISTRIBUTION_FILENAME: evidence_distribution,
        EVIDENCE_VALUE_COUNTS_FILENAME: evidence_values,
    }
    if set(tables) != set(OUTPUT_SCHEMAS):
        raise RdxAnalysisError("derived RDX-003 table set differs from the fixed contract")
    return tables, merged


def _matching_row(
    rows: Sequence[Mapping[str, Any]],
    **criteria: object,
) -> Mapping[str, Any]:
    matched = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matched) != 1:
        rendered = ", ".join(f"{key}={value!r}" for key, value in criteria.items())
        raise RdxAnalysisError(f"analysis table lacks exactly one row for {rendered}")
    return matched[0]


def _sum_matching(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_column: str,
    **criteria: object,
) -> int:
    return sum(
        int(row[value_column])
        for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    )


def _build_contract(inputs: RdxAnalysisInputs) -> dict[str, object]:
    return {
        "version": RDX_ANALYSIS_VERSION,
        "source_rdx_bundle": {
            "path": str(inputs.root),
            "evaluation_version": RDX_EVALUATION_VERSION,
            "protocol_version": RDX_PROTOCOL_VERSION,
            "bundle_digest": inputs.manifest["bundle_digest"],
            "manifest_sha256": _file_sha256(inputs.root / RDX_MANIFEST_FILENAME),
            "contract_sha256": _file_sha256(inputs.root / "rdx_contract.json"),
            "summary_sha256": _file_sha256(inputs.root / "rdx_summary.json"),
            "validated_before_analysis": True,
            "expected_row_counts": EXPECTED_ROW_COUNTS,
        },
        "analysis_contract": {
            "source_only": "validated RDX-002 bundle; no raw data or model operation",
            "rdx_a_primary_population": "evaluator-HARMFUL decision windows",
            "comparative_unit": "rotation x seed",
            "window_counts": "descriptive, not independent inferential replicates",
            "core_pathways": "overlapping indicators and directly supported combinations",
            "oracle_scope": "frozen A0-A4 one-step outcomes, not global recoverability",
            "oracle_numeric_scope": "selected branches only; no unselected numeric imputation",
            "recovery_censoring": "excluded from recovery-failure denominators",
            "evidence_interpretation": "descriptive association only, not causal",
            "quantiles": "linear interpolation at (n-1)p",
            "new_experiment_authorized": False,
            "danids_hypotheses_modified": False,
            "rdx001_definitions_modified": False,
        },
        "outputs": {
            "csv_schemas": {
                name: list(columns) for name, columns in sorted(OUTPUT_SCHEMAS.items())
            },
            "json_files": [CONTRACT_FILENAME, SUMMARY_FILENAME, MANIFEST_FILENAME],
            "markdown_files": [REPORT_FILENAME],
            "canonical_json": "sorted_keys_indent_2_lf_no_nan",
            "canonical_csv": "fixed_columns_lf_blank_for_null",
        },
    }


def _build_summary(
    inputs: RdxAnalysisInputs,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, object]:
    recognition = tables[RECOGNITION_FILENAME]
    pathways = tables[CORE_PATHWAY_SUMMARY_FILENAME]
    oracle_classes = tables[ORACLE_CLASS_FILENAME]
    recovery = tables[RECOVERY_SUMMARY_FILENAME]
    evidence = tables[EVIDENCE_DISTRIBUTION_FILENAME]
    evidence_values = tables[EVIDENCE_VALUE_COUNTS_FILENAME]

    overall_recognition = {
        state: int(
            _matching_row(
                recognition,
                stratum_type="OVERALL",
                predicted_health_state=state,
            )["count"]
        )
        for state in PREDICTED_STATES
    }
    core_pathways = {
        indicator: int(_matching_row(pathways, indicator=indicator)["count"])
        for indicator in (
            "RECOGNISED_HARM",
            "MODEL_CHANGING_ACTION_ATTEMPTED",
            "A1_STRUCTURALLY_INFEASIBLE",
            "ACCEPTED_CANDIDATE",
            "ROLLBACK",
            "RECOGNISED_HARM_AND_EVIDENCE_AND_NO_MODEL_CHANGE_ATTEMPT",
        )
    }
    overall_oracle_classes = {
        diagnostic_class: int(
            _matching_row(
                oracle_classes,
                stratum_type="OVERALL",
                diagnostic_class=diagnostic_class,
            )["count"]
        )
        for diagnostic_class in ORACLE_CLASSES
    }
    successful_classes = ORACLE_CLASSES[:3]
    any_oracle_success = sum(overall_oracle_classes[item] for item in successful_classes)
    harmful_oracle_success = sum(
        _sum_matching(
            oracle_classes,
            value_column="count",
            stratum_type="CURRENT_EVALUATOR_STATE",
            current_evaluator_state="HARMFUL",
            diagnostic_class=item,
        )
        for item in successful_classes
    )
    harmful_oracle_total = int(
        _matching_row(
            oracle_classes,
            stratum_type="CURRENT_EVALUATOR_STATE",
            current_evaluator_state="HARMFUL",
            diagnostic_class=ORACLE_CLASSES[0],
        )["denominator"]
    )

    immediate = {
        outcome: int(
            _matching_row(
                recovery,
                stratum_type="OVERALL",
                horizon="IMMEDIATE",
                outcome=outcome,
            )["count"]
        )
        for outcome in IMMEDIATE_OUTCOMES
    }
    sustained = {
        outcome: int(
            _matching_row(
                recovery,
                stratum_type="OVERALL",
                horizon="TWO_WINDOW_SUSTAINED",
                outcome=outcome,
            )["count"]
        )
        for outcome in SUSTAINED_OUTCOMES
    }
    sustained_assessable = int(
        _matching_row(
            recovery,
            stratum_type="OVERALL",
            horizon="TWO_WINDOW_SUSTAINED",
            outcome="SAFE",
        )["assessable_interventions"]
    )

    evidence_headline: dict[str, object] = {}
    for metric in EVIDENCE_METRICS:
        row = _matching_row(
            evidence,
            stratum_type="OVERALL",
            stratum_value="ALL",
            metric=metric,
        )
        evidence_headline[metric] = {
            "median": row["median"],
            "q1": row["q1"],
            "q3": row["q3"],
            "minimum": row["minimum"],
            "maximum": row["maximum"],
            "zero_count": int(row["zero_count"]),
        }
    pending_false = int(
        _matching_row(evidence_values, metric="query_pending", value="false")["count"]
    )
    return {
        "version": RDX_ANALYSIS_VERSION,
        "status": "VALIDATED_ARTIFACT_ONLY_SCIENTIFIC_ANALYSIS",
        "source_bundle": {
            "path": str(inputs.root),
            "bundle_digest": inputs.manifest["bundle_digest"],
            "source_version": RDX_EVALUATION_VERSION,
            "protocol_version": RDX_PROTOCOL_VERSION,
            "row_counts": EXPECTED_ROW_COUNTS,
            "validated_before_analysis": True,
        },
        "rdx_a_recognition": {
            "denominator_evaluator_harmful_windows": sum(overall_recognition.values()),
            "counts_by_predicted_state": overall_recognition,
        },
        "rdx_ad_core_pathway": {
            "denominator_core_evaluator_harmful_windows": len(inputs.tables[CORE_PATHWAY_FILENAME]),
            "headline_counts": core_pathways,
        },
        "rdx_b_oracle": {
            "denominator_decisions": len(inputs.tables[ORACLE_DECISION_FILENAME]),
            "class_counts": overall_oracle_classes,
            "any_confirmed_one_step_success_count": any_oracle_success,
            "current_harmful_decisions": harmful_oracle_total,
            "current_harmful_any_success_count": harmful_oracle_success,
        },
        "rdx_c_deployed_recovery": {
            "accepted_interventions": len(inputs.tables[RECOVERY_FILENAME]),
            "immediate_counts": immediate,
            "sustained_counts": sustained,
            "sustained_assessable_interventions": sustained_assessable,
        },
        "rdx_d_evidence": {
            "distributions": evidence_headline,
            "query_pending_false_count": pending_false,
        },
        "diagnostic_synthesis": {
            "upstream_or_downstream": (
                "Substantial downstream failure remains after same-window harm recognition; "
                "the artifact cannot assign a causal share to policy gating versus action efficacy."
            ),
            "oracle_one_step_map": (
                "Confirmed A0-A4 one-step success is sparse, especially when the current "
                "evaluator state is HARMFUL."
            ),
            "failure_location": (
                "Explicit feasible-execution failure is absent, while many executed candidates "
                "lack one-step success and accepted deployed actions rarely recover."
            ),
            "unresolved_causal_question": (
                "Whether more legitimately released training evidence expands recoverability, "
                "or whether the frozen model/action capability remains the limiting factor."
            ),
        },
        "recommended_next_controlled_question": {
            "selection": "INCREASED_TRAINING_LABEL_BUDGET",
            "question": (
                "Does increasing the legitimately released training-label budget improve "
                "one-step and sustained recovery under otherwise unchanged chronology, audit, "
                "and A0-A4 semantics?"
            ),
            "competing_explanations": (
                "training-evidence limitation versus model/action-space capability limitation"
            ),
            "recommendation_only": True,
        },
        "new_experiment_authorized": False,
        "danids_hypotheses_modified": False,
        "rdx001_definitions_modified": False,
        "source_artifacts_mutated": False,
    }


def _percent_text(proportion: Any) -> str:
    if proportion in {None, ""}:
        return "NA"
    return f"{100.0 * float(proportion):.2f}%"


def _count_rate(row: Mapping[str, Any]) -> str:
    return f"{int(row['count']):,}/{int(row['denominator']):,} ({_percent_text(row['proportion'])})"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    rendered = ["| " + " | ".join(headers) + " |"]
    rendered.append("| " + " | ".join("---" for _ in headers) + " |")
    rendered.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(rendered)


def _recognition_table(rows: Sequence[Mapping[str, Any]], stratum_type: str) -> str:
    keys: list[tuple[Any, ...]]
    headers: tuple[str, ...]
    selected = [row for row in rows if row["stratum_type"] == stratum_type]
    if stratum_type == "METHOD":
        keys = [(str(row["method"]),) for row in selected]
        headers = ("Method",)
    elif stratum_type == "ROTATION":
        keys = [(str(row["sequence"]),) for row in selected]
        headers = ("Rotation",)
    elif stratum_type == "CURRENT_DOMAIN":
        keys = [(str(row["current_domain"]),) for row in selected]
        headers = ("Domain",)
    elif stratum_type == "ROTATION_SEED":
        keys = [(str(row["sequence"]), row["seed"]) for row in selected]
        headers = ("Rotation", "Seed")
    else:
        raise RdxAnalysisError(f"unsupported recognition report stratum {stratum_type}")
    unique_keys = sorted(set(keys))
    output: list[list[object]] = []
    for key in unique_keys:
        criteria: dict[str, object] = {"stratum_type": stratum_type}
        if stratum_type == "METHOD":
            criteria["method"] = key[0]
        elif stratum_type == "ROTATION":
            criteria["sequence"] = key[0]
        elif stratum_type == "CURRENT_DOMAIN":
            criteria["current_domain"] = key[0]
        else:
            criteria.update(sequence=key[0], seed=key[1])
        state_rows = [
            _matching_row(rows, **criteria, predicted_health_state=state)
            for state in PREDICTED_STATES
        ]
        output.append([*key, *(_count_rate(row) for row in state_rows)])
    return _markdown_table(
        (*headers, "PREDICTED_HARMFUL", "PREDICTED_UNCERTAIN", "PREDICTED_SAFE"),
        output,
    )


def _oracle_class_table(rows: Sequence[Mapping[str, Any]], stratum_type: str) -> str:
    keys: list[tuple[Any, ...]]
    headers: tuple[str, ...]
    selected = [row for row in rows if row["stratum_type"] == stratum_type]
    if stratum_type == "OVERALL":
        keys = [("Overall",)]
        headers = ("Stratum",)
    elif stratum_type == "ROTATION":
        keys = sorted({(str(row["sequence"]),) for row in selected})
        headers = ("Rotation",)
    elif stratum_type == "CURRENT_DOMAIN":
        keys = sorted({(str(row["current_domain"]),) for row in selected})
        headers = ("Domain",)
    elif stratum_type == "ROTATION_SEED":
        keys = sorted({(str(row["sequence"]), row["seed"]) for row in selected})
        headers = ("Rotation", "Seed")
    else:
        raise RdxAnalysisError(f"unsupported Oracle report stratum {stratum_type}")
    output: list[list[object]] = []
    for key in keys:
        criteria: dict[str, object] = {"stratum_type": stratum_type}
        if stratum_type == "ROTATION":
            criteria["sequence"] = key[0]
        elif stratum_type == "CURRENT_DOMAIN":
            criteria["current_domain"] = key[0]
        elif stratum_type == "ROTATION_SEED":
            criteria.update(sequence=key[0], seed=key[1])
        class_rows = [
            _matching_row(rows, **criteria, diagnostic_class=item) for item in ORACLE_CLASSES
        ]
        output.append([*key, *(_count_rate(row) for row in class_rows)])
    return _markdown_table(
        (
            *headers,
            "A0 success",
            "Core-accessible success",
            "A3-only success",
            "Executed/no success",
            "Feasible/execution failed",
            "No feasible model change",
        ),
        output,
    )


def _recovery_table(rows: Sequence[Mapping[str, Any]], stratum_type: str) -> str:
    keys: list[tuple[Any, ...]]
    headers: tuple[str, ...]
    selected = [row for row in rows if row["stratum_type"] == stratum_type]
    context_column = {
        "OVERALL": None,
        "METHOD": "method",
        "ACTION": "action",
        "ROTATION": "sequence",
        "CURRENT_DOMAIN": "current_domain",
    }.get(stratum_type)
    if stratum_type == "ROTATION_SEED":
        keys = sorted({(str(row["sequence"]), row["seed"]) for row in selected})
        headers = ("Rotation", "Seed")
    elif stratum_type == "OVERALL":
        keys = [("Overall",)]
        headers = ("Stratum",)
    elif context_column is not None:
        keys = sorted({(str(row[context_column]),) for row in selected})
        headers = (context_column.replace("_", " ").title(),)
    else:
        raise RdxAnalysisError(f"unsupported recovery report stratum {stratum_type}")

    output: list[list[object]] = []
    for key in keys:
        criteria: dict[str, object] = {"stratum_type": stratum_type}
        if stratum_type == "ROTATION_SEED":
            criteria.update(sequence=key[0], seed=key[1])
        elif context_column is not None:
            criteria[context_column] = key[0]
        immediate = [
            _matching_row(rows, **criteria, horizon="IMMEDIATE", outcome=outcome)
            for outcome in IMMEDIATE_OUTCOMES
        ]
        sustained = [
            _matching_row(rows, **criteria, horizon="TWO_WINDOW_SUSTAINED", outcome=outcome)
            for outcome in SUSTAINED_OUTCOMES
        ]
        immediate_cells = [
            f"{int(row['count']):,}/{int(row['assessable_interventions']):,} "
            f"({_percent_text(row['assessable_proportion'])})"
            if row["outcome"] != "NOT_ASSESSED_CENSORED"
            else f"{int(row['count']):,}/{int(row['total_accepted_interventions']):,}"
            for row in immediate
        ]
        sustained_cells = [
            f"{int(row['count']):,}/{int(row['assessable_interventions']):,} "
            f"({_percent_text(row['assessable_proportion'])})"
            if row["outcome"] != "NOT_ASSESSED_CENSORED"
            else f"{int(row['count']):,}/{int(row['total_accepted_interventions']):,}"
            for row in sustained
        ]
        output.append([*key, *immediate_cells, *sustained_cells])
    return _markdown_table(
        (
            *headers,
            "Immediate SAFE",
            "Immediate UNCERTAIN",
            "Immediate HARMFUL",
            "Immediate censored",
            "Sustained SAFE",
            "Sustained NOT_SAFE",
            "Sustained censored",
        ),
        output,
    )


def _report_text(
    inputs: RdxAnalysisInputs,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    summary: Mapping[str, Any],
) -> str:
    source = tables[SOURCE_VALIDATION_FILENAME]
    recognition = tables[RECOGNITION_FILENAME]
    pathway = tables[CORE_PATHWAY_SUMMARY_FILENAME]
    attempts = tables[CORE_ATTEMPT_AUDIT_FILENAME]
    oracle_classes = tables[ORACLE_CLASS_FILENAME]
    oracle_candidates = tables[ORACLE_CANDIDATE_SUMMARY_FILENAME]
    overlaps = tables[ORACLE_SUCCESS_OVERLAP_FILENAME]
    recovery = tables[RECOVERY_SUMMARY_FILENAME]
    evidence = tables[EVIDENCE_DISTRIBUTION_FILENAME]
    evidence_values = tables[EVIDENCE_VALUE_COUNTS_FILENAME]

    source_table = _markdown_table(
        ("Source table", "Rows", "Expected", "Validation"),
        [
            (
                row["source_table"],
                f"{int(row['observed_row_count']):,}",
                f"{int(row['expected_row_count']):,}",
                row["validation_status"],
            )
            for row in source
        ],
    )
    overall_recognition_rows = [
        _matching_row(
            recognition,
            stratum_type="OVERALL",
            predicted_health_state=state,
        )
        for state in PREDICTED_STATES
    ]
    recognition_overall = _markdown_table(
        ("State", "Interpretation", "Count", "Denominator", "Proportion"),
        [
            (
                row["predicted_health_state"],
                row["recognition_category"],
                f"{int(row['count']):,}",
                f"{int(row['denominator']):,}",
                _percent_text(row["proportion"]),
            )
            for row in overall_recognition_rows
        ],
    )
    pathway_table = _markdown_table(
        ("Type", "Indicator or combination", "Count", "Denominator", "Proportion"),
        [
            (
                row["indicator_type"],
                row["indicator"],
                f"{int(row['count']):,}",
                f"{int(row['denominator']):,}",
                _percent_text(row["proportion"]),
            )
            for row in pathway
        ],
    )
    attempt_table = _markdown_table(
        ("Population unit", "Metric", "Count", "Denominator", "Proportion"),
        [
            (
                row["population_unit"],
                row["metric"],
                f"{int(row['count']):,}",
                f"{int(row['denominator']):,}",
                _percent_text(row["proportion"]),
            )
            for row in attempts
        ],
    )
    candidate_overall = [row for row in oracle_candidates if row["stratum_type"] == "OVERALL"]
    candidate_table = _markdown_table(
        (
            "Action",
            "Feasible/all",
            "Execution/feasible",
            "Audit-admissible/executed",
            "Confirmed success/all",
            "Selected",
        ),
        [
            (
                row["action"],
                f"{int(row['feasible_count']):,}/{int(row['candidate_count']):,} "
                f"({_percent_text(row['feasible_proportion_all'])})",
                f"{int(row['execution_success_count']):,}/"
                f"{int(row['execution_success_denominator_feasible']):,} "
                f"({_percent_text(row['execution_success_proportion_given_feasible'])})",
                f"{int(row['audit_admissible_count']):,}/"
                f"{int(row['audit_admissible_denominator_executed']):,} "
                f"({_percent_text(row['audit_admissible_proportion_given_executed'])})",
                f"{int(row['confirmed_success_count']):,}/{int(row['candidate_count']):,} "
                f"({_percent_text(row['confirmed_success_proportion_all'])})",
                f"{int(row['selected_count']):,}",
            )
            for row in candidate_overall
        ],
    )
    overlap_table = _markdown_table(
        ("Successful candidate pattern", "Decisions", "Denominator", "Proportion"),
        [
            (
                row["successful_action_pattern"],
                f"{int(row['decision_count']):,}",
                f"{int(row['denominator_decisions']):,}",
                _percent_text(row["proportion"]),
            )
            for row in overlaps
        ],
    )
    evidence_overall = [
        row
        for row in evidence
        if row["stratum_type"] == "OVERALL" and row["stratum_value"] == "ALL"
    ]
    evidence_table = _markdown_table(
        ("Metric", "N", "Median", "IQR", "Range", "Zero count", "Unique values"),
        [
            (
                row["metric"],
                f"{int(row['n_available']):,}",
                row["median"],
                f"{row['q1']} to {row['q3']}",
                f"{row['minimum']} to {row['maximum']}",
                f"{int(row['zero_count']):,}",
                row["unique_count"],
            )
            for row in evidence_overall
        ],
    )

    recognised_core = _matching_row(pathway, indicator="RECOGNISED_HARM")
    no_attempt_after_evidence = _matching_row(
        pathway,
        indicator="RECOGNISED_HARM_AND_EVIDENCE_AND_NO_MODEL_CHANGE_ATTEMPT",
    )
    accepted = _matching_row(pathway, indicator="ACCEPTED_CANDIDATE")
    execution_failure = _matching_row(
        attempts,
        population_unit="MODEL_CHANGE_ATTEMPT",
        metric="EXPLICIT_EXECUTION_FAILURE",
    )
    oracle_successful = sum(
        int(
            _matching_row(
                oracle_classes,
                stratum_type="OVERALL",
                diagnostic_class=item,
            )["count"]
        )
        for item in ORACLE_CLASSES[:3]
    )
    oracle_total = int(
        _matching_row(
            oracle_classes,
            stratum_type="OVERALL",
            diagnostic_class=ORACLE_CLASSES[0],
        )["denominator"]
    )
    harmful_success = sum(
        int(
            _matching_row(
                oracle_classes,
                stratum_type="CURRENT_EVALUATOR_STATE",
                current_evaluator_state="HARMFUL",
                diagnostic_class=item,
            )["count"]
        )
        for item in ORACLE_CLASSES[:3]
    )
    harmful_total = int(
        _matching_row(
            oracle_classes,
            stratum_type="CURRENT_EVALUATOR_STATE",
            current_evaluator_state="HARMFUL",
            diagnostic_class=ORACLE_CLASSES[0],
        )["denominator"]
    )
    immediate_safe = _matching_row(
        recovery,
        stratum_type="OVERALL",
        horizon="IMMEDIATE",
        outcome="SAFE",
    )
    sustained_safe = _matching_row(
        recovery,
        stratum_type="OVERALL",
        horizon="TWO_WINDOW_SUSTAINED",
        outcome="SAFE",
    )
    sustained_censored = _matching_row(
        recovery,
        stratum_type="OVERALL",
        horizon="TWO_WINDOW_SUSTAINED",
        outcome="NOT_ASSESSED_CENSORED",
    )
    budget_zero = _matching_row(evidence_values, metric="remaining_label_budget", value="0")
    optimizer_80 = _matching_row(
        evidence_values,
        metric="optimizer_eligible_released_label_count",
        value="80",
    )
    query_not_pending = _matching_row(
        evidence_values,
        metric="query_pending",
        value="false",
    )
    immediate_safe_fraction = (
        f"{int(immediate_safe['count']):,}/{int(immediate_safe['assessable_interventions']):,}"
    )
    sustained_safe_fraction = (
        f"{int(sustained_safe['count']):,}/{int(sustained_safe['assessable_interventions']):,}"
    )
    sustained_censored_fraction = (
        f"{int(sustained_censored['count']):,}/"
        f"{int(sustained_censored['total_accepted_interventions']):,}"
    )
    optimizer_80_fraction = f"{int(optimizer_80['count']):,}/{int(optimizer_80['denominator']):,}"
    budget_zero_fraction = f"{int(budget_zero['count']):,}/{int(budget_zero['denominator']):,}"
    query_not_pending_fraction = (
        f"{int(query_not_pending['count']):,}/{int(query_not_pending['denominator']):,}"
    )
    oracle_success_result = (
        f"{oracle_successful:,}/{oracle_total:,} "
        f"({_percent_text(_optional_ratio(oracle_successful, oracle_total))})"
    )
    harmful_success_result = (
        f"{harmful_success:,}/{harmful_total:,} "
        f"({_percent_text(_optional_ratio(harmful_success, harmful_total))})"
    )
    immediate_safe_result = (
        f"{immediate_safe_fraction} ({_percent_text(immediate_safe['assessable_proportion'])})"
    )
    sustained_safe_result = (
        f"{sustained_safe_fraction} ({_percent_text(sustained_safe['assessable_proportion'])})"
    )

    report = f"""# RDX-003 recoverability diagnostic analysis

This is a post-freeze, artifact-only scientific diagnostic of the validated RDX-002 bundle.
It changes no DANIDS 2.0 hypothesis or RDX-001 definition, performs no experiment or model
operation, and treats window/event counts as descriptive rather than independent replicates.

## Source validation and provenance

The source-backed validator passed before analysis. The exact RDX-002 bundle digest is
`{inputs.manifest["bundle_digest"]}` and matches the frozen RDX-003 input identity.

{source_table}

## RDX-A: same-window harm recognition

The primary population is evaluator-HARMFUL decision windows. “Recognised” means
PREDICTED_HARMFUL in the same window; this is not anticipatory early warning.

{recognition_overall}

### By method

{_recognition_table(recognition, "METHOD")}

### By rotation

{_recognition_table(recognition, "ROTATION")}

### By current domain

{_recognition_table(recognition, "CURRENT_DOMAIN")}

### By rotation x seed

{_recognition_table(recognition, "ROTATION_SEED")}

## RDX-A/D: Core pathway

The indicators below use all {int(recognised_core["denominator"]):,} Core
evaluator-HARMFUL windows and deliberately overlap. They are state/path combinations, not a
mutually exclusive causal decomposition.

{pathway_table}

Explicit attempt and panel evidence separates audit outcomes from mere non-acceptance:

{attempt_table}

Audit UNCERTAIN is admissible under the frozen candidate guard because no regression was
demonstrated; it is not certified safety. The A1 structural-infeasibility records are
decision-local explicit preflight outcomes and are not generalised to windows where A1 was not
considered.

Core recognised {_count_rate(recognised_core)} of its harmful windows. A directly supported
combination shows {_count_rate(no_attempt_after_evidence)} were recognised, had non-zero
optimizer-eligible released evidence, and had no model-changing action attempt. Only
{_count_rate(accepted)} contained an accepted candidate, while explicit execution failure was
{_count_rate(execution_failure)}. These are descriptive facts and do not identify why the
frozen controller did or did not act.

## RDX-B: frozen A0-A4 one-step recoverability map

These are first-fresh-successor outcomes under frozen A0-A4 semantics. They are neither a
global recoverability result nor a claim that later recovery was impossible.

### Overall six-class map

{_oracle_class_table(oracle_classes, "OVERALL")}

### By rotation

{_oracle_class_table(oracle_classes, "ROTATION")}

### By current domain

{_oracle_class_table(oracle_classes, "CURRENT_DOMAIN")}

### By rotation x seed

{_oracle_class_table(oracle_classes, "ROTATION_SEED")}

Overall, {oracle_success_result} decisions had any confirmed one-step success class. RDX-B is not
restricted to currently HARMFUL decisions: only {harmful_success_result} currently HARMFUL
decisions had any successful class.

### Candidate-level categorical outcomes

{candidate_table}

Per-action candidate success counts overlap and therefore must not be added; the mutually
exclusive pattern table shows that overlap:

{overlap_table}

Categorical candidate outcomes are complete for every branch. Numeric successor metrics are
selection-limited and are not analysed here; their availability and preserved null counts are
recorded separately in `{ORACLE_NUMERIC_AVAILABILITY_FILENAME}`.

## RDX-C: actual deployed recovery

Immediate and sustained outcomes are reported separately. Censored interventions appear as a
category but are never included in a recovery-failure denominator.

### Overall

{_recovery_table(recovery, "OVERALL")}

### By method

{_recovery_table(recovery, "METHOD")}

### By action

{_recovery_table(recovery, "ACTION")}

### By rotation

{_recovery_table(recovery, "ROTATION")}

### By current domain

{_recovery_table(recovery, "CURRENT_DOMAIN")}

### By rotation x seed

{_recovery_table(recovery, "ROTATION_SEED")}

Among accepted interventions, immediate SAFE occurred in {immediate_safe_result} assessable
cases. Sustained SAFE occurred in {sustained_safe_result} assessable cases; the other
{sustained_censored_fraction} interventions were censored and are not failures. Post-accept audit
admissibility is conceptually separate from these later deployed outcomes.

## RDX-D: decision-time evidence characterisation

All quantities are observed before the Core decision at evaluator-HARMFUL windows. Quantiles use
linear interpolation at `(n-1)p`; stratified rows in the machine table are descriptive
associations only.

{evidence_table}

The full discrete table shows {optimizer_80_fraction} snapshots with 80 optimizer-eligible
released labels, {budget_zero_fraction} with zero remaining label budget, and
{query_not_pending_fraction} with no query pending. These observations do not show
that label quantity caused any later outcome.

## Diagnostic synthesis

1. **Recognition versus downstream failure.** Recognition misses are substantial, but failure is
   not predominantly explained away upstream. Core recognised {_count_rate(recognised_core)},
   yet model-changing attempts and accepted candidates were sparse, and later accepted-action
   recovery was rarer still. The artifacts cannot causally separate controller gating from
   action efficacy.
2. **Oracle opportunity.** Confirmed one-step success was sparse even with evaluator visibility,
   especially from a currently HARMFUL state ({harmful_success:,}/{harmful_total:,}). This is a
   one-step A0-A4 result only.
3. **Feasibility, execution, audit, and post-accept outcomes.** Explicit feasible-execution
   failures were absent. Audit non-admissibility existed, but many feasible, successfully
   executed candidates still lacked confirmed one-step success. Accepted deployed interventions
   also rarely reached SAFE, so failure remains after apparently legitimate acceptance.
4. **Observed deployed recovery.** Immediate SAFE was {immediate_safe_fraction}; sustained SAFE
   was {sustained_safe_fraction}, with censoring excluded.
5. **Unresolved causal question.** The evidence cannot distinguish whether the frozen B100
   training evidence was insufficient to learn an effective repair or whether the compact model
   and A0-A4 action capability imposed the dominant ceiling.

## Strongest new scientific insight

Under the frozen B100/D1 and A0-A4 regime, operational harm recognition and recoverability are
empirically separable: many harmful windows were recognised, but evaluator-informed one-step
success and deployed post-accept recovery both remained sparse. The evidence localises a major
downstream capability gap without establishing its cause.

## Recommended next controlled question

**Increased training-label budget.** Holding chronology, delay, audit rules, model, action
semantics, and evaluation fixed, ask whether more legitimately released training labels improve
one-step and sustained recovery. This has the highest information value among the proposed
options because perfect recognition cannot resolve the demonstrated downstream gap, explicit
execution failure was absent, and additional audit evidence would primarily test certification
rather than whether a trained action can repair the detector. No query was pending in these
pre-decision snapshots, but that state does not establish whether a query was issued in the
current window or whether delay mattered elsewhere. The controlled contrast would distinguish a
training-evidence limitation from a model/action-space capability limitation.

This is a recommendation for a future protocol only. **No new experiment is authorised.**

## Important limitations

- Window and event counts are descriptive and are not independent experimental replicates.
- Rotation x seed remains the higher-level comparative unit; no significance test is added.
- Oracle results are one-step and evaluator-informed, not global recoverability.
- Censored recovery is not failure, and unavailable numeric candidate values remain null.
- Evidence-pathway associations are not causal.
- The diagnostics are bounded to the frozen model, B100/D1 supervision, and A0-A4 actions.
"""
    return report


def _write_bundle(
    inputs: RdxAnalysisInputs,
    output_dir: str | Path,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    contract: Mapping[str, Any],
    summary: Mapping[str, Any],
    report: str,
) -> Path:
    output = Path(output_dir).resolve()
    if output == inputs.root or inputs.root in output.parents:
        raise RdxAnalysisError("RDX-003 output must not be inside the immutable RDX-002 bundle")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite RDX-003 analysis directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    for name in sorted(OUTPUT_SCHEMAS):
        (output / name).write_text(
            _csv_text(OUTPUT_SCHEMAS[name], tables[name]),
            encoding="utf-8",
            newline="",
        )
    (output / CONTRACT_FILENAME).write_text(_json_text(contract), encoding="utf-8", newline="")
    (output / SUMMARY_FILENAME).write_text(_json_text(summary), encoding="utf-8", newline="")
    (output / REPORT_FILENAME).write_text(report, encoding="utf-8", newline="")
    files = _tree_digests(output)
    if set(files) != ALL_OUTPUT_FILES.difference({MANIFEST_FILENAME}):
        raise RdxAnalysisError("RDX-003 pre-manifest file set differs from the contract")
    manifest = {
        "version": RDX_ANALYSIS_VERSION,
        "source_rdx_bundle_digest": inputs.manifest["bundle_digest"],
        "files": files,
        "bundle_digest": _digest(files),
    }
    (output / MANIFEST_FILENAME).write_text(_json_text(manifest), encoding="utf-8", newline="")
    return output


def _validate_analysis_with_inputs(output: Path, inputs: RdxAnalysisInputs) -> None:
    entries = list(output.iterdir())
    if any(not entry.is_file() for entry in entries) or {entry.name for entry in entries} != (
        ALL_OUTPUT_FILES
    ):
        raise RdxAnalysisError("RDX-003 output file set differs from the fixed contract")
    manifest = _load_json(output / MANIFEST_FILENAME)
    if (output / MANIFEST_FILENAME).read_bytes() != _json_text(manifest).encode("utf-8"):
        raise RdxAnalysisError("RDX-003 manifest is not canonically serialized")
    files = _tree_digests(output)
    if (
        set(manifest)
        != {
            "version",
            "source_rdx_bundle_digest",
            "files",
            "bundle_digest",
        }
        or manifest.get("version") != RDX_ANALYSIS_VERSION
        or manifest.get("source_rdx_bundle_digest") != EXPECTED_SOURCE_BUNDLE_DIGEST
        or manifest.get("files") != files
        or manifest.get("bundle_digest") != _digest(files)
    ):
        raise RdxAnalysisError("RDX-003 manifest differs from its persisted files")

    tables, _ = _derive_analysis_tables(inputs)
    contract = _build_contract(inputs)
    summary = _build_summary(inputs, tables)
    report = _report_text(inputs, tables, summary)
    expected_text: dict[str, str] = {
        name: _csv_text(OUTPUT_SCHEMAS[name], rows) for name, rows in tables.items()
    }
    expected_text.update(
        {
            CONTRACT_FILENAME: _json_text(contract),
            SUMMARY_FILENAME: _json_text(summary),
            REPORT_FILENAME: report,
        }
    )
    for name, text in expected_text.items():
        if (output / name).read_bytes() != text.encode("utf-8"):
            raise RdxAnalysisError(
                f"persisted {name} differs from deterministic source-backed analysis"
            )


def validate_rdx_analysis(output_dir: str | Path) -> None:
    """Independently revalidate RDX-002 and source-recompute the RDX-003 package."""

    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise RdxAnalysisError(f"RDX-003 output directory does not exist: {output}")
    contract = _load_json(output / CONTRACT_FILENAME)
    source = contract.get("source_rdx_bundle")
    if not isinstance(source, dict):
        raise RdxAnalysisError("RDX-003 contract lacks source RDX-002 provenance")
    inputs = load_rdx_analysis_inputs(Path(str(source.get("path", ""))))
    _validate_analysis_with_inputs(output, inputs)


def analyze_rdx_recoverability(
    source_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Validate RDX-002, derive RDX-003, and seal a new write-once analysis package."""

    inputs = load_rdx_analysis_inputs(source_dir)
    tables, _ = _derive_analysis_tables(inputs)
    contract = _build_contract(inputs)
    summary = _build_summary(inputs, tables)
    report = _report_text(inputs, tables, summary)
    output = _write_bundle(inputs, output_dir, tables, contract, summary, report)
    _validate_analysis_with_inputs(output, inputs)
    return output


__all__ = [
    "ALL_OUTPUT_FILES",
    "EXPECTED_ROW_COUNTS",
    "EXPECTED_SOURCE_BUNDLE_DIGEST",
    "OUTPUT_SCHEMAS",
    "RDX_ANALYSIS_VERSION",
    "RdxAnalysisError",
    "analyze_rdx_recoverability",
    "build_core_pathway_summaries",
    "build_evidence_summaries",
    "build_oracle_candidate_summaries",
    "build_oracle_class_summary",
    "build_recognition_summary",
    "build_recovery_summary",
    "load_rdx_analysis_inputs",
    "validate_rdx_analysis",
]
