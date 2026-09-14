"""Deterministic artifact-only analysis of the frozen confirmatory Study-4 matrix."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import matplotlib
import numpy as np
import pandas as pd

from danids.config.continual import FROZEN_ROTATIONS
from danids.config.study4 import STUDY4_CONFIRMATORY_SEEDS, Study4Method
from danids.evaluation.study4 import Study4ArtifactError, validate_study4_evaluation

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ANALYSIS_VERSION: Final = "task006-e4-confirmatory-analysis-v1"
METHOD_ORDER: Final = (
    Study4Method.STATIC.value,
    Study4Method.ALWAYS_ADAPT.value,
    Study4Method.DANIDS_CORE.value,
    Study4Method.OFFLINE_ORACLE.value,
)
ACTION_ORDER: Final = (
    "A0_NO_OP",
    "A1_RECALIBRATE",
    "A2_HEAD_UPDATE",
    "A3_FULL_FINE_TUNE",
    "A4_REPLAY_UPDATE",
)
RETENTION_METRICS: Final = (
    "pr_auc",
    "roc_auc",
    "tpr",
    "fpr_budget_ratio",
    "operational_tpr_forgetting",
)
SOURCE_FILES: Final = (
    "study4_summary.json",
    "evaluation_contract.json",
    "artifact_manifest.json",
    "study4_runs.csv",
    "study4_windows.csv",
    "study4_holdouts.csv",
    "study4_interventions.csv",
    "study4_method_summary.csv",
    "study4_safety_compliance.csv",
    "study4_label_query_usage.csv",
    "study4_action_distribution.csv",
    "study4_rollback_summary.csv",
    "study4_retention_summary.csv",
    "study4_core_vs_always_paired.csv",
)
TABLE_FILES: Final = tuple(name for name in SOURCE_FILES if name.endswith(".csv"))
OUTPUT_FILES: Final = (
    "study4_confirmatory_results.json",
    "study4_confirmatory_results.md",
    "study4_hypothesis_verdicts.md",
    "study4_discussion_notes.md",
    "table_study4_overall.csv",
    "table_study4_overall.md",
    "table_study4_core_vs_always.csv",
    "table_study4_core_vs_always.md",
    "fig_study4_safety_compliance.png",
    "fig_study4_resource_tradeoff.png",
    "fig_study4_core_vs_always_paired.png",
    "fig_study4_action_distribution.png",
    "fig_study4_retention.png",
    "fig_study4_oracle_gap.png",
)


class Study4AnalysisError(ValueError):
    """Raised when confirmatory Study-4 evidence cannot support the analysis package."""


@dataclass(frozen=True, slots=True)
class Study4AnalysisInputs:
    evaluation_dir: Path
    summary: dict[str, Any]
    contract: dict[str, Any]
    manifest: dict[str, Any]
    tables: dict[str, pd.DataFrame]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Study4AnalysisError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise Study4AnalysisError(f"{path.name} must contain a JSON object")
    return cast(dict[str, Any], value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _to_float(value: object) -> float:
    return float(cast(Any, value))


def _to_int(value: object) -> int:
    return int(cast(Any, value))


def load_study4_analysis_inputs(evaluation_dir: str | Path) -> Study4AnalysisInputs:
    """Validate and load only the canonical Study-4 evaluation artifact."""

    root = Path(evaluation_dir).resolve()
    try:
        validate_study4_evaluation(root)
    except (Study4ArtifactError, FileNotFoundError, OSError) as exc:
        raise Study4AnalysisError(f"Study-4 evaluation artifact is invalid: {exc}") from exc
    missing = [name for name in SOURCE_FILES if not (root / name).is_file()]
    if missing:
        raise Study4AnalysisError(f"Study-4 evaluation artifact lacks required files: {missing}")
    summary = _load_json(root / "study4_summary.json")
    expected_methods = set(METHOD_ORDER)
    expected_rotations = {"-".join(rotation) for rotation in FROZEN_ROTATIONS}
    if (
        summary.get("status") != "complete"
        or int(summary.get("run_count", -1)) != 48
        or int(summary.get("smoke_run_count", -1)) != 0
        or set(summary.get("methods_present", ())) != expected_methods
        or set(summary.get("rotations_present", ())) != expected_rotations
        or set(summary.get("seeds_present", ())) != set(STUDY4_CONFIRMATORY_SEEDS)
        or int(summary.get("paired_core_vs_always_count", -1)) != 12
    ):
        raise Study4AnalysisError(
            "confirmatory analysis requires the exact complete 48-run Study-4 matrix"
        )
    tables = {name: pd.read_csv(root / name) for name in TABLE_FILES}
    _validate_matrix_rows(tables["study4_runs.csv"])
    return Study4AnalysisInputs(
        evaluation_dir=root,
        summary=summary,
        contract=_load_json(root / "evaluation_contract.json"),
        manifest=_load_json(root / "artifact_manifest.json"),
        tables=tables,
    )


def _validate_matrix_rows(runs: pd.DataFrame) -> None:
    required = {"method", "sequence", "seed", "smoke"}
    if not required.issubset(runs.columns) or len(runs) != 48:
        raise Study4AnalysisError("canonical Study-4 run table is not the complete matrix")
    observed = {
        (str(row.sequence), _to_int(row.seed), str(row.method))
        for row in runs.loc[:, ["sequence", "seed", "method"]].itertuples(index=False)
    }
    expected = {
        ("-".join(rotation), seed, method)
        for rotation in FROZEN_ROTATIONS
        for seed in STUDY4_CONFIRMATORY_SEEDS
        for method in METHOD_ORDER
    }
    if observed != expected or len(observed) != len(runs):
        raise Study4AnalysisError("canonical Study-4 run identities differ from the frozen matrix")
    smoke = runs["smoke"].astype(str).str.casefold()
    if smoke.isin({"true", "1"}).any():
        raise Study4AnalysisError("smoke runs cannot enter confirmatory Study-4 analysis")


def _method_records(frame: pd.DataFrame, name: str) -> dict[str, dict[str, Any]]:
    if "method" not in frame.columns or frame["method"].duplicated().any():
        raise Study4AnalysisError(f"{name} must contain exactly one row per method")
    records = cast(list[dict[str, Any]], frame.to_dict(orient="records"))
    result = {str(row["method"]): row for row in records}
    if set(result) != set(METHOD_ORDER):
        raise Study4AnalysisError(f"{name} method coverage differs from the frozen matrix")
    return result


def build_overall_table(inputs: Study4AnalysisInputs) -> pd.DataFrame:
    """Build the canonical method-level thesis table from independent source summaries."""

    method_summary = _method_records(
        inputs.tables["study4_method_summary.csv"], "study4_method_summary.csv"
    )
    safety = _method_records(
        inputs.tables["study4_safety_compliance.csv"], "study4_safety_compliance.csv"
    )
    labels = _method_records(
        inputs.tables["study4_label_query_usage.csv"], "study4_label_query_usage.csv"
    )
    rollbacks = _method_records(
        inputs.tables["study4_rollback_summary.csv"], "study4_rollback_summary.csv"
    )
    retention = inputs.tables["study4_retention_summary.csv"]
    if set(retention.columns) != {"method", "metric", "mean_final"}:
        raise Study4AnalysisError("Study-4 retention summary schema differs")
    if retention.duplicated(["method", "metric"]).any():
        raise Study4AnalysisError("Study-4 retention summary contains duplicate metrics")
    retention_map = {
        (str(row.method), str(row.metric)): _to_float(row.mean_final)
        for row in retention.itertuples(index=False)
    }
    expected_retention = {
        (method, metric) for method in METHOD_ORDER for metric in RETENTION_METRICS
    }
    if set(retention_map) != expected_retention:
        raise Study4AnalysisError("Study-4 retention metric coverage differs")

    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        run_count = int(method_summary[method]["run_count"])
        if run_count != 12:
            raise Study4AnalysisError(f"{method} does not contain 12 confirmatory runs")
        labels_per_run = float(labels[method]["labels_requested"]) / run_count
        accepted_per_run = float(rollbacks[method]["accepted_updates"]) / run_count
        unsafe_per_run = float(safety[method]["unsafe_exposure_windows"]) / run_count
        missed_per_run = float(safety[method]["missed_harmful_windows"]) / run_count
        expected_values = (
            (labels_per_run, float(method_summary[method]["mean_labels_requested"])),
            (accepted_per_run, float(method_summary[method]["mean_accepted_updates"])),
            (unsafe_per_run, float(method_summary[method]["mean_unsafe_exposure_windows"])),
            (missed_per_run, float(method_summary[method]["mean_missed_harmful_windows"])),
        )
        if any(
            not np.isclose(left, right, rtol=1e-12, atol=1e-12) for left, right in expected_values
        ):
            raise Study4AnalysisError(f"{method} source summaries do not reconcile")
        rows.append(
            {
                "method": method,
                "labels_per_run": labels_per_run,
                "accepted_updates_per_run": accepted_per_run,
                "unsafe_exposure_per_run": unsafe_per_run,
                "missed_harmful_per_run": missed_per_run,
                "compliance_rate": float(safety[method]["operating_envelope_compliance_rate"]),
                "final_pr_auc": retention_map[(method, "pr_auc")],
                "final_roc_auc": retention_map[(method, "roc_auc")],
                "final_tpr": retention_map[(method, "tpr")],
                "final_fpr_budget_ratio": retention_map[(method, "fpr_budget_ratio")],
                "operational_tpr_forgetting": retention_map[(method, "operational_tpr_forgetting")],
            }
        )
    return pd.DataFrame(rows)


def build_core_vs_always_table(inputs: Study4AnalysisInputs) -> pd.DataFrame:
    """Return all paired differences plus explicitly labelled mean and median rows."""

    source = inputs.tables["study4_core_vs_always_paired.csv"]
    required = (
        "sequence",
        "seed",
        "core_minus_always_labels",
        "core_minus_always_updates",
        "core_minus_always_unsafe_exposure",
        "core_minus_always_missed_harm",
    )
    if tuple(source.columns) != required or len(source) != 12:
        raise Study4AnalysisError("Core-versus-Always paired table schema/count differs")
    expected_pairs = {
        ("-".join(rotation), seed)
        for rotation in FROZEN_ROTATIONS
        for seed in STUDY4_CONFIRMATORY_SEEDS
    }
    observed_pairs = {
        (str(row.sequence), _to_int(row.seed))
        for row in source.loc[:, ["sequence", "seed"]].itertuples(index=False)
    }
    if observed_pairs != expected_pairs or source.duplicated(["sequence", "seed"]).any():
        raise Study4AnalysisError("Core-versus-Always pairs differ from the frozen matrix")
    renamed = source.rename(
        columns={
            "core_minus_always_labels": "label_difference",
            "core_minus_always_updates": "update_difference",
            "core_minus_always_unsafe_exposure": "unsafe_exposure_difference",
            "core_minus_always_missed_harm": "missed_harm_difference",
        }
    ).copy()
    renamed.insert(0, "row_type", "pair")
    metric_columns = [
        "label_difference",
        "update_difference",
        "unsafe_exposure_difference",
        "missed_harm_difference",
    ]
    summary_rows = []
    for row_type, operation in (("aggregate_mean", "mean"), ("aggregate_median", "median")):
        values = getattr(renamed[metric_columns], operation)(axis=0)
        summary_rows.append(
            {
                "row_type": row_type,
                "sequence": "ALL",
                "seed": None,
                **{column: float(values[column]) for column in metric_columns},
            }
        )
    return pd.concat([renamed, pd.DataFrame(summary_rows)], ignore_index=True)


def determine_hypothesis_verdicts(overall: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Apply the frozen H4/H10 component logic without a post-hoc similarity margin."""

    indexed = overall.set_index("method")
    core = indexed.loc[Study4Method.DANIDS_CORE.value]
    always = indexed.loc[Study4Method.ALWAYS_ADAPT.value]
    fewer_updates = _to_float(core["accepted_updates_per_run"]) < _to_float(
        always["accepted_updates_per_run"]
    )
    less_supervision = _to_float(core["labels_per_run"]) < _to_float(always["labels_per_run"])
    comparable_safety = _to_float(core["unsafe_exposure_per_run"]) <= _to_float(
        always["unsafe_exposure_per_run"]
    ) and _to_float(core["compliance_rate"]) >= _to_float(always["compliance_rate"])
    favourable_forgetting = _to_float(core["operational_tpr_forgetting"]) < _to_float(
        always["operational_tpr_forgetting"]
    )
    h4_supported = fewer_updates and less_supervision and comparable_safety
    h10_supported = (
        fewer_updates and less_supervision and comparable_safety and favourable_forgetting
    )
    return {
        "H4": {
            "verdict": "SUPPORTED" if h4_supported else "NOT_SUPPORTED",
            "update_reduction_supported": fewer_updates,
            "label_reduction_supported": less_supervision,
            "comparable_safety_supported": comparable_safety,
            "basis": "confirmatory frozen Study-4 aggregation",
        },
        "H10": {
            "verdict": (
                "SUPPORTED"
                if h10_supported
                else "PARTIALLY_SUPPORTED"
                if fewer_updates or favourable_forgetting
                else "NOT_SUPPORTED"
            ),
            "fewer_updates_supported": fewer_updates,
            "less_supervision_supported": less_supervision,
            "comparable_safety_supported": comparable_safety,
            "operational_forgetting_descriptively_favourable": favourable_forgetting,
            "central_safety_efficiency_proposition_supported": h10_supported,
            "basis": (
                "component-level support only; the central safety-efficiency proposition failed"
            ),
        },
    }


def _method_row(overall: pd.DataFrame, method: str) -> dict[str, Any]:
    records = cast(
        list[dict[str, Any]], overall.loc[overall["method"] == method].to_dict(orient="records")
    )
    if len(records) != 1:
        raise Study4AnalysisError(f"overall table lacks exactly one {method} row")
    return records[0]


def _difference(left: Mapping[str, Any], right: Mapping[str, Any], field: str) -> dict[str, Any]:
    left_value = float(left[field])
    right_value = float(right[field])
    absolute = left_value - right_value
    return {
        "left": left_value,
        "right": right_value,
        "absolute_difference": absolute,
        "percentage_difference": None if right_value == 0 else 100.0 * absolute / right_value,
    }


def _rotation_differences(paired: pd.DataFrame) -> list[dict[str, Any]]:
    source = paired.loc[paired["row_type"] == "pair"].copy()
    rows: list[dict[str, Any]] = []
    for sequence in sorted(source["sequence"].astype(str).unique()):
        group = source.loc[source["sequence"] == sequence]
        unsafe = group["unsafe_exposure_difference"].astype(float)
        rows.append(
            {
                "sequence": sequence,
                "mean_label_difference": float(group["label_difference"].mean()),
                "mean_update_difference": float(group["update_difference"].mean()),
                "mean_unsafe_exposure_difference": float(unsafe.mean()),
                "median_unsafe_exposure_difference": float(unsafe.median()),
                "mean_missed_harm_difference": float(group["missed_harm_difference"].mean()),
                "core_better_unsafe_count": int((unsafe < 0).sum()),
                "core_equal_unsafe_count": int((unsafe == 0).sum()),
                "core_worse_unsafe_count": int((unsafe > 0).sum()),
            }
        )
    return rows


def build_headline_results(
    inputs: Study4AnalysisInputs,
    overall: pd.DataFrame,
    paired: pd.DataFrame,
) -> dict[str, Any]:
    """Build machine-readable confirmatory results and labelled descriptive findings."""

    core = _method_row(overall, Study4Method.DANIDS_CORE.value)
    always = _method_row(overall, Study4Method.ALWAYS_ADAPT.value)
    static = _method_row(overall, Study4Method.STATIC.value)
    oracle = _method_row(overall, Study4Method.OFFLINE_ORACLE.value)
    paired_rows = paired.loc[paired["row_type"] == "pair"]
    unsafe_differences = paired_rows["unsafe_exposure_difference"].astype(float)
    rotations = _rotation_differences(paired)

    actions = inputs.tables["study4_action_distribution.csv"]
    expected_action_rows = {(method, action) for method in METHOD_ORDER for action in ACTION_ORDER}
    observed_action_rows = {
        (str(row.method), str(row.action))
        for row in actions.loc[:, ["method", "action"]].itertuples(index=False)
    }
    if (
        observed_action_rows != expected_action_rows
        or actions.duplicated(["method", "action"]).any()
    ):
        raise Study4AnalysisError("Study-4 action distribution coverage differs")
    action_totals = {
        method: {
            action: int(
                actions.loc[
                    (actions["method"] == method) & (actions["action"] == action),
                    "attempt_count",
                ].iloc[0]
            )
            for action in ACTION_ORDER
        }
        for method in METHOD_ORDER
    }
    labels = _method_records(
        inputs.tables["study4_label_query_usage.csv"], "study4_label_query_usage.csv"
    )
    rollbacks = _method_records(
        inputs.tables["study4_rollback_summary.csv"], "study4_rollback_summary.csv"
    )
    verdicts = determine_hypothesis_verdicts(overall)
    return {
        "version": ANALYSIS_VERSION,
        "source_evaluation": {
            "evaluation_version": inputs.summary["version"],
            "evaluation_bundle_digest": inputs.manifest["bundle_digest"],
            "evaluation_contract_sha256": _file_sha256(
                inputs.evaluation_dir / "evaluation_contract.json"
            ),
            "evaluation_contract_digest": _canonical_digest(inputs.contract),
            "validated_before_analysis": True,
            "source_artifacts_mutated": False,
        },
        "experimental_completion": {
            "status": inputs.summary["status"],
            "run_count": int(inputs.summary["run_count"]),
            "method_count": len(inputs.summary["methods_present"]),
            "rotation_count": len(inputs.summary["rotations_present"]),
            "seeds": inputs.summary["seeds_present"],
            "paired_core_vs_always_count": int(inputs.summary["paired_core_vs_always_count"]),
            "smoke_run_count": int(inputs.summary["smoke_run_count"]),
        },
        "confirmatory_hypotheses": verdicts,
        "confirmatory_comparisons": {
            "core_minus_always": {
                "labels_per_run": _difference(core, always, "labels_per_run"),
                "accepted_updates_per_run": _difference(core, always, "accepted_updates_per_run"),
                "unsafe_exposure_per_run": _difference(core, always, "unsafe_exposure_per_run"),
                "missed_harmful_per_run": _difference(core, always, "missed_harmful_per_run"),
                "compliance_rate": {
                    **_difference(core, always, "compliance_rate"),
                    "absolute_percentage_point_difference": 100.0
                    * (float(core["compliance_rate"]) - float(always["compliance_rate"])),
                },
            },
            "always_minus_static": {
                "labels_per_run": _difference(always, static, "labels_per_run"),
                "accepted_updates_per_run": _difference(always, static, "accepted_updates_per_run"),
                "unsafe_exposure_per_run": _difference(always, static, "unsafe_exposure_per_run"),
                "missed_harmful_per_run": _difference(always, static, "missed_harmful_per_run"),
                "compliance_rate": {
                    **_difference(always, static, "compliance_rate"),
                    "absolute_percentage_point_difference": 100.0
                    * (float(always["compliance_rate"]) - float(static["compliance_rate"])),
                },
            },
            "core_minus_oracle": {
                "labels_per_run": _difference(core, oracle, "labels_per_run"),
                "accepted_updates_per_run": _difference(core, oracle, "accepted_updates_per_run"),
                "unsafe_exposure_per_run": _difference(core, oracle, "unsafe_exposure_per_run"),
                "missed_harmful_per_run": _difference(core, oracle, "missed_harmful_per_run"),
                "compliance_rate": {
                    **_difference(core, oracle, "compliance_rate"),
                    "absolute_percentage_point_difference": 100.0
                    * (float(core["compliance_rate"]) - float(oracle["compliance_rate"])),
                },
            },
        },
        "resource_totals": {
            method: {
                "labels_requested": int(labels[method]["labels_requested"]),
                "labels_released": int(labels[method]["labels_released"]),
                "query_events": int(labels[method]["query_events"]),
                "attempted_updates": int(rollbacks[method]["attempted_updates"]),
                "accepted_updates": int(rollbacks[method]["accepted_updates"]),
                "rejected_updates": int(rollbacks[method]["rejected_updates"]),
                "rollback_rate": (
                    None
                    if pd.isna(rollbacks[method]["rollback_rate"])
                    else float(rollbacks[method]["rollback_rate"])
                ),
            }
            for method in METHOD_ORDER
        },
        "final_retention": {
            str(row["method"]): {
                key: float(row[key])
                for key in (
                    "final_pr_auc",
                    "final_roc_auc",
                    "final_tpr",
                    "final_fpr_budget_ratio",
                    "operational_tpr_forgetting",
                )
            }
            for row in cast(list[dict[str, Any]], overall.to_dict(orient="records"))
        },
        "paired_core_vs_always": {
            "unsafe_exposure": {
                "core_better_count": int((unsafe_differences < 0).sum()),
                "equal_count": int((unsafe_differences == 0).sum()),
                "core_worse_count": int((unsafe_differences > 0).sum()),
            },
            "rotation_differences": rotations,
        },
        "action_attempt_totals": action_totals,
        "descriptive_exploratory_findings": {
            "always_adapt_vs_static": (
                "Always-Adapt was slightly worse than Static on aggregate unsafe exposure and "
                "compliance despite using the full supervision budget."
            ),
            "oracle_interpretation": (
                "The non-deployable one-step Oracle shows that sparse beneficial interventions "
                "exist, but its low absolute compliance leaves most harm unresolved under the "
                "frozen B100/action regime."
            ),
            "oracle_a3_observation": (
                f"{action_totals[Study4Method.OFFLINE_ORACLE.value]['A3_FULL_FINE_TUNE']} of "
                f"{sum(action_totals[Study4Method.OFFLINE_ORACLE.value].values())} Oracle "
                "model-changing selections were A3. This is exploratory/mechanistic evidence "
                "and must not be used to retrospectively change Core."
            ),
        },
    }


def _markdown_table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    def clean(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(clean(value) for value in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(clean(value) for value in row) + " |" for row in rows]
    return "\n".join([header, rule, *body]) + "\n"


def _overall_markdown(overall: pd.DataFrame) -> str:
    columns = [
        "Method",
        "Labels/run",
        "Accepted updates/run",
        "Unsafe exposure/run",
        "Missed harmful/run",
        "Compliance",
        "Final PR-AUC",
        "Final ROC-AUC",
        "Final TPR",
        "Final FPR budget ratio",
        "Operational TPR forgetting",
    ]
    rows = [
        [
            str(row.method),
            f"{_to_float(row.labels_per_run):.2f}",
            f"{_to_float(row.accepted_updates_per_run):.2f}",
            f"{_to_float(row.unsafe_exposure_per_run):.2f}",
            f"{_to_float(row.missed_harmful_per_run):.2f}",
            f"{100.0 * _to_float(row.compliance_rate):.2f}%",
            f"{_to_float(row.final_pr_auc):.4f}",
            f"{_to_float(row.final_roc_auc):.4f}",
            f"{_to_float(row.final_tpr):.4f}",
            f"{_to_float(row.final_fpr_budget_ratio):.2f}",
            f"{_to_float(row.operational_tpr_forgetting):.4f}",
        ]
        for row in overall.itertuples(index=False)
    ]
    return _markdown_table(columns, rows)


def _paired_markdown(paired: pd.DataFrame) -> str:
    columns = [
        "Row",
        "Sequence",
        "Seed",
        "Label difference",
        "Update difference",
        "Unsafe-exposure difference",
        "Missed-harm difference",
    ]
    rows = []
    for row in paired.itertuples(index=False):
        rows.append(
            [
                str(row.row_type),
                str(row.sequence),
                "" if pd.isna(row.seed) else str(_to_int(row.seed)),
                f"{_to_float(row.label_difference):.2f}",
                f"{_to_float(row.update_difference):.2f}",
                f"{_to_float(row.unsafe_exposure_difference):.2f}",
                f"{_to_float(row.missed_harm_difference):.2f}",
            ]
        )
    return _markdown_table(columns, rows)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.15g")


def _annotate_bars(axis: Any, values: Sequence[float], *, suffix: str = "") -> None:
    for index, value in enumerate(values):
        axis.text(index, value, f"{value:.2f}{suffix}", ha="center", va="bottom", fontsize=8)


def _save_safety_figure(overall: pd.DataFrame, output: Path) -> None:
    values = (overall["compliance_rate"].astype(float) * 100.0).tolist()
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(overall["method"], values)
    axis.set_ylabel("Operating-envelope compliance (%)")
    axis.set_title("Study 4 operating-envelope compliance")
    axis.set_ylim(0.0, max(values) * 1.25)
    axis.tick_params(axis="x", rotation=15)
    _annotate_bars(axis, values, suffix="%")
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def _save_resource_figure(overall: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.2))
    axis.scatter(overall["accepted_updates_per_run"], overall["unsafe_exposure_per_run"], s=55)
    for row in overall.itertuples(index=False):
        axis.annotate(
            f"{row.method}\nlabels/run={_to_float(row.labels_per_run):.0f}",
            (
                _to_float(row.accepted_updates_per_run),
                _to_float(row.unsafe_exposure_per_run),
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Accepted updates per run")
    axis.set_ylabel("Unsafe exposure windows per run")
    axis.set_title("Study 4 intervention and safety trade-off")
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def _save_paired_figure(paired: pd.DataFrame, output: Path) -> None:
    source = paired.loc[paired["row_type"] == "pair"].copy()
    labels = [f"{row.sequence} s{_to_int(row.seed)}" for row in source.itertuples(index=False)]
    values = source["unsafe_exposure_difference"].astype(float).tolist()
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(np.arange(len(values)), values)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(len(values)), labels, rotation=45, ha="right")
    axis.set_ylabel("Core minus Always unsafe-exposure windows")
    axis.set_title("Paired Core versus Always-Adapt safety differences")
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def _save_action_figure(actions: pd.DataFrame, output: Path) -> None:
    displayed = ACTION_ORDER[1:]
    x_values = np.arange(len(METHOD_ORDER), dtype=float)
    width = 0.18
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for offset, action in enumerate(displayed):
        values = [
            int(
                actions.loc[
                    (actions["method"] == method) & (actions["action"] == action),
                    "attempt_count",
                ].iloc[0]
            )
            for method in METHOD_ORDER
        ]
        axis.bar(x_values + (offset - 1.5) * width, values, width=width, label=action)
    axis.set_xticks(x_values, METHOD_ORDER, rotation=15)
    axis.set_ylabel("Recorded action attempts")
    axis.set_title("Study 4 intervention action distribution")
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def _save_retention_figure(overall: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    x_values = np.arange(len(METHOD_ORDER), dtype=float)
    width = 0.24
    for offset, (column, label) in enumerate(
        (("final_pr_auc", "PR-AUC"), ("final_roc_auc", "ROC-AUC"), ("final_tpr", "TPR"))
    ):
        axes[0].bar(
            x_values + (offset - 1) * width,
            overall[column].astype(float),
            width=width,
            label=label,
        )
    axes[0].set_xticks(x_values, METHOD_ORDER, rotation=20)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Final mean metric")
    axes[0].set_title("Threshold-free and recall retention")
    axes[0].legend(fontsize=8)
    ratios = overall["final_fpr_budget_ratio"].astype(float).tolist()
    axes[1].bar(METHOD_ORDER, ratios)
    axes[1].set_ylabel("Final FPR budget ratio")
    axes[1].set_title("Operational false-positive behaviour")
    axes[1].tick_params(axis="x", rotation=20)
    _annotate_bars(axes[1], ratios)
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def _save_oracle_gap_figure(overall: pd.DataFrame, output: Path) -> None:
    unsafe = overall["unsafe_exposure_per_run"].astype(float).tolist()
    compliance = (overall["compliance_rate"].astype(float) * 100.0).tolist()
    core = _method_row(overall, Study4Method.DANIDS_CORE.value)
    oracle = _method_row(overall, Study4Method.OFFLINE_ORACLE.value)
    unsafe_gap = float(core["unsafe_exposure_per_run"]) - float(oracle["unsafe_exposure_per_run"])
    compliance_gap = 100.0 * (float(core["compliance_rate"]) - float(oracle["compliance_rate"]))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].bar(METHOD_ORDER, unsafe)
    axes[0].set_ylabel("Unsafe exposure windows per run")
    axes[0].set_title(f"Safety exposure (Core minus Oracle: {unsafe_gap:+.2f})")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(METHOD_ORDER, compliance)
    axes[1].set_ylabel("Operating-envelope compliance (%)")
    axes[1].set_title(f"Compliance (Core minus Oracle: {compliance_gap:+.2f} pp)")
    axes[1].tick_params(axis="x", rotation=20)
    figure.suptitle("Study 4 Core-to-Oracle opportunity gap")
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def _confirmatory_results_text(
    inputs: Study4AnalysisInputs,
    overall: pd.DataFrame,
    headline: Mapping[str, Any],
) -> str:
    core = _method_row(overall, Study4Method.DANIDS_CORE.value)
    always = _method_row(overall, Study4Method.ALWAYS_ADAPT.value)
    static = _method_row(overall, Study4Method.STATIC.value)
    oracle = _method_row(overall, Study4Method.OFFLINE_ORACLE.value)
    core_always = cast(Mapping[str, Any], headline["confirmatory_comparisons"]["core_minus_always"])
    core_oracle = cast(Mapping[str, Any], headline["confirmatory_comparisons"]["core_minus_oracle"])
    paired = cast(Mapping[str, Any], headline["paired_core_vs_always"])
    unsafe_counts = cast(Mapping[str, Any], paired["unsafe_exposure"])
    rotations = cast(Sequence[Mapping[str, Any]], paired["rotation_differences"])
    rotation_text = "; ".join(
        f"{row['sequence']}: {float(row['mean_unsafe_exposure_difference']):+.2f} mean "
        f"Core-minus-Always unsafe windows"
        for row in rotations
    )
    h4 = cast(Mapping[str, Any], headline["confirmatory_hypotheses"]["H4"])
    h10 = cast(Mapping[str, Any], headline["confirmatory_hypotheses"]["H10"])
    return f"""# Study 4 confirmatory results

This report separates confirmatory findings under the frozen Study-4 protocol from
descriptive and mechanistic interpretation. It adds no post-hoc thresholds, tuning, or
significance tests.

## Confirmatory results

### Experimental completion

The validated evaluation contains {int(inputs.summary["run_count"])} runs across four methods,
four frozen rotations, and seeds 42, 43, and 44. It contains
{int(inputs.summary["paired_core_vs_always_count"])} paired Core-versus-Always comparisons and
no smoke runs.

### Overall safety

Static, Always-Adapt, Core, and the non-deployable Offline Oracle achieved aggregate
operating-envelope compliance rates of {100.0 * float(static["compliance_rate"]):.2f}%,
{100.0 * float(always["compliance_rate"]):.2f}%,
{100.0 * float(core["compliance_rate"]):.2f}%, and
{100.0 * float(oracle["compliance_rate"]):.2f}%, respectively. Their mean unsafe-exposure
counts were {float(static["unsafe_exposure_per_run"]):.2f},
{float(always["unsafe_exposure_per_run"]):.2f},
{float(core["unsafe_exposure_per_run"]):.2f}, and
{float(oracle["unsafe_exposure_per_run"]):.2f} windows per run.

### Supervision and intervention cost

Core and Always-Adapt each requested 300 labels per run, or 3,600 labels across 12 runs.
Core therefore achieved intervention sparsity but not supervision sparsity. Core accepted
{float(core["accepted_updates_per_run"]):.2f} updates per run compared with
{float(always["accepted_updates_per_run"]):.2f} for Always-Adapt, an absolute difference of
{float(core_always["accepted_updates_per_run"]["absolute_difference"]):+.2f} and a reduction of
{abs(float(core_always["accepted_updates_per_run"]["percentage_difference"])):.2f}%.

### Core versus Always-Adapt

Core did not preserve Always-Adapt operating-envelope compliance. Core had
{float(core_always["unsafe_exposure_per_run"]["absolute_difference"]):+.2f} unsafe windows per
run and {float(core_always["compliance_rate"]["absolute_percentage_point_difference"]):+.2f}
percentage points of compliance relative to Always-Adapt. Across the 12 paired runs, Core had
fewer unsafe windows in {int(unsafe_counts["core_better_count"])}, equal unsafe windows in
{int(unsafe_counts["equal_count"])}, and more unsafe windows in
{int(unsafe_counts["core_worse_count"])} cases.

### Offline Oracle upper-bound comparison

The Offline Oracle is a non-deployable, evaluator-informed one-step upper-bound comparator.
It accepted {float(oracle["accepted_updates_per_run"]):.2f} updates per run while reducing unsafe
exposure by {abs(float(core_oracle["unsafe_exposure_per_run"]["absolute_difference"])):.2f}
windows per run relative to Core and improving compliance by
{abs(float(core_oracle["compliance_rate"]["absolute_percentage_point_difference"])):.2f}
percentage points. Its absolute compliance remained only
{100.0 * float(oracle["compliance_rate"]):.2f}%.

### Retention

Core's final mean PR-AUC, ROC-AUC, TPR, and FPR-budget ratio were
{float(core["final_pr_auc"]):.4f}, {float(core["final_roc_auc"]):.4f},
{float(core["final_tpr"]):.4f}, and {float(core["final_fpr_budget_ratio"]):.2f}. Its operational
TPR forgetting was {float(core["operational_tpr_forgetting"]):.4f}, compared with
{float(always["operational_tpr_forgetting"]):.4f} for Always-Adapt. This forgetting comparison
is descriptively favourable for Core; it does not offset the failed safety comparison.

### Rotation dependence

Core's paired safety result depended on deployment sequence. {rotation_text}. This pattern is
descriptive and does not add a new confirmatory hypothesis.

### H4 verdict

**{h4["verdict"]}.** Update reduction was supported, label reduction was not supported, and
comparable safety was not supported. Core used the same supervision as Always-Adapt and had
lower aggregate operating-envelope compliance.

### H10 verdict

**{h10["verdict"]}.** This denotes component-level support only: Core used fewer accepted
updates and had descriptively favourable operational forgetting, but it did not reduce
supervision or approach Always-Adapt safety. The central safety-efficiency proposition failed.

### Main Study-4 conclusion

The frozen Core policy reduced accepted update frequency by approximately 30% but did not
reduce label use and did not maintain Always-Adapt operating-envelope compliance. Minimum
intervention remains mechanistically plausible, as shown by the Oracle, but the deployable
Core treatment did not satisfy the central confirmatory safety-efficiency requirements.
"""


def _hypothesis_verdicts_text(headline: Mapping[str, Any]) -> str:
    hypotheses = cast(Mapping[str, Mapping[str, Any]], headline["confirmatory_hypotheses"])
    h4 = hypotheses["H4"]
    h10 = hypotheses["H10"]
    return f"""# Study 4 hypothesis verdicts

These verdicts are derived from the frozen confirmatory Study-4 results.

## H4

- Update reduction: supported
- Label reduction: not supported
- Comparable safety: not supported
- Overall: **{h4["verdict"]}**

## H10

- Fewer updates: supported
- Less supervision: not supported
- Operational forgetting: descriptively favourable
- Comparable safety: not supported
- Overall: **{h10["verdict"]}**

The central safety-efficiency proposition was not supported. The partial verdict denotes
component-level support only and does not override the failed safety and supervision conditions.
"""


def _discussion_notes_text(
    overall: pd.DataFrame,
    headline: Mapping[str, Any],
) -> str:
    static = _method_row(overall, Study4Method.STATIC.value)
    always = _method_row(overall, Study4Method.ALWAYS_ADAPT.value)
    oracle = _method_row(overall, Study4Method.OFFLINE_ORACLE.value)
    comparisons = cast(Mapping[str, Mapping[str, Any]], headline["confirmatory_comparisons"])
    always_static = comparisons["always_minus_static"]
    core_always = comparisons["core_minus_always"]
    resources = cast(Mapping[str, Mapping[str, Any]], headline["resource_totals"])
    action_totals = cast(Mapping[str, Mapping[str, int]], headline["action_attempt_totals"])
    paired = cast(Mapping[str, Any], headline["paired_core_vs_always"])
    rotations = cast(Sequence[Mapping[str, Any]], paired["rotation_differences"])
    rotation_lines = "\n".join(
        f"- {row['sequence']}: {float(row['mean_unsafe_exposure_difference']):+.2f} mean "
        "Core-minus-Always unsafe windows"
        for row in rotations
    )
    oracle_actions = action_totals[Study4Method.OFFLINE_ORACLE.value]
    oracle_total = sum(oracle_actions.values())
    oracle_a3 = oracle_actions["A3_FULL_FINE_TUNE"]
    return f"""# Study 4 discussion notes

These observations are descriptive or exploratory. They do not modify the frozen Study-4
hypotheses, controller, or scientific protocol.

## Always-Adapt versus Static

Always-Adapt was slightly worse than Static in aggregate safety despite using
{int(resources[Study4Method.ALWAYS_ADAPT.value]["labels_requested"]):,} labels. Its mean unsafe
exposure was {float(always["unsafe_exposure_per_run"]):.2f} rather than
{float(static["unsafe_exposure_per_run"]):.2f} windows per run, a difference of
{float(always_static["unsafe_exposure_per_run"]["absolute_difference"]):+.2f}. Its compliance
was {float(always_static["compliance_rate"]["absolute_percentage_point_difference"]):+.2f}
percentage points relative to Static. The frozen Always-Adapt mechanism therefore provided no
aggregate safety advantage over no adaptation.

## Core sparsity

Core achieved intervention sparsity but not supervision sparsity. It accepted
{int(resources[Study4Method.DANIDS_CORE.value]["accepted_updates"])} updates versus
{int(resources[Study4Method.ALWAYS_ADAPT.value]["accepted_updates"])} for Always-Adapt, a
{abs(float(core_always["accepted_updates_per_run"]["percentage_difference"])):.2f}% reduction,
but both methods requested
{int(resources[Study4Method.DANIDS_CORE.value]["labels_requested"]):,} labels. Health-aware
action selection was sparser; health-aware querying was not selective under the frozen rule.

## Offline Oracle and intervention capability

The non-deployable Offline Oracle's {oracle_total} sparse accepted updates support the
minimum-intervention motivation. Nevertheless, its absolute operating-envelope compliance was
only {100.0 * float(oracle["compliance_rate"]):.2f}%, meaning that most harm remained
unrecoverable under the frozen B100, delayed-supervision, and A0--A4 capability regime.

## Sequence dependence

Core's paired safety behaviour depended on deployment sequence:

{rotation_lines}

This is a descriptive pattern, not a new confirmatory hypothesis.

## Exploratory A3 observation

{oracle_a3}/{oracle_total} Oracle model-changing selections were A3 full fine-tuning. A3 was
prospectively excluded from normal Core. This observation is exploratory and mechanistic only;
it must not be used to retrospectively alter Core or its confirmatory interpretation.
"""


def validate_study4_confirmatory_analysis(output_dir: str | Path) -> Path:
    """Verify the canonical analysis file set and every persisted content digest."""

    output = Path(output_dir).resolve()
    manifest = _load_json(output / "analysis_manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(OUTPUT_FILES):
        raise Study4AnalysisError("Study-4 analysis manifest file set differs")
    for name in OUTPUT_FILES:
        expected_digest = files[name]
        path = output / name
        if not isinstance(expected_digest, str) or not path.is_file():
            raise Study4AnalysisError(f"Study-4 analysis manifest entry is invalid: {name}")
        if _file_sha256(path) != expected_digest:
            raise Study4AnalysisError(f"Study-4 analysis file digest differs: {name}")
    if manifest.get("bundle_digest") != _canonical_digest(files):
        raise Study4AnalysisError("Study-4 analysis bundle digest differs")
    return output


def analyze_study4_confirmatory(
    evaluation_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Validate the frozen evaluation and write a deterministic thesis analysis package."""

    inputs = load_study4_analysis_inputs(evaluation_dir)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Study-4 analysis directory: {output}")
    overall = build_overall_table(inputs)
    paired = build_core_vs_always_table(inputs)
    headline = build_headline_results(inputs, overall, paired)
    results_text = _confirmatory_results_text(inputs, overall, headline)
    verdicts_text = _hypothesis_verdicts_text(headline)
    discussion_text = _discussion_notes_text(overall, headline)

    output.mkdir(parents=True)
    _write_csv(output / "table_study4_overall.csv", overall)
    (output / "table_study4_overall.md").write_text(
        _overall_markdown(overall), encoding="utf-8", newline="\n"
    )
    _write_csv(output / "table_study4_core_vs_always.csv", paired)
    (output / "table_study4_core_vs_always.md").write_text(
        _paired_markdown(paired), encoding="utf-8", newline="\n"
    )
    (output / "study4_confirmatory_results.json").write_text(
        json.dumps(headline, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / "study4_confirmatory_results.md").write_text(
        results_text, encoding="utf-8", newline="\n"
    )
    (output / "study4_hypothesis_verdicts.md").write_text(
        verdicts_text, encoding="utf-8", newline="\n"
    )
    (output / "study4_discussion_notes.md").write_text(
        discussion_text, encoding="utf-8", newline="\n"
    )
    _save_safety_figure(overall, output / "fig_study4_safety_compliance.png")
    _save_resource_figure(overall, output / "fig_study4_resource_tradeoff.png")
    _save_paired_figure(paired, output / "fig_study4_core_vs_always_paired.png")
    _save_action_figure(
        inputs.tables["study4_action_distribution.csv"],
        output / "fig_study4_action_distribution.png",
    )
    _save_retention_figure(overall, output / "fig_study4_retention.png")
    _save_oracle_gap_figure(overall, output / "fig_study4_oracle_gap.png")
    produced = sorted(path.name for path in output.iterdir() if path.is_file())
    if produced != sorted(OUTPUT_FILES):
        raise Study4AnalysisError("Study-4 analysis output file set differs")
    files = {name: _file_sha256(output / name) for name in produced}
    manifest = {
        "version": ANALYSIS_VERSION,
        "source_evaluation_bundle_digest": inputs.manifest["bundle_digest"],
        "files": files,
        "bundle_digest": _canonical_digest(files),
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return validate_study4_confirmatory_analysis(output)


__all__ = [
    "ANALYSIS_VERSION",
    "Study4AnalysisError",
    "Study4AnalysisInputs",
    "analyze_study4_confirmatory",
    "build_core_vs_always_table",
    "build_headline_results",
    "build_overall_table",
    "determine_hypothesis_verdicts",
    "load_study4_analysis_inputs",
    "validate_study4_confirmatory_analysis",
]
