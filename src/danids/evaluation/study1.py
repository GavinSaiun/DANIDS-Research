"""Strict artifact-only aggregation for the Study-1 static transfer matrix."""

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

import yaml

CANONICAL_DOMAINS = ("U", "T", "C", "B")
FROZEN_ROTATIONS = (
    ("U", "T", "C", "B"),
    ("T", "C", "B", "U"),
    ("C", "B", "U", "T"),
    ("B", "U", "T", "C"),
)
TARGET_FPR = 0.001
WINDOW_SIZE = 50_000
FLOAT_TOLERANCE = 1e-12
REQUIRED_FILES = (
    "summary.json",
    "config.resolved.yaml",
    "provenance.json",
    "holdout_metrics.csv",
    "retention_matrix.csv",
    "native_attack_metrics.csv",
)
CORE_METRICS = (
    "pr_auc",
    "pr_auc_minus_prevalence",
    "roc_auc",
    "tpr",
    "fpr",
    "threshold_transfer_ratio",
)
MATRIX_METRICS = {
    "pr_auc": "pr_auc",
    "pr_auc_minus_prevalence": "pr_auc_minus_prevalence",
    "roc_auc": "roc_auc",
    "tpr": "tpr",
    "fpr": "fpr",
    "ttr": "threshold_transfer_ratio",
}
ASYMMETRY_METRICS = (
    "roc_auc",
    "pr_auc_minus_prevalence",
    "fpr",
    "tpr",
)


class Study1AggregationError(ValueError):
    """Raised when an input artifact violates the frozen Study-1 contract."""


@dataclass(frozen=True, slots=True)
class ValidatedRun:
    path: Path
    seed: int
    source: str
    sequence: tuple[str, ...]
    dataset_fingerprints: tuple[tuple[str, str], ...]
    contract_signature: str
    final_holdouts: tuple[dict[str, str], ...]
    final_native_rows: tuple[dict[str, str], ...]


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Study1AggregationError(f"{context} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise Study1AggregationError(f"{context} is missing {key!r}")
    return mapping[key]


def _nested(mapping: Mapping[str, Any], keys: Sequence[str], context: str) -> Any:
    current: Any = mapping
    for key in keys:
        current = _required(_mapping(current, context), key, context)
    return current


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Study1AggregationError(f"cannot read {path}: {exc}") from exc
    return _mapping(value, str(path))


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Study1AggregationError(f"cannot read {path}: {exc}") from exc
    return _mapping(value, str(path))


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise Study1AggregationError(f"{path} has no CSV header")
            rows = []
            for raw in reader:
                if None in raw:
                    raise Study1AggregationError(f"{path} has rows wider than its header")
                rows.append({key: value or "" for key, value in raw.items()})
    except OSError as exc:
        raise Study1AggregationError(f"cannot read {path}: {exc}") from exc
    return rows


def _require_columns(rows: Sequence[Mapping[str, str]], columns: set[str], path: Path) -> None:
    if not rows:
        raise Study1AggregationError(f"{path} contains no rows")
    missing = columns.difference(rows[0])
    if missing:
        raise Study1AggregationError(f"{path} is missing columns: {', '.join(sorted(missing))}")


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool):
        raise Study1AggregationError(f"{context} must be an integer")
    try:
        result = int(str(value))
    except (TypeError, ValueError) as exc:
        raise Study1AggregationError(f"{context} must be an integer") from exc
    return result


def _number(value: object, context: str, *, optional: bool = False) -> float | None:
    if optional and (value is None or str(value).strip().lower() in {"", "none", "nan"}):
        return None
    try:
        result = float(str(value))
    except (TypeError, ValueError) as exc:
        raise Study1AggregationError(f"{context} must be numeric") from exc
    if not math.isfinite(result):
        if optional:
            return None
        raise Study1AggregationError(f"{context} must be finite")
    return result


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE)


def _same_csv_value(left: str, right: str) -> bool:
    if left == right:
        return True
    try:
        left_number = float(left)
        right_number = float(right)
    except ValueError:
        return False
    if not math.isfinite(left_number) or not math.isfinite(right_number):
        return False
    return _close(left_number, right_number)


def _validate_frozen(summary: Mapping[str, Any], run_dir: Path) -> None:
    frozen = _mapping(
        _required(summary, "frozen_state_evidence", str(run_dir / "summary.json")),
        f"{run_dir}/summary.json frozen_state_evidence",
    )
    for state in ("model", "preprocessor", "threshold"):
        if frozen.get(f"{state}_unchanged") is not True:
            raise Study1AggregationError(f"{run_dir}: {state} was not frozen")
        if frozen.get(f"{state}_before") != frozen.get(f"{state}_after"):
            raise Study1AggregationError(f"{run_dir}: {state} digests differ")
    if _integer(frozen.get("target_optimizer_steps"), f"{run_dir}: target optimizer steps") != 0:
        raise Study1AggregationError(f"{run_dir}: target_optimizer_steps must be zero")


def _validate_triangle(
    rows: Sequence[dict[str, str]], sequence: tuple[str, ...], path: Path
) -> tuple[dict[str, str], ...]:
    required = {
        "stage",
        "holdout_dataset_id",
        "row_count",
        "attack_count",
        "benign_count",
        "attack_prevalence",
        "pr_auc",
        "roc_auc",
        "tpr",
        "fpr",
        "precision",
        "macro_f1",
        "false_positives_per_million",
        "source_holdout_fpr",
        "threshold_transfer_ratio",
        "threshold_transfer_undefined_reason",
    }
    _require_columns(rows, required, path)
    by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
    for stage in range(1, 5):
        stage_rows = [row for row in rows if _integer(row["stage"], f"{path}: stage") == stage]
        targets = [row["holdout_dataset_id"] for row in stage_rows]
        expected = list(sequence[:stage])
        if sorted(targets) != sorted(expected) or len(targets) != len(set(targets)):
            raise Study1AggregationError(
                f"{path}: stage {stage} must contain exactly {expected}, got {targets}"
            )
        for row in stage_rows:
            by_target[row["holdout_dataset_id"]].append(row)
    if len(rows) != 10:
        raise Study1AggregationError(f"{path}: complete triangular run must contain 10 rows")
    for target, repeated in by_target.items():
        first = repeated[0]
        for current in repeated[1:]:
            for column in set(first).difference({"stage", "holdout_dataset_id"}):
                if not _same_csv_value(first[column], current[column]):
                    raise Study1AggregationError(
                        f"{path}: static holdout metric {column!r} changed across stages "
                        f"for {target}"
                    )
    final = tuple(row for row in rows if _integer(row["stage"], f"{path}: stage") == 4)
    if len(final) != 4 or {row["holdout_dataset_id"] for row in final} != set(CANONICAL_DOMAINS):
        raise Study1AggregationError(f"{path}: final stage must contain U, T, C and B exactly once")
    return tuple(sorted(final, key=lambda row: CANONICAL_DOMAINS.index(row["holdout_dataset_id"])))


def _validate_retention(
    rows: Sequence[dict[str, str]],
    sequence: tuple[str, ...],
    holdouts: Sequence[dict[str, str]],
    path: Path,
) -> None:
    required = {"stage", "holdout_dataset_id", "pr_auc", "roc_auc", "fpr", "tpr"}
    _require_columns(rows, required, path)
    if len(rows) != 10:
        raise Study1AggregationError(f"{path}: complete retention triangle must contain 10 rows")
    holdout_lookup = {
        (_integer(row["stage"], "holdout stage"), row["holdout_dataset_id"]): row
        for row in holdouts
    }
    for stage in range(1, 5):
        stage_rows = [row for row in rows if _integer(row["stage"], f"{path}: stage") == stage]
        if {row["holdout_dataset_id"] for row in stage_rows} != set(sequence[:stage]):
            raise Study1AggregationError(f"{path}: retention stage {stage} has wrong domains")
        for row in stage_rows:
            key = (stage, row["holdout_dataset_id"])
            held = holdout_lookup.get(key)
            if held is None:
                raise Study1AggregationError(f"{path}: retention cell {key} lacks holdout cell")
            for metric in ("pr_auc", "roc_auc", "fpr", "tpr"):
                if not _same_csv_value(row[metric], held[metric]):
                    raise Study1AggregationError(
                        f"{path}: retention {metric} disagrees with holdout metrics for {key}"
                    )


def _contract_signature(config: Mapping[str, Any], provenance: Mapping[str, Any]) -> str:
    signature = {
        "study": _required(config, "study", "config"),
        "boundary_mode": _nested(config, ("stream", "boundary_mode"), "config"),
        "splits": _required(config, "splits", "config"),
        "model": _required(config, "model", "config"),
        "training": _required(config, "training", "config"),
        "feature_contract_version": _required(provenance, "feature_contract_version", "provenance"),
        "feature_count": _required(provenance, "feature_count", "provenance"),
        "preprocessor_version": _required(provenance, "preprocessor_version", "provenance"),
        "materializer_version": provenance.get("materializer_version"),
        "split_version": _required(provenance, "split_version", "provenance"),
        "config_split_version": _nested(config, ("datasets", "split_version"), "config"),
    }
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def _dataset_fingerprints(
    provenance: Mapping[str, Any], run_dir: Path
) -> tuple[tuple[str, str], ...]:
    raw = _required(provenance, "dataset_fingerprints", f"{run_dir}/provenance.json")
    fingerprints = _mapping(raw, f"{run_dir}/provenance.json dataset_fingerprints")
    actual_domains = set(fingerprints)
    expected_domains = set(CANONICAL_DOMAINS)
    if actual_domains != expected_domains:
        missing = sorted(expected_domains.difference(actual_domains))
        extra = sorted(actual_domains.difference(expected_domains))
        raise Study1AggregationError(
            f"{run_dir}: dataset_fingerprints must contain exactly U, T, C and B "
            f"(missing={missing}, extra={extra})"
        )
    result: list[tuple[str, str]] = []
    for dataset_id in CANONICAL_DOMAINS:
        value = fingerprints[dataset_id]
        if not isinstance(value, str) or not value.strip():
            raise Study1AggregationError(
                f"{run_dir}: dataset fingerprint for {dataset_id} must be a non-empty string"
            )
        result.append((dataset_id, value))
    return tuple(result)


def validate_static_study1_run(run_dir: str | Path) -> ValidatedRun:
    """Load and strictly validate one complete, non-smoke TASK-002 static run."""

    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise Study1AggregationError(f"run directory does not exist: {root}")
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise Study1AggregationError(f"{root}: missing required files: {', '.join(missing)}")
    summary = _load_json(root / "summary.json")
    config = _load_yaml(root / "config.resolved.yaml")
    provenance = _load_json(root / "provenance.json")
    if summary.get("smoke") is not False:
        raise Study1AggregationError(f"{root}: smoke runs cannot be aggregated")
    if config.get("smoke") not in (None, False):
        raise Study1AggregationError(f"{root}: resolved config records smoke limits")

    raw_sequence = _nested(config, ("datasets", "sequence"), "config")
    if not isinstance(raw_sequence, list) or not all(
        isinstance(item, str) for item in raw_sequence
    ):
        raise Study1AggregationError(f"{root}: datasets.sequence must be a string list")
    sequence = tuple(raw_sequence)
    if sequence not in FROZEN_ROTATIONS:
        raise Study1AggregationError(f"{root}: sequence is not a frozen Study-1 rotation")
    summary_sequence = summary.get("sequence")
    if summary_sequence != list(sequence):
        raise Study1AggregationError(f"{root}: summary/config sequences disagree")
    source = sequence[0]

    seed = _integer(_required(config, "seed", "config"), f"{root}: config seed")
    provenance_seed = _integer(
        _required(provenance, "seed", "provenance"), f"{root}: provenance seed"
    )
    if seed != provenance_seed:
        raise Study1AggregationError(f"{root}: config/provenance seeds disagree")
    if "seed" in summary and _integer(summary["seed"], f"{root}: summary seed") != seed:
        raise Study1AggregationError(f"{root}: config/summary seeds disagree")
    target_fpr = _number(
        _nested(config, ("operating_envelope", "target_fpr"), "config"),
        f"{root}: target FPR",
    )
    if target_fpr is None or not _close(target_fpr, TARGET_FPR):
        raise Study1AggregationError(f"{root}: target FPR must be {TARGET_FPR}")
    window_size = _integer(
        _nested(config, ("stream", "window_size"), "config"), f"{root}: window size"
    )
    if window_size != WINDOW_SIZE:
        raise Study1AggregationError(f"{root}: window size must be {WINDOW_SIZE}")
    if _integer(provenance.get("feature_count"), f"{root}: feature count") != 47:
        raise Study1AggregationError(f"{root}: primary feature count must be 47")
    config_split = _nested(config, ("datasets", "split_version"), "config")
    if provenance.get("split_version") != config_split:
        raise Study1AggregationError(f"{root}: config/provenance split versions disagree")
    _validate_frozen(summary, root)

    all_holdouts = _read_csv(root / "holdout_metrics.csv")
    final_holdouts = _validate_triangle(all_holdouts, sequence, root / "holdout_metrics.csv")
    self_row = next(row for row in final_holdouts if row["holdout_dataset_id"] == source)
    self_ttr = _number(
        self_row["threshold_transfer_ratio"],
        f"{root}: self threshold transfer ratio",
        optional=True,
    )
    if self_ttr is not None and not _close(self_ttr, 1.0):
        raise Study1AggregationError(f"{root}: defined source/self TTR must equal 1")
    _validate_retention(
        _read_csv(root / "retention_matrix.csv"),
        sequence,
        all_holdouts,
        root / "retention_matrix.csv",
    )

    native_rows = _read_csv(root / "native_attack_metrics.csv")
    native_required = {
        "scope",
        "stage",
        "holdout_dataset_id",
        "native_attack_label",
        "support",
        "recall",
        "macro_native_attack_recall",
        "worst_native_attack_recall",
    }
    _require_columns(native_rows, native_required, root / "native_attack_metrics.csv")
    final_native = tuple(
        row
        for row in native_rows
        if row["scope"] == "holdout" and _integer(row["stage"], "native stage") == 4
    )
    if not final_native:
        raise Study1AggregationError(f"{root}: no final-stage holdout native attack rows")
    unknown_targets = {row["holdout_dataset_id"] for row in final_native}.difference(
        CANONICAL_DOMAINS
    )
    if unknown_targets:
        raise Study1AggregationError(f"{root}: native rows contain unknown targets")
    native_targets = {row["holdout_dataset_id"] for row in final_native}
    if native_targets != set(CANONICAL_DOMAINS):
        raise Study1AggregationError(f"{root}: final native rows must cover all four domains")

    return ValidatedRun(
        path=root,
        seed=seed,
        source=source,
        sequence=sequence,
        dataset_fingerprints=_dataset_fingerprints(provenance, root),
        contract_signature=_contract_signature(config, provenance),
        final_holdouts=final_holdouts,
        final_native_rows=final_native,
    )


def _optional_metric(row: Mapping[str, str], name: str, context: str) -> float | None:
    return _number(row[name], f"{context}: {name}", optional=True)


def _transfer_row(run: ValidatedRun, raw: Mapping[str, str]) -> dict[str, Any]:
    target = raw["holdout_dataset_id"]
    row_count = _integer(raw["row_count"], "row count")
    attack_count = _integer(raw["attack_count"], "attack count")
    benign_count = _integer(raw["benign_count"], "benign count")
    prevalence = _number(raw["attack_prevalence"], "attack prevalence")
    assert prevalence is not None
    if row_count != attack_count + benign_count:
        raise Study1AggregationError(
            f"{run.path}: {target} row count differs from attack plus benign support"
        )
    if row_count <= 0 or not _close(prevalence, attack_count / row_count):
        raise Study1AggregationError(f"{run.path}: {target} attack prevalence is inconsistent")
    pr_auc = _optional_metric(raw, "pr_auc", "holdout")
    roc_auc = _optional_metric(raw, "roc_auc", "holdout")
    fpr = _optional_metric(raw, "fpr", "holdout")
    return {
        "seed": run.seed,
        "source_domain": run.source,
        "target_domain": target,
        "transfer_type": "self" if run.source == target else "transfer",
        "row_count": row_count,
        "attack_count": attack_count,
        "benign_count": benign_count,
        "attack_prevalence": prevalence,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "tpr": _optional_metric(raw, "tpr", "holdout"),
        "fpr": fpr,
        "precision": _optional_metric(raw, "precision", "holdout"),
        "macro_f1": _optional_metric(raw, "macro_f1", "holdout"),
        "false_positives_per_million": _optional_metric(
            raw, "false_positives_per_million", "holdout"
        ),
        "source_holdout_fpr": _optional_metric(raw, "source_holdout_fpr", "holdout"),
        "threshold_transfer_ratio": _optional_metric(raw, "threshold_transfer_ratio", "holdout"),
        "threshold_transfer_undefined_reason": raw["threshold_transfer_undefined_reason"],
        "pr_auc_random_baseline": prevalence,
        "pr_auc_minus_prevalence": None if pr_auc is None else pr_auc - prevalence,
        "fpr_budget_ratio": None if fpr is None else fpr / TARGET_FPR,
        "roc_auc_below_chance": None if roc_auc is None else roc_auc < 0.5,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_matrix(
    path: Path, cells: Mapping[tuple[int, str, str], Mapping[str, Any]], seed: int, metric: str
) -> None:
    rows: list[dict[str, Any]] = []
    for source in CANONICAL_DOMAINS:
        row: dict[str, Any] = {"source_domain": source}
        for target in CANONICAL_DOMAINS:
            row[target] = cells[(seed, source, target)][metric]
        rows.append(row)
    _write_csv(path, rows, ("source_domain", *CANONICAL_DOMAINS))


def _seed_summary(transfer_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for source in CANONICAL_DOMAINS:
        for target in CANONICAL_DOMAINS:
            matching = [
                row
                for row in transfer_rows
                if row["source_domain"] == source and row["target_domain"] == target
            ]
            if not matching:
                continue
            for metric in CORE_METRICS:
                values = [float(row[metric]) for row in matching if row[metric] is not None]
                summaries.append(
                    {
                        "source_domain": source,
                        "target_domain": target,
                        "metric": metric,
                        "n_seeds": len(values),
                        "mean": statistics.fmean(values) if values else None,
                        "sample_std": statistics.stdev(values) if len(values) >= 2 else None,
                        "minimum": min(values) if values else None,
                        "maximum": max(values) if values else None,
                    }
                )
    return summaries


def _asymmetry(
    cells: Mapping[tuple[int, str, str], Mapping[str, Any]], seeds: Sequence[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for left_index, domain_a in enumerate(CANONICAL_DOMAINS):
            for domain_b in CANONICAL_DOMAINS[left_index + 1 :]:
                a_to_b = cells.get((seed, domain_a, domain_b))
                b_to_a = cells.get((seed, domain_b, domain_a))
                if a_to_b is None or b_to_a is None:
                    continue
                for metric in ASYMMETRY_METRICS:
                    left = a_to_b[metric]
                    right = b_to_a[metric]
                    if left is None or right is None:
                        continue
                    rows.append(
                        {
                            "seed": seed,
                            "domain_a": domain_a,
                            "domain_b": domain_b,
                            "metric": metric,
                            "a_to_b": left,
                            "b_to_a": right,
                            "signed_difference": float(left) - float(right),
                        }
                    )
    return rows


def aggregate_static_study1(run_dirs: Sequence[str | Path], output_dir: str | Path) -> Path:
    """Validate static runs and write deterministic Study-1 tables without raw-data access."""

    if not run_dirs:
        raise Study1AggregationError("at least one --run-dir is required")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite aggregation directory: {output}")
    runs = [validate_static_study1_run(path) for path in run_dirs]
    seen: set[tuple[int, str]] = set()
    for run in runs:
        identity = (run.seed, run.source)
        if identity in seen:
            raise Study1AggregationError(f"duplicate Study-1 run for seed/source {identity}")
        seen.add(identity)
    reference = runs[0]
    for run in runs[1:]:
        if run.dataset_fingerprints != reference.dataset_fingerprints:
            reference_fingerprints = dict(reference.dataset_fingerprints)
            run_fingerprints = dict(run.dataset_fingerprints)
            differing = [
                dataset_id
                for dataset_id in CANONICAL_DOMAINS
                if reference_fingerprints[dataset_id] != run_fingerprints[dataset_id]
            ]
            raise Study1AggregationError(
                "dataset provenance differs for "
                + ", ".join(differing)
                + f" between {reference.path} and {run.path}"
            )
    signatures = {run.contract_signature for run in runs}
    if len(signatures) != 1:
        raise Study1AggregationError(
            "input runs have inconsistent model/training/preprocessing/materializer/feature/split "
            "contracts"
        )
    runs.sort(key=lambda run: (run.seed, CANONICAL_DOMAINS.index(run.source)))

    transfer_rows = [_transfer_row(run, raw) for run in runs for raw in run.final_holdouts]
    cells = {
        (int(row["seed"]), str(row["source_domain"]), str(row["target_domain"])): row
        for row in transfer_rows
    }
    native_rows: list[dict[str, Any]] = []
    for run in runs:
        for raw in sorted(
            run.final_native_rows,
            key=lambda row: (
                CANONICAL_DOMAINS.index(row["holdout_dataset_id"]),
                row["native_attack_label"],
            ),
        ):
            native_rows.append(
                {
                    "seed": run.seed,
                    "source_domain": run.source,
                    "target_domain": raw["holdout_dataset_id"],
                    "native_attack_label": raw["native_attack_label"],
                    "support": _integer(raw["support"], "native attack support"),
                    "recall": _optional_metric(raw, "recall", "native attack"),
                    "macro_native_attack_recall": _optional_metric(
                        raw, "macro_native_attack_recall", "native attack"
                    ),
                    "worst_native_attack_recall": _optional_metric(
                        raw, "worst_native_attack_recall", "native attack"
                    ),
                }
            )

    seeds = sorted({run.seed for run in runs})
    sources_by_seed = {seed: {run.source for run in runs if run.seed == seed} for seed in seeds}
    complete_seeds = [seed for seed in seeds if sources_by_seed[seed] == set(CANONICAL_DOMAINS)]
    missing = {
        str(seed): [domain for domain in CANONICAL_DOMAINS if domain not in sources_by_seed[seed]]
        for seed in seeds
        if seed not in complete_seeds
    }
    summary = {
        "status": "complete" if len(complete_seeds) == len(seeds) else "incomplete",
        "complete": len(complete_seeds) == len(seeds),
        "run_count": len(runs),
        "cell_count": len(transfer_rows),
        "seeds": seeds,
        "complete_seeds": complete_seeds,
        "missing_sources_by_seed": missing,
        "canonical_domain_order": list(CANONICAL_DOMAINS),
        "matrix_orientation": "rows=source_domain, columns=target_domain",
        "asymmetry_definition": "a_to_b - b_to_a (signed; descriptive, not inferential)",
        "directional_asymmetry": _asymmetry(cells, seeds),
        "interpretation": (
            "Each row source is an independently trained frozen static model. Later sequence "
            "position has no causal effect; these outputs describe transfer asymmetry, not "
            "continual-learning order sensitivity."
        ),
    }

    output.mkdir(parents=True)
    transfer_columns = tuple(transfer_rows[0])
    _write_csv(output / "study1_transfer_long.csv", transfer_rows, transfer_columns)
    native_columns = tuple(native_rows[0])
    _write_csv(output / "study1_native_attack_long.csv", native_rows, native_columns)
    seed_rows = _seed_summary(transfer_rows)
    _write_csv(output / "study1_seed_summary.csv", seed_rows, tuple(seed_rows[0]))
    (output / "study1_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for seed in complete_seeds:
        for filename, metric in MATRIX_METRICS.items():
            _write_matrix(output / f"{filename}_matrix_s{seed}.csv", cells, seed, metric)
    return output


__all__ = [
    "CANONICAL_DOMAINS",
    "Study1AggregationError",
    "ValidatedRun",
    "aggregate_static_study1",
    "validate_static_study1_run",
]
