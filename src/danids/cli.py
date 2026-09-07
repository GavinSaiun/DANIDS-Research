"""Thin command-line interface for dataset validation and benchmark dry runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from danids.config.experiment import ExperimentConfig, load_experiment_config
from danids.data.manifests import SplitManifest, generate_split_manifest
from danids.data.registry import DatasetRegistry
from danids.data.schema import discover_core_feature_contract, read_csv_header, validate_schema
from danids.utils.reproducibility import set_global_seed


def _registry_and_experiment(args: argparse.Namespace) -> tuple[DatasetRegistry, ExperimentConfig]:
    registry = DatasetRegistry.from_yaml(args.datasets_config)
    experiment = load_experiment_config(args.experiment_config)
    for dataset_id in experiment.sequence:
        registry[dataset_id]
    return registry, experiment


def _manifest_sequence(
    registry: DatasetRegistry, experiment: ExperimentConfig
) -> list[SplitManifest]:
    contract = discover_core_feature_contract(registry)
    manifests: list[SplitManifest] = []
    for stage, dataset_id in enumerate(experiment.sequence):
        manifests.append(
            generate_split_manifest(
                registry[dataset_id],
                contract,
                role="initial" if stage == 0 else "later",
                split_version=experiment.split_version,
                seed=experiment.seed,
                splits=experiment.splits,
            )
        )
    return manifests


def _validate(args: argparse.Namespace) -> int:
    registry = DatasetRegistry.from_yaml(args.datasets_config)
    requested: tuple[str, ...] = tuple(args.ids) if args.ids else tuple(registry)
    rows: list[dict[str, Any]] = []
    for dataset_id in requested:
        spec = registry[dataset_id]
        columns = read_csv_header(spec.path)
        validate_schema(spec, columns)
        rows.append(
            {
                "dataset_id": dataset_id,
                "name": spec.name,
                "path": str(spec.path),
                "column_count": len(columns),
                "status": "valid",
            }
        )
    print(json.dumps({"datasets": rows}, indent=2))
    return 0


def _generate(args: argparse.Namespace) -> int:
    registry, experiment = _registry_and_experiment(args)
    set_global_seed(experiment.seed)
    output_dir = Path(args.output_dir)
    written: list[str] = []
    for stage, manifest in enumerate(_manifest_sequence(registry, experiment), start=1):
        output = output_dir / f"stage-{stage:02d}-{manifest.dataset_id}.json"
        manifest.write(output)
        written.append(str(output))
    print(json.dumps({"experiment_id": experiment.experiment_id, "manifests": written}, indent=2))
    return 0


def _dry_run(args: argparse.Namespace) -> int:
    registry, experiment = _registry_and_experiment(args)
    set_global_seed(experiment.seed)
    summaries: list[dict[str, Any]] = []
    for stage, manifest in enumerate(_manifest_sequence(registry, experiment), start=1):
        if manifest.online_stream is None:
            windows = 0
            stream_rows = 0
        else:
            stream_rows = manifest.online_stream.size
            windows = (stream_rows + experiment.window_size - 1) // experiment.window_size
        summaries.append(
            {
                "stage": stage,
                "dataset_id": manifest.dataset_id,
                "role": manifest.domain_role,
                "rows": manifest.row_count,
                "stream_rows": stream_rows,
                "windows": windows,
                "holdout_rows": manifest.permanent_holdout.size,
                "chronological_start": manifest.chronological_start,
                "chronological_end": manifest.chronological_end,
            }
        )
    print(
        json.dumps(
            {
                "experiment_id": experiment.experiment_id,
                "seed": experiment.seed,
                "sequence": list(experiment.sequence),
                "window_size": experiment.window_size,
                "target_fpr": experiment.target_fpr,
                "stages": summaries,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="danids", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate configured dataset schemas")
    validate.add_argument("--datasets-config", required=True, type=Path)
    validate.add_argument("--ids", nargs="*", help="dataset IDs (default: every configured ID)")
    validate.set_defaults(handler=_validate)

    generate = subparsers.add_parser(
        "generate-manifests", help="generate deterministic chronological manifests"
    )
    generate.add_argument("--datasets-config", required=True, type=Path)
    generate.add_argument("--experiment-config", required=True, type=Path)
    generate.add_argument("--output-dir", required=True, type=Path)
    generate.set_defaults(handler=_generate)

    dry_run = subparsers.add_parser("dry-run", help="print sequence partitions and window counts")
    dry_run.add_argument("--datasets-config", required=True, type=Path)
    dry_run.add_argument("--experiment-config", required=True, type=Path)
    dry_run.set_defaults(handler=_dry_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
