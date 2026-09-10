"""Thin command-line interface for dataset validation and benchmark dry runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from danids.config.continual import load_continual_experiment_config
from danids.config.core import load_study4_core_config
from danids.config.experiment import ExperimentConfig, load_experiment_config
from danids.config.health import load_health_experiment_config
from danids.config.static import load_static_experiment_config
from danids.continual.supervision import load_or_create_supervision_schedule
from danids.data.manifests import SplitManifest, generate_split_manifest
from danids.data.registry import DatasetRegistry
from danids.data.schema import discover_core_feature_contract, read_csv_header, validate_schema
from danids.evaluation.study1 import aggregate_static_study1
from danids.evaluation.study2 import aggregate_continual_study2
from danids.evaluation.study3 import evaluate_health_study3
from danids.experiments.continual import ContinualSmokeLimits, run_continual_experiment
from danids.experiments.health import HealthSmokeLimits, run_health_experiment
from danids.experiments.static import (
    SmokeLimits,
    load_or_generate_static_manifests,
    run_static_experiment,
)
from danids.policy.health_artifact import (
    build_health_model_artifact,
    validate_health_model_artifact,
)
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


def _run_static(args: argparse.Namespace) -> int:
    registry = DatasetRegistry.from_yaml(args.datasets_config)
    config = load_static_experiment_config(args.experiment_config).with_runtime_overrides(
        seed=args.seed, maximum_epochs=args.maximum_epochs
    )
    for dataset_id in config.experiment.sequence:
        registry[dataset_id]
    smoke = None
    if args.smoke:
        smoke = SmokeLimits(
            training_rows=args.smoke_training_rows,
            validation_rows=args.smoke_validation_rows,
            holdout_rows=args.smoke_holdout_rows,
            later_windows=args.smoke_later_windows,
        )
    output = run_static_experiment(
        registry,
        config,
        manifest_dir=args.manifest_dir,
        output_root=args.output_dir,
        device_name=args.device,
        smoke=smoke,
    )
    print(json.dumps({"run_directory": str(output)}, indent=2))
    return 0


def _aggregate_static_study1(args: argparse.Namespace) -> int:
    output = aggregate_static_study1(args.run_dirs, args.output_dir)
    summary = json.loads((output / "study1_summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"output_directory": str(output), **summary}, indent=2))
    return 0


def _generate_supervision_study2(args: argparse.Namespace) -> int:
    registry = DatasetRegistry.from_yaml(args.datasets_config)
    config = load_continual_experiment_config(args.experiment_config).with_runtime_overrides(
        seed=args.seed
    )
    contract = discover_core_feature_contract(registry)
    manifests = tuple(
        load_or_generate_static_manifests(registry, contract, config, args.manifest_dir)
    )
    schedule = load_or_create_supervision_schedule(
        args.output, manifests, seed=config.experiment.seed
    )
    print(json.dumps(schedule.to_dict(), indent=2))
    return 0


def _run_continual(args: argparse.Namespace) -> int:
    registry = DatasetRegistry.from_yaml(args.datasets_config)
    config = load_continual_experiment_config(args.experiment_config).with_runtime_overrides(
        seed=args.seed, adaptation_epochs=args.adaptation_epochs
    )
    smoke = None
    if args.smoke:
        smoke = ContinualSmokeLimits(
            later_stages=args.smoke_later_stages,
            later_windows=args.smoke_later_windows,
            holdout_rows=args.smoke_holdout_rows,
            source_state_rows=args.smoke_source_state_rows,
        )
    output = run_continual_experiment(
        registry,
        config,
        initial_run=args.initial_run,
        manifest_dir=args.manifest_dir,
        schedule_path=args.schedule,
        output_root=args.output_dir,
        device_name=args.device,
        smoke=smoke,
    )
    print(json.dumps({"run_directory": str(output)}, indent=2))
    return 0


def _aggregate_continual_study2(args: argparse.Namespace) -> int:
    output = aggregate_continual_study2(args.static_runs, args.run_dirs, args.output_dir)
    summary = json.loads((output / "study2_summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"output_directory": str(output), **summary}, indent=2))
    return 0


def _run_health_study3(args: argparse.Namespace) -> int:
    registry = DatasetRegistry.from_yaml(args.datasets_config)
    config = load_health_experiment_config(args.experiment_config).with_runtime_seed(args.seed)
    smoke = None
    if args.smoke:
        smoke = HealthSmokeLimits(
            source_control_windows=args.smoke_source_control_windows,
            later_stages=args.smoke_later_stages,
            later_windows=args.smoke_later_windows,
        )
    output = run_health_experiment(
        registry,
        config,
        initial_run=args.initial_run,
        manifest_dir=args.manifest_dir,
        output_root=args.output_dir,
        device_name=args.device,
        smoke=smoke,
    )
    print(json.dumps({"run_directory": str(output)}, indent=2))
    return 0


def _evaluate_health_study3(args: argparse.Namespace) -> int:
    output = evaluate_health_study3(args.run_dirs, args.output_dir)
    summary = json.loads((output / "study3_summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"output_directory": str(output), **summary}, indent=2))
    return 0


def _validate_core_config_study4(args: argparse.Namespace) -> int:
    config = load_study4_core_config(args.core_config)
    print(json.dumps(config.to_dict(), indent=2))
    return 0


def _build_health_model_study4(args: argparse.Namespace) -> int:
    config = load_study4_core_config(args.core_config)
    output = build_health_model_artifact(args.study3_dataset, args.output_dir)
    frozen = validate_health_model_artifact(
        output,
        study3_dataset_path=args.study3_dataset,
    )
    print(
        json.dumps(
            {
                "experiment_id": config.experiment.experiment_id,
                "artifact_directory": str(output),
                "artifact_identity_sha256": frozen.artifact_identity,
                "serialized_model_sha256": frozen.serialized_model_sha256,
                "tau_safe": config.health.tau_safe,
                "tau_harmful": config.health.tau_harmful,
            },
            indent=2,
        )
    )
    return 0


def _validate_health_model_study4(args: argparse.Namespace) -> int:
    frozen = validate_health_model_artifact(
        args.artifact_dir,
        study3_dataset_path=args.study3_dataset,
    )
    print(
        json.dumps(
            {
                "artifact_directory": str(Path(args.artifact_dir).resolve()),
                "artifact_identity_sha256": frozen.artifact_identity,
                "serialized_model_sha256": frozen.serialized_model_sha256,
                "status": "valid",
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

    run_static = subparsers.add_parser(
        "run-static", help="train and evaluate the frozen TASK-002 static MLP"
    )
    run_static.add_argument("--datasets-config", required=True, type=Path)
    run_static.add_argument("--experiment-config", required=True, type=Path)
    run_static.add_argument("--manifest-dir", required=True, type=Path)
    run_static.add_argument("--output-dir", required=True, type=Path)
    run_static.add_argument("--device", default="auto")
    run_static.add_argument("--seed", type=int)
    run_static.add_argument("--maximum-epochs", type=int)
    run_static.add_argument("--smoke", action="store_true")
    run_static.add_argument("--smoke-training-rows", type=int, default=20_000)
    run_static.add_argument("--smoke-validation-rows", type=int, default=10_000)
    run_static.add_argument("--smoke-holdout-rows", type=int, default=10_000)
    run_static.add_argument("--smoke-later-windows", type=int, default=1)
    run_static.set_defaults(handler=_run_static)

    aggregate = subparsers.add_parser(
        "aggregate-static-study1",
        help="strictly aggregate completed static run artifacts for Study 1",
    )
    aggregate.add_argument("--run-dir", dest="run_dirs", required=True, action="append", type=Path)
    aggregate.add_argument("--output-dir", required=True, type=Path)
    aggregate.set_defaults(handler=_aggregate_static_study1)

    generate_supervision = subparsers.add_parser(
        "generate-supervision-study2",
        help="generate/reuse the shared deterministic TASK-004 label schedule",
    )
    generate_supervision.add_argument("--datasets-config", required=True, type=Path)
    generate_supervision.add_argument("--experiment-config", required=True, type=Path)
    generate_supervision.add_argument("--manifest-dir", required=True, type=Path)
    generate_supervision.add_argument("--output", required=True, type=Path)
    generate_supervision.add_argument("--seed", type=int)
    generate_supervision.set_defaults(handler=_generate_supervision_study2)

    run_continual = subparsers.add_parser(
        "run-continual", help="run one controlled TASK-004 continual baseline"
    )
    run_continual.add_argument("--datasets-config", required=True, type=Path)
    run_continual.add_argument("--experiment-config", required=True, type=Path)
    run_continual.add_argument("--initial-run", required=True, type=Path)
    run_continual.add_argument("--manifest-dir", required=True, type=Path)
    run_continual.add_argument("--schedule", required=True, type=Path)
    run_continual.add_argument("--output-dir", required=True, type=Path)
    run_continual.add_argument("--device", default="auto")
    run_continual.add_argument("--seed", type=int)
    run_continual.add_argument("--adaptation-epochs", type=int)
    run_continual.add_argument("--smoke", action="store_true")
    run_continual.add_argument("--smoke-later-stages", type=int, default=1)
    run_continual.add_argument("--smoke-later-windows", type=int, default=2)
    run_continual.add_argument("--smoke-holdout-rows", type=int, default=1_000)
    run_continual.add_argument("--smoke-source-state-rows", type=int, default=1_000)
    run_continual.set_defaults(handler=_run_continual)

    aggregate_continual = subparsers.add_parser(
        "aggregate-continual-study2",
        help="strictly aggregate completed E1/E2 artifacts for Study 2",
    )
    aggregate_continual.add_argument(
        "--static-run", dest="static_runs", required=True, action="append", type=Path
    )
    aggregate_continual.add_argument(
        "--run-dir", dest="run_dirs", required=True, action="append", type=Path
    )
    aggregate_continual.add_argument("--output-dir", required=True, type=Path)
    aggregate_continual.set_defaults(handler=_aggregate_continual_study2)

    run_health = subparsers.add_parser(
        "run-health-study3", help="extract one frozen-model TASK-005 health episode"
    )
    run_health.add_argument("--datasets-config", required=True, type=Path)
    run_health.add_argument("--experiment-config", required=True, type=Path)
    run_health.add_argument("--initial-run", type=Path)
    run_health.add_argument("--manifest-dir", required=True, type=Path)
    run_health.add_argument("--output-dir", required=True, type=Path)
    run_health.add_argument("--device", default="auto")
    run_health.add_argument("--seed", type=int)
    run_health.add_argument("--smoke", action="store_true")
    run_health.add_argument("--smoke-source-control-windows", type=int, default=2)
    run_health.add_argument("--smoke-later-stages", type=int, default=1)
    run_health.add_argument("--smoke-later-windows", type=int, default=3)
    run_health.set_defaults(handler=_run_health_study3)

    evaluate_health = subparsers.add_parser(
        "evaluate-health-study3", help="artifact-only grouped Study-3 health evaluation"
    )
    evaluate_health.add_argument(
        "--run-dir", dest="run_dirs", required=True, action="append", type=Path
    )
    evaluate_health.add_argument("--output-dir", required=True, type=Path)
    evaluate_health.set_defaults(handler=_evaluate_health_study3)

    validate_core = subparsers.add_parser(
        "validate-core-config-study4",
        help="validate the prospectively frozen TASK-006 Core configuration",
    )
    validate_core.add_argument("--core-config", required=True, type=Path)
    validate_core.set_defaults(handler=_validate_core_config_study4)

    build_core_health = subparsers.add_parser(
        "build-health-model-study4",
        help="build the frozen write-once TASK-006 Core health-model artifact",
    )
    build_core_health.add_argument("--core-config", required=True, type=Path)
    build_core_health.add_argument("--study3-dataset", required=True, type=Path)
    build_core_health.add_argument("--output-dir", required=True, type=Path)
    build_core_health.set_defaults(handler=_build_health_model_study4)

    validate_core_health = subparsers.add_parser(
        "validate-health-model-study4",
        help="strictly validate a frozen TASK-006 Core health-model artifact",
    )
    validate_core_health.add_argument("--artifact-dir", required=True, type=Path)
    validate_core_health.add_argument("--study3-dataset", type=Path)
    validate_core_health.set_defaults(handler=_validate_health_model_study4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
