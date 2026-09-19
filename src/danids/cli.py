"""DANIDS Study 1--5 execution, artifact evaluation, and thesis-release utilities."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from danids.attacks import (
    load_study5_contract,
    validate_configured_dataset_inventory,
    validate_materialized_inventory,
)
from danids.config.continual import load_continual_experiment_config
from danids.config.core import load_study4_core_config
from danids.config.experiment import ExperimentConfig, load_experiment_config
from danids.config.health import load_health_experiment_config
from danids.config.policy_development import load_policy_development_config
from danids.config.rdx_training_evidence import RDX004_ROTATIONS, RDX004Budget
from danids.config.rdx_training_execution import load_rdx006_execution_config
from danids.config.static import load_static_experiment_config
from danids.config.study4 import load_study4_execution_config
from danids.continual.supervision import load_or_create_supervision_schedule
from danids.data.manifests import SplitManifest, generate_split_manifest
from danids.data.registry import DatasetRegistry
from danids.data.schema import discover_core_feature_contract, read_csv_header, validate_schema
from danids.evaluation.policy_development import evaluate_policy_development
from danids.evaluation.policy_qualification import evaluate_policy_qualification
from danids.evaluation.rdx import evaluate_rdx_recoverability
from danids.evaluation.rdx_analysis import analyze_rdx_recoverability
from danids.evaluation.rdx_training_analysis import (
    SUMMARY_FILENAME,
    RdxTrainingAnalysisError,
    analyze_rdx004_training_evidence,
)
from danids.evaluation.study1 import aggregate_static_study1
from danids.evaluation.study2 import aggregate_continual_study2
from danids.evaluation.study3 import evaluate_health_study3
from danids.evaluation.study4 import evaluate_study4
from danids.evaluation.study5 import evaluate_study5_threats
from danids.evaluation.study5b import evaluate_study5b_replay_robustness
from danids.evaluation.thesis_assets import generate_thesis_assets
from danids.experiments.continual import ContinualSmokeLimits, run_continual_experiment
from danids.experiments.health import HealthSmokeLimits, run_health_experiment
from danids.experiments.policy_development import (
    PolicyDevelopmentSmokeLimits,
    run_policy_development,
)
from danids.experiments.rdx_training_evidence import (
    RdxTrainingEvidencePreflightError,
    build_rdx004_training_evidence_preflight,
)
from danids.experiments.static import (
    SmokeLimits,
    load_or_generate_static_manifests,
    run_static_experiment,
)
from danids.experiments.study4 import Study4SmokeLimits, run_study4_experiment
from danids.experiments.study5b import (
    Study5BLauncherError,
    Study5BLaunchMode,
    format_launcher_report,
    run_study5b_launcher,
)
from danids.policy.development import RollIn
from danids.policy.health_artifact import (
    build_health_model_artifact,
    validate_health_model_artifact,
)
from danids.utils.evidence_archive import (
    build_frozen_evidence_archive,
    validate_frozen_evidence_archive,
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


def _run_study4(args: argparse.Namespace) -> int:
    registry = DatasetRegistry.from_yaml(args.datasets_config)
    config = load_study4_execution_config(args.experiment_config)
    smoke = None
    if args.smoke:
        smoke = Study4SmokeLimits(
            later_stages=args.smoke_later_stages,
            later_windows=args.smoke_later_windows,
            holdout_rows=args.smoke_holdout_rows,
        )
    output = run_study4_experiment(
        registry,
        config,
        initial_run=args.initial_run,
        health_artifact_dir=args.health_artifact_dir,
        manifest_dir=args.manifest_dir,
        output_root=args.output_dir,
        policy_qualification_dir=args.policy_qualification_dir,
        device_name=args.device,
        smoke=smoke,
    )
    print(json.dumps({"run_directory": str(output)}, indent=2))
    return 0


def _evaluate_study4(args: argparse.Namespace) -> int:
    output = evaluate_study4(
        args.run_dirs,
        args.output_dir,
        allow_incomplete=args.allow_incomplete,
        allow_smoke=args.allow_smoke,
    )
    summary = json.loads((output / "study4_summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"output_directory": str(output), **summary}, indent=2))
    return 0


def _evaluate_rdx_recoverability(args: argparse.Namespace) -> int:
    output = evaluate_rdx_recoverability(
        args.study4_evaluation_root,
        args.output_dir,
    )
    summary = json.loads((output / "rdx_summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"output_directory": str(output), **summary}, indent=2))
    return 0


def _analyze_rdx_recoverability(args: argparse.Namespace) -> int:
    output = analyze_rdx_recoverability(args.rdx_bundle_root, args.output_dir)
    summary = json.loads((output / "rdx_analysis_summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"output_directory": str(output), **summary}, indent=2))
    return 0


def _analyze_rdx004_training_evidence(args: argparse.Namespace) -> int:
    output = analyze_rdx004_training_evidence(
        args.study4_evaluation_dir,
        args.preflight_dir,
        args.run_root,
        args.output_dir,
    )
    summary = json.loads((output / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    print(json.dumps({"output_directory": str(output), **summary}, indent=2))
    return 0


def _preflight_rdx004_training_evidence(args: argparse.Namespace) -> int:
    output = build_rdx004_training_evidence_preflight(
        args.config,
        args.output_dir,
        study4_evaluation_root=args.study4_evaluation_root,
        manifest_root=args.manifest_root,
    )
    summary = json.loads((output / "rdx004_preflight_summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"output_directory": str(output), **summary}, indent=2))
    return 0


def _run_rdx004_training_evidence(args: argparse.Namespace) -> int:
    if args.execute is not True:
        raise ValueError("RDX-004 execution requires explicit --execute")
    rotation = tuple(args.rotation)
    if rotation not in RDX004_ROTATIONS:
        raise ValueError("RDX-004 rotation must be one exact frozen rotation")
    registry = DatasetRegistry.from_yaml(args.datasets_config)
    config = load_rdx006_execution_config(args.execution_config)

    # Keep the execution implementation out of the artifact-only preflight
    # import path.  The runner independently repeats every authorization and
    # provenance check before it can materialize a run.
    from danids.experiments.rdx_training_execution import run_rdx004_training_evidence

    output = run_rdx004_training_evidence(
        registry,
        config,
        budget=args.budget,
        rotation=rotation,
        seed=args.seed,
        preflight_dir=args.preflight_dir,
        manifest_dir=args.manifest_dir,
        output_root=args.output_root,
        device_name=args.device,
        smoke=args.smoke,
        execute=args.execute,
    )
    print(json.dumps({"run_directory": str(output)}, indent=2))
    return 0


def _run_policy_development(args: argparse.Namespace) -> int:
    registry = DatasetRegistry.from_yaml(args.datasets_config)
    config = load_policy_development_config(args.experiment_config).with_runtime_seed(args.seed)
    smoke = None
    if args.smoke:
        smoke = PolicyDevelopmentSmokeLimits(
            later_stages=args.smoke_later_stages,
            later_windows=args.smoke_later_windows,
            maximum_anchors=args.smoke_maximum_anchors,
            maximum_actions=5,
        )
    output = run_policy_development(
        registry,
        config,
        roll_in=RollIn(args.roll_in),
        initial_run=args.initial_run,
        health_artifact_dir=args.health_artifact_dir,
        manifest_dir=args.manifest_dir,
        output_root=args.output_dir,
        device_name=args.device,
        smoke=smoke,
    )
    print(json.dumps({"run_directory": str(output)}, indent=2))
    return 0


def _evaluate_policy_development(args: argparse.Namespace) -> int:
    output = evaluate_policy_development(
        args.run_dirs, args.output_dir, allow_smoke=args.allow_smoke
    )
    summary = json.loads((output / "policy_development_summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"output_directory": str(output), **summary}, indent=2))
    return 0


def _qualify_policy(args: argparse.Namespace) -> int:
    output = evaluate_policy_qualification(args.evaluation_dir, args.output_dir)
    summary = json.loads((output / "policy_qualification.json").read_text(encoding="utf-8"))
    print(json.dumps({"output_directory": str(output), **summary}, indent=2))
    return 0


def _validate_study5_contract(args: argparse.Namespace) -> int:
    contract = load_study5_contract(args.contract)
    registry = DatasetRegistry.from_yaml(args.datasets_config)
    raw = validate_configured_dataset_inventory(
        contract,
        registry,
        chunk_rows=args.chunk_rows,
    )
    materialized = validate_materialized_inventory(
        contract,
        registry,
        args.manifest_dir,
        args.materialization_root,
        chunk_rows=args.chunk_rows,
    )
    print(
        json.dumps(
            {
                "status": "valid",
                "contract_version": contract.contract_version,
                "contract_sha256": contract.contract_sha256,
                "raw_datasets": [item.to_dict() for item in raw],
                "materialized_datasets": [item.to_dict() for item in materialized],
            },
            indent=2,
        )
    )
    return 0


def _evaluate_study5_threats(args: argparse.Namespace) -> int:
    output = evaluate_study5_threats(
        contract_path=args.contract,
        study1_run_dirs=args.study1_run_dirs,
        study1_evaluation_dir=args.study1_evaluation_dir,
        study2_static_run_dirs=args.study2_static_run_dirs,
        study2_run_dirs=args.study2_run_dirs,
        study2_evaluation_dir=args.study2_evaluation_dir,
        study4_run_dirs=args.study4_run_dirs,
        study4_evaluation_dir=args.study4_evaluation_dir,
        output_dir=args.output_dir,
    )
    summary = json.loads((output / "study5_summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"output_directory": str(output), **summary}, indent=2))
    return 0


def _launch_study5b_replay_retention(args: argparse.Namespace) -> int:
    mode = Study5BLaunchMode.PREFLIGHT_ONLY if args.preflight_only else Study5BLaunchMode.LAUNCH
    report = run_study5b_launcher(
        contract_path=args.contract,
        mode=mode,
        repo_root=args.repo_root,
        datasets_config=args.datasets_config,
        study1_run_root=args.study1_run_root,
        manifest_root=args.manifest_root,
        schedule_root=args.schedule_root,
        output_root=args.output_root,
        log_root=args.log_root,
        quarantine_root=args.quarantine_root,
        generate_schedules=args.generate_schedules,
        device_name=args.device,
        workers=args.workers,
    )
    print(format_launcher_report(report), end="")
    return 0


def _evaluate_study5b_replay_retention(args: argparse.Namespace) -> int:
    output = evaluate_study5b_replay_robustness(
        contract_path=args.contract,
        task008_dir=args.task008_dir,
        study1_run_dirs=args.study1_run_dirs,
        study2_run_dirs=args.study2_run_dirs,
        output_dir=args.output_dir,
    )
    summary = json.loads((output / "study5b_summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"output_directory": str(output), **summary}, indent=2))
    return 0


def _generate_thesis_assets(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root or Path.cwd()).resolve()
    output_root = Path(args.output_dir).resolve() if args.output_dir else None
    manifest = generate_thesis_assets(repo_root, output_root)
    resolved_output = output_root or repo_root / "thesis" / "assets"
    print(
        json.dumps(
            {
                "output_directory": str(resolved_output),
                "display_count": manifest["display_count"],
                "figure_count": manifest["figure_count"],
                "table_count": manifest["table_count"],
                "status": "valid",
            },
            indent=2,
        )
    )
    return 0


def _build_frozen_evidence_archive(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root or Path.cwd()).resolve()
    result = build_frozen_evidence_archive(repo_root, Path(args.output_dir))
    validated = validate_frozen_evidence_archive(result.archive_path, result.sidecar_path)
    if validated != result:
        raise RuntimeError("frozen evidence archive validation result differs from build result")
    print(
        json.dumps(
            {
                "archive": str(result.archive_path),
                "archive_size_bytes": result.archive_size_bytes,
                "archive_sha256": result.archive_sha256,
                "authority_sha256": result.authority_sha256,
                "internal_file_count": result.source_file_count,
                "manifest_sha256": result.manifest_sha256,
                "sidecar": str(result.sidecar_path),
                "status": "valid",
            },
            indent=2,
        )
    )
    return 0


def _export_explorer(args: argparse.Namespace) -> int:
    from danids.evaluation.explorer_export import export_explorer_data

    output = export_explorer_data(args.repo_root, args.output_dir, check=args.check)
    print(f"Explorer evidence verified: {output}")
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

    run_study4 = subparsers.add_parser(
        "run-study4", help="run one chronological Study-4 E4 method/rotation/seed"
    )
    run_study4.add_argument("--datasets-config", required=True, type=Path)
    run_study4.add_argument("--experiment-config", required=True, type=Path)
    run_study4.add_argument("--initial-run", required=True, type=Path)
    run_study4.add_argument("--health-artifact-dir", required=True, type=Path)
    run_study4.add_argument("--manifest-dir", required=True, type=Path)
    run_study4.add_argument("--output-dir", required=True, type=Path)
    run_study4.add_argument("--policy-qualification-dir", type=Path)
    run_study4.add_argument("--device", default="auto")
    run_study4.add_argument("--smoke", action="store_true")
    run_study4.add_argument("--smoke-later-stages", type=int, default=1)
    run_study4.add_argument("--smoke-later-windows", type=int, default=2)
    run_study4.add_argument("--smoke-holdout-rows", type=int, default=1_000)
    run_study4.set_defaults(handler=_run_study4)

    evaluate_study4_parser = subparsers.add_parser(
        "evaluate-study4", help="artifact-only aggregation of completed Study-4 E4 runs"
    )
    evaluate_study4_parser.add_argument(
        "--run-dir", dest="run_dirs", required=True, action="append", type=Path
    )
    evaluate_study4_parser.add_argument("--output-dir", required=True, type=Path)
    evaluate_study4_parser.add_argument("--allow-incomplete", action="store_true")
    evaluate_study4_parser.add_argument("--allow-smoke", action="store_true")
    evaluate_study4_parser.set_defaults(handler=_evaluate_study4)

    evaluate_rdx = subparsers.add_parser(
        "evaluate-rdx-recoverability",
        help="derive the frozen artifact-only RDX recoverability diagnostics",
    )
    evaluate_rdx.add_argument("--study4-evaluation-root", required=True, type=Path)
    evaluate_rdx.add_argument("--output-dir", required=True, type=Path)
    evaluate_rdx.set_defaults(handler=_evaluate_rdx_recoverability)

    analyze_rdx = subparsers.add_parser(
        "analyze-rdx-recoverability",
        help="artifact-only scientific analysis of the validated RDX diagnostics",
    )
    analyze_rdx.add_argument("--rdx-bundle-root", required=True, type=Path)
    analyze_rdx.add_argument("--output-dir", required=True, type=Path)
    analyze_rdx.set_defaults(handler=_analyze_rdx_recoverability)

    analyze_rdx004 = subparsers.add_parser(
        "analyze-rdx004-training-evidence",
        help="validate and analyze the frozen RDX-004 B100/B400/B1600 corpus",
    )
    analyze_rdx004.add_argument("--study4-evaluation-dir", required=True, type=Path)
    analyze_rdx004.add_argument("--preflight-dir", required=True, type=Path)
    analyze_rdx004.add_argument("--run-root", required=True, type=Path)
    analyze_rdx004.add_argument("--output-dir", required=True, type=Path)
    analyze_rdx004.set_defaults(handler=_analyze_rdx004_training_evidence)

    preflight_rdx004 = subparsers.add_parser(
        "preflight-rdx004-training-evidence",
        help="construct and validate the artifact-only RDX-004 24-run preflight",
    )
    preflight_rdx004.add_argument("--config", required=True, type=Path)
    preflight_rdx004.add_argument("--study4-evaluation-root", type=Path)
    preflight_rdx004.add_argument("--manifest-root", type=Path)
    preflight_rdx004.add_argument("--output-dir", required=True, type=Path)
    preflight_rdx004.set_defaults(handler=_preflight_rdx004_training_evidence)

    run_rdx004 = subparsers.add_parser(
        "run-rdx004-training-evidence",
        help="explicitly execute one authorized RDX-004 B400/B1600 run",
    )
    run_rdx004.add_argument("--execution-config", required=True, type=Path)
    run_rdx004.add_argument("--datasets-config", required=True, type=Path)
    run_rdx004.add_argument("--preflight-dir", required=True, type=Path)
    run_rdx004.add_argument(
        "--budget",
        required=True,
        type=RDX004Budget,
        choices=[RDX004Budget.B400, RDX004Budget.B1600],
    )
    run_rdx004.add_argument(
        "--rotation",
        required=True,
        nargs=4,
        choices=["U", "T", "C", "B"],
        metavar=("D1", "D2", "D3", "D4"),
    )
    run_rdx004.add_argument("--seed", required=True, type=int, choices=[42, 43, 44])
    run_rdx004.add_argument("--manifest-dir", required=True, type=Path)
    run_rdx004.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="repository/output root; the frozen confirmatory or smoke namespace is appended",
    )
    run_rdx004.add_argument("--device", default="auto")
    run_rdx004.add_argument("--smoke", action="store_true")
    run_rdx004.add_argument("--execute", action="store_true")
    run_rdx004.set_defaults(handler=_run_rdx004_training_evidence)

    run_policy = subparsers.add_parser(
        "run-policy-development",
        help="run one POLICY_DEVELOPMENT_V1 rotation/seed/roll-in unit",
    )
    run_policy.add_argument("--datasets-config", required=True, type=Path)
    run_policy.add_argument("--experiment-config", required=True, type=Path)
    run_policy.add_argument("--roll-in", required=True, choices=[value.value for value in RollIn])
    run_policy.add_argument("--initial-run", required=True, type=Path)
    run_policy.add_argument("--health-artifact-dir", required=True, type=Path)
    run_policy.add_argument("--manifest-dir", required=True, type=Path)
    run_policy.add_argument("--output-dir", required=True, type=Path)
    run_policy.add_argument("--device", default="auto")
    run_policy.add_argument("--seed", type=int)
    run_policy.add_argument("--smoke", action="store_true")
    run_policy.add_argument("--smoke-later-stages", type=int, default=1)
    run_policy.add_argument("--smoke-later-windows", type=int, default=3)
    run_policy.add_argument("--smoke-maximum-anchors", type=int, default=1)
    run_policy.set_defaults(handler=_run_policy_development)

    evaluate_policy = subparsers.add_parser(
        "evaluate-policy-development",
        help="artifact-only aggregation of POLICY_DEVELOPMENT_V1 action trials",
    )
    evaluate_policy.add_argument(
        "--run-dir", dest="run_dirs", required=True, action="append", type=Path
    )
    evaluate_policy.add_argument("--output-dir", required=True, type=Path)
    evaluate_policy.add_argument("--allow-smoke", action="store_true")
    evaluate_policy.set_defaults(handler=_evaluate_policy_development)

    qualify_policy = subparsers.add_parser(
        "qualify-policy",
        help="artifact-only DANIDS-Policy oracle qualification gate",
    )
    qualify_policy.add_argument("--evaluation-dir", required=True, type=Path)
    qualify_policy.add_argument("--output-dir", required=True, type=Path)
    qualify_policy.set_defaults(handler=_qualify_policy)

    validate_study5 = subparsers.add_parser(
        "validate-study5-contract",
        help="validate the frozen Study-5 ontology against raw and materialized labels",
    )
    validate_study5.add_argument("--contract", required=True, type=Path)
    validate_study5.add_argument("--datasets-config", required=True, type=Path)
    validate_study5.add_argument("--manifest-dir", required=True, type=Path)
    validate_study5.add_argument("--materialization-root", required=True, type=Path)
    validate_study5.add_argument("--chunk-rows", type=int, default=100_000)
    validate_study5.set_defaults(handler=_validate_study5_contract)

    evaluate_study5 = subparsers.add_parser(
        "evaluate-study5-threats",
        help="artifact-only Study-5A native and semantic threat audit",
    )
    evaluate_study5.add_argument("--contract", required=True, type=Path)
    evaluate_study5.add_argument(
        "--study1-run",
        dest="study1_run_dirs",
        required=True,
        action="append",
        type=Path,
    )
    evaluate_study5.add_argument("--study1-evaluation-dir", required=True, type=Path)
    evaluate_study5.add_argument(
        "--study2-static-run",
        dest="study2_static_run_dirs",
        required=True,
        action="append",
        type=Path,
    )
    evaluate_study5.add_argument(
        "--study2-run",
        dest="study2_run_dirs",
        required=True,
        action="append",
        type=Path,
    )
    evaluate_study5.add_argument("--study2-evaluation-dir", required=True, type=Path)
    evaluate_study5.add_argument(
        "--study4-run",
        dest="study4_run_dirs",
        required=True,
        action="append",
        type=Path,
    )
    evaluate_study5.add_argument("--study4-evaluation-dir", required=True, type=Path)
    evaluate_study5.add_argument("--output-dir", required=True, type=Path)
    evaluate_study5.set_defaults(handler=_evaluate_study5_threats)

    launch_study5b = subparsers.add_parser(
        "launch-study5b-replay-retention",
        help="preflight or serially run the frozen 27-run TASK-009 extension",
    )
    launch_study5b.add_argument("--contract", required=True, type=Path)
    launch_study5b.add_argument("--repo-root", type=Path)
    launch_study5b.add_argument("--datasets-config", type=Path)
    launch_study5b.add_argument("--study1-run-root", type=Path)
    launch_study5b.add_argument("--manifest-root", type=Path)
    launch_study5b.add_argument("--schedule-root", type=Path)
    launch_study5b.add_argument("--output-root", type=Path)
    launch_study5b.add_argument("--log-root", type=Path)
    launch_study5b.add_argument("--quarantine-root", type=Path)
    launch_study5b.add_argument("--generate-schedules", action="store_true")
    launch_study5b.add_argument("--preflight-only", action="store_true")
    launch_study5b.add_argument("--device", default="cpu", choices=["cpu"])
    launch_study5b.add_argument("--workers", default=1, type=int, choices=[1])
    launch_study5b.set_defaults(handler=_launch_study5b_replay_retention)

    evaluate_study5b = subparsers.add_parser(
        "evaluate-study5b-replay-robustness",
        help="artifact-only all-order TASK-009 replay-retention evaluation",
    )
    evaluate_study5b.add_argument("--contract", required=True, type=Path)
    evaluate_study5b.add_argument("--task008-dir", required=True, type=Path)
    evaluate_study5b.add_argument(
        "--study1-run",
        dest="study1_run_dirs",
        required=True,
        action="append",
        type=Path,
    )
    evaluate_study5b.add_argument(
        "--study2-run",
        dest="study2_run_dirs",
        required=True,
        action="append",
        type=Path,
    )
    evaluate_study5b.add_argument("--output-dir", required=True, type=Path)
    evaluate_study5b.set_defaults(handler=_evaluate_study5b_replay_retention)

    thesis_assets = subparsers.add_parser(
        "generate-thesis-assets",
        help="generate and verify the frozen artifact-only DANIDS thesis display pack",
    )
    thesis_assets.add_argument("--repo-root", type=Path)
    thesis_assets.add_argument("--output-dir", type=Path)
    thesis_assets.set_defaults(handler=_generate_thesis_assets)

    evidence_archive = subparsers.add_parser(
        "build-frozen-evidence-archive",
        help="build and validate the deterministic DANIDS 2.0 frozen-evidence archive",
    )
    evidence_archive.add_argument("--repo-root", type=Path)
    evidence_archive.add_argument("--output-dir", required=True, type=Path)
    evidence_archive.set_defaults(handler=_build_frozen_evidence_archive)
    explorer = subparsers.add_parser(
        "export-explorer-data", help="export or check read-only public Explorer projections"
    )
    explorer.add_argument("--repo-root", type=Path, default=Path.cwd())
    explorer.add_argument("--output-dir", type=Path, required=True)
    explorer.add_argument("--check", action="store_true")
    explorer.set_defaults(handler=_export_explorer)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        Study5BLauncherError,
        RdxTrainingAnalysisError,
        RdxTrainingEvidencePreflightError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
