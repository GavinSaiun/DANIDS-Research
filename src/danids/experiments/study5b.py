"""Closed, serial launcher for the TASK-009 Study-5B prospective run matrix."""

from __future__ import annotations

import contextlib
import json
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, cast

from danids.config.continual import (
    METHOD_TOKENS,
    ContinualExperimentConfig,
    ContinualMethod,
    load_continual_experiment_config,
)
from danids.config.study5b import (
    TASK009_EXPECTED_NEW_RUNS,
    TASK009_EXTENSION_ROTATIONS,
    TASK009_METHODS,
    TASK009_SEEDS,
    Study5BContract,
    load_study5b_contract,
)
from danids.continual.initial_state import ImportedInitialState, load_study1_initial_state
from danids.continual.supervision import SupervisionSchedule, load_or_create_supervision_schedule
from danids.data.manifests import SourceFingerprintCache, SplitManifest
from danids.data.materialized import open_existing_materialized_dataset
from danids.data.registry import DatasetRegistry
from danids.data.schema import FeatureContract, discover_core_feature_contract
from danids.evaluation.study2 import StaticReference, validate_continual_run
from danids.experiments.continual import run_continual_experiment
from danids.experiments.static import load_or_generate_static_manifests

TASK009_SCHEDULE_DIRECTORY: Final = Path("schedules/task009-study5b")
TASK009_DEFAULT_LOG_DIRECTORY: Final = Path("logs/task009-study5b")
TASK009_DEFAULT_QUARANTINE_DIRECTORY: Final = Path("runs/task009-study5b-quarantine")


class Study5BLauncherError(RuntimeError):
    """Raised when the closed launcher cannot safely continue."""


class Study5BLaunchMode(StrEnum):
    PREFLIGHT_ONLY = "preflight_only"
    LAUNCH = "launch"


class ExistingOutputState(StrEnum):
    ABSENT = "ABSENT"
    SKIPPED_VALID = "SKIPPED_VALID"
    INVALID_EXISTING = "INVALID_EXISTING"


class FinalRunState(StrEnum):
    READY = "READY"
    SKIPPED_VALID = "SKIPPED_VALID"
    COMPLETED_VALID = "COMPLETED_VALID"


@dataclass(frozen=True, slots=True)
class Study5BRunIdentity:
    rotation: tuple[str, ...]
    seed: int
    method: ContinualMethod


# This literal roster is intentional: TASK-009 must never discover or expand work from disk.
TASK009_RUN_IDENTITIES: Final[tuple[Study5BRunIdentity, ...]] = (
    Study5BRunIdentity(("T", "C", "B", "U"), 42, "naive_ft"),
    Study5BRunIdentity(("T", "C", "B", "U"), 42, "er"),
    Study5BRunIdentity(("T", "C", "B", "U"), 42, "ft_mem"),
    Study5BRunIdentity(("T", "C", "B", "U"), 43, "naive_ft"),
    Study5BRunIdentity(("T", "C", "B", "U"), 43, "er"),
    Study5BRunIdentity(("T", "C", "B", "U"), 43, "ft_mem"),
    Study5BRunIdentity(("T", "C", "B", "U"), 44, "naive_ft"),
    Study5BRunIdentity(("T", "C", "B", "U"), 44, "er"),
    Study5BRunIdentity(("T", "C", "B", "U"), 44, "ft_mem"),
    Study5BRunIdentity(("C", "B", "U", "T"), 42, "naive_ft"),
    Study5BRunIdentity(("C", "B", "U", "T"), 42, "er"),
    Study5BRunIdentity(("C", "B", "U", "T"), 42, "ft_mem"),
    Study5BRunIdentity(("C", "B", "U", "T"), 43, "naive_ft"),
    Study5BRunIdentity(("C", "B", "U", "T"), 43, "er"),
    Study5BRunIdentity(("C", "B", "U", "T"), 43, "ft_mem"),
    Study5BRunIdentity(("C", "B", "U", "T"), 44, "naive_ft"),
    Study5BRunIdentity(("C", "B", "U", "T"), 44, "er"),
    Study5BRunIdentity(("C", "B", "U", "T"), 44, "ft_mem"),
    Study5BRunIdentity(("B", "U", "T", "C"), 42, "naive_ft"),
    Study5BRunIdentity(("B", "U", "T", "C"), 42, "er"),
    Study5BRunIdentity(("B", "U", "T", "C"), 42, "ft_mem"),
    Study5BRunIdentity(("B", "U", "T", "C"), 43, "naive_ft"),
    Study5BRunIdentity(("B", "U", "T", "C"), 43, "er"),
    Study5BRunIdentity(("B", "U", "T", "C"), 43, "ft_mem"),
    Study5BRunIdentity(("B", "U", "T", "C"), 44, "naive_ft"),
    Study5BRunIdentity(("B", "U", "T", "C"), 44, "er"),
    Study5BRunIdentity(("B", "U", "T", "C"), 44, "ft_mem"),
)


@dataclass(frozen=True, slots=True)
class Study5BRunSpec:
    identity: Study5BRunIdentity
    experiment_id: str
    base_config_path: Path
    source_run_path: Path
    manifest_dir: Path
    schedule_path: Path
    output_path: Path
    stdout_log_path: Path
    stderr_log_path: Path

    @property
    def rotation(self) -> tuple[str, ...]:
        return self.identity.rotation

    @property
    def seed(self) -> int:
        return self.identity.seed

    @property
    def method(self) -> ContinualMethod:
        return self.identity.method


@dataclass(frozen=True, slots=True)
class Study5BLauncherPaths:
    repo_root: Path
    datasets_config: Path
    study1_run_root: Path
    manifest_root: Path
    schedule_root: Path
    output_root: Path
    log_root: Path
    quarantine_root: Path


@dataclass(frozen=True, slots=True)
class RunPreflightRecord:
    experiment_id: str
    rotation: str
    seed: int
    method: ContinualMethod
    schedule_path: str
    schedule_sha256: str
    source_run_path: str
    output_path: str
    existing_output_state: ExistingOutputState
    invalid_reason: str | None


@dataclass(frozen=True, slots=True)
class RunLaunchRecord:
    experiment_id: str
    final_state: FinalRunState
    quarantined_path: str | None
    stdout_log_path: str | None
    stderr_log_path: str | None


@dataclass(frozen=True, slots=True)
class Study5BLauncherReport:
    mode: Study5BLaunchMode
    contract_version: str
    contract_sha256: str
    workers: int
    device: str
    roster_count: int
    schedule_count: int
    generated_schedule_count: int
    preflight_records: tuple[RunPreflightRecord, ...]
    launch_records: tuple[RunLaunchRecord, ...]

    @property
    def skipped_valid_count(self) -> int:
        return sum(
            record.existing_output_state is ExistingOutputState.SKIPPED_VALID
            for record in self.preflight_records
        )

    @property
    def restartable_invalid_count(self) -> int:
        return sum(
            record.existing_output_state is ExistingOutputState.INVALID_EXISTING
            for record in self.preflight_records
        )

    @property
    def absent_count(self) -> int:
        return sum(
            record.existing_output_state is ExistingOutputState.ABSENT
            for record in self.preflight_records
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "contract_version": self.contract_version,
            "contract_sha256": self.contract_sha256,
            "workers": self.workers,
            "device": self.device,
            "roster_count": self.roster_count,
            "schedule_count": self.schedule_count,
            "generated_schedule_count": self.generated_schedule_count,
            "absent_count": self.absent_count,
            "skipped_valid_count": self.skipped_valid_count,
            "restartable_invalid_count": self.restartable_invalid_count,
            "preflight_records": [asdict(record) for record in self.preflight_records],
            "launch_records": [asdict(record) for record in self.launch_records],
        }


@dataclass(frozen=True, slots=True)
class _PreparedUnit:
    manifests: tuple[SplitManifest, ...]
    static_reference: StaticReference
    schedule: SupervisionSchedule


@dataclass(frozen=True, slots=True)
class _PreflightState:
    contract: Study5BContract
    roster: tuple[Study5BRunSpec, ...]
    configs: Mapping[str, ContinualExperimentConfig]
    units: Mapping[tuple[tuple[str, ...], int], _PreparedUnit]
    generated_schedules: tuple[Path, ...]
    records: tuple[RunPreflightRecord, ...]


def _safe_direct_child(root: Path, name: str, label: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / name).resolve()
    if candidate.parent != resolved_root:
        raise Study5BLauncherError(f"{label} escapes its configured root: {candidate}")
    return candidate


def _resolve_paths(
    *,
    repo_root: str | Path,
    datasets_config: str | Path | None,
    study1_run_root: str | Path | None,
    manifest_root: str | Path | None,
    schedule_root: str | Path | None,
    output_root: str | Path | None,
    log_root: str | Path | None,
    quarantine_root: str | Path | None,
) -> Study5BLauncherPaths:
    root = Path(repo_root).resolve()

    def chosen(value: str | Path | None, default: Path) -> Path:
        path = default if value is None else Path(value)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    paths = Study5BLauncherPaths(
        repo_root=root,
        datasets_config=chosen(datasets_config, Path("configs/datasets.local.yaml")),
        study1_run_root=chosen(study1_run_root, Path("runs")),
        manifest_root=chosen(manifest_root, Path("manifests")),
        schedule_root=chosen(schedule_root, TASK009_SCHEDULE_DIRECTORY),
        output_root=chosen(output_root, Path("runs")),
        log_root=chosen(log_root, TASK009_DEFAULT_LOG_DIRECTORY),
        quarantine_root=chosen(quarantine_root, TASK009_DEFAULT_QUARANTINE_DIRECTORY),
    )
    expected_schedule_root = (root / TASK009_SCHEDULE_DIRECTORY).resolve()
    if paths.schedule_root != expected_schedule_root:
        raise Study5BLauncherError(
            f"TASK-009 schedules must use the frozen directory {expected_schedule_root}"
        )
    if paths.quarantine_root.drive.lower() != paths.output_root.drive.lower():
        raise Study5BLauncherError("quarantine and run roots must be on the same filesystem")
    return paths


def _config_filename(method: ContinualMethod, rotation: tuple[str, ...]) -> str:
    return f"task009_{method}_{'-'.join(rotation).lower()}.yaml"


def _experiment_id(identity: Study5BRunIdentity) -> str:
    return (
        f"E2_{METHOD_TOKENS[identity.method]}_{'-'.join(identity.rotation)}_"
        f"B100_D1_s{identity.seed}"
    )


def build_task009_roster(
    contract: Study5BContract, paths: Study5BLauncherPaths
) -> tuple[Study5BRunSpec, ...]:
    """Resolve the literal 27-run roster without filesystem discovery."""

    if (
        contract.methods != TASK009_METHODS
        or contract.seeds != TASK009_SEEDS
        or contract.extension_rotations != TASK009_EXTENSION_ROTATIONS
        or contract.expected_new_run_count != TASK009_EXPECTED_NEW_RUNS
    ):
        raise Study5BLauncherError("TASK-009 contract cannot populate the frozen launcher roster")
    expected_identities = {
        Study5BRunIdentity(rotation, seed, method)
        for rotation in TASK009_EXTENSION_ROTATIONS
        for seed in TASK009_SEEDS
        for method in TASK009_METHODS
    }
    if (
        len(TASK009_RUN_IDENTITIES) != TASK009_EXPECTED_NEW_RUNS
        or set(TASK009_RUN_IDENTITIES) != expected_identities
    ):
        raise Study5BLauncherError("literal TASK-009 launcher roster differs from the freeze")

    result: list[Study5BRunSpec] = []
    for identity in TASK009_RUN_IDENTITIES:
        rotation = "-".join(identity.rotation)
        experiment_id = _experiment_id(identity)
        result.append(
            Study5BRunSpec(
                identity=identity,
                experiment_id=experiment_id,
                base_config_path=_safe_direct_child(
                    paths.repo_root / "configs" / "experiments",
                    _config_filename(identity.method, identity.rotation),
                    "base experiment config",
                ),
                source_run_path=_safe_direct_child(
                    paths.study1_run_root,
                    f"E1_STATIC_MLP_{rotation}_s{identity.seed}",
                    "Study-1 source run",
                ),
                manifest_dir=_safe_direct_child(
                    paths.manifest_root,
                    f"study1-s{identity.seed}",
                    "manifest directory",
                ),
                schedule_path=_safe_direct_child(
                    paths.schedule_root,
                    f"{rotation}_s{identity.seed}.json",
                    "supervision schedule",
                ),
                output_path=_safe_direct_child(paths.output_root, experiment_id, "TASK-009 output"),
                stdout_log_path=_safe_direct_child(
                    paths.log_root, f"{experiment_id}.stdout.log", "stdout log"
                ),
                stderr_log_path=_safe_direct_child(
                    paths.log_root, f"{experiment_id}.stderr.log", "stderr log"
                ),
            )
        )
    if len({item.experiment_id for item in result}) != TASK009_EXPECTED_NEW_RUNS:
        raise Study5BLauncherError("TASK-009 experiment IDs are not unique")
    if len({item.schedule_path for item in result}) != 9:
        raise Study5BLauncherError("TASK-009 must resolve exactly nine shared schedules")
    if len({item.base_config_path for item in result}) != 9:
        raise Study5BLauncherError("TASK-009 must resolve exactly nine base configurations")
    return tuple(result)


def _validate_base_config(
    spec: Study5BRunSpec, contract: Study5BContract
) -> ContinualExperimentConfig:
    base = load_continual_experiment_config(spec.base_config_path)
    frozen = contract.to_dict()
    execution = cast(Mapping[str, Any], frozen["execution"])
    reuse = cast(Mapping[str, Any], frozen["study2_reuse"])
    supervision = cast(Mapping[str, Any], reuse["supervision"])
    adaptation = cast(Mapping[str, Any], reuse["adaptation"])
    memory = cast(Mapping[str, Any], reuse["memory"])
    operating = cast(Mapping[str, Any], reuse["operating_envelope"])
    splits = base.experiment.splits
    expected_cache_root = (spec.base_config_path.parents[2] / "data" / "materialized").resolve()
    comparisons: tuple[tuple[str, object, object], ...] = (
        ("study", base.experiment.study, execution["study"]),
        ("base_seed", base.experiment.seed, execution["base_seed"]),
        ("datasets.sequence", base.experiment.sequence, spec.rotation),
        ("datasets.split_version", base.experiment.split_version, reuse["split_version"]),
        ("stream.window_size", base.experiment.window_size, reuse["window_size"]),
        ("stream.boundary_mode", base.experiment.boundary_mode, reuse["boundary_mode"]),
        ("splits.initial.train", splits.initial_train, 0.60),
        ("splits.initial.validation", splits.initial_validation, 0.20),
        ("splits.initial.holdout", splits.initial_holdout, 0.20),
        ("splits.later.online", splits.later_online, 0.80),
        ("splits.later.holdout", splits.later_holdout, 0.20),
        (
            "supervision.label_budget_per_later_domain",
            base.supervision.label_budget_per_later_domain,
            supervision["label_budget_per_later_domain"],
        ),
        (
            "supervision.label_delay_windows",
            base.supervision.label_delay_windows,
            supervision["label_delay_windows"],
        ),
        ("supervision.schedule", base.supervision.schedule, supervision["schedule"]),
        (
            "operating_envelope.target_fpr",
            base.experiment.target_fpr,
            operating["target_fpr"],
        ),
        ("adaptation.method", base.adaptation.method, spec.method),
        ("adaptation.optimizer", base.adaptation.optimizer, adaptation["optimizer"]),
        (
            "adaptation.learning_rate",
            base.adaptation.learning_rate,
            adaptation["learning_rate"],
        ),
        ("adaptation.weight_decay", base.adaptation.weight_decay, adaptation["weight_decay"]),
        ("adaptation.batch_size", base.adaptation.batch_size, adaptation["batch_size"]),
        ("adaptation.epochs", base.adaptation.epochs, adaptation["epochs"]),
        # TASK-004's shared schema retains this inert value even though EWC is excluded.
        ("adaptation.ewc_lambda", base.adaptation.ewc_lambda, 100.0),
        ("memory.replay_per_domain", base.memory.replay_per_domain, memory["replay_per_domain"]),
        ("memory.audit_per_domain", base.memory.audit_per_domain, memory["audit_per_domain"]),
        ("materialization.cache_root", base.materialization.cache_root, expected_cache_root),
        ("materialization.csv_chunk_rows", base.materialization.csv_chunk_rows, 100_000),
    )
    drifts = [
        f"{name}: actual={actual!r}, expected={expected!r}"
        for name, actual, expected in comparisons
        if actual != expected
    ]
    if drifts:
        raise Study5BLauncherError(
            f"{spec.base_config_path}: frozen TASK-009 config differs: " + "; ".join(drifts)
        )
    runtime = base.with_runtime_overrides(seed=spec.seed)
    if runtime.experiment.experiment_id != spec.experiment_id:
        raise Study5BLauncherError(f"{spec.base_config_path}: runtime experiment ID differs")
    return runtime


def _load_runtime_configs(
    roster: Sequence[Study5BRunSpec], contract: Study5BContract
) -> dict[str, ContinualExperimentConfig]:
    configs = {spec.experiment_id: _validate_base_config(spec, contract) for spec in roster}
    if len(configs) != TASK009_EXPECTED_NEW_RUNS:
        raise Study5BLauncherError("TASK-009 did not resolve exactly 27 runtime configs")
    return configs


def _manifest_paths(spec: Study5BRunSpec) -> tuple[Path, ...]:
    return tuple(
        spec.manifest_dir / f"stage-{stage:02d}-{domain}.json"
        for stage, domain in enumerate(spec.rotation, start=1)
    )


def _static_reference(spec: Study5BRunSpec, initial: ImportedInitialState) -> StaticReference:
    return StaticReference(
        path=spec.source_run_path,
        seed=spec.seed,
        sequence=spec.rotation,
        fingerprints=initial.dataset_fingerprints,
        checkpoint_sha256=initial.checkpoint_sha256,
        model_digest=initial.model_digest,
        preprocessor_digest=initial.preprocessor_digest,
        threshold_digest=initial.threshold_digest,
    )


def _prepare_unit(
    *,
    registry: DatasetRegistry,
    feature_contract: FeatureContract,
    fingerprint_cache: SourceFingerprintCache,
    representative: Study5BRunSpec,
    config: ContinualExperimentConfig,
    generate_missing_schedule: bool,
) -> tuple[_PreparedUnit, bool]:
    if not representative.manifest_dir.is_dir():
        raise Study5BLauncherError(
            f"Study-1 manifest directory is missing: {representative.manifest_dir}"
        )
    missing_manifests = [path for path in _manifest_paths(representative) if not path.is_file()]
    if missing_manifests:
        raise Study5BLauncherError(
            "TASK-009 must reuse existing Study-1 manifests; missing: "
            + ", ".join(str(path) for path in missing_manifests)
        )
    manifests = tuple(
        load_or_generate_static_manifests(
            registry,
            feature_contract,
            config,
            representative.manifest_dir,
            fingerprint_cache=fingerprint_cache,
        )
    )
    for manifest in manifests:
        open_existing_materialized_dataset(config.materialization.cache_root, manifest)
    initial = load_study1_initial_state(
        representative.source_run_path, config, feature_contract, manifests
    )
    static = _static_reference(representative, initial)
    schedule_existed = representative.schedule_path.is_file()
    if not schedule_existed and not generate_missing_schedule:
        raise Study5BLauncherError(
            f"TASK-009 schedule is missing (pass generate_schedules=True): "
            f"{representative.schedule_path}"
        )
    schedule = load_or_create_supervision_schedule(
        representative.schedule_path, manifests, seed=representative.seed
    )
    schedule.validate(manifests)
    return _PreparedUnit(manifests, static, schedule), not schedule_existed


def _existing_output(
    spec: Study5BRunSpec, static: StaticReference
) -> tuple[ExistingOutputState, str | None]:
    if not spec.output_path.exists():
        return ExistingOutputState.ABSENT, None
    if not spec.output_path.is_dir():
        return ExistingOutputState.INVALID_EXISTING, "output path exists but is not a directory"
    try:
        validated = validate_continual_run(spec.output_path, static)
        if (
            validated.path != spec.output_path
            or validated.method != spec.method
            or validated.seed != spec.seed
            or validated.sequence != spec.rotation
        ):
            raise Study5BLauncherError("validated output identity differs from launcher roster")
    except Exception as exc:
        return ExistingOutputState.INVALID_EXISTING, f"{type(exc).__name__}: {exc}"
    return ExistingOutputState.SKIPPED_VALID, None


def _preflight(
    *,
    contract: Study5BContract,
    paths: Study5BLauncherPaths,
    registry: DatasetRegistry,
    generate_schedules: bool,
) -> _PreflightState:
    roster = build_task009_roster(contract, paths)
    configs = _load_runtime_configs(roster, contract)
    feature_contract = discover_core_feature_contract(registry)
    if len(feature_contract.feature_columns) != 47:
        raise Study5BLauncherError(
            f"TASK-009 requires 47 primary features, got {len(feature_contract.feature_columns)}"
        )

    representatives: dict[tuple[tuple[str, ...], int], Study5BRunSpec] = {}
    for spec in roster:
        representatives.setdefault((spec.rotation, spec.seed), spec)
    if len(representatives) != 9:
        raise Study5BLauncherError("TASK-009 must contain exactly nine rotation-seed units")

    fingerprint_cache = SourceFingerprintCache()
    units: dict[tuple[tuple[str, ...], int], _PreparedUnit] = {}
    generated: list[Path] = []
    for key, representative in representatives.items():
        prepared, was_generated = _prepare_unit(
            registry=registry,
            feature_contract=feature_contract,
            fingerprint_cache=fingerprint_cache,
            representative=representative,
            config=configs[representative.experiment_id],
            generate_missing_schedule=generate_schedules,
        )
        units[key] = prepared
        if was_generated:
            generated.append(representative.schedule_path)

    records: list[RunPreflightRecord] = []
    for spec in roster:
        unit = units[(spec.rotation, spec.seed)]
        state, reason = _existing_output(spec, unit.static_reference)
        records.append(
            RunPreflightRecord(
                experiment_id=spec.experiment_id,
                rotation="-".join(spec.rotation),
                seed=spec.seed,
                method=spec.method,
                schedule_path=str(spec.schedule_path),
                schedule_sha256=unit.schedule.digest(),
                source_run_path=str(spec.source_run_path),
                output_path=str(spec.output_path),
                existing_output_state=state,
                invalid_reason=reason,
            )
        )
    for key in units:
        paired = [
            record for record in records if (tuple(record.rotation.split("-")), record.seed) == key
        ]
        if len(paired) != 3 or len({record.schedule_sha256 for record in paired}) != 1:
            raise Study5BLauncherError(f"TASK-009 paired methods do not share one schedule: {key}")
    return _PreflightState(
        contract=contract,
        roster=roster,
        configs=configs,
        units=units,
        generated_schedules=tuple(generated),
        records=tuple(records),
    )


def _unique_quarantine_path(root: Path, experiment_id: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    base = _safe_direct_child(root, f"{experiment_id}.invalid-{stamp}", "quarantine path")
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = _safe_direct_child(
            root, f"{base.name}-{suffix:02d}", "quarantine collision path"
        )
        suffix += 1
    return candidate


def quarantine_invalid_output(spec: Study5BRunSpec, quarantine_root: str | Path) -> Path:
    """Atomically retain one exact invalid output before its whole-run restart."""

    source = spec.output_path.resolve()
    if not source.exists():
        raise Study5BLauncherError(f"cannot quarantine absent output: {source}")
    expected_parent = spec.output_path.parent.resolve()
    if source.parent != expected_parent or source.name != spec.experiment_id:
        raise Study5BLauncherError(f"refusing to quarantine unexpected output target: {source}")
    root = Path(quarantine_root).resolve()
    if root.drive.lower() != source.drive.lower():
        raise Study5BLauncherError("quarantine requires an atomic same-filesystem rename")
    root.mkdir(parents=True, exist_ok=True)
    destination = _unique_quarantine_path(root, spec.experiment_id)
    source.rename(destination)
    return destination


def _launch_one(
    *,
    registry: DatasetRegistry,
    config: ContinualExperimentConfig,
    spec: Study5BRunSpec,
    device_name: str,
) -> None:
    spec.stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"\n[{datetime.now(UTC).isoformat()}] TASK-009 launch {spec.experiment_id}\n"
    with (
        spec.stdout_log_path.open("a", encoding="utf-8", newline="\n") as stdout_handle,
        spec.stderr_log_path.open("a", encoding="utf-8", newline="\n") as stderr_handle,
    ):
        stdout_handle.write(header)
        stderr_handle.write(header)
        stdout_handle.flush()
        stderr_handle.flush()
        try:
            with (
                contextlib.redirect_stdout(stdout_handle),
                contextlib.redirect_stderr(stderr_handle),
            ):
                result = run_continual_experiment(
                    registry,
                    config,
                    initial_run=spec.source_run_path,
                    manifest_dir=spec.manifest_dir,
                    schedule_path=spec.schedule_path,
                    output_root=spec.output_path.parent,
                    device_name=device_name,
                    smoke=None,
                )
            if result.resolve() != spec.output_path:
                raise Study5BLauncherError(
                    f"runner returned unexpected output path for {spec.experiment_id}: {result}"
                )
        except Exception:
            traceback.print_exc(file=stderr_handle)
            raise


def _coerce_mode(value: Study5BLaunchMode | str) -> Study5BLaunchMode:
    try:
        return Study5BLaunchMode(value)
    except ValueError as exc:
        raise Study5BLauncherError(f"unsupported TASK-009 launcher mode: {value}") from exc


def run_study5b_launcher(
    *,
    contract_path: str | Path,
    mode: Study5BLaunchMode | str,
    repo_root: str | Path | None = None,
    datasets_config: str | Path | None = None,
    study1_run_root: str | Path | None = None,
    manifest_root: str | Path | None = None,
    schedule_root: str | Path | None = None,
    output_root: str | Path | None = None,
    log_root: str | Path | None = None,
    quarantine_root: str | Path | None = None,
    generate_schedules: bool = False,
    device_name: str = "cpu",
    workers: int = 1,
) -> Study5BLauncherReport:
    """Preflight or serially launch the frozen 27-run extension.

    Preflight may create only the nine ignored deterministic schedules when
    ``generate_schedules`` is true. Launch mode also permits that generation, quarantines
    invalid outputs, skips strictly valid outputs, and fails immediately on the first run error.
    """

    resolved_mode = _coerce_mode(mode)
    if workers != 1:
        raise Study5BLauncherError("TASK-009 is frozen to one serial worker")
    if device_name != "cpu":
        raise Study5BLauncherError("TASK-009 launcher is frozen to CPU on the target machine")
    contract = load_study5b_contract(contract_path)
    inferred_root = contract.path.parents[2]
    paths = _resolve_paths(
        repo_root=inferred_root if repo_root is None else repo_root,
        datasets_config=datasets_config,
        study1_run_root=study1_run_root,
        manifest_root=manifest_root,
        schedule_root=schedule_root,
        output_root=output_root,
        log_root=log_root,
        quarantine_root=quarantine_root,
    )
    registry = DatasetRegistry.from_yaml(paths.datasets_config)
    state = _preflight(
        contract=contract,
        paths=paths,
        registry=registry,
        generate_schedules=generate_schedules or resolved_mode is Study5BLaunchMode.LAUNCH,
    )
    if resolved_mode is Study5BLaunchMode.PREFLIGHT_ONLY:
        return Study5BLauncherReport(
            mode=resolved_mode,
            contract_version=contract.contract_version,
            contract_sha256=contract.contract_sha256,
            workers=workers,
            device=device_name,
            roster_count=len(state.roster),
            schedule_count=len(state.units),
            generated_schedule_count=len(state.generated_schedules),
            preflight_records=state.records,
            launch_records=(),
        )

    launched: list[RunLaunchRecord] = []
    for spec in state.roster:
        unit = state.units[(spec.rotation, spec.seed)]
        current_state, current_reason = _existing_output(spec, unit.static_reference)
        if current_state is ExistingOutputState.SKIPPED_VALID:
            launched.append(
                RunLaunchRecord(
                    spec.experiment_id,
                    FinalRunState.SKIPPED_VALID,
                    None,
                    None,
                    None,
                )
            )
            continue
        quarantined: Path | None = None
        if current_state is ExistingOutputState.INVALID_EXISTING:
            quarantined = quarantine_invalid_output(spec, paths.quarantine_root)
        try:
            _launch_one(
                registry=registry,
                config=state.configs[spec.experiment_id],
                spec=spec,
                device_name=device_name,
            )
            final_state, invalid_reason = _existing_output(spec, unit.static_reference)
            if final_state is not ExistingOutputState.SKIPPED_VALID:
                reason = invalid_reason or "completed output did not validate"
                raise Study5BLauncherError(f"{spec.experiment_id}: {reason}")
        except Exception as exc:
            prior_reason = f"; prior invalid reason: {current_reason}" if current_reason else ""
            raise Study5BLauncherError(
                f"TASK-009 stopped at {spec.experiment_id}{prior_reason}; "
                f"stdout={spec.stdout_log_path}; stderr={spec.stderr_log_path}"
            ) from exc
        launched.append(
            RunLaunchRecord(
                spec.experiment_id,
                FinalRunState.COMPLETED_VALID,
                None if quarantined is None else str(quarantined),
                str(spec.stdout_log_path),
                str(spec.stderr_log_path),
            )
        )
    return Study5BLauncherReport(
        mode=resolved_mode,
        contract_version=contract.contract_version,
        contract_sha256=contract.contract_sha256,
        workers=workers,
        device=device_name,
        roster_count=len(state.roster),
        schedule_count=len(state.units),
        generated_schedule_count=len(state.generated_schedules),
        preflight_records=state.records,
        launch_records=tuple(launched),
    )


def preflight_study5b_launcher(
    **kwargs: Any,
) -> Study5BLauncherReport:
    """CLI-friendly alias for a no-experiment TASK-009 preflight."""

    return run_study5b_launcher(mode=Study5BLaunchMode.PREFLIGHT_ONLY, **kwargs)


def launch_study5b(**kwargs: Any) -> Study5BLauncherReport:
    """CLI-friendly alias for the fail-fast serial TASK-009 launch mode."""

    return run_study5b_launcher(mode=Study5BLaunchMode.LAUNCH, **kwargs)


def format_launcher_report(report: Study5BLauncherReport) -> str:
    """Return stable JSON suitable for a thin CLI handler."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


__all__ = [
    "TASK009_DEFAULT_LOG_DIRECTORY",
    "TASK009_DEFAULT_QUARANTINE_DIRECTORY",
    "TASK009_RUN_IDENTITIES",
    "TASK009_SCHEDULE_DIRECTORY",
    "ExistingOutputState",
    "FinalRunState",
    "RunLaunchRecord",
    "RunPreflightRecord",
    "Study5BLaunchMode",
    "Study5BLauncherError",
    "Study5BLauncherPaths",
    "Study5BLauncherReport",
    "Study5BRunIdentity",
    "Study5BRunSpec",
    "build_task009_roster",
    "format_launcher_report",
    "launch_study5b",
    "preflight_study5b_launcher",
    "quarantine_invalid_output",
    "run_study5b_launcher",
]
