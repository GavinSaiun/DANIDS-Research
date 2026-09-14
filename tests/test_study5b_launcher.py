from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from danids.config.continual import load_continual_experiment_config
from danids.config.experiment import ExperimentConfigError
from danids.config.study5b import (
    TASK009_CONTRACT_SHA256,
    TASK009_CONTRACT_VERSION,
    TASK009_EXTENSION_ROTATIONS,
    TASK009_METHODS,
    TASK009_SEEDS,
    Study5BContractError,
    load_study5b_contract,
)
from danids.continual.supervision import (
    SupervisionSchedule,
    generate_supervision_schedule_from_identity,
)
from danids.data.registry import DatasetRegistry
from danids.experiments import study5b
from danids.experiments.study5b import (
    ExistingOutputState,
    Study5BLauncherError,
    Study5BLauncherPaths,
    Study5BLaunchMode,
    build_task009_roster,
    quarantine_invalid_output,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _contract_path() -> Path:
    return _repo_root() / "configs" / "study5" / "task009_h7_all_order_v1.yaml"


def _paths(root: Path) -> Study5BLauncherPaths:
    return Study5BLauncherPaths(
        repo_root=root,
        datasets_config=root / "configs" / "datasets.local.yaml",
        study1_run_root=root / "runs",
        manifest_root=root / "manifests",
        schedule_root=root / "schedules" / "task009-study5b",
        output_root=root / "runs",
        log_root=root / "logs" / "task009-study5b",
        quarantine_root=root / "runs" / "task009-study5b-quarantine",
    )


def test_contract_and_nine_base_configs_freeze_the_exact_matrix() -> None:
    root = _repo_root()
    contract = load_study5b_contract(_contract_path())
    assert contract.contract_version == TASK009_CONTRACT_VERSION
    assert contract.contract_sha256 == TASK009_CONTRACT_SHA256
    assert contract.methods == TASK009_METHODS
    assert contract.extension_rotations == TASK009_EXTENSION_ROTATIONS
    assert contract.seeds == TASK009_SEEDS
    assert contract.expected_new_run_count == 27
    assert contract.task008_bundle_digest == (
        "9e339bc3dce92a72d666a5b750e49e2bcb7c3247b184d17bbe0410153ab87fea"
    )

    roster = build_task009_roster(contract, _paths(root))
    assert len(roster) == 27
    assert len({spec.experiment_id for spec in roster}) == 27
    assert len({spec.base_config_path for spec in roster}) == 9
    assert len({spec.schedule_path for spec in roster}) == 9
    assert {spec.method for spec in roster} == {"naive_ft", "er", "ft_mem"}
    assert all("ewc" not in spec.experiment_id.lower() for spec in roster)
    assert {spec.schedule_path.name for spec in roster} == {
        f"{'-'.join(rotation)}_s{seed}.json"
        for rotation in TASK009_EXTENSION_ROTATIONS
        for seed in TASK009_SEEDS
    }

    for base_path in {spec.base_config_path for spec in roster}:
        base = load_continual_experiment_config(base_path)
        assert base.experiment.seed == 42
        assert base.adaptation.method in TASK009_METHODS
        assert base.experiment.sequence in TASK009_EXTENSION_ROTATIONS
        assert base.adaptation.epochs == 20


def test_contract_tamper_and_ewc_roster_are_rejected(tmp_path: Path) -> None:
    contract = load_study5b_contract(_contract_path())
    raw = _contract_path().read_text(encoding="utf-8")
    changed = tmp_path / "changed.yaml"
    changed.write_text(
        raw.replace("primary_minimum_physical_support: 50", "primary_minimum_physical_support: 49"),
        encoding="utf-8",
    )
    with pytest.raises(Study5BContractError, match="declared contract digest"):
        load_study5b_contract(changed)

    with pytest.raises(Study5BLauncherError, match="contract cannot populate"):
        build_task009_roster(
            replace(contract, methods=cast(Any, (*contract.methods, "ewc"))),
            _paths(tmp_path),
        )


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    (
        (
            "label_budget_per_later_domain: 100",
            "label_budget_per_later_domain: 99",
            "label budget",
        ),
        ("label_delay_windows: 1", "label_delay_windows: 2", "label delay"),
        ("optimizer: AdamW", "optimizer: SGD", "adaptation requires"),
        ("replay_per_domain: 400", "replay_per_domain: 399", "replay memory"),
        ("window_size: 50000", "window_size: 25000", "window size"),
        ("split_version: task001-v1", "split_version: drift-v1", "datasets.split_version"),
        ("cache_root: data/materialized", "cache_root: data/other", "cache_root"),
    ),
)
def test_base_config_drift_fails_before_launch(
    tmp_path: Path,
    original: str,
    replacement: str,
    message: str,
) -> None:
    contract = load_study5b_contract(_contract_path())
    paths = _paths(_repo_root())
    spec = build_task009_roster(contract, paths)[0]
    changed_path = tmp_path / "configs" / "experiments" / spec.base_config_path.name
    changed_path.parent.mkdir(parents=True)
    text = spec.base_config_path.read_text(encoding="utf-8")
    assert text.count(original) == 1
    changed_path.write_text(text.replace(original, replacement), encoding="utf-8")

    with pytest.raises((ExperimentConfigError, Study5BLauncherError), match=message):
        study5b._validate_base_config(
            replace(spec, base_config_path=changed_path),
            contract,
        )


def test_preflight_groups_all_27_runs_into_exactly_nine_shared_schedules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = load_study5b_contract(_contract_path())
    paths = _paths(tmp_path)
    calls: list[tuple[tuple[str, ...], int, bool]] = []

    monkeypatch.setattr(
        study5b,
        "_load_runtime_configs",
        lambda roster, _contract: {spec.experiment_id: object() for spec in roster},
    )
    monkeypatch.setattr(
        study5b,
        "discover_core_feature_contract",
        lambda _registry: SimpleNamespace(feature_columns=tuple(f"F{i}" for i in range(47))),
    )

    def prepare(**kwargs: Any) -> tuple[Any, bool]:
        representative = kwargs["representative"]
        calls.append(
            (
                representative.rotation,
                representative.seed,
                kwargs["generate_missing_schedule"],
            )
        )
        schedule = SimpleNamespace(
            digest=lambda: f"{''.join(representative.rotation)}-{representative.seed}"
        )
        return SimpleNamespace(static_reference=object(), schedule=schedule), True

    monkeypatch.setattr(study5b, "_prepare_unit", prepare)
    monkeypatch.setattr(
        study5b,
        "_existing_output",
        lambda _spec, _static, _schedule: (ExistingOutputState.ABSENT, None),
    )
    state = study5b._preflight(
        contract=contract,
        paths=paths,
        registry=cast(DatasetRegistry, object()),
        generate_schedules=True,
    )

    assert len(state.roster) == 27
    assert len(state.records) == 27
    assert len(state.units) == 9
    assert len(state.generated_schedules) == 9
    assert len(calls) == 9
    assert all(generate for _rotation, _seed, generate in calls)
    for rotation in TASK009_EXTENSION_ROTATIONS:
        for seed in TASK009_SEEDS:
            records = [
                row
                for row in state.records
                if row.rotation == "-".join(rotation) and row.seed == seed
            ]
            assert len(records) == 3
            assert len({row.schedule_path for row in records}) == 1
            assert len({row.schedule_sha256 for row in records}) == 1


def _synthetic_schedule(sequence: tuple[str, ...], seed: int) -> SupervisionSchedule:
    later = sequence[1:]
    return generate_supervision_schedule_from_identity(
        sequence=sequence,
        seed=seed,
        dataset_fingerprints={domain: domain.lower() * 64 for domain in later},
        online_stream_ranges={domain: (0, 100_000) for domain in later},
    )


def _different_valid_schedule(schedule: SupervisionSchedule) -> SupervisionSchedule:
    first = schedule.entries[0]
    positions = set(first.chronological_positions)
    replacement = next(position for position in range(50_000) if position not in positions)
    changed_positions = tuple(sorted((positions - {min(positions)}) | {replacement}))
    return replace(
        schedule,
        entries=(replace(first, chronological_positions=changed_positions), *schedule.entries[1:]),
    )


def test_existing_output_must_match_prepared_canonical_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = load_study5b_contract(_contract_path())
    spec = build_task009_roster(contract, _paths(tmp_path))[0]
    spec.output_path.mkdir(parents=True)
    canonical = _synthetic_schedule(spec.rotation, spec.seed)
    spec.schedule_path.parent.mkdir(parents=True)
    spec.schedule_path.write_text(canonical.to_json(), encoding="utf-8")
    noncanonical = _different_valid_schedule(canonical)
    noncanonical.validate()
    (spec.output_path / "supervision_schedule.json").write_text(
        noncanonical.to_json(), encoding="utf-8"
    )

    validated = SimpleNamespace(
        path=spec.output_path,
        method=spec.method,
        seed=spec.seed,
        sequence=spec.rotation,
        schedule_digest=noncanonical.digest(),
    )
    monkeypatch.setattr(study5b, "validate_continual_run", lambda _path, _static: validated)

    state, reason = study5b._existing_output(spec, cast(Any, object()), canonical)
    assert state is ExistingOutputState.INVALID_EXISTING
    assert reason is not None and "canonical schedule" in reason
    assert noncanonical.digest() != canonical.digest()
    assert (
        noncanonical.entries[0].chronological_positions
        != canonical.entries[0].chronological_positions
    )

    quarantined = quarantine_invalid_output(spec, tmp_path / "quarantine")
    assert not spec.output_path.exists()
    assert SupervisionSchedule.from_json(quarantined / "supervision_schedule.json") == noncanonical

    spec.output_path.mkdir(parents=True)
    (spec.output_path / "supervision_schedule.json").write_text(
        canonical.to_json(), encoding="utf-8"
    )
    validated.schedule_digest = canonical.digest()
    state, reason = study5b._existing_output(spec, cast(Any, object()), canonical)
    assert state is ExistingOutputState.SKIPPED_VALID
    assert reason is None


def test_preflight_only_mode_never_calls_the_experiment_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = load_study5b_contract(_contract_path())
    paths = _paths(tmp_path)
    roster = build_task009_roster(contract, paths)
    records = tuple(
        study5b.RunPreflightRecord(
            experiment_id=spec.experiment_id,
            rotation="-".join(spec.rotation),
            seed=spec.seed,
            method=spec.method,
            schedule_path=str(spec.schedule_path),
            schedule_sha256="a" * 64,
            source_run_path=str(spec.source_run_path),
            output_path=str(spec.output_path),
            existing_output_state=ExistingOutputState.ABSENT,
            invalid_reason=None,
        )
        for spec in roster
    )
    fake_state = study5b._PreflightState(
        contract=contract,
        roster=roster,
        configs={},
        units={
            (rotation, seed): cast(Any, object())
            for rotation in TASK009_EXTENSION_ROTATIONS
            for seed in TASK009_SEEDS
        },
        generated_schedules=tuple(
            paths.schedule_root / f"{'-'.join(rotation)}_s{seed}.json"
            for rotation in TASK009_EXTENSION_ROTATIONS
            for seed in TASK009_SEEDS
        ),
        records=records,
    )
    monkeypatch.setattr(study5b, "load_study5b_contract", lambda _path: contract)
    monkeypatch.setattr(study5b.DatasetRegistry, "from_yaml", lambda _path: object())
    monkeypatch.setattr(study5b, "_preflight", lambda **_kwargs: fake_state)
    monkeypatch.setattr(
        study5b,
        "run_continual_experiment",
        lambda *_args, **_kwargs: pytest.fail("preflight launched an experiment"),
    )

    report = study5b.run_study5b_launcher(
        contract_path=_contract_path(),
        repo_root=tmp_path,
        mode=Study5BLaunchMode.PREFLIGHT_ONLY,
        generate_schedules=True,
    )
    assert report.mode is Study5BLaunchMode.PREFLIGHT_ONLY
    assert report.roster_count == 27
    assert report.schedule_count == 9
    assert report.generated_schedule_count == 9
    assert report.absent_count == 27
    assert not report.launch_records


def test_invalid_output_is_quarantined_intact_for_whole_run_restart(tmp_path: Path) -> None:
    contract = load_study5b_contract(_contract_path())
    paths = _paths(tmp_path)
    spec = build_task009_roster(contract, paths)[0]
    spec.output_path.mkdir(parents=True)
    partial = spec.output_path / "partial.txt"
    partial.write_text("retain exactly\n", encoding="utf-8")

    destination = quarantine_invalid_output(spec, paths.quarantine_root)

    assert not spec.output_path.exists()
    assert destination.parent == paths.quarantine_root
    assert (destination / "partial.txt").read_text(encoding="utf-8") == "retain exactly\n"


def test_launcher_rejects_parallel_workers_and_non_cpu_device() -> None:
    with pytest.raises(Study5BLauncherError, match="one serial worker"):
        study5b.run_study5b_launcher(
            contract_path=_contract_path(),
            mode="preflight_only",
            workers=2,
        )
    with pytest.raises(Study5BLauncherError, match="frozen to CPU"):
        study5b.run_study5b_launcher(
            contract_path=_contract_path(),
            mode="preflight_only",
            device_name="auto",
        )
