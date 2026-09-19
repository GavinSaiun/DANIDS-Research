from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from danids.config.rdx_training_evidence import (
    RDX004_PROSPECTIVE_BUDGETS,
    RDX004_PROTOCOL_SHA256,
    RDX004_PROTOCOL_VERSION,
    RDX004_ROTATIONS,
    RDX004_SEEDS,
    RDX004RunIdentity,
)
from danids.config.rdx_training_execution import (
    RDX006_EXECUTION_IMPLEMENTATION_VERSION,
    RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST,
    RDX004ExecutionIdentity,
)
from danids.config.study4 import Study4Method
from danids.evaluation import rdx_training_analysis as analysis
from danids.evaluation import rdx_training_execution as validation


def test_batch_reuses_one_authority_and_standalone_keeps_independent_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = (tmp_path / "run-a", tmp_path / "run-b")
    authority = object()
    load_calls: list[Path] = []
    calls: list[tuple[Path, bool, Path, object | None]] = []
    snapshot = object()
    asserted: list[object] = []

    def fake_load(preflight_dir: str | Path) -> object:
        load_calls.append(Path(preflight_dir))
        return authority

    def fake_validate(
        run_dir: str | Path,
        *,
        expected_smoke: bool,
        preflight_dir: str | Path,
        authority: object | None,
    ) -> str:
        path = Path(run_dir)
        calls.append((path, expected_smoke, Path(preflight_dir), authority))
        return path.name

    monkeypatch.setattr(validation, "_load_preflight_authority", fake_load)
    monkeypatch.setattr(validation, "_validate_rdx004_run_with_authority", fake_validate)
    monkeypatch.setattr(
        validation,
        "_capture_dependency_snapshot",
        lambda _authority, _paths: snapshot,
    )
    monkeypatch.setattr(
        validation,
        "_assert_dependency_snapshot_unchanged",
        asserted.append,
    )

    standalone = tuple(
        validation.validate_rdx004_run(path, preflight_dir=tmp_path / "preflight") for path in paths
    )
    batch = validation.validate_rdx004_runs(
        paths,
        preflight_dir=tmp_path / "preflight",
    )

    assert standalone == batch == ("run-a", "run-b")
    assert [call[3] for call in calls[:2]] == [None, None]
    assert [call[3] for call in calls[2:]] == [authority, authority]
    assert load_calls == [tmp_path / "preflight"]
    assert asserted == [snapshot]


def test_standalone_preflight_reference_loads_a_fresh_authority_each_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    authorities = iter((object(), object()))
    loaded: list[Path] = []
    projected: list[tuple[object, RDX004ExecutionIdentity, str]] = []
    identity = RDX004ExecutionIdentity(
        scientific_identity=RDX004RunIdentity(
            budget=RDX004_PROSPECTIVE_BUDGETS[0],
            rotation=RDX004_ROTATIONS[0],
            seed=RDX004_SEEDS[0],
        ),
        smoke=False,
    )

    def fake_load(path: str | Path) -> object:
        loaded.append(Path(path))
        return next(authorities)

    def fake_project(
        authority: object,
        projected_identity: RDX004ExecutionIdentity,
        *,
        declared_bundle_digest: str,
    ) -> object:
        projected.append((authority, projected_identity, declared_bundle_digest))
        return authority

    monkeypatch.setattr(validation, "_load_preflight_authority", fake_load)
    monkeypatch.setattr(validation, "_preflight_reference_from_authority", fake_project)

    first = validation._load_preflight_reference(
        tmp_path / "preflight",
        identity,
        declared_bundle_digest="1" * 64,
    )
    second = validation._load_preflight_reference(
        tmp_path / "preflight",
        identity,
        declared_bundle_digest="1" * 64,
    )

    assert first is not second
    assert loaded == [tmp_path / "preflight", tmp_path / "preflight"]
    assert [entry[0] for entry in projected] == [first, second]
    assert all(entry[1:] == (identity, "1" * 64) for entry in projected)


def test_paired_b100_source_is_validated_once_per_scoped_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    authority = validation._PreflightAuthority(
        root=tmp_path / "preflight",
        bundle_digest="1" * 64,
        contract={},
        run_rows=(),
        starting_rows=(),
        query_rows=(),
        nested_rows=(),
        audit_rows=(),
        allocation_rows=(),
        paired_sources={},
    )
    paired_run = (tmp_path / "paired-b100").resolve()
    validation_calls: list[Path] = []
    pipeline_fields = {
        "feature_contract_version": "v1",
        "feature_columns": ["x"],
        "dataset_fingerprints": {},
        "manifest_partition_ranges": {},
        "window_size": 50_000,
        "boundary_mode": "task_free",
        "health_artifact_identity": "2" * 64,
        "health_model_sha256": "3" * 64,
        "health_feature_contract_digest": "4" * 64,
    }

    def fake_validate(path: str | Path, *, allow_smoke: bool) -> SimpleNamespace:
        assert allow_smoke is False
        resolved = Path(path).resolve()
        validation_calls.append(resolved)
        return SimpleNamespace(
            path=resolved,
            experiment_id="E4_OFFLINE_ORACLE_U-T-C-B_s42",
            method=Study4Method.OFFLINE_ORACLE,
            sequence=("U", "T", "C", "B"),
            seed=42,
            smoke=False,
            contract_digest="5" * 64,
            source_checkpoint_sha256="6" * 64,
        )

    def fake_json(path: Path, *, canonical: bool = True) -> dict[str, Any]:
        assert canonical is True
        if path.name == validation.MANIFEST_FILENAME:
            return {"bundle_digest": "7" * 64}
        if path.name == validation.STUDY4_PROVENANCE_FILENAME:
            return dict(pipeline_fields)
        if path.name == validation.MEMORY_FILENAME:
            return {"source_selection": {}}
        raise AssertionError(path)

    monkeypatch.setattr(validation, "validate_study4_run", fake_validate)
    monkeypatch.setattr(validation, "_load_json", fake_json)
    monkeypatch.setattr(validation, "_load_yaml", lambda _path: {"config": "source"})
    monkeypatch.setattr(validation, "_file_sha256", lambda _path: "8" * 64)

    first = validation._load_paired_source(authority, paired_run)
    second = validation._load_paired_source(authority, paired_run)

    assert first is second
    assert validation_calls == [paired_run]
    assert authority.paired_sources == {paired_run: first}


@pytest.mark.parametrize(
    "mutation",
    ["content", "add", "delete", "empty_directory", "explicit"],
)
def test_dependency_snapshot_detects_content_and_file_set_tampering(
    mutation: str,
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    target = nested / "artifact.txt"
    target.write_text("original\n", encoding="utf-8")
    explicit = tmp_path / "explicit.txt"
    explicit.write_text("sentinel\n", encoding="utf-8")
    snapshot = validation._DependencyDigestSnapshot(
        roots=(root,),
        explicit_files=(explicit,),
        digests=validation._dependency_digests((root,), (explicit,)),
    )

    if mutation == "content":
        target.write_text("tampered\n", encoding="utf-8")
    elif mutation == "add":
        (root / "unexpected.txt").write_text("extra\n", encoding="utf-8")
    elif mutation == "delete":
        target.unlink()
    elif mutation == "empty_directory":
        (root / "unexpected-directory").mkdir()
    else:
        explicit.write_text("tampered sentinel\n", encoding="utf-8")

    with pytest.raises(validation.RDX004RunArtifactError, match="dependency changed"):
        validation._assert_dependency_snapshot_unchanged(snapshot)


def test_rdx007_prospective_roster_uses_one_batch_validation_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    preflight_dir = tmp_path / "preflight"
    run_root.mkdir()
    identities = tuple(
        RDX004RunIdentity(budget=budget, rotation=rotation, seed=seed)
        for budget in RDX004_PROSPECTIVE_BUDGETS
        for rotation in RDX004_ROTATIONS
        for seed in RDX004_SEEDS
    )
    paths = tuple((run_root / identity.run_id).resolve() for identity in identities)
    for path in paths:
        path.mkdir()
    batch_calls: list[tuple[tuple[Path, ...], bool, Path]] = []

    def fake_batch(
        run_dirs: tuple[Path, ...],
        *,
        expected_smoke: bool,
        preflight_dir: str | Path,
    ) -> tuple[SimpleNamespace, ...]:
        resolved = tuple(Path(path).resolve() for path in run_dirs)
        batch_calls.append((resolved, expected_smoke, Path(preflight_dir)))
        return tuple(
            SimpleNamespace(
                run_id=identity.run_id,
                expected_confirmatory_run_id=identity.run_id,
                budget=identity.budget,
                rotation=identity.rotation,
                seed=identity.seed,
                smoke=False,
                paired_b100_experiment_id=identity.paired_b100_experiment_id,
                paired_study1_experiment_id=RDX004ExecutionIdentity(
                    scientific_identity=identity,
                    smoke=False,
                ).paired_study1_experiment_id,
                preflight_bundle_digest=RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST,
                artifact_bundle_digest="9" * 64,
                query_event_count=12,
                scope_closure_count=3,
                oracle_decision_count=1,
            )
            for identity in identities
        )

    monkeypatch.setattr(analysis, "validate_rdx004_runs", fake_batch)
    monkeypatch.setattr(analysis, "_manifest_identity", lambda _path: ("8" * 64, "9" * 64))
    monkeypatch.setattr(
        analysis,
        "_load_json",
        lambda _path: {
            "code": {
                "commit": analysis.RDX006_EXECUTION_COMMIT,
                "working_tree_dirty": False,
            },
            "execution_config": {
                "implementation_version": RDX006_EXECUTION_IMPLEMENTATION_VERSION,
            },
            "protocol": {
                "version": RDX004_PROTOCOL_VERSION,
                "sha256": RDX004_PROTOCOL_SHA256,
            },
        },
    )

    runs = analysis._validate_prospective_sources(run_root, preflight_dir)

    assert len(runs) == 24
    assert batch_calls == [(paths, False, preflight_dir)]
