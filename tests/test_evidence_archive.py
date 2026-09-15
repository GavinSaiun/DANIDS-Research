from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from danids.utils import evidence_archive as evidence_archive_module
from danids.utils.evidence_archive import (
    EVIDENCE_ROOTS,
    FROZEN_EVIDENCE_ARCHIVE_NAME,
    FROZEN_EVIDENCE_AUTHORITY_NAME,
    FROZEN_EVIDENCE_MANIFEST_NAME,
    FROZEN_EVIDENCE_SIDECAR_NAME,
    EvidenceArchiveError,
    build_frozen_evidence_archive,
    validate_frozen_evidence_archive,
)


def _synthetic_evidence(root: Path) -> None:
    for index, requirement in enumerate(EVIDENCE_ROOTS):
        evidence_root = root.joinpath(*requirement.path.split("/"))
        evidence_root.mkdir(parents=True)
        for anchor_index, anchor_name in enumerate(requirement.anchors):
            anchor = evidence_root.joinpath(*anchor_name.split("/"))
            anchor.parent.mkdir(parents=True, exist_ok=True)
            anchor.write_bytes(f"anchor-{index}-{anchor_index}\n".encode())
        nested = evidence_root / "nested" / f"payload-{index}.txt"
        nested.parent.mkdir()
        nested.write_bytes((f"evidence-{index}\n" * (index + 1)).encode())
    _seal_nested_manifests(root)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal_nested_manifests(root: Path) -> None:
    manifest_paths = (
        "study4/policy-qualification-v1-final/policy_qualification_manifest.json",
        "study4/e4-confirmatory-final/artifact_manifest.json",
        "study4/e4-confirmatory-analysis/analysis_manifest.json",
        "study5/threat-audit-v1/artifact_manifest.json",
        "study5/task009-study5b-all-order-replay-v1/artifact_manifest.json",
    )
    for relative in manifest_paths:
        manifest_path = root / relative
        files = {
            path.relative_to(manifest_path.parent).as_posix(): _digest(path)
            for path in sorted(manifest_path.parent.rglob("*"))
            if path.is_file() and path != manifest_path
        }
        bundle = hashlib.sha256(
            json.dumps(files, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        manifest_path.write_bytes(
            _canonical_json({"bundle_digest": bundle, "files": files, "version": "test"})
        )

    health_manifest = root / "study4/frozen-core-health/health_model_manifest.json"
    model = health_manifest.with_name("health_model.pkl")
    health_manifest.write_bytes(
        _canonical_json(
            {
                "model": {
                    "filename": model.name,
                    "serialized_sha256": _digest(model),
                    "serialized_size_bytes": model.stat().st_size,
                },
                "version": "test",
            }
        )
    )


def _write_authority(source: Path, path: Path) -> Path:
    files = []
    for requirement in EVIDENCE_ROOTS:
        root = source.joinpath(*requirement.path.split("/"))
        for member in sorted(root.rglob("*")):
            if member.is_file():
                files.append(
                    {
                        "path": member.relative_to(source).as_posix(),
                        "sha256": _digest(member),
                        "size_bytes": member.stat().st_size,
                    }
                )
    files.sort(key=lambda item: item["path"])
    inventory_sha = hashlib.sha256(_canonical_json(files)).hexdigest()
    path.write_bytes(
        _canonical_json(
            {
                "evidence_roots": [item.path for item in EVIDENCE_ROOTS],
                "format_version": 1,
                "inventory_sha256": inventory_sha,
                "source_file_count": len(files),
            }
        )
    )
    return path


def _source_snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    snapshot: dict[str, tuple[bytes, int]] = {}
    for requirement in EVIDENCE_ROOTS:
        evidence_root = root.joinpath(*requirement.path.split("/"))
        for path in evidence_root.rglob("*"):
            if path.is_file():
                snapshot[path.relative_to(root).as_posix()] = (
                    path.read_bytes(),
                    path.stat().st_mtime_ns,
                )
    return snapshot


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _reseal_archive(
    archive_path: Path,
    sidecar_path: Path,
    files: dict[str, bytes],
    manifest: dict[str, object],
) -> None:
    manifest_bytes = _canonical_json(manifest)
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(FROZEN_EVIDENCE_MANIFEST_NAME, manifest_bytes)
        for name in sorted(files):
            archive.writestr(name, files[name])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    archive_bytes = archive_path.read_bytes()
    sidecar["archive_sha256"] = hashlib.sha256(archive_bytes).hexdigest()
    sidecar["archive_size_bytes"] = len(archive_bytes)
    sidecar["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    raw_manifest_files = manifest["files"]
    assert isinstance(raw_manifest_files, list)
    sidecar["source_file_count"] = len(raw_manifest_files)
    sidecar_path.write_bytes(_canonical_json(sidecar))


def test_build_uses_exact_roots_preserves_layout_and_validates(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _synthetic_evidence(source)
    authority = _write_authority(source, tmp_path / "authority.json")

    result = build_frozen_evidence_archive(source, tmp_path / "output", authority_path=authority)

    assert result.archive_path.name == FROZEN_EVIDENCE_ARCHIVE_NAME
    assert result.sidecar_path.name == FROZEN_EVIDENCE_SIDECAR_NAME
    assert result.archive_size_bytes == result.archive_path.stat().st_size
    assert result.archive_sha256 == hashlib.sha256(result.archive_path.read_bytes()).hexdigest()

    with zipfile.ZipFile(result.archive_path) as archive:
        manifest = json.loads(archive.read(FROZEN_EVIDENCE_MANIFEST_NAME))
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
        assert manifest["evidence_roots"] == [item.path for item in EVIDENCE_ROOTS]
        expected_paths = sorted(
            path.relative_to(source).as_posix()
            for requirement in EVIDENCE_ROOTS
            for path in source.joinpath(*requirement.path.split("/")).rglob("*")
            if path.is_file()
        )
        assert [item["path"] for item in manifest["files"]] == expected_paths
        assert set(archive.namelist()) == {FROZEN_EVIDENCE_MANIFEST_NAME, *expected_paths}

    assert validate_frozen_evidence_archive(result.archive_path, authority_path=authority) == result


@pytest.mark.parametrize("failure", ["root", "anchor", "empty_anchor"])
def test_build_rejects_missing_root_or_anchor(tmp_path: Path, failure: str) -> None:
    source = tmp_path / "source"
    _synthetic_evidence(source)
    authority = _write_authority(source, tmp_path / "authority.json")
    requirement = EVIDENCE_ROOTS[0]
    evidence_root = source.joinpath(*requirement.path.split("/"))
    anchor = evidence_root.joinpath(*requirement.anchors[0].split("/"))
    if failure == "root":
        for path in sorted(evidence_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        evidence_root.rmdir()
    elif failure == "anchor":
        anchor.unlink()
    else:
        anchor.write_bytes(b"")

    with pytest.raises(EvidenceArchiveError, match="required evidence"):
        build_frozen_evidence_archive(source, tmp_path / "output", authority_path=authority)


def test_build_is_byte_deterministic_and_does_not_mutate_sources(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _synthetic_evidence(source)
    authority = _write_authority(source, tmp_path / "authority.json")
    before = _source_snapshot(source)

    first = build_frozen_evidence_archive(source, tmp_path / "output-one", authority_path=authority)
    second = build_frozen_evidence_archive(
        source, tmp_path / "output-two", authority_path=authority
    )

    assert first.archive_path.read_bytes() == second.archive_path.read_bytes()
    assert first.sidecar_path.read_bytes() == second.sidecar_path.read_bytes()
    assert (
        first.archive_sha256 == "f7d752389e12f7503d88ae5e30b7d3c1e4a02981a8987e27c91fbdb8f7e97942"
    )
    assert _source_snapshot(source) == before


def test_build_outputs_are_write_once(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _synthetic_evidence(source)
    authority = _write_authority(source, tmp_path / "authority.json")
    original = build_frozen_evidence_archive(source, output, authority_path=authority)
    archive_bytes = original.archive_path.read_bytes()
    sidecar_bytes = original.sidecar_path.read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        build_frozen_evidence_archive(source, output, authority_path=authority)

    assert original.archive_path.read_bytes() == archive_bytes
    assert original.sidecar_path.read_bytes() == sidecar_bytes


def test_validation_rejects_archive_corruption(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _synthetic_evidence(source)
    authority = _write_authority(source, tmp_path / "authority.json")
    result = build_frozen_evidence_archive(source, tmp_path / "output", authority_path=authority)
    archive_bytes = bytearray(result.archive_path.read_bytes())
    archive_bytes[len(archive_bytes) // 2] ^= 0x01
    result.archive_path.write_bytes(archive_bytes)

    with pytest.raises(EvidenceArchiveError, match="sidecar"):
        validate_frozen_evidence_archive(result.archive_path, authority_path=authority)


def test_validation_rejects_sidecar_corruption(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _synthetic_evidence(source)
    authority = _write_authority(source, tmp_path / "authority.json")
    result = build_frozen_evidence_archive(source, tmp_path / "output", authority_path=authority)
    sidecar = json.loads(result.sidecar_path.read_text(encoding="utf-8"))
    sidecar["archive_size_bytes"] += 1
    result.sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EvidenceArchiveError, match="sidecar"):
        validate_frozen_evidence_archive(result.archive_path, authority_path=authority)


def test_validation_rejects_resealed_member_outside_required_roots(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _synthetic_evidence(source)
    authority = _write_authority(source, tmp_path / "authority.json")
    result = build_frozen_evidence_archive(source, tmp_path / "output", authority_path=authority)
    with zipfile.ZipFile(result.archive_path) as archive:
        manifest = json.loads(archive.read(FROZEN_EVIDENCE_MANIFEST_NAME))
        files = {
            name: archive.read(name)
            for name in archive.namelist()
            if name != FROZEN_EVIDENCE_MANIFEST_NAME
        }
    smuggled = b"not frozen evidence\n"
    files["unexpected/payload.txt"] = smuggled
    manifest["files"].append(
        {
            "path": "unexpected/payload.txt",
            "sha256": hashlib.sha256(smuggled).hexdigest(),
            "size_bytes": len(smuggled),
        }
    )
    manifest["files"] = sorted(manifest["files"], key=lambda item: item["path"])
    _reseal_archive(result.archive_path, result.sidecar_path, files, manifest)

    with pytest.raises(EvidenceArchiveError, match="frozen authority"):
        validate_frozen_evidence_archive(result.archive_path, authority_path=authority)


def test_validation_rejects_resealed_archive_missing_anchor(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _synthetic_evidence(source)
    authority = _write_authority(source, tmp_path / "authority.json")
    result = build_frozen_evidence_archive(source, tmp_path / "output", authority_path=authority)
    requirement = EVIDENCE_ROOTS[0]
    anchor_path = f"{requirement.path}/{requirement.anchors[0]}"
    with zipfile.ZipFile(result.archive_path) as archive:
        manifest = json.loads(archive.read(FROZEN_EVIDENCE_MANIFEST_NAME))
        files = {
            name: archive.read(name)
            for name in archive.namelist()
            if name not in {FROZEN_EVIDENCE_MANIFEST_NAME, anchor_path}
        }
    manifest["files"] = [item for item in manifest["files"] if item["path"] != anchor_path]
    _reseal_archive(result.archive_path, result.sidecar_path, files, manifest)

    with pytest.raises(EvidenceArchiveError, match="frozen authority"):
        validate_frozen_evidence_archive(result.archive_path, authority_path=authority)


def test_build_rejects_missing_non_anchor_from_authoritative_inventory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _synthetic_evidence(source)
    authority = _write_authority(source, tmp_path / "authority.json")
    (source / EVIDENCE_ROOTS[0].path / "nested" / "payload-0.txt").unlink()

    with pytest.raises(EvidenceArchiveError, match="file count differs"):
        build_frozen_evidence_archive(source, tmp_path / "output", authority_path=authority)


def test_build_rejects_unexpected_extra_from_authoritative_inventory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _synthetic_evidence(source)
    authority = _write_authority(source, tmp_path / "authority.json")
    (source / EVIDENCE_ROOTS[0].path / "unexpected.txt").write_text(
        "not frozen\n", encoding="utf-8"
    )

    with pytest.raises(EvidenceArchiveError, match="file count differs"):
        build_frozen_evidence_archive(source, tmp_path / "output", authority_path=authority)


def test_build_rejects_stale_nested_manifest_even_under_resealed_authority(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _synthetic_evidence(source)
    payload = source / "study5/threat-audit-v1/nested/payload-8.txt"
    payload.write_text("tampered after bundle sealing\n", encoding="utf-8")
    authority = _write_authority(source, tmp_path / "authority.json")

    with pytest.raises(EvidenceArchiveError, match="nested manifest file digest differs"):
        build_frozen_evidence_archive(source, tmp_path / "output", authority_path=authority)


def test_validation_rejects_authority_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _synthetic_evidence(source)
    authority = _write_authority(source, tmp_path / "authority.json")
    result = build_frozen_evidence_archive(source, tmp_path / "output", authority_path=authority)
    authority_data = json.loads(authority.read_text(encoding="utf-8"))
    authority_data["inventory_sha256"] = "0" * 64
    foreign_authority = tmp_path / "foreign-authority.json"
    foreign_authority.write_bytes(_canonical_json(authority_data))

    with pytest.raises(EvidenceArchiveError, match="authority identity differs"):
        validate_frozen_evidence_archive(result.archive_path, authority_path=foreign_authority)


def test_default_authority_is_canonical_and_declared_as_package_data() -> None:
    authority_path = Path(evidence_archive_module.__file__).with_name(
        FROZEN_EVIDENCE_AUTHORITY_NAME
    )
    raw = authority_path.read_bytes()
    authority = json.loads(raw)

    assert raw == _canonical_json(authority)
    assert authority["evidence_roots"] == [item.path for item in EVIDENCE_ROOTS]
    assert authority["source_file_count"] == 119
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    assert "utils/frozen_evidence_authority.json" in pyproject.read_text(encoding="utf-8")
