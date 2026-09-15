"""Deterministic, write-once packaging of the frozen DANIDS 2.0 evidence tree."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

FROZEN_EVIDENCE_ARCHIVE_NAME: Final = "DANIDS-2.0-frozen-evidence.zip"
FROZEN_EVIDENCE_MANIFEST_NAME: Final = "DANIDS-2.0-frozen-evidence-manifest.json"
FROZEN_EVIDENCE_SIDECAR_NAME: Final = "DANIDS-2.0-frozen-evidence.archive.json"
FROZEN_EVIDENCE_AUTHORITY_NAME: Final = "frozen_evidence_authority.json"

_FORMAT_VERSION: Final = 1
_BUFFER_SIZE: Final = 1024 * 1024
_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
_NESTED_FILE_MANIFESTS: Final = (
    "study4/policy-qualification-v1-final/policy_qualification_manifest.json",
    "study4/e4-confirmatory-final/artifact_manifest.json",
    "study4/e4-confirmatory-analysis/analysis_manifest.json",
    "study5/threat-audit-v1/artifact_manifest.json",
    "study5/task009-study5b-all-order-replay-v1/artifact_manifest.json",
)
_HEALTH_MODEL_MANIFEST: Final = "study4/frozen-core-health/health_model_manifest.json"


@dataclass(frozen=True)
class EvidenceRoot:
    """One required evidence directory and the files proving that it is complete."""

    path: str
    anchors: tuple[str, ...]


EVIDENCE_ROOTS: Final[tuple[EvidenceRoot, ...]] = (
    EvidenceRoot("study1/static-s42-s44", ("study1_summary.json",)),
    EvidenceRoot("study2/u-t-c-b-s42-s44", ("study2_summary.json",)),
    EvidenceRoot(
        "study3/health-s42-s44",
        ("study3_summary.json", "evaluation_contract.json"),
    ),
    EvidenceRoot(
        "study4/frozen-core-health",
        ("health_model.pkl", "health_model_manifest.json"),
    ),
    EvidenceRoot(
        "study4/policy-development-v1-final",
        ("policy_development_summary.json", "evaluation_contract.json", "leakage_checks.json"),
    ),
    EvidenceRoot(
        "study4/policy-qualification-v1-final",
        ("policy_qualification.json", "policy_qualification_manifest.json"),
    ),
    EvidenceRoot(
        "study4/e4-confirmatory-final",
        ("study4_summary.json", "artifact_manifest.json", "evaluation_contract.json"),
    ),
    EvidenceRoot(
        "study4/e4-confirmatory-analysis",
        ("study4_confirmatory_results.json", "analysis_manifest.json"),
    ),
    EvidenceRoot(
        "study5/threat-audit-v1",
        ("study5_summary.json", "study5_contract.json", "artifact_manifest.json"),
    ),
    EvidenceRoot(
        "study5/task009-study5b-all-order-replay-v1",
        ("study5b_summary.json", "task009_contract.json", "artifact_manifest.json"),
    ),
)


class EvidenceArchiveError(ValueError):
    """Raised when frozen evidence is incomplete, changes, or fails validation."""


@dataclass(frozen=True)
class FrozenEvidenceArchive:
    """Paths and integrity metadata for a built or validated evidence archive."""

    archive_path: Path
    sidecar_path: Path
    archive_size_bytes: int
    archive_sha256: str
    authority_sha256: str
    manifest_sha256: str
    source_file_count: int


@dataclass(frozen=True)
class _SourceFile:
    path: Path
    archive_path: str
    size_bytes: int
    sha256: str


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_path(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_BUFFER_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _root_members(root: Path) -> tuple[Path, ...]:
    members: list[Path] = []
    resolved_root = root.resolve(strict=True)
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise EvidenceArchiveError(f"symbolic links are not allowed in evidence: {candidate}")
        resolved_candidate = candidate.resolve(strict=True)
        if not _is_relative_to(resolved_candidate, resolved_root):
            raise EvidenceArchiveError(
                f"evidence entry resolves outside its required directory: {candidate}"
            )
        if candidate.is_file():
            members.append(candidate)
        elif not candidate.is_dir():
            raise EvidenceArchiveError(f"unsupported evidence filesystem entry: {candidate}")
    return tuple(sorted(members, key=lambda path: path.relative_to(root).as_posix()))


def _collect_sources(source_root: Path, output_dir: Path) -> tuple[_SourceFile, ...]:
    source_root = source_root.resolve(strict=True)
    if not source_root.is_dir():
        raise EvidenceArchiveError(f"source root is not a directory: {source_root}")

    resolved_output = output_dir.resolve(strict=False)
    source_files: list[_SourceFile] = []
    archive_names: set[str] = set()
    casefolded_names: set[str] = set()

    for requirement in EVIDENCE_ROOTS:
        evidence_root = source_root.joinpath(*PurePosixPath(requirement.path).parts)
        if not evidence_root.is_dir():
            raise EvidenceArchiveError(
                f"required evidence directory is missing: {requirement.path}"
            )
        if evidence_root.is_symlink():
            raise EvidenceArchiveError(
                f"required evidence directory must not be a symbolic link: {requirement.path}"
            )
        resolved_evidence_root = evidence_root.resolve(strict=True)
        if not _is_relative_to(resolved_evidence_root, source_root):
            raise EvidenceArchiveError(
                f"required evidence directory resolves outside source root: {requirement.path}"
            )
        if _is_relative_to(resolved_output, resolved_evidence_root):
            raise EvidenceArchiveError(
                f"output directory must not be inside evidence directory: {requirement.path}"
            )

        for anchor_name in requirement.anchors:
            anchor = evidence_root.joinpath(*PurePosixPath(anchor_name).parts)
            if not anchor.is_file() or anchor.is_symlink():
                raise EvidenceArchiveError(
                    f"required evidence anchor is missing: {requirement.path}/{anchor_name}"
                )
            if anchor.stat().st_size == 0:
                raise EvidenceArchiveError(
                    f"required evidence anchor is empty: {requirement.path}/{anchor_name}"
                )

        members = _root_members(evidence_root)
        if not members:
            raise EvidenceArchiveError(f"required evidence directory is empty: {requirement.path}")
        for path in members:
            relative = path.relative_to(source_root).as_posix()
            lowered = relative.casefold()
            if relative in archive_names or lowered in casefolded_names:
                raise EvidenceArchiveError(f"duplicate evidence archive path: {relative}")
            digest, size = _sha256_path(path)
            source_files.append(
                _SourceFile(
                    path=path,
                    archive_path=relative,
                    size_bytes=size,
                    sha256=digest,
                )
            )
            archive_names.add(relative)
            casefolded_names.add(lowered)

    return tuple(sorted(source_files, key=lambda item: item.archive_path))


def _manifest(source_files: Iterable[_SourceFile]) -> dict[str, object]:
    files = [
        {"path": item.archive_path, "sha256": item.sha256, "size_bytes": item.size_bytes}
        for item in source_files
    ]
    return {
        "archive_name": FROZEN_EVIDENCE_ARCHIVE_NAME,
        "authority_sha256": "",
        "evidence_roots": [requirement.path for requirement in EVIDENCE_ROOTS],
        "files": files,
        "format_version": _FORMAT_VERSION,
        "inventory_sha256": _inventory_sha256(files),
    }


def _inventory_sha256(files: object) -> str:
    """Digest the exact ordered path/size/content inventory."""

    return hashlib.sha256(_canonical_json(files)).hexdigest()


def _authority_file(path: str | Path | None) -> Path:
    return (
        Path(path) if path is not None else Path(__file__).with_name(FROZEN_EVIDENCE_AUTHORITY_NAME)
    )


def _load_authority(path: str | Path | None) -> tuple[dict[str, Any], str]:
    authority_path = _authority_file(path)
    try:
        raw = authority_path.read_bytes()
    except OSError as exc:
        raise EvidenceArchiveError(
            f"cannot read frozen evidence authority: {authority_path}"
        ) from exc
    authority = _load_canonical_json(raw, "frozen evidence authority")
    if set(authority) != {
        "evidence_roots",
        "format_version",
        "inventory_sha256",
        "source_file_count",
    }:
        raise EvidenceArchiveError("frozen evidence authority fields do not match the schema")
    if authority["format_version"] != _FORMAT_VERSION:
        raise EvidenceArchiveError("unsupported frozen evidence authority format version")
    expected_roots = [requirement.path for requirement in EVIDENCE_ROOTS]
    if authority["evidence_roots"] != expected_roots:
        raise EvidenceArchiveError("frozen evidence authority does not list the exact roots")
    _required_sha256(authority["inventory_sha256"], "authority.inventory_sha256")
    _required_int(authority["source_file_count"], "authority.source_file_count")
    return authority, hashlib.sha256(raw).hexdigest()


def _assert_authorized_inventory(
    source_files: tuple[_SourceFile, ...], authority: dict[str, Any]
) -> None:
    files = [
        {"path": item.archive_path, "sha256": item.sha256, "size_bytes": item.size_bytes}
        for item in source_files
    ]
    if len(files) != authority["source_file_count"]:
        raise EvidenceArchiveError("evidence file count differs from frozen authority")
    if _inventory_sha256(files) != authority["inventory_sha256"]:
        raise EvidenceArchiveError("evidence inventory differs from frozen authority")


def _decode_json(raw: bytes, description: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceArchiveError(f"invalid {description} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceArchiveError(f"{description} must be a JSON object")
    return value


def _validate_nested_manifests(
    files: dict[str, tuple[str, int]],
    read_bytes: Callable[[str], bytes],
) -> None:
    """Verify the artifact bundles' own persisted file-integrity declarations."""

    for manifest_path in _NESTED_FILE_MANIFESTS:
        manifest = _decode_json(read_bytes(manifest_path), manifest_path)
        declared = manifest.get("files")
        if not isinstance(declared, dict) or not all(
            isinstance(name, str) and isinstance(digest, str) for name, digest in declared.items()
        ):
            raise EvidenceArchiveError(f"invalid nested files map: {manifest_path}")
        prefix = manifest_path.rsplit("/", 1)[0] + "/"
        actual_peers = {
            path.removeprefix(prefix)
            for path in files
            if path.startswith(prefix) and path != manifest_path
        }
        if set(declared) != actual_peers:
            raise EvidenceArchiveError(f"nested manifest membership differs: {manifest_path}")
        for name, expected_digest in declared.items():
            _required_sha256(expected_digest, f"{manifest_path}.files[{name}]")
            if files[prefix + name][0] != expected_digest:
                raise EvidenceArchiveError(f"nested manifest file digest differs: {prefix + name}")
        if "bundle_digest" in manifest:
            expected_bundle = _required_sha256(
                manifest["bundle_digest"], f"{manifest_path}.bundle_digest"
            )
            canonical_files = json.dumps(
                declared, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            if hashlib.sha256(canonical_files).hexdigest() != expected_bundle:
                raise EvidenceArchiveError(f"nested bundle digest differs: {manifest_path}")

    health = _decode_json(read_bytes(_HEALTH_MODEL_MANIFEST), _HEALTH_MODEL_MANIFEST)
    model = health.get("model")
    if not isinstance(model, dict):
        raise EvidenceArchiveError("health model manifest lacks model integrity metadata")
    filename = model.get("filename")
    if not isinstance(filename, str) or PurePosixPath(filename).name != filename:
        raise EvidenceArchiveError("health model manifest has invalid model filename")
    model_path = _HEALTH_MODEL_MANIFEST.rsplit("/", 1)[0] + "/" + filename
    expected_sha = _required_sha256(
        model.get("serialized_sha256"), "health_model_manifest.model.serialized_sha256"
    )
    expected_size = _required_int(
        model.get("serialized_size_bytes"),
        "health_model_manifest.model.serialized_size_bytes",
    )
    if files.get(model_path) != (expected_sha, expected_size):
        raise EvidenceArchiveError("health model manifest does not match serialized model")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.extra = b""
    info.comment = b""
    return info


def _write_zip(path: Path, manifest_bytes: bytes, source_files: tuple[_SourceFile, ...]) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(_zip_info(FROZEN_EVIDENCE_MANIFEST_NAME), manifest_bytes)
        for source in source_files:
            digest = hashlib.sha256()
            size = 0
            with (
                source.path.open("rb") as input_stream,
                archive.open(
                    _zip_info(source.archive_path), mode="w", force_zip64=True
                ) as output_stream,
            ):
                while chunk := input_stream.read(_BUFFER_SIZE):
                    output_stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if digest.hexdigest() != source.sha256 or size != source.size_bytes:
                raise EvidenceArchiveError(f"evidence file changed while archiving: {source.path}")


def _publish_write_once(temp_path: Path, destination: Path) -> None:
    try:
        os.link(temp_path, destination)
    except FileExistsError:
        raise FileExistsError(f"frozen evidence output already exists: {destination}") from None


def build_frozen_evidence_archive(
    source_root: str | Path,
    output_dir: str | Path,
    *,
    authority_path: str | Path | None = None,
) -> FrozenEvidenceArchive:
    """Build the fixed DANIDS 2.0 evidence archive without altering source artifacts.

    Both final files are write-once. Reproducibility checks should therefore build into
    distinct output directories and compare the returned archive digests or bytes.
    """

    source = Path(source_root)
    output = Path(output_dir)
    archive_path = output / FROZEN_EVIDENCE_ARCHIVE_NAME
    sidecar_path = output / FROZEN_EVIDENCE_SIDECAR_NAME
    if archive_path.exists() or sidecar_path.exists():
        existing = archive_path if archive_path.exists() else sidecar_path
        raise FileExistsError(f"frozen evidence output already exists: {existing}")

    authority, authority_sha256 = _load_authority(authority_path)
    source_files = _collect_sources(source, output)
    _assert_authorized_inventory(source_files, authority)
    source_file_index = {item.archive_path: (item.sha256, item.size_bytes) for item in source_files}
    source_paths = {item.archive_path: item.path for item in source_files}
    _validate_nested_manifests(
        source_file_index,
        lambda archive_path: source_paths[archive_path].read_bytes(),
    )
    initial_names = tuple(item.archive_path for item in source_files)
    manifest = _manifest(source_files)
    manifest["authority_sha256"] = authority_sha256
    manifest_bytes = _canonical_json(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    output.mkdir(parents=True, exist_ok=True)
    archive_temp: Path | None = None
    sidecar_temp: Path | None = None
    archive_published = False
    try:
        with tempfile.NamedTemporaryFile(
            dir=output,
            prefix=f".{FROZEN_EVIDENCE_ARCHIVE_NAME}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            archive_temp = Path(stream.name)
        _write_zip(archive_temp, manifest_bytes, source_files)

        final_sources = _collect_sources(source, output)
        _assert_authorized_inventory(final_sources, authority)
        final_names = tuple(item.archive_path for item in final_sources)
        final_fingerprints = tuple(
            (item.archive_path, item.size_bytes, item.sha256) for item in final_sources
        )
        initial_fingerprints = tuple(
            (item.archive_path, item.size_bytes, item.sha256) for item in source_files
        )
        if final_names != initial_names or final_fingerprints != initial_fingerprints:
            raise EvidenceArchiveError("evidence tree changed while archiving")

        archive_sha256, archive_size_bytes = _sha256_path(archive_temp)
        sidecar_bytes = _canonical_json(
            {
                "archive_name": FROZEN_EVIDENCE_ARCHIVE_NAME,
                "archive_sha256": archive_sha256,
                "archive_size_bytes": archive_size_bytes,
                "authority_sha256": authority_sha256,
                "format_version": _FORMAT_VERSION,
                "manifest_name": FROZEN_EVIDENCE_MANIFEST_NAME,
                "manifest_sha256": manifest_sha256,
                "source_file_count": len(source_files),
            }
        )
        with tempfile.NamedTemporaryFile(
            dir=output,
            prefix=f".{FROZEN_EVIDENCE_SIDECAR_NAME}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            sidecar_temp = Path(stream.name)
            stream.write(sidecar_bytes)
            stream.flush()
            os.fsync(stream.fileno())

        _publish_write_once(archive_temp, archive_path)
        archive_published = True
        try:
            _publish_write_once(sidecar_temp, sidecar_path)
        except BaseException:
            archive_path.unlink(missing_ok=True)
            archive_published = False
            raise
    finally:
        if archive_temp is not None:
            archive_temp.unlink(missing_ok=True)
        if sidecar_temp is not None:
            sidecar_temp.unlink(missing_ok=True)
        if archive_published and not sidecar_path.exists():
            archive_path.unlink(missing_ok=True)

    return FrozenEvidenceArchive(
        archive_path=archive_path,
        sidecar_path=sidecar_path,
        archive_size_bytes=archive_size_bytes,
        archive_sha256=archive_sha256,
        authority_sha256=authority_sha256,
        manifest_sha256=manifest_sha256,
        source_file_count=len(source_files),
    )


def _load_canonical_json(raw: bytes, description: str) -> dict[str, Any]:
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceArchiveError(f"invalid {description} JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise EvidenceArchiveError(f"{description} must be a JSON object")
    if raw != _canonical_json(decoded):
        raise EvidenceArchiveError(f"{description} is not canonical JSON")
    return decoded


def _required_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise EvidenceArchiveError(f"invalid non-negative integer for {field}")
    return value


def _required_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvidenceArchiveError(f"invalid SHA256 for {field}")
    return value


def validate_frozen_evidence_archive(
    archive_path: str | Path,
    sidecar_path: str | Path | None = None,
    *,
    authority_path: str | Path | None = None,
) -> FrozenEvidenceArchive:
    """Reopen a frozen archive and verify its sidecar, layout, sizes, and all digests."""

    archive = Path(archive_path)
    sidecar = (
        Path(sidecar_path)
        if sidecar_path is not None
        else archive.with_name(FROZEN_EVIDENCE_SIDECAR_NAME)
    )
    try:
        sidecar_bytes = sidecar.read_bytes()
    except OSError as exc:
        raise EvidenceArchiveError(f"cannot read archive sidecar: {sidecar}") from exc
    sidecar_data = _load_canonical_json(sidecar_bytes, "archive sidecar")

    authority, authority_sha256 = _load_authority(authority_path)
    expected_sidecar_keys = {
        "archive_name",
        "archive_sha256",
        "archive_size_bytes",
        "authority_sha256",
        "format_version",
        "manifest_name",
        "manifest_sha256",
        "source_file_count",
    }
    if set(sidecar_data) != expected_sidecar_keys:
        raise EvidenceArchiveError("archive sidecar fields do not match the frozen schema")
    if sidecar_data["archive_name"] != archive.name:
        raise EvidenceArchiveError("archive filename does not match its sidecar")
    if sidecar_data["manifest_name"] != FROZEN_EVIDENCE_MANIFEST_NAME:
        raise EvidenceArchiveError("sidecar names an unexpected internal manifest")
    if sidecar_data["format_version"] != _FORMAT_VERSION:
        raise EvidenceArchiveError("unsupported archive sidecar format version")
    if sidecar_data["authority_sha256"] != authority_sha256:
        raise EvidenceArchiveError("archive authority identity differs from frozen authority")
    expected_archive_sha = _required_sha256(
        sidecar_data["archive_sha256"], "sidecar.archive_sha256"
    )
    expected_archive_size = _required_int(
        sidecar_data["archive_size_bytes"], "sidecar.archive_size_bytes"
    )
    expected_manifest_sha = _required_sha256(
        sidecar_data["manifest_sha256"], "sidecar.manifest_sha256"
    )
    expected_file_count = _required_int(
        sidecar_data["source_file_count"], "sidecar.source_file_count"
    )

    try:
        archive_sha, archive_size = _sha256_path(archive)
    except OSError as exc:
        raise EvidenceArchiveError(f"cannot read evidence archive: {archive}") from exc
    if archive_size != expected_archive_size or archive_sha != expected_archive_sha:
        raise EvidenceArchiveError("archive size or SHA256 does not match its sidecar")

    try:
        with zipfile.ZipFile(archive, mode="r") as zip_file:
            infos = zip_file.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise EvidenceArchiveError("archive contains duplicate member names")
            if any(info.is_dir() for info in infos):
                raise EvidenceArchiveError("archive contains a directory member")
            if FROZEN_EVIDENCE_MANIFEST_NAME not in names:
                raise EvidenceArchiveError("archive is missing its internal manifest")
            manifest_bytes = zip_file.read(FROZEN_EVIDENCE_MANIFEST_NAME)
            if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha:
                raise EvidenceArchiveError("internal manifest SHA256 does not match the sidecar")
            manifest = _load_canonical_json(manifest_bytes, "internal manifest")
            if set(manifest) != {
                "archive_name",
                "authority_sha256",
                "evidence_roots",
                "files",
                "format_version",
                "inventory_sha256",
            }:
                raise EvidenceArchiveError(
                    "internal manifest fields do not match the frozen schema"
                )
            if manifest["archive_name"] != archive.name:
                raise EvidenceArchiveError("internal manifest names an unexpected archive")
            if manifest["format_version"] != _FORMAT_VERSION:
                raise EvidenceArchiveError("unsupported internal manifest format version")
            if manifest["authority_sha256"] != authority_sha256:
                raise EvidenceArchiveError(
                    "internal manifest authority differs from frozen authority"
                )
            expected_roots = [requirement.path for requirement in EVIDENCE_ROOTS]
            if manifest["evidence_roots"] != expected_roots:
                raise EvidenceArchiveError(
                    "internal manifest does not list the exact evidence roots"
                )

            raw_files = manifest["files"]
            if not isinstance(raw_files, list) or len(raw_files) != expected_file_count:
                raise EvidenceArchiveError("manifest file count does not match the sidecar")
            if expected_file_count != authority["source_file_count"]:
                raise EvidenceArchiveError("sidecar file count differs from frozen authority")
            if manifest["inventory_sha256"] != authority["inventory_sha256"]:
                raise EvidenceArchiveError(
                    "internal manifest inventory differs from frozen authority"
                )
            if _inventory_sha256(raw_files) != authority["inventory_sha256"]:
                raise EvidenceArchiveError("archive inventory differs from frozen authority")
            expected_names: list[str] = []
            manifest_sizes: dict[str, int] = {}
            verified_digests: dict[str, str] = {}
            files_by_root: dict[str, int] = {requirement.path: 0 for requirement in EVIDENCE_ROOTS}
            for index, raw_file in enumerate(raw_files):
                if not isinstance(raw_file, dict) or set(raw_file) != {
                    "path",
                    "sha256",
                    "size_bytes",
                }:
                    raise EvidenceArchiveError(f"invalid manifest file entry at index {index}")
                member_name = raw_file["path"]
                if not isinstance(member_name, str):
                    raise EvidenceArchiveError(f"invalid member path at manifest index {index}")
                pure_path = PurePosixPath(member_name)
                if (
                    pure_path.is_absolute()
                    or ".." in pure_path.parts
                    or pure_path.as_posix() != member_name
                ):
                    raise EvidenceArchiveError(
                        f"unsafe or non-canonical member path: {member_name}"
                    )
                matching_roots = [
                    requirement.path
                    for requirement in EVIDENCE_ROOTS
                    if pure_path.parts[: len(PurePosixPath(requirement.path).parts)]
                    == PurePosixPath(requirement.path).parts
                    and len(pure_path.parts) > len(PurePosixPath(requirement.path).parts)
                ]
                if len(matching_roots) != 1:
                    raise EvidenceArchiveError(
                        f"archive member is not under exactly one required root: {member_name}"
                    )
                files_by_root[matching_roots[0]] += 1
                expected_names.append(member_name)
                expected_size = _required_int(raw_file["size_bytes"], f"files[{index}].size_bytes")
                manifest_sizes[member_name] = expected_size
                expected_sha = _required_sha256(raw_file["sha256"], f"files[{index}].sha256")
                digest = hashlib.sha256()
                size = 0
                with zip_file.open(member_name, mode="r") as stream:
                    while chunk := stream.read(_BUFFER_SIZE):
                        digest.update(chunk)
                        size += len(chunk)
                if size != expected_size or digest.hexdigest() != expected_sha:
                    raise EvidenceArchiveError(f"archived evidence digest mismatch: {member_name}")
                verified_digests[member_name] = expected_sha

            if expected_names != sorted(expected_names) or len(expected_names) != len(
                set(expected_names)
            ):
                raise EvidenceArchiveError("manifest paths are not unique and sorted")
            if set(names) != {FROZEN_EVIDENCE_MANIFEST_NAME, *expected_names}:
                raise EvidenceArchiveError("archive members do not exactly match the manifest")
            for requirement in EVIDENCE_ROOTS:
                if files_by_root[requirement.path] == 0:
                    raise EvidenceArchiveError(
                        f"manifest contains no files for required root: {requirement.path}"
                    )
                for anchor in requirement.anchors:
                    anchor_path = f"{requirement.path}/{anchor}"
                    if manifest_sizes.get(anchor_path, 0) == 0:
                        raise EvidenceArchiveError(
                            f"manifest is missing non-empty required anchor: {anchor_path}"
                        )
            archived_index = {
                name: (verified_digests[name], manifest_sizes[name]) for name in expected_names
            }
            _validate_nested_manifests(archived_index, zip_file.read)
    except EvidenceArchiveError:
        raise
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise EvidenceArchiveError(f"invalid evidence archive: {exc}") from exc

    return FrozenEvidenceArchive(
        archive_path=archive,
        sidecar_path=sidecar,
        archive_size_bytes=archive_size,
        archive_sha256=archive_sha,
        authority_sha256=authority_sha256,
        manifest_sha256=expected_manifest_sha,
        source_file_count=expected_file_count,
    )


__all__ = [
    "EVIDENCE_ROOTS",
    "FROZEN_EVIDENCE_ARCHIVE_NAME",
    "FROZEN_EVIDENCE_AUTHORITY_NAME",
    "FROZEN_EVIDENCE_MANIFEST_NAME",
    "FROZEN_EVIDENCE_SIDECAR_NAME",
    "EvidenceArchiveError",
    "EvidenceRoot",
    "FrozenEvidenceArchive",
    "build_frozen_evidence_archive",
    "validate_frozen_evidence_archive",
]
