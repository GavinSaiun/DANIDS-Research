"""Strict artifact-only validation for prospective RDX-004 executions.

The validator deliberately has no dependency on the execution harness.  It
accepts only a sealed, complete run bundle, binds it to the independently
sealed RDX-005 preflight, and cross-checks the scientific provenance carried
by the run.  It never opens raw datasets, trains a model, or evaluates a new
counterfactual.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Final, cast

import yaml

from danids.adaptation.actions import (
    ACTION_ORDER,
    InterventionAction,
    InterventionRecord,
    replay_flat_positions_digest,
    replay_scoped_positions_digest,
)
from danids.adaptation.memory import SOURCE_PARTITION_SELECTOR_VERSION
from danids.config.rdx_training_evidence import (
    RDX004_BASE_SELECTOR_VERSION,
    RDX004_METHOD,
    RDX004_PROTOCOL_SHA256,
    RDX004_PROTOCOL_VERSION,
    RDX004_REPLAY_PROJECTION_VERSION,
    RDX004_SELECTOR_VERSION,
    RDX004Budget,
    RDX004RunIdentity,
    rdx004_treatment,
)
from danids.config.rdx_training_execution import (
    RDX006_EXECUTION_CONFIG_SHA256,
    RDX006_EXECUTION_CONFIG_VERSION,
    RDX006_EXECUTION_IMPLEMENTATION_VERSION,
    RDX006_EXECUTION_STATUS,
    RDX006_EXPLICIT_EXECUTE_FLAG,
    RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST,
    RDX006_REQUIRED_PREFLIGHT_VERDICT,
    RDX004ExecutionIdentity,
)
from danids.config.study4 import Study4Method
from danids.continual.supervision import row_positions_digest
from danids.evaluation.study4 import (
    PROVENANCE_FILENAME as STUDY4_PROVENANCE_FILENAME,
)
from danids.evaluation.study4 import validate_study4_run
from danids.experiments.rdx_training_evidence import (
    ALL_OUTPUT_FILES as PREFLIGHT_OUTPUT_FILES,
)
from danids.experiments.rdx_training_evidence import (
    AUDIT_IDENTITY_FILENAME,
    NESTED_SELECTION_FILENAME,
    QUERY_MATRIX_FILENAME,
    RUN_MATRIX_FILENAME,
    STARTING_STATE_FILENAME,
    TRAINING_ALLOCATION_FILENAME,
    validate_rdx004_training_evidence_preflight,
)
from danids.experiments.rdx_training_evidence import (
    CONTRACT_FILENAME as PREFLIGHT_CONTRACT_FILENAME,
)
from danids.experiments.rdx_training_evidence import (
    MANIFEST_FILENAME as PREFLIGHT_MANIFEST_FILENAME,
)
from danids.experiments.rdx_training_evidence import (
    SUMMARY_FILENAME as PREFLIGHT_SUMMARY_FILENAME,
)
from danids.policy.artifacts import CORE_SOURCE_MEMORY_SELECTION_VERSION
from danids.policy.oracle import (
    OFFLINE_ORACLE_VERSION,
    ORACLE_HORIZON,
    ORACLE_INFORMATION_POLICY,
    OracleCandidate,
    select_offline_oracle,
)
from danids.policy.rdx_training_evidence import (
    RDX004_EXECUTOR_VERSION,
    RDX004_FIXED_AUDIT_VERSION,
    RDX004_RETAIN_ALL_REPLAY_VERSION,
    RDX004_SCOPE_CLOSURE_VERSION,
    RDX004_SUPERVISION_VERSION,
    rdx_action_capacity,
)

RDX004_RUN_ARTIFACT_VERSION: Final = "rdx004-training-evidence-run-v1"
RDX004_SMOKE_PLAN_VERSION: Final = "rdx006-two-scope-capability-smoke-v1"

MANIFEST_FILENAME: Final = "artifact_manifest.json"
CONFIG_FILENAME: Final = "config.resolved.yaml"
EXECUTION_CONTRACT_FILENAME: Final = "execution_contract.json"
QUERY_EVIDENCE_FILENAME: Final = "query_evidence.json"
BRANCH_INTEGRITY_FILENAME: Final = "branch_integrity.json"
ORACLE_COUNTERFACTUALS_FILENAME: Final = "oracle_counterfactuals.json"
INTERVENTION_FILENAME: Final = "intervention_log.csv"
MEMORY_FILENAME: Final = "replay_audit_manifests.json"
REFERENCE_FILENAME: Final = "reference_state_history.json"
WINDOW_FILENAME: Final = "window_metrics.csv"
HOLDOUT_FILENAME: Final = "holdout_metrics.csv"
RESOURCE_FILENAME: Final = "resource_metrics.json"
SUMMARY_FILENAME: Final = "summary.json"

RDX004_RUN_OUTPUT_FILES: Final = frozenset(
    {
        MANIFEST_FILENAME,
        CONFIG_FILENAME,
        EXECUTION_CONTRACT_FILENAME,
        QUERY_EVIDENCE_FILENAME,
        BRANCH_INTEGRITY_FILENAME,
        ORACLE_COUNTERFACTUALS_FILENAME,
        INTERVENTION_FILENAME,
        MEMORY_FILENAME,
        REFERENCE_FILENAME,
        WINDOW_FILENAME,
        HOLDOUT_FILENAME,
        RESOURCE_FILENAME,
        SUMMARY_FILENAME,
    }
)

_JSON_FILENAMES: Final = frozenset(
    {
        MANIFEST_FILENAME,
        EXECUTION_CONTRACT_FILENAME,
        QUERY_EVIDENCE_FILENAME,
        BRANCH_INTEGRITY_FILENAME,
        ORACLE_COUNTERFACTUALS_FILENAME,
        MEMORY_FILENAME,
        REFERENCE_FILENAME,
        RESOURCE_FILENAME,
        SUMMARY_FILENAME,
    }
)
_CSV_FILENAMES: Final = frozenset({INTERVENTION_FILENAME, WINDOW_FILENAME, HOLDOUT_FILENAME})
_ACTION_VALUES: Final = tuple(action.value for action in ACTION_ORDER)
_INTERVENTION_FIELDS: Final = frozenset(item.name for item in fields(InterventionRecord))
_RDX_OPTIMIZER_BATCH_SIZE: Final = 64
_RDX_OPTIMIZER_EPOCHS: Final = 20
_SMOKE_PLAN: Final = {
    "version": RDX004_SMOKE_PLAN_VERSION,
    "scientific_admissible": False,
    "stage_1_windows": 5,
    "stage_2_windows": 2,
    "stage_2_successor_window": 2,
    "terminal_pending_query_expected": True,
}


class RDX004RunArtifactError(ValueError):
    """Raised when a prospective RDX-004 run artifact fails closed."""


@dataclass(frozen=True, slots=True)
class ValidatedRDX004Run:
    """Identity and validated evidence exposed by one sealed RDX run."""

    path: Path
    run_id: str
    expected_confirmatory_run_id: str
    budget: RDX004Budget
    rotation: tuple[str, str, str, str]
    seed: int
    smoke: bool
    paired_b100_experiment_id: str
    paired_study1_experiment_id: str
    preflight_bundle_digest: str
    artifact_bundle_digest: str
    query_event_count: int
    scope_closure_count: int
    oracle_decision_count: int
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _PreflightReference:
    root: Path
    bundle_digest: str
    run_row: dict[str, str]
    starting_row: dict[str, str]
    query_rows: tuple[dict[str, str], ...]
    nested_rows: dict[tuple[int, str, int], dict[str, str]]
    audit_rows: dict[tuple[int, str, int], dict[str, str]]
    allocation_rows: dict[tuple[int, str, int], dict[str, str]]
    paired_pipeline: dict[str, Any]
    paired_source_selection: dict[str, Any]
    paired_config_source: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _ValidatedPairedSource:
    path: Path
    experiment_id: str
    method: Study4Method
    sequence: tuple[str, ...]
    seed: int
    smoke: bool
    contract_digest: str
    source_checkpoint_sha256: str
    artifact_manifest_sha256: str
    manifest_bundle_digest: str
    pipeline: dict[str, Any]
    source_selection: dict[str, Any]
    config: dict[str, Any]


@dataclass(slots=True)
class _PreflightAuthority:
    """One process-local, fully validated view of the common RDX-005 authority."""

    root: Path
    bundle_digest: str
    contract: dict[str, Any]
    run_rows: tuple[dict[str, str], ...]
    starting_rows: tuple[dict[str, str], ...]
    query_rows: tuple[dict[str, str], ...]
    nested_rows: tuple[dict[str, str], ...]
    audit_rows: tuple[dict[str, str], ...]
    allocation_rows: tuple[dict[str, str], ...]
    paired_sources: dict[Path, _ValidatedPairedSource]


@dataclass(frozen=True, slots=True)
class _DependencyDigestSnapshot:
    """Exact file-set and content identity for one scoped batch validation."""

    roots: tuple[Path, ...]
    explicit_files: tuple[Path, ...]
    digests: dict[str, str]


@dataclass(frozen=True, slots=True)
class _HistoricalMemoryEvidence:
    replay_scoped_positions: tuple[Mapping[str, object], ...]
    replay_rows: int
    replay_digest: str
    audit_scoped_positions: tuple[Mapping[str, object], ...]
    audit_rows: int
    audit_digest: str
    scope_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class _ExpectedWindow:
    stage: int
    domain: str
    window_id: int
    prediction_index: int
    row_start: int
    row_stop: int
    oracle_evaluated: bool
    successor_row_start: int | None
    successor_row_stop: int | None


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise RDX004RunArtifactError("YAML mapping keys must be scalar") from exc
        if duplicate:
            raise RDX004RunArtifactError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise RDX004RunArtifactError(f"cannot hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _tree_digests(root: Path, *, manifest_name: str) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != manifest_name
    }


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise RDX004RunArtifactError(f"{name} must be a string-keyed mapping")
    return {str(key): item for key, item in value.items()}


def _records(value: object, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise RDX004RunArtifactError(f"{name} must be a list")
    return [_mapping(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _require_keys(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    missing = fields.difference(value)
    if missing:
        raise RDX004RunArtifactError(f"{name} is missing fields: {sorted(missing)}")


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise RDX004RunArtifactError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise RDX004RunArtifactError(f"{name} must be an integer") from exc
    raise RDX004RunArtifactError(f"{name} must be an integer")


def _as_bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    raise RDX004RunArtifactError(f"{name} must be a boolean")


def _sha256(value: object, name: str, *, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    if not isinstance(value, str) or len(value) != 64:
        raise RDX004RunArtifactError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RDX004RunArtifactError(f"{name} must be a SHA-256 digest") from exc
    return value.lower()


def _positions(value: object, name: str) -> tuple[int, ...]:
    raw: object = value
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RDX004RunArtifactError(f"{name} is not canonical position JSON") from exc
    if not isinstance(raw, list):
        raise RDX004RunArtifactError(f"{name} must be a position list")
    result = tuple(_as_int(item, name) for item in raw)
    if result != tuple(sorted(set(result))):
        raise RDX004RunArtifactError(f"{name} must be chronological and distinct")
    return result


def _load_json(path: Path, *, canonical: bool = True) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RDX004RunArtifactError(f"cannot load JSON artifact {path}: {exc}") from exc
    result = _mapping(value, str(path))
    if canonical and raw != _json_text(result).encode("utf-8"):
        raise RDX004RunArtifactError(f"JSON artifact is not canonically serialized: {path}")
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RDX004RunArtifactError(f"cannot load YAML artifact {path}: {exc}") from exc
    return _mapping(value, str(path))


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RDX004RunArtifactError(f"cannot load CSV artifact {path}: {exc}") from exc
    if "\r" in text or (text and not text.endswith("\n")):
        raise RDX004RunArtifactError(f"CSV artifact is not LF-canonical: {path}")
    if text == "\n":
        return (), []
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise RDX004RunArtifactError(f"CSV header is absent or duplicated: {path}")
    rows: list[dict[str, str]] = []
    try:
        for raw in reader:
            if None in raw or any(value is None for value in raw.values()):
                raise RDX004RunArtifactError(f"CSV row differs from its header: {path}")
            rows.append({str(key): cast(str, value) for key, value in raw.items()})
    except csv.Error as exc:
        raise RDX004RunArtifactError(f"invalid CSV artifact {path}: {exc}") from exc
    columns = tuple(reader.fieldnames)
    rendered = io.StringIO(newline="")
    writer = csv.DictWriter(rendered, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    if rendered.getvalue() != text:
        raise RDX004RunArtifactError(f"CSV artifact is not canonically serialized: {path}")
    return columns, rows


def _seal_manifest_identity(
    root: Path,
    *,
    expected_files: frozenset[str],
    manifest_name: str,
    expected_version: str | None,
) -> tuple[dict[str, Any], str]:
    entries = list(root.iterdir())
    if (
        any(not entry.is_file() for entry in entries)
        or {entry.name for entry in entries} != expected_files
    ):
        raise RDX004RunArtifactError(f"artifact file set differs from exact contract under {root}")
    manifest = _load_json(root / manifest_name)
    files = _tree_digests(root, manifest_name=manifest_name)
    if (
        set(manifest) != {"version", "files", "bundle_digest"}
        or (expected_version is not None and manifest.get("version") != expected_version)
        or manifest.get("files") != files
        or manifest.get("bundle_digest") != _digest(files)
        or set(files) != expected_files.difference({manifest_name})
    ):
        raise RDX004RunArtifactError("artifact manifest differs from its exact bundle")
    return manifest, _sha256(manifest.get("bundle_digest"), "artifact bundle digest")


def _row_key(row: Mapping[str, object]) -> tuple[int, str, int]:
    return (
        _as_int(row.get("stage"), "event stage"),
        str(row.get("current_domain", "")),
        _as_int(row.get("query_ordinal"), "query ordinal"),
    )


def _read_preflight_csv(root: Path, name: str) -> list[dict[str, str]]:
    _columns, rows = _read_csv(root / name)
    return rows


def _load_preflight_authority(preflight_dir: str | Path) -> _PreflightAuthority:
    root = Path(preflight_dir).resolve()
    if not root.is_dir():
        raise RDX004RunArtifactError(f"RDX-005 preflight directory is absent: {root}")
    try:
        validate_rdx004_training_evidence_preflight(root)
    except Exception as exc:
        raise RDX004RunArtifactError(
            "RDX-005 preflight failed independent immutable-source validation"
        ) from exc
    manifest, bundle_digest = _seal_manifest_identity(
        root,
        expected_files=PREFLIGHT_OUTPUT_FILES,
        manifest_name=PREFLIGHT_MANIFEST_FILENAME,
        expected_version=None,
    )
    if bundle_digest != RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST:
        raise RDX004RunArtifactError("RDX-005 preflight bundle identity differs")
    if manifest.get("version") != "rdx004-training-evidence-preflight-v1":
        raise RDX004RunArtifactError("RDX-005 preflight artifact version differs")
    summary = _load_json(root / PREFLIGHT_SUMMARY_FILENAME)
    contract = _load_json(root / PREFLIGHT_CONTRACT_FILENAME)
    verdict = str(summary.get("verdict", ""))
    if (
        summary.get("status") != "complete"
        or verdict not in {"GO", "RDX-004 PRE-FLIGHT GO"}
        or summary.get("execution_authorized") is not False
        or summary.get("b400_b1600_runs_executed") is not False
        or contract.get("version") != "rdx004-training-evidence-preflight-v1"
    ):
        raise RDX004RunArtifactError("RDX-005 preflight is not the complete GO artifact")
    protocol = _mapping(contract.get("protocol"), "preflight protocol")
    if (
        protocol.get("version") != RDX004_PROTOCOL_VERSION
        or protocol.get("sha256") != RDX004_PROTOCOL_SHA256
    ):
        raise RDX004RunArtifactError("RDX-005 preflight protocol identity differs")

    return _PreflightAuthority(
        root=root,
        bundle_digest=bundle_digest,
        contract=contract,
        run_rows=tuple(_read_preflight_csv(root, RUN_MATRIX_FILENAME)),
        starting_rows=tuple(_read_preflight_csv(root, STARTING_STATE_FILENAME)),
        query_rows=tuple(_read_preflight_csv(root, QUERY_MATRIX_FILENAME)),
        nested_rows=tuple(_read_preflight_csv(root, NESTED_SELECTION_FILENAME)),
        audit_rows=tuple(_read_preflight_csv(root, AUDIT_IDENTITY_FILENAME)),
        allocation_rows=tuple(_read_preflight_csv(root, TRAINING_ALLOCATION_FILENAME)),
        paired_sources={},
    )


def _load_paired_source(
    authority: _PreflightAuthority,
    paired_run: Path,
) -> _ValidatedPairedSource:
    cached = authority.paired_sources.get(paired_run)
    if cached is not None:
        return cached
    try:
        validated = validate_study4_run(paired_run, allow_smoke=False)
        source_manifest = _load_json(paired_run / MANIFEST_FILENAME)
        source_provenance = _load_json(paired_run / STUDY4_PROVENANCE_FILENAME)
        source_memory = _load_json(paired_run / MEMORY_FILENAME)
        source_config = _load_yaml(paired_run / CONFIG_FILENAME)
        manifest_sha256 = _file_sha256(paired_run / MANIFEST_FILENAME)
    except Exception as exc:
        raise RDX004RunArtifactError("paired canonical B100 run failed validation") from exc
    paired_pipeline_fields = (
        "feature_contract_version",
        "feature_columns",
        "dataset_fingerprints",
        "manifest_partition_ranges",
        "window_size",
        "boundary_mode",
        "health_artifact_identity",
        "health_model_sha256",
        "health_feature_contract_digest",
    )
    if any(field not in source_provenance for field in paired_pipeline_fields):
        raise RDX004RunArtifactError("paired canonical B100 pipeline provenance is incomplete")
    source = _ValidatedPairedSource(
        path=validated.path,
        experiment_id=validated.experiment_id,
        method=validated.method,
        sequence=tuple(validated.sequence),
        seed=validated.seed,
        smoke=validated.smoke,
        contract_digest=validated.contract_digest,
        source_checkpoint_sha256=validated.source_checkpoint_sha256,
        artifact_manifest_sha256=manifest_sha256,
        manifest_bundle_digest=_sha256(
            source_manifest.get("bundle_digest"), "paired B100 bundle digest"
        ),
        pipeline={field: source_provenance[field] for field in paired_pipeline_fields},
        source_selection=_mapping(
            source_memory.get("source_selection"), "paired B100 source selection"
        ),
        config=source_config,
    )
    authority.paired_sources[paired_run] = source
    return source


def _preflight_reference_from_authority(
    authority: _PreflightAuthority,
    identity: RDX004ExecutionIdentity,
    *,
    declared_bundle_digest: str,
) -> _PreflightReference:
    if declared_bundle_digest != authority.bundle_digest:
        raise RDX004RunArtifactError("RDX-005 preflight bundle identity differs")

    confirmatory_id = identity.expected_confirmatory_run_id
    run_matches = [row for row in authority.run_rows if row.get("run_id") == confirmatory_id]
    starting_matches = [
        row for row in authority.starting_rows if row.get("run_id") == confirmatory_id
    ]
    if len(run_matches) != 1 or len(starting_matches) != 1:
        raise RDX004RunArtifactError("execution identity has no unique preflight pairing")
    run_row = run_matches[0]
    starting_row = starting_matches[0]
    if (
        run_row.get("status") != "PREFLIGHT_READY"
        or run_row.get("budget") != identity.budget.value
        or run_row.get("sequence") != identity.sequence_token
        or _as_int(run_row.get("seed"), "preflight seed") != identity.seed
        or run_row.get("method") != RDX004_METHOD
        or run_row.get("paired_b100_experiment_id") != identity.paired_b100_experiment_id
        or starting_row.get("status") != "STARTING_STATE_PAIRED"
        or starting_row.get("paired_b100_experiment_id") != identity.paired_b100_experiment_id
        or starting_row.get("study1_experiment_id") != identity.paired_study1_experiment_id
        or _as_bool(
            starting_row.get("source_detector_retrained"),
            "preflight source_detector_retrained",
        )
    ):
        raise RDX004RunArtifactError("execution identity differs from its preflight row")

    paired_run = Path(str(run_row.get("paired_b100_run_path", ""))).resolve()
    validated_source = _load_paired_source(authority, paired_run)
    if (
        validated_source.path != paired_run
        or validated_source.experiment_id != identity.paired_b100_experiment_id
        or validated_source.method is not Study4Method.OFFLINE_ORACLE
        or tuple(validated_source.sequence) != identity.rotation
        or validated_source.seed != identity.seed
        or validated_source.smoke
        or validated_source.contract_digest != run_row.get("paired_b100_scientific_contract_digest")
        or validated_source.source_checkpoint_sha256 != run_row.get("source_checkpoint_sha256")
        or validated_source.artifact_manifest_sha256
        != run_row.get("paired_b100_artifact_manifest_sha256")
        or validated_source.manifest_bundle_digest
        != run_row.get("paired_b100_manifest_bundle_digest")
    ):
        raise RDX004RunArtifactError("paired canonical B100 identity differs from preflight")

    query_rows = tuple(
        sorted(
            (row for row in authority.query_rows if row.get("run_id") == confirmatory_id),
            key=_row_key,
        )
    )
    if len(query_rows) != 12 or len({_row_key(row) for row in query_rows}) != 12:
        raise RDX004RunArtifactError("preflight run does not contain 12 unique query events")
    paired_id = identity.paired_b100_experiment_id
    nested_rows = {
        _row_key(row): row
        for row in authority.nested_rows
        if row.get("paired_b100_experiment_id") == paired_id
    }
    audit_rows = {
        _row_key(row): row
        for row in authority.audit_rows
        if row.get("paired_b100_experiment_id") == paired_id
    }
    allocation_rows = {
        _row_key(row): row
        for row in authority.allocation_rows
        if row.get("paired_b100_experiment_id") == paired_id
        and row.get("budget") == identity.budget.value
    }
    keys = {_row_key(row) for row in query_rows}
    if set(nested_rows) != keys or set(audit_rows) != keys or set(allocation_rows) != keys:
        raise RDX004RunArtifactError("preflight query/allocation coverage is inconsistent")
    return _PreflightReference(
        root=authority.root,
        bundle_digest=authority.bundle_digest,
        run_row=run_row,
        starting_row=starting_row,
        query_rows=query_rows,
        nested_rows=nested_rows,
        audit_rows=audit_rows,
        allocation_rows=allocation_rows,
        paired_pipeline=validated_source.pipeline,
        paired_source_selection=validated_source.source_selection,
        paired_config_source=validated_source.config,
    )


def _load_preflight_reference(
    preflight_dir: str | Path,
    identity: RDX004ExecutionIdentity,
    *,
    declared_bundle_digest: str,
) -> _PreflightReference:
    authority = _load_preflight_authority(preflight_dir)
    return _preflight_reference_from_authority(
        authority,
        identity,
        declared_bundle_digest=declared_bundle_digest,
    )


def _required_dependency_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RDX004RunArtifactError(f"{name} dependency path is absent")
    return Path(value).resolve()


def _authority_dependency_scope(
    authority: _PreflightAuthority,
    run_dirs: Sequence[str | Path],
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    inputs = _mapping(authority.contract.get("inputs"), "preflight inputs")
    roots = {
        authority.root,
        _required_dependency_path(inputs.get("study4_evaluation_root"), "Study-4 evaluation"),
        _required_dependency_path(inputs.get("manifest_root"), "split-manifest root"),
        *(Path(path).resolve() for path in run_dirs),
    }
    sources = _records(
        authority.contract.get("canonical_b100_sources"),
        "canonical B100 sources",
    )
    for source in sources:
        roots.add(_required_dependency_path(source.get("source_run_path"), "canonical B100 run"))
        roots.add(_required_dependency_path(source.get("study1_run_path"), "Study-1 run"))
    sentinels = _mapping(
        authority.contract.get("source_immutability_sentinels"),
        "preflight source immutability sentinels",
    )
    explicit_files = {_required_dependency_path(path, "source sentinel") for path in sentinels}
    return (
        tuple(sorted(roots, key=lambda path: str(path).casefold())),
        tuple(sorted(explicit_files, key=lambda path: str(path).casefold())),
    )


def _dependency_digests(roots: Sequence[Path], explicit_files: Sequence[Path]) -> dict[str, str]:
    entries: dict[str, str] = {}
    for path in explicit_files:
        if not path.is_file():
            raise RDX004RunArtifactError(f"batch validation dependency file is absent: {path}")
        entries[str(path)] = f"FILE:{_file_sha256(path)}"
    for root in roots:
        if not root.is_dir():
            raise RDX004RunArtifactError(f"batch validation dependency root is absent: {root}")
        entries[str(root)] = "DIRECTORY"
        try:
            for path in root.rglob("*"):
                if path.is_file():
                    entries[str(path)] = f"FILE:{_file_sha256(path)}"
                elif path.is_dir():
                    entries[str(path)] = "DIRECTORY"
                else:
                    raise RDX004RunArtifactError(
                        f"unsupported batch validation dependency entry: {path}"
                    )
        except OSError as exc:
            raise RDX004RunArtifactError(
                f"cannot enumerate batch validation dependency root {root}: {exc}"
            ) from exc
    return dict(sorted(entries.items(), key=lambda item: item[0].casefold()))


def _capture_dependency_snapshot(
    authority: _PreflightAuthority,
    run_dirs: Sequence[str | Path],
) -> _DependencyDigestSnapshot:
    roots, explicit_files = _authority_dependency_scope(authority, run_dirs)
    return _DependencyDigestSnapshot(
        roots=roots,
        explicit_files=explicit_files,
        digests=_dependency_digests(roots, explicit_files),
    )


def _assert_dependency_snapshot_unchanged(snapshot: _DependencyDigestSnapshot) -> None:
    current = _dependency_digests(snapshot.roots, snapshot.explicit_files)
    if current != snapshot.digests:
        raise RDX004RunArtifactError("batch validation dependency changed during scoped validation")


def _identity_from_contract(
    contract: Mapping[str, Any], *, expected_smoke: bool
) -> RDX004ExecutionIdentity:
    _require_keys(
        contract,
        {
            "version",
            "identity",
            "method",
            "protocol",
            "execution_config",
            "preflight",
            "starting_state",
            "information_boundary",
            "treatment",
            "selector",
        },
        "execution contract",
    )
    if contract.get("version") != RDX004_RUN_ARTIFACT_VERSION:
        raise RDX004RunArtifactError("RDX run artifact version differs")
    raw_identity = _mapping(contract.get("identity"), "execution identity")
    _require_keys(
        raw_identity,
        {
            "budget",
            "rotation",
            "seed",
            "run_id",
            "expected_confirmatory_run_id",
            "output_namespace",
            "smoke",
            "confirmatory_eligible",
            "paired_b100_experiment_id",
            "paired_study1_experiment_id",
        },
        "execution identity",
    )
    try:
        budget = RDX004Budget(str(raw_identity.get("budget", "")))
        raw_rotation = raw_identity.get("rotation")
        if not isinstance(raw_rotation, list) or len(raw_rotation) != 4:
            raise ValueError("rotation")
        rotation = cast(tuple[str, str, str, str], tuple(str(item) for item in raw_rotation))
        scientific = RDX004RunIdentity(
            budget=budget,
            rotation=rotation,
            seed=_as_int(raw_identity.get("seed"), "execution seed"),
        )
        smoke = _as_bool(raw_identity.get("smoke"), "execution smoke marker")
        identity = RDX004ExecutionIdentity(scientific_identity=scientific, smoke=smoke)
    except (TypeError, ValueError) as exc:
        raise RDX004RunArtifactError("execution identity is outside the frozen roster") from exc
    if smoke != expected_smoke:
        kind = "smoke" if smoke else "confirmatory"
        raise RDX004RunArtifactError(f"unexpected {kind} RDX artifact")
    if raw_identity != identity.to_dict():
        raise RDX004RunArtifactError("persisted execution identity differs from recomputation")
    if contract.get("method") != RDX004_METHOD:
        raise RDX004RunArtifactError("RDX execution method must be OFFLINE_ORACLE")
    return identity


def _pipeline_online_ranges(
    pipeline: Mapping[str, Any], identity: RDX004ExecutionIdentity
) -> dict[str, tuple[int, int]]:
    raw_ranges = _mapping(pipeline.get("manifest_partition_ranges"), "manifest partition ranges")
    if set(raw_ranges) != set(identity.rotation):
        raise RDX004RunArtifactError("pipeline partition-range domains differ")
    expected_partitions = {
        "initial_train",
        "validation",
        "online_stream",
        "permanent_holdout",
    }
    result: dict[str, tuple[int, int]] = {}
    for domain in identity.rotation:
        partitions = _mapping(raw_ranges.get(domain), f"{domain} partition ranges")
        if set(partitions) != expected_partitions:
            raise RDX004RunArtifactError(f"{domain} partition-range fields differ")
        for name, raw in partitions.items():
            if raw is None:
                continue
            if not isinstance(raw, list) or len(raw) != 2:
                raise RDX004RunArtifactError(f"{domain} {name} range is invalid")
            start = _as_int(raw[0], f"{domain} {name} start")
            stop = _as_int(raw[1], f"{domain} {name} stop")
            if start < 0 or stop <= start:
                raise RDX004RunArtifactError(f"{domain} {name} range is invalid")
        online = partitions.get("online_stream")
        if domain == identity.rotation[0] and online is None:
            continue
        if not isinstance(online, list) or len(online) != 2:
            raise RDX004RunArtifactError(f"{domain} online-stream range is absent")
        result[domain] = (
            _as_int(online[0], f"{domain} online-stream start"),
            _as_int(online[1], f"{domain} online-stream stop"),
        )
    return result


def _expected_window_plan(
    identity: RDX004ExecutionIdentity,
    pipeline: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> tuple[_ExpectedWindow, ...]:
    window_size = _as_int(pipeline.get("window_size"), "window size")
    if window_size <= 0:
        raise RDX004RunArtifactError("window size must be positive")
    ranges = _pipeline_online_ranges(pipeline, identity)
    stages = (1, 2) if identity.smoke else (1, 2, 3)
    result: list[_ExpectedWindow] = []
    for stage in stages:
        stage_events = [
            event for event in events if _as_int(event.get("stage"), "event stage") == stage
        ]
        offsets = {
            _as_int(event.get("prediction_index"), "event prediction")
            - _as_int(event.get("selection_window_id"), "event selection window")
            for event in stage_events
        }
        if len(offsets) != 1:
            raise RDX004RunArtifactError(
                "query evidence does not define one stage prediction offset"
            )
        stage_offset = next(iter(offsets))
        domain = identity.rotation[stage]
        online_start, online_stop = ranges[domain]
        full_count = (online_stop - online_start + window_size - 1) // window_size
        process_count = (5 if stage == 1 else 2) if identity.smoke else full_count
        if full_count < process_count or (identity.smoke and stage == 2 and full_count < 3):
            raise RDX004RunArtifactError("smoke plan exceeds the canonical online stream")
        for window_id in range(process_count):
            row_start = online_start + window_id * window_size
            row_stop = min(row_start + window_size, online_stop)
            oracle_evaluated = (
                stage == 2 and window_id == 1 if identity.smoke else window_id + 1 < full_count
            )
            successor_start = (
                online_start + (window_id + 1) * window_size if oracle_evaluated else None
            )
            result.append(
                _ExpectedWindow(
                    stage=stage,
                    domain=domain,
                    window_id=window_id,
                    prediction_index=stage_offset + window_id,
                    row_start=row_start,
                    row_stop=row_stop,
                    oracle_evaluated=oracle_evaluated,
                    successor_row_start=successor_start,
                    successor_row_stop=(
                        min(successor_start + window_size, online_stop)
                        if successor_start is not None
                        else None
                    ),
                )
            )
    return tuple(result)


def _validate_execution_contract(
    contract: Mapping[str, Any],
    identity: RDX004ExecutionIdentity,
    preflight: _PreflightReference,
) -> None:
    expected_top = {
        "version",
        "identity",
        "method",
        "protocol",
        "execution_config",
        "preflight",
        "starting_state",
        "information_boundary",
        "treatment",
        "executor",
        "selector",
        "pipeline",
        "code",
    }
    if set(contract) != expected_top:
        raise RDX004RunArtifactError("execution contract fields differ")
    protocol = _mapping(contract.get("protocol"), "execution protocol")
    if (
        protocol.get("version") != RDX004_PROTOCOL_VERSION
        or protocol.get("sha256") != RDX004_PROTOCOL_SHA256
    ):
        raise RDX004RunArtifactError("execution protocol identity differs")
    execution = _mapping(contract.get("execution_config"), "execution config identity")
    expected_execution = {
        "contract_version": RDX006_EXECUTION_CONFIG_VERSION,
        "contract_sha256": RDX006_EXECUTION_CONFIG_SHA256,
        "implementation_version": RDX006_EXECUTION_IMPLEMENTATION_VERSION,
        "implementation_status": RDX006_EXECUTION_STATUS,
        "execution_authorized": True,
        "explicit_execute_flag": RDX006_EXPLICIT_EXECUTE_FLAG,
        "explicit_execute": True,
    }
    if execution != expected_execution:
        raise RDX004RunArtifactError("RDX-006 execution authorization identity differs")
    preflight_record = _mapping(contract.get("preflight"), "execution preflight identity")
    _require_keys(
        preflight_record,
        {"path", "bundle_digest", "verdict"},
        "execution preflight identity",
    )
    if (
        Path(str(preflight_record.get("path", ""))).resolve() != preflight.root
        or preflight_record.get("bundle_digest") != preflight.bundle_digest
        or preflight_record.get("verdict") != RDX006_REQUIRED_PREFLIGHT_VERDICT
    ):
        raise RDX004RunArtifactError("execution does not bind the exact GO preflight")

    starting = _mapping(contract.get("starting_state"), "execution starting state")
    _require_keys(
        starting,
        {
            "paired_b100_experiment_id",
            "paired_study1_experiment_id",
            "source_domain",
            "paired_b100_run_path",
            "source_checkpoint_sha256",
            "initial_model_digest",
            "preprocessor_digest",
            "threshold_digest",
            "source_split_manifest_set_digest",
            "study1_run_path",
            "source_detector_retrained",
            "paired_scientific_contract_digest",
        },
        "execution starting state",
    )
    expected_starting = {
        "paired_b100_experiment_id": identity.paired_b100_experiment_id,
        "paired_study1_experiment_id": identity.paired_study1_experiment_id,
        "source_domain": preflight.starting_row.get("source_domain"),
        "paired_b100_run_path": preflight.run_row.get("paired_b100_run_path"),
        "source_checkpoint_sha256": preflight.starting_row.get("source_checkpoint_sha256"),
        "initial_model_digest": preflight.starting_row.get("initial_model_digest"),
        "preprocessor_digest": preflight.starting_row.get("preprocessor_digest"),
        "threshold_digest": preflight.starting_row.get("threshold_digest"),
        "source_split_manifest_set_digest": preflight.starting_row.get(
            "source_split_manifest_set_digest"
        ),
        "study1_run_path": preflight.starting_row.get("study1_run_path"),
        "source_detector_retrained": False,
    }
    if set(starting) != {*expected_starting, "paired_scientific_contract_digest"} or any(
        starting.get(key) != value for key, value in expected_starting.items()
    ):
        raise RDX004RunArtifactError("starting state differs from the paired preflight state")
    _sha256(
        starting.get("paired_scientific_contract_digest"),
        "paired scientific contract",
    )
    for name in (
        "source_checkpoint_sha256",
        "initial_model_digest",
        "preprocessor_digest",
        "threshold_digest",
        "source_split_manifest_set_digest",
    ):
        _sha256(starting.get(name), f"starting state {name}")

    boundary = _mapping(contract.get("information_boundary"), "information boundary")
    expected_boundary = {
        "evaluator_truth_oracle_visible": True,
        "oracle_non_deployable_upper_bound": True,
        "permanent_holdout_used_for_query": False,
        "permanent_holdout_used_for_training": False,
        "permanent_holdout_used_for_action_choice": False,
    }
    if boundary != expected_boundary:
        raise RDX004RunArtifactError("RDX information boundary differs from the freeze")

    treatment = rdx004_treatment(identity.budget)
    treatment_record = _mapping(contract.get("treatment"), "execution treatment")
    expected_treatment = asdict(treatment)
    expected_treatment["budget"] = identity.budget.value
    if treatment_record != expected_treatment:
        raise RDX004RunArtifactError("execution treatment differs from RDX-004")
    executor = _mapping(contract.get("executor"), "RDX executor")
    executor_payload = {
        "version": RDX004_EXECUTOR_VERSION,
        "budget": identity.budget.value,
        "target_row_limit": treatment.maximum_training_rows,
        "requested_label_limit": treatment.label_budget_per_scope,
        "action_capacities": [
            rdx_action_capacity(treatment, action).to_dict() for action in InterventionAction
        ],
    }
    if executor != {**executor_payload, "manifest_digest": _digest(executor_payload)}:
        raise RDX004RunArtifactError("RDX executor capacity evidence differs")
    selector = _mapping(contract.get("selector"), "execution selector")
    if selector != {
        "version": RDX004_SELECTOR_VERSION,
        "inherited_version": RDX004_BASE_SELECTOR_VERSION,
        "fixed_b100_audit": True,
    }:
        raise RDX004RunArtifactError("execution selector identity differs")

    pipeline = _mapping(contract.get("pipeline"), "execution pipeline")
    _require_keys(
        pipeline,
        {
            "feature_contract_version",
            "feature_columns",
            "dataset_fingerprints",
            "dataset_fingerprints_digest",
            "manifest_partition_ranges",
            "split_manifest_sha256",
            "split_manifest_set_digest",
            "window_size",
            "boundary_mode",
            "health_artifact_identity",
            "health_model_sha256",
            "health_feature_contract_digest",
        },
        "execution pipeline",
    )
    fingerprints = _mapping(pipeline.get("dataset_fingerprints"), "dataset fingerprints")
    manifest_hashes = _mapping(pipeline.get("split_manifest_sha256"), "split manifest hashes")
    if (
        set(fingerprints) != set(identity.rotation)
        or any(not _sha256(value, "dataset fingerprint") for value in fingerprints.values())
        or pipeline.get("dataset_fingerprints_digest")
        != preflight.starting_row.get("dataset_fingerprints_digest")
        or _digest(fingerprints) != pipeline.get("dataset_fingerprints_digest")
        or set(manifest_hashes)
        != {
            f"stage-{stage:02d}-{domain}.json"
            for stage, domain in enumerate(identity.rotation, start=1)
        }
        or any(not _sha256(value, "split manifest hash") for value in manifest_hashes.values())
        or pipeline.get("split_manifest_set_digest")
        != preflight.run_row.get("source_split_manifest_set_digest")
        or _digest(manifest_hashes) != pipeline.get("split_manifest_set_digest")
        or _as_int(pipeline.get("window_size"), "window size") <= 0
    ):
        raise RDX004RunArtifactError("execution pipeline identity differs")
    if any(pipeline.get(field) != value for field, value in preflight.paired_pipeline.items()):
        raise RDX004RunArtifactError(
            "execution pipeline differs from the paired canonical B100 provenance"
        )
    _pipeline_online_ranges(pipeline, identity)
    _sha256(pipeline.get("health_model_sha256"), "health model")
    _sha256(
        pipeline.get("health_feature_contract_digest"),
        "health feature contract",
    )
    code = _mapping(contract.get("code"), "execution code identity")
    _require_keys(
        code,
        {"commit", "working_tree_dirty", "python", "platform", "torch", "device"},
        "execution code identity",
    )
    _as_bool(code.get("working_tree_dirty"), "working-tree dirty marker")


def _validate_resolved_config(
    config: Mapping[str, Any],
    identity: RDX004ExecutionIdentity,
    contract: Mapping[str, Any],
    preflight: _PreflightReference,
) -> None:
    if set(config) != {
        "execution",
        "paired_study4_config",
        "paired_config_source",
        "smoke",
    }:
        raise RDX004RunArtifactError("resolved execution config fields differ")
    if (
        _mapping(config.get("execution"), "resolved execution contract") != contract
        or _as_bool(config.get("smoke"), "resolved smoke marker") != identity.smoke
    ):
        raise RDX004RunArtifactError("resolved execution contract identity differs")
    paired = _mapping(config.get("paired_study4_config"), "paired Study-4 config")
    source = _mapping(config.get("paired_config_source"), "paired config source")
    if source != preflight.paired_config_source:
        raise RDX004RunArtifactError("paired config source differs from canonical B100")
    for name, value in (("paired Study-4 config", paired), ("paired config source", source)):
        if (
            value.get("experiment_id") != identity.paired_b100_experiment_id
            or value.get("method") != RDX004_METHOD
            or value.get("sequence") != list(identity.rotation)
            or _as_int(value.get("seed"), f"{name} seed") != identity.seed
        ):
            raise RDX004RunArtifactError(f"{name} identity differs")
    scientific = {
        "core_contract": source.get("core_contract"),
        "health_signal_contract": source.get("health_signal_contract"),
        "always_adapt": source.get("always_adapt"),
    }
    starting = _mapping(contract.get("starting_state"), "execution starting state")
    core_contract = _mapping(source.get("core_contract"), "paired Core contract")
    adaptation = _mapping(core_contract.get("adaptation"), "paired adaptation contract")
    if (
        paired.get("core_contract") != scientific["core_contract"]
        or paired.get("health_signal_contract") != scientific["health_signal_contract"]
        or paired.get("always_adapt") != scientific["always_adapt"]
        or starting.get("paired_scientific_contract_digest") != _digest(scientific)
        or _as_int(adaptation.get("batch_size"), "adaptation batch size")
        != _RDX_OPTIMIZER_BATCH_SIZE
        or _as_int(adaptation.get("epochs"), "adaptation epochs") != _RDX_OPTIMIZER_EPOCHS
    ):
        raise RDX004RunArtifactError("paired scientific contract identity differs")


def _expected_event(
    preflight: _PreflightReference,
    query_row: Mapping[str, str],
    budget: RDX004Budget,
) -> dict[str, object]:
    key = _row_key(query_row)
    nested = preflight.nested_rows[key]
    audit = preflight.audit_rows[key]
    allocation = preflight.allocation_rows[key]
    prefix = "q400" if budget is RDX004Budget.B400 else "q1600"
    b100_query = _positions(nested["q100_positions_json"], "preflight B100 query")
    audit_positions = _positions(audit["fixed_audit_positions_json"], "preflight audit positions")
    b100_training = tuple(
        position for position in b100_query if position not in set(audit_positions)
    )
    nested_selection = {
        "selector_version": nested["selector_version"]
        if "selector_version" in nested
        else query_row["selector_version"],
        "inherited_selector_version": query_row["inherited_selector_version"],
        "seed": _as_int(query_row["seed"], "preflight selector seed"),
        "opaque_scope_token": query_row["opaque_scope_token"],
        "window_id": _as_int(query_row["selection_window_id"], "preflight window"),
        "query_ordinal": key[2],
        "current_row_positions_digest": query_row["candidate_population_digest"],
        "candidate_count": _as_int(query_row["candidate_population_size"], "candidate size"),
        "b100_positions": list(b100_query),
        "b100_positions_digest": nested["q100_digest"],
        "b400_positions": list(_positions(nested["q400_positions_json"], "preflight B400")),
        "b400_positions_digest": nested["q400_digest"],
        "b1600_positions": list(_positions(nested["q1600_positions_json"], "preflight B1600")),
        "b1600_positions_digest": nested["q1600_digest"],
    }
    fixed_audit_plan = {
        "dataset_fingerprint": query_row["dataset_fingerprint"],
        "opaque_scope_token": query_row["opaque_scope_token"],
        "selection_window_id": _as_int(query_row["selection_window_id"], "preflight window"),
        "query_ordinal": key[2],
        "prediction_index": _as_int(query_row["prediction_index"], "preflight prediction"),
        "release_prediction_index": _as_int(
            query_row["release_prediction_index"], "preflight release"
        ),
        "b100_query_positions": list(b100_query),
        "b100_query_positions_digest": nested["q100_digest"],
        "b100_training_positions": list(b100_training),
        "b100_training_positions_digest": row_positions_digest(b100_training),
        "audit_positions": list(audit_positions),
        "audit_positions_digest": audit["fixed_audit_digest"],
    }
    return {
        "stage": key[0],
        "current_domain": key[1],
        "opaque_scope_token": query_row["opaque_scope_token"],
        "query_ordinal": key[2],
        "selection_window_id": _as_int(
            query_row["selection_window_id"], "preflight selection window"
        ),
        "prediction_index": _as_int(query_row["prediction_index"], "preflight prediction index"),
        "release_prediction_index": _as_int(
            query_row["release_prediction_index"], "preflight release index"
        ),
        "candidate_population_size": _as_int(
            query_row["candidate_population_size"], "preflight candidate size"
        ),
        "candidate_population_digest": query_row["candidate_population_digest"],
        "query_positions": _positions(
            nested[f"{prefix}_positions_json"], "preflight query positions"
        ),
        "query_positions_digest": query_row["query_set_digest"],
        "audit_positions": _positions(
            audit["fixed_audit_positions_json"], "preflight audit positions"
        ),
        "audit_positions_digest": query_row["audit_set_digest"],
        "training_positions": _positions(
            allocation["training_positions_json"], "preflight training positions"
        ),
        "training_positions_digest": query_row["training_set_digest"],
        "b100_query_positions_digest": query_row["b100_reference_query_digest"],
        "selector_version": query_row["selector_version"],
        "inherited_selector_version": query_row["inherited_selector_version"],
        "nested_selection": nested_selection,
        "fixed_audit_plan": fixed_audit_plan,
    }


def _validate_query_evidence(
    payload: Mapping[str, Any],
    identity: RDX004ExecutionIdentity,
    preflight: _PreflightReference,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_payload_fields = {
        "version",
        "run_id",
        "budget",
        "smoke",
        "canonical_event_count",
        "exercised_event_count",
        "released_event_count",
        "smoke_plan",
        "events",
        "closures",
    }
    if set(payload) != expected_payload_fields:
        raise RDX004RunArtifactError("query evidence fields differ")
    events = _records(payload.get("events"), "query evidence events")
    closures = _records(payload.get("closures"), "query evidence closures")
    if (
        payload.get("version") != RDX004_RUN_ARTIFACT_VERSION
        or payload.get("run_id") != identity.run_id
        or payload.get("budget") != identity.budget.value
        or _as_bool(payload.get("smoke"), "query evidence smoke") != identity.smoke
        or _as_int(payload.get("canonical_event_count"), "canonical event count") != 12
        or _as_int(payload.get("exercised_event_count"), "exercised event count") != len(events)
        or payload.get("smoke_plan") != (_SMOKE_PLAN if identity.smoke else None)
    ):
        raise RDX004RunArtifactError("query evidence identity/count differs")
    expected_event_count = 6 if identity.smoke else 12
    expected_released_count = 5 if identity.smoke else 12
    if (
        len(events) != expected_event_count
        or _as_int(payload.get("released_event_count"), "released event count")
        != expected_released_count
    ):
        raise RDX004RunArtifactError("run does not exercise the frozen full/smoke event plan")
    if not identity.smoke and len(events) != 12:
        raise RDX004RunArtifactError("confirmatory run must exercise all 12 query events")

    references = {
        _row_key(row): _expected_event(preflight, row, identity.budget)
        for row in preflight.query_rows
    }
    required_event_fields = {
        *next(iter(references.values())),
        "prediction_before_query",
        "available_on_query_prediction",
        "delay_prediction_count",
        "permanent_holdout_excluded",
        "released",
        "actual_release_prediction_index",
        "allocation",
    }
    seen_keys: set[tuple[int, str, int]] = set()
    queried_by_scope: dict[tuple[int, str], set[int]] = {}
    audit_by_scope: dict[tuple[int, str], set[int]] = {}
    training_by_scope: dict[tuple[int, str], set[int]] = {}
    previous_prediction = -1
    for event in events:
        if set(event) != required_event_fields:
            raise RDX004RunArtifactError("query event fields differ")
        key = _row_key(event)
        expected = references.get(key)
        if expected is None or key in seen_keys:
            raise RDX004RunArtifactError("query event is duplicated or absent from preflight")
        seen_keys.add(key)
        for field, expected_value in expected.items():
            if field in {"nested_selection", "fixed_audit_plan"}:
                actual_value: object = _mapping(event.get(field), field)
            elif field.endswith("_positions"):
                actual_value = _positions(event.get(field), field)
            elif field in {
                "stage",
                "query_ordinal",
                "selection_window_id",
                "prediction_index",
                "release_prediction_index",
                "candidate_population_size",
            }:
                actual_value = _as_int(event.get(field), field)
            else:
                actual_value = event.get(field)
            if actual_value != expected_value:
                raise RDX004RunArtifactError(
                    f"query event {key} differs from preflight field {field}"
                )
        query = _positions(event.get("query_positions"), "query positions")
        audit = _positions(event.get("audit_positions"), "audit positions")
        training = _positions(event.get("training_positions"), "training positions")
        prediction_index = _as_int(event.get("prediction_index"), "prediction index")
        release_index = _as_int(event.get("release_prediction_index"), "release prediction index")
        if (
            len(query) != identity.budget.selected_per_query
            or len(audit) != identity.budget.audit_per_query
            or len(training) != identity.budget.training_per_query
            or set(training).intersection(audit)
            or set(training).union(audit) != set(query)
            or _as_bool(event.get("prediction_before_query"), "prediction-before-query") is not True
            or _as_bool(
                event.get("available_on_query_prediction"),
                "same-prediction availability",
            )
            is not False
            or _as_int(event.get("delay_prediction_count"), "delay count") != 1
            or release_index != prediction_index + 1
            or _as_bool(event.get("permanent_holdout_excluded"), "holdout exclusion") is not True
        ):
            raise RDX004RunArtifactError(
                "query event violates candidate, allocation, D1, or holdout invariants"
            )
        if (
            row_positions_digest(query) != event.get("query_positions_digest")
            or row_positions_digest(audit) != event.get("audit_positions_digest")
            or row_positions_digest(training) != event.get("training_positions_digest")
        ):
            raise RDX004RunArtifactError("query event position digest differs")
        released = _as_bool(event.get("released"), "query released marker")
        allocation_raw = event.get("allocation")
        actual_release = event.get("actual_release_prediction_index")
        if released:
            if _as_int(
                actual_release, "actual release prediction"
            ) != release_index or not isinstance(allocation_raw, Mapping):
                raise RDX004RunArtifactError("released query lacks its exact D1 allocation")
            allocation = _mapping(allocation_raw, "query release allocation")
            if (
                allocation.get("version") != RDX004_FIXED_AUDIT_VERSION
                or allocation.get("budget") != identity.budget.value
                or _as_int(allocation.get("query_ordinal"), "allocation ordinal") != key[2]
                or _as_int(allocation.get("query_window"), "allocation query window")
                != prediction_index
                or _as_int(allocation.get("release_window"), "allocation release window")
                != release_index
                or _positions(allocation.get("queried_positions"), "allocation queried") != query
                or _positions(allocation.get("training_positions"), "allocation training")
                != training
                or _positions(allocation.get("audit_positions"), "allocation audit") != audit
            ):
                raise RDX004RunArtifactError("released allocation differs from query evidence")
        elif (
            not identity.smoke
            or key != (2, identity.rotation[2], 1)
            or actual_release is not None
            or allocation_raw is not None
        ):
            raise RDX004RunArtifactError("only the frozen smoke terminal query may remain pending")
        scope_key = (key[0], key[1])
        prior = queried_by_scope.setdefault(scope_key, set())
        if prior.intersection(query):
            raise RDX004RunArtifactError("query event repeats a physical row in its scope")
        prior.update(query)
        audit_by_scope.setdefault(scope_key, set()).update(audit)
        training_by_scope.setdefault(scope_key, set()).update(training)
        if prediction_index <= previous_prediction:
            raise RDX004RunArtifactError("query events are not globally chronological")
        previous_prediction = prediction_index
    expected_keys = list(references)[:expected_event_count]
    if [_row_key(event) for event in events] != expected_keys:
        raise RDX004RunArtifactError("query events differ from the frozen chronological plan")
    if not identity.smoke and seen_keys != set(references):
        raise RDX004RunArtifactError("confirmatory query events differ from full preflight")
    for scope_key in queried_by_scope:
        if audit_by_scope[scope_key].intersection(training_by_scope[scope_key]):
            raise RDX004RunArtifactError("scope audit evidence enters training")

    closure_by_scope: dict[tuple[int, str], dict[str, Any]] = {}
    for wrapper in closures:
        if set(wrapper) != {"stage", "current_domain", "closure", "activation"}:
            raise RDX004RunArtifactError("query closure fields differ")
        scope_key = (
            _as_int(wrapper.get("stage"), "closure stage"),
            str(wrapper.get("current_domain", "")),
        )
        if scope_key in closure_by_scope or scope_key not in queried_by_scope:
            raise RDX004RunArtifactError("query closure is duplicated or unexercised")
        closure = _mapping(wrapper.get("closure"), "scope closure")
        activation = _mapping(wrapper.get("activation"), "historical activation")
        closure_by_scope[scope_key] = wrapper
        scope_events = [event for event in events if _row_key(event)[:2] == scope_key]
        queried = tuple(
            sorted(
                position
                for event in scope_events
                for position in _positions(event["query_positions"], "query positions")
            )
        )
        training = tuple(
            sorted(
                position
                for event in scope_events
                for position in _positions(event["training_positions"], "training positions")
            )
        )
        audit = tuple(
            sorted(
                position
                for event in scope_events
                for position in _positions(event["audit_positions"], "audit positions")
            )
        )
        if (
            closure.get("version") != RDX004_SCOPE_CLOSURE_VERSION
            or closure.get("budget") != identity.budget.value
            or closure.get("opaque_scope_token") != scope_events[0].get("opaque_scope_token")
            or _as_int(closure.get("successful_query_count"), "closure query count")
            != len(scope_events)
            or _as_int(closure.get("released_label_count"), "closure queried count") != len(queried)
            or closure.get("released_row_positions_digest") != row_positions_digest(queried)
            or _as_int(closure.get("activation_boundary_index"), "closure activation index")
            < max(
                _as_int(event["release_prediction_index"], "event release")
                for event in scope_events
            )
            or closure.get("closure_digest")
            != _digest({key: value for key, value in closure.items() if key != "closure_digest"})
            or _as_int(activation.get("activation_prediction_index"), "activation prediction")
            != _as_int(closure.get("activation_boundary_index"), "closure activation")
            or _as_bool(activation.get("permanent_holdout_used"), "activation holdout use")
            is not False
        ):
            raise RDX004RunArtifactError("query closure differs from exercised releases")
    expected_closed_scopes = {
        scope_key
        for scope_key, scope_events in _events_by_scope(events).items()
        if all(_as_bool(event.get("released"), "query released") for event in scope_events)
    }
    if set(closure_by_scope) != expected_closed_scopes:
        raise RDX004RunArtifactError("activated scope closures differ from exercised scopes")
    if not identity.smoke and (
        len(closures) != 3
        or any(
            _as_int(
                _mapping(item.get("closure"), "closure").get("successful_query_count"),
                "closure query count",
            )
            != 4
            for item in closures
        )
    ):
        raise RDX004RunArtifactError("confirmatory run lacks three complete closures")
    if identity.smoke and len(closures) != 1:
        raise RDX004RunArtifactError("smoke run must activate exactly its first scope")
    return events, closures


def _events_by_scope(
    events: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], list[Mapping[str, Any]]]:
    result: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for event in events:
        result.setdefault(_row_key(event)[:2], []).append(event)
    for values in result.values():
        values.sort(key=lambda item: _row_key(item)[2])
    return result


def _validate_memory_evidence(
    payload: Mapping[str, Any],
    identity: RDX004ExecutionIdentity,
    preflight: _PreflightReference,
    events: Sequence[Mapping[str, Any]],
    closures: Sequence[Mapping[str, Any]],
) -> dict[int, _HistoricalMemoryEvidence]:
    if set(payload) != {
        "version",
        "run_id",
        "budget",
        "smoke",
        "source_selection",
        "scopes",
        "final_memory",
    }:
        raise RDX004RunArtifactError("replay/audit manifest fields differ")
    if (
        payload.get("version") != RDX004_RUN_ARTIFACT_VERSION
        or payload.get("run_id") != identity.run_id
        or payload.get("budget") != identity.budget.value
        or _as_bool(payload.get("smoke"), "replay/audit smoke") != identity.smoke
    ):
        raise RDX004RunArtifactError("replay/audit manifest identity differs")
    source_selection = _mapping(payload.get("source_selection"), "source selection")
    if source_selection != preflight.paired_source_selection:
        raise RDX004RunArtifactError("source replay/audit selection differs from canonical B100")
    expected_source_fields = {
        "source_scope",
        "selector_version",
        "initial_train_start",
        "initial_train_stop",
        "initial_train_labels_digest",
        "source_dataset_fingerprint",
        "benign_support",
        "attack_support",
        "replay_seed",
        "audit_seed",
        "replay_positions",
        "replay_binary_labels",
        "audit_positions",
        "audit_binary_labels",
        "audit_true_positives_at_learned_state",
        "audit_learned_recall",
        "detector_model_digest",
        "detector_threshold_digest",
        "preprocessor_digest",
        "evidence_digest",
    }
    if set(source_selection) != expected_source_fields:
        raise RDX004RunArtifactError("source-memory selection fields differ")
    source_payload = {
        key: value for key, value in source_selection.items() if key != "evidence_digest"
    }
    source_replay_positions = _positions(
        source_selection.get("replay_positions"), "source replay positions"
    )
    source_audit_positions = _positions(
        source_selection.get("audit_positions"), "source audit positions"
    )
    replay_labels = source_selection.get("replay_binary_labels")
    audit_labels = source_selection.get("audit_binary_labels")
    if not isinstance(replay_labels, list) or not isinstance(audit_labels, list):
        raise RDX004RunArtifactError("source-memory binary labels must be lists")
    replay_binary = tuple(_as_int(value, "source replay label") for value in replay_labels)
    audit_binary = tuple(_as_int(value, "source audit label") for value in audit_labels)
    start = _as_int(source_selection.get("initial_train_start"), "source training start")
    stop = _as_int(source_selection.get("initial_train_stop"), "source training stop")
    benign = _as_int(source_selection.get("benign_support"), "source benign support")
    attacks = _as_int(source_selection.get("attack_support"), "source attack support")
    audit_true_positives = _as_int(
        source_selection.get("audit_true_positives_at_learned_state"),
        "source audit true positives",
    )
    learned_recall = source_selection.get("audit_learned_recall")
    if (
        source_selection.get("evidence_digest") != _digest(source_payload)
        or source_selection.get("source_scope") != identity.rotation[0]
        or source_selection.get("selector_version") != CORE_SOURCE_MEMORY_SELECTION_VERSION
        or start < 0
        or stop <= start
        or benign < 0
        or attacks < 0
        or benign + attacks != stop - start
        or len(source_replay_positions) != 400
        or len(source_audit_positions) != 100
        or len(replay_binary) != len(source_replay_positions)
        or len(audit_binary) != len(source_audit_positions)
        or any(value not in {0, 1} for value in (*replay_binary, *audit_binary))
        or not set(source_replay_positions).isdisjoint(source_audit_positions)
        or not all(
            start <= value < stop for value in (*source_replay_positions, *source_audit_positions)
        )
        or not 0 <= audit_true_positives <= sum(audit_binary)
        or (
            learned_recall
            != (None if sum(audit_binary) == 0 else audit_true_positives / sum(audit_binary))
        )
        or _as_int(source_selection.get("replay_seed"), "source replay seed")
        != identity.seed + 10_000
        or _as_int(source_selection.get("audit_seed"), "source audit seed")
        != identity.seed + 20_000
    ):
        raise RDX004RunArtifactError("source replay/audit selection differs")
    for field in (
        "initial_train_labels_digest",
        "source_dataset_fingerprint",
        "detector_model_digest",
        "detector_threshold_digest",
        "preprocessor_digest",
    ):
        _sha256(source_selection.get(field), f"source-memory {field}")

    scopes = _records(payload.get("scopes"), "memory scopes")
    by_scope = _events_by_scope(events)
    closure_by_scope = {
        (_as_int(item.get("stage"), "closure stage"), str(item.get("current_domain", ""))): item
        for item in closures
    }
    if len(scopes) != len(by_scope):
        raise RDX004RunArtifactError("replay/audit scope count differs")
    seen: set[tuple[int, str]] = set()
    activated: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    activated_by_stage: dict[int, str] = {}
    treatment = rdx004_treatment(identity.budget)
    for scope in scopes:
        if set(scope) != {
            "stage",
            "current_domain",
            "opaque_scope_token",
            "closed",
            "supervision",
            "allocation",
            "closure",
            "activation",
        }:
            raise RDX004RunArtifactError("memory scope fields differ")
        key = (_as_int(scope.get("stage"), "scope stage"), str(scope.get("current_domain", "")))
        scope_events = by_scope.get(key)
        if scope_events is None or key in seen:
            raise RDX004RunArtifactError("memory scope is duplicate or unexercised")
        seen.add(key)
        token = str(scope.get("opaque_scope_token", ""))
        if token != scope_events[0].get("opaque_scope_token"):
            raise RDX004RunArtifactError("memory scope token differs")
        released_events = [
            item for item in scope_events if _as_bool(item.get("released"), "released")
        ]
        queried = tuple(
            sorted(
                position
                for item in released_events
                for position in _positions(item["query_positions"], "query")
            )
        )
        training = tuple(
            sorted(
                position
                for item in released_events
                for position in _positions(item["training_positions"], "training")
            )
        )
        audit = tuple(
            sorted(
                position
                for item in released_events
                for position in _positions(item["audit_positions"], "audit")
            )
        )
        pending_count = len(scope_events) - len(released_events)

        supervision = _mapping(scope.get("supervision"), "scope supervision")
        supervision_payload = {
            field: value for field, value in supervision.items() if field != "manifest_digest"
        }
        selections = _records(supervision.get("selections"), "supervision selections")
        provenance = _records(supervision.get("queries"), "supervision queries")
        if (
            supervision.get("version") != RDX004_SUPERVISION_VERSION
            or supervision.get("budget") != identity.budget.value
            or _as_int(supervision.get("seed"), "supervision seed") != identity.seed
            or supervision.get("opaque_scope_token") != token
            or _as_int(supervision.get("label_budget"), "label budget")
            != treatment.label_budget_per_scope
            or _as_int(supervision.get("selected_per_query"), "selected/query")
            != treatment.selected_per_query
            or _as_int(supervision.get("delay_windows"), "delay") != 1
            or _as_int(supervision.get("maximum_queries"), "maximum queries") != 4
            or _as_int(supervision.get("query_count"), "query count") != len(scope_events)
            or _as_int(supervision.get("pending_count"), "pending count") != pending_count
            or _as_int(supervision.get("used_budget"), "used budget")
            != len(scope_events) * treatment.selected_per_query
            or _as_int(supervision.get("available_count"), "available count") != len(queried)
            or _as_int(supervision.get("remaining_budget"), "remaining budget")
            != treatment.label_budget_per_scope - len(scope_events) * treatment.selected_per_query
            or len(selections) != len(scope_events)
            or len(provenance) != len(scope_events)
            or supervision.get("manifest_digest") != _digest(supervision_payload)
        ):
            raise RDX004RunArtifactError("supervision manifest differs from D1 evidence")
        for event, selection, query in zip(scope_events, selections, provenance, strict=True):
            positions = _positions(event["query_positions"], "event query")
            if (
                selection != _mapping(event.get("nested_selection"), "event nested selection")
                or query.get("scope_token") != token
                or _as_int(query.get("query_window"), "provenance query")
                != _as_int(event["prediction_index"], "prediction")
                or _as_int(query.get("release_window"), "provenance release")
                != _as_int(event["release_prediction_index"], "release")
                or _as_int(query.get("count"), "provenance count") != len(positions)
                or _positions(query.get("row_positions"), "provenance positions") != positions
                or query.get("row_positions_digest") != event.get("query_positions_digest")
            ):
                raise RDX004RunArtifactError("supervision selection/provenance differs")

        allocation = _mapping(scope.get("allocation"), "scope allocation")
        allocation_payload = {
            field: value for field, value in allocation.items() if field != "manifest_digest"
        }
        plans = _records(allocation.get("fixed_audit_plans"), "fixed audit plans")
        releases = _records(allocation.get("releases"), "release allocations")
        is_closed = _as_bool(scope.get("closed"), "scope closed")
        if (
            allocation.get("version") != RDX004_FIXED_AUDIT_VERSION
            or allocation.get("budget") != identity.budget.value
            or allocation.get("scope_id") != token
            or _as_int(allocation.get("release_count"), "release count") != len(released_events)
            or _as_int(allocation.get("used_label_budget"), "allocation used budget")
            != len(queried)
            or _as_int(allocation.get("remaining_label_budget"), "allocation remaining")
            != treatment.label_budget_per_scope - len(queried)
            or _positions(allocation.get("queried_positions"), "allocation queried") != queried
            or _positions(allocation.get("training_positions"), "allocation training") != training
            or _positions(allocation.get("audit_positions"), "allocation audit") != audit
            or allocation.get("queried_positions_digest")
            != (row_positions_digest(queried) if queried else None)
            or allocation.get("training_positions_digest")
            != (row_positions_digest(training) if training else None)
            or allocation.get("audit_positions_digest")
            != (row_positions_digest(audit) if audit else None)
            or len(plans) != 4
            or releases
            != [_mapping(item.get("allocation"), "event allocation") for item in released_events]
            or _as_bool(allocation.get("historical"), "historical marker") != is_closed
            or allocation.get("manifest_digest") != _digest(allocation_payload)
        ):
            raise RDX004RunArtifactError("allocation manifest differs from released evidence")
        for ordinal, plan in enumerate(plans):
            expected = _expected_event(
                preflight,
                next(
                    row
                    for row in preflight.query_rows
                    if _row_key(row) == (key[0], key[1], ordinal)
                ),
                identity.budget,
            )["fixed_audit_plan"]
            if plan != expected:
                raise RDX004RunArtifactError("fixed audit plan differs from preflight")

        wrapper = closure_by_scope.get(key)
        if is_closed:
            if (
                wrapper is None
                or scope.get("closure") != wrapper.get("closure")
                or scope.get("activation") != wrapper.get("activation")
            ):
                raise RDX004RunArtifactError("closed scope differs from query closure")
            capabilities = _mapping(
                allocation.get("historical_capabilities"), "historical capabilities"
            )
            projection = _mapping(capabilities.get("projection"), "historical projection")
            output = _positions(projection.get("output_positions"), "projection output")
            projected = len(training) > 400
            expected_selector = (
                RDX004_REPLAY_PROJECTION_VERSION if projected else RDX004_RETAIN_ALL_REPLAY_VERSION
            )
            closure = _mapping(scope.get("closure"), "scope closure")
            if (
                capabilities.get("budget") != identity.budget.value
                or capabilities.get("scope_id") != token
                or capabilities.get("closure_digest") != closure.get("closure_digest")
                or capabilities.get("audit_selection") != RDX004_FIXED_AUDIT_VERSION
                or _positions(capabilities.get("audit_positions"), "historical audit") != audit
                or capabilities.get("audit_positions_digest") != row_positions_digest(audit)
                or projection.get("selector_version") != expected_selector
                or projection.get("domain_id") != token
                or _as_int(projection.get("experiment_seed"), "projection seed identity")
                != identity.seed
                or _as_int(projection.get("projection_seed"), "projection seed")
                != identity.seed + 10_000
                or _as_int(projection.get("capacity"), "projection capacity") != 400
                or _as_bool(projection.get("projected"), "projection marker") != projected
                or _positions(projection.get("input_positions"), "projection input") != training
                or projection.get("input_positions_digest") != row_positions_digest(training)
                or projection.get("output_positions_digest") != row_positions_digest(output)
                or len(output) != min(400, len(training))
                or not set(output).issubset(training)
                or set(output).intersection(audit)
                or (not projected and output != training)
                or capabilities.get("manifest_digest")
                != _digest(
                    {
                        field: value
                        for field, value in capabilities.items()
                        if field != "manifest_digest"
                    }
                )
            ):
                raise RDX004RunArtifactError("historical replay/audit capability differs")
            activated[token] = (output, audit)
            activated_by_stage[key[0]] = token
        elif (
            wrapper is not None
            or scope.get("closure") is not None
            or scope.get("activation") is not None
            or allocation.get("historical_capabilities") is not None
        ):
            raise RDX004RunArtifactError("open smoke scope contains historical capability")
    if seen != set(by_scope):
        raise RDX004RunArtifactError("replay/audit scopes differ from query scopes")

    final_memory = _mapping(payload.get("final_memory"), "final memory")
    expected_memory_fields = {
        "replay_capacity_per_domain",
        "audit_capacity_per_domain",
        "replay_size",
        "audit_size",
        "total_bytes",
        "replay_domains",
        "audit_domains",
    }
    if set(final_memory) != expected_memory_fields:
        raise RDX004RunArtifactError("final memory fields differ")
    replay_domains = _records(final_memory.get("replay_domains"), "final replay domains")
    audit_domains = _records(final_memory.get("audit_domains"), "final audit domains")
    if (
        _as_int(final_memory.get("replay_capacity_per_domain"), "replay capacity") != 400
        or _as_int(final_memory.get("audit_capacity_per_domain"), "audit capacity") != 100
    ):
        raise RDX004RunArtifactError("final memory capacity differs")
    replay_by_domain = {str(item.get("domain_id")): item for item in replay_domains}
    audit_by_domain = {str(item.get("domain_id")): item for item in audit_domains}
    if len(replay_by_domain) != len(replay_domains) or len(audit_by_domain) != len(audit_domains):
        raise RDX004RunArtifactError("final memory repeats a domain")
    source_scope = str(source_selection.get("source_scope", ""))
    expected_domains = {source_scope, *activated}
    if set(replay_by_domain) != expected_domains or set(audit_by_domain) != expected_domains:
        raise RDX004RunArtifactError("final memory contains an absent or extra scope")
    for capability, records in (("replay", replay_domains), ("audit", audit_domains)):
        partition_field = "partition_kind" if capability == "replay" else "source_partition_kind"
        for item in records:
            positions = _positions(item.get("row_positions"), f"{capability} positions")
            expected_partition = (
                "initial_train" if item.get("domain_id") == source_scope else "online_stream"
            )
            if (
                _as_int(item.get("size"), f"{capability} size") != len(positions)
                or item.get("row_positions_digest") != row_positions_digest(positions)
                or item.get(partition_field) != expected_partition
            ):
                raise RDX004RunArtifactError("final memory domain evidence differs")
    source_replay = _positions(source_selection.get("replay_positions"), "source replay")
    source_audit = _positions(source_selection.get("audit_positions"), "source audit")
    if (
        not source_scope
        or source_scope != identity.rotation[0]
        or source_replay
        != _positions(replay_by_domain[source_scope].get("row_positions"), "final source replay")
        or source_audit
        != _positions(audit_by_domain[source_scope].get("row_positions"), "final source audit")
        or replay_by_domain[source_scope].get("selection") != SOURCE_PARTITION_SELECTOR_VERSION
        or audit_by_domain[source_scope].get("selection") != SOURCE_PARTITION_SELECTOR_VERSION
        or set(source_replay).intersection(source_audit)
    ):
        raise RDX004RunArtifactError("final source memory differs from source selection evidence")
    for token, (replay, audit) in activated.items():
        if (
            token not in replay_by_domain
            or token not in audit_by_domain
            or _positions(replay_by_domain[token].get("row_positions"), "activated replay")
            != replay
            or _positions(audit_by_domain[token].get("row_positions"), "activated audit") != audit
        ):
            raise RDX004RunArtifactError("final memory omits an activated scope")
    if (
        _as_int(final_memory.get("replay_size"), "final replay size")
        != sum(_as_int(item.get("size"), "replay size") for item in replay_domains)
        or _as_int(final_memory.get("audit_size"), "final audit size")
        != sum(_as_int(item.get("size"), "audit size") for item in audit_domains)
        or _as_int(final_memory.get("total_bytes"), "final memory bytes") <= 0
    ):
        raise RDX004RunArtifactError("final memory row accounting differs")
    _walk_information_boundary(payload, context="replay_audit_manifests")
    evidence_by_stage: dict[int, _HistoricalMemoryEvidence] = {}
    for stage in (1, 2, 3):
        scope_ids = {
            source_scope,
            *(token for prior_stage, token in activated_by_stage.items() if prior_stage < stage),
        }
        ordered_scope_ids = sorted(scope_ids)
        replay_scoped = tuple(
            {
                "domain_id": scope_id,
                "positions": list(
                    _positions(
                        replay_by_domain[scope_id].get("row_positions"),
                        "stage replay positions",
                    )
                ),
            }
            for scope_id in ordered_scope_ids
        )
        audit_scoped = tuple(
            {
                "domain_id": scope_id,
                "positions": list(
                    _positions(
                        audit_by_domain[scope_id].get("row_positions"),
                        "stage audit positions",
                    )
                ),
            }
            for scope_id in ordered_scope_ids
        )
        evidence_by_stage[stage] = _HistoricalMemoryEvidence(
            replay_scoped_positions=replay_scoped,
            replay_rows=sum(len(item["positions"]) for item in replay_scoped),
            replay_digest=_digest(replay_scoped),
            audit_scoped_positions=audit_scoped,
            audit_rows=sum(len(item["positions"]) for item in audit_scoped),
            audit_digest=_digest(audit_scoped),
            scope_ids=frozenset(scope_ids),
        )
    return evidence_by_stage


def _record_positions(record: Mapping[str, Any], field: str) -> tuple[int, ...]:
    return _positions(record.get(field, []), field)


def _position_multiset(value: object, name: str) -> tuple[int, ...]:
    raw = value
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RDX004RunArtifactError(f"{name} is not position JSON") from exc
    if not isinstance(raw, list):
        raise RDX004RunArtifactError(f"{name} must be a position list")
    result = tuple(_as_int(item, name) for item in raw)
    if result != tuple(sorted(result)):
        raise RDX004RunArtifactError(f"{name} must be chronological")
    return result


def _expected_optimizer_steps(action: InterventionAction, target_rows: int) -> int:
    if action in {InterventionAction.HEAD_UPDATE, InterventionAction.FULL_FINE_TUNE}:
        rows_per_step = _RDX_OPTIMIZER_BATCH_SIZE
    elif action is InterventionAction.REPLAY_UPDATE:
        rows_per_step = _RDX_OPTIMIZER_BATCH_SIZE // 2
    else:
        return 0
    return _RDX_OPTIMIZER_EPOCHS * ((target_rows + rows_per_step - 1) // rows_per_step)


def _validate_candidate_intervention(
    candidate: Mapping[str, Any],
    *,
    action: InterventionAction,
    identity: RDX004ExecutionIdentity,
    stage: int,
    domain: str,
    window_id: int,
    available_target: tuple[int, ...],
    current_audit: set[int],
    expected_replay_available: int,
    expected_replay_scope_ids: set[str],
    expected_replay_scoped_positions: Sequence[Mapping[str, object]] | None = None,
) -> None:
    feasible = _as_bool(candidate.get("feasible"), "candidate feasibility")
    succeeded = _as_bool(candidate.get("execution_succeeded"), "candidate execution success")
    target_count = _as_int(candidate.get("target_rows"), "candidate target rows")
    replay_count = _as_int(candidate.get("replay_rows"), "candidate replay rows")
    steps = _as_int(candidate.get("optimizer_steps"), "candidate optimizer steps")
    intervention_raw = candidate.get("intervention_record")
    if intervention_raw is None:
        if (
            feasible
            or target_count
            or replay_count
            or steps
            or candidate.get("target_positions_digest_consumed") is not None
            or candidate.get("historical_replay_positions_digest_consumed") is not None
            or (
                identity.smoke
                and action
                in {
                    InterventionAction.HEAD_UPDATE,
                    InterventionAction.FULL_FINE_TUNE,
                    InterventionAction.REPLAY_UPDATE,
                }
            )
        ):
            raise RDX004RunArtifactError(
                f"Oracle {action.value} lacks its required intervention attempt"
            )
        return
    if not feasible:
        raise RDX004RunArtifactError("infeasible Oracle candidate has intervention evidence")

    intervention = _mapping(intervention_raw, "candidate intervention record")
    if set(intervention) != _INTERVENTION_FIELDS:
        raise RDX004RunArtifactError("candidate intervention fields differ")
    target = _record_positions(intervention, "target_row_positions")
    calibration = _record_positions(intervention, "calibration_row_positions")
    replay = _position_multiset(intervention.get("replay_row_positions"), "replay positions")
    try:
        scoped_value = json.loads(str(intervention.get("replay_scoped_row_positions", "[]")))
    except json.JSONDecodeError as exc:
        raise RDX004RunArtifactError("scoped replay provenance is invalid") from exc
    scoped = _records(scoped_value, "scoped replay provenance")
    scoped_positions = tuple(
        sorted(
            position
            for item in scoped
            for position in _positions(item.get("row_positions"), "scoped replay positions")
        )
    )
    expected_target = () if action is InterventionAction.NO_OP else available_target
    expected_calibration = available_target if action is InterventionAction.RECALIBRATE else ()
    expected_replay_rows = (
        expected_replay_available if action is InterventionAction.REPLAY_UPDATE else 0
    )
    expected_steps = _expected_optimizer_steps(action, len(expected_target)) if succeeded else 0
    accepted = _as_bool(intervention.get("accepted"), "intervention accepted")
    rolled_back = _as_bool(intervention.get("rolled_back"), "intervention rollback")
    rejection_reason = str(intervention.get("rejection_reason", ""))
    audit_admissible = _as_bool(candidate.get("audit_admissible"), "candidate audit")
    expected_succeeded = not rejection_reason.startswith("action failed safely:")
    if (
        identity.smoke
        and action
        in {
            InterventionAction.HEAD_UPDATE,
            InterventionAction.FULL_FINE_TUNE,
            InterventionAction.REPLAY_UPDATE,
        }
        and not succeeded
    ):
        raise RDX004RunArtifactError(
            f"smoke {action.value} did not exercise its optimizer capability"
        )
    if (
        intervention.get("sequence") != identity.sequence_token
        or _as_int(intervention.get("seed"), "intervention seed") != identity.seed
        or _as_int(intervention.get("domain_stage"), "intervention stage") != stage
        or intervention.get("current_domain") != domain
        or _as_int(intervention.get("window_id"), "intervention window") != window_id
        or intervention.get("action_attempted") != action.value
        or _as_int(intervention.get("action_rank"), "intervention rank") != action.rank
        or _as_int(intervention.get("action_cost_proxy"), "intervention cost") != action.rank
        or _as_int(intervention.get("labels_requested"), "intervention labels requested") != 0
        or _as_int(intervention.get("labels_available"), "intervention labels available")
        != len(expected_target)
        or _as_int(intervention.get("target_rows"), "intervention target rows")
        != len(expected_target)
        or target != expected_target
        or target_count != len(expected_target)
        or _as_int(intervention.get("calibration_rows"), "intervention calibration rows")
        != len(expected_calibration)
        or calibration != expected_calibration
        or _as_int(intervention.get("replay_rows"), "intervention replay rows")
        != expected_replay_rows
        or replay_count != expected_replay_rows
        or replay != scoped_positions
        or len(replay) != expected_replay_rows
        or _as_int(intervention.get("optimizer_steps"), "intervention optimizer steps")
        != expected_steps
        or steps != expected_steps
        or intervention.get("target_row_positions_digest")
        != (row_positions_digest(target) if target else "")
        or intervention.get("calibration_row_positions_digest")
        != (row_positions_digest(calibration) if calibration else "")
        or intervention.get("replay_row_positions_digest")
        != (replay_flat_positions_digest(replay) if replay else "")
        or intervention.get("replay_scoped_row_positions_digest")
        != (replay_scoped_positions_digest(scoped) if scoped else "")
        or candidate.get("target_positions_digest_consumed")
        != intervention.get("target_row_positions_digest")
        or candidate.get("historical_replay_positions_digest_consumed")
        != intervention.get("replay_scoped_row_positions_digest")
        or set(target).intersection(current_audit)
        or set(calibration).intersection(current_audit)
        or intervention.get("preprocessor_digest_before")
        != intervention.get("preprocessor_digest_after")
        or candidate.get("incoming_model_digest") != intervention.get("model_digest_before")
        or candidate.get("deployed_model_digest") != intervention.get("model_digest_after")
        or candidate.get("deployed_threshold_digest") != intervention.get("threshold_digest_after")
        or succeeded != expected_succeeded
        or audit_admissible != (succeeded and accepted)
        or candidate.get("audit_state") != intervention.get("audit_result")
        or rolled_back == accepted
        or (
            not accepted
            and (
                intervention.get("model_digest_after") != intervention.get("model_digest_before")
                or intervention.get("threshold_digest_after")
                != intervention.get("threshold_digest_before")
            )
        )
        or _as_bool(candidate.get("confirmed_success"), "candidate confirmed success")
        != (
            feasible
            and succeeded
            and audit_admissible
            and candidate.get("successor_evaluator_state") == "SAFE"
        )
    ):
        raise RDX004RunArtifactError(
            f"Oracle {action.value} intervention semantics differ from the frozen executor"
        )
    if (
        not 0
        <= _as_int(intervention.get("remaining_label_budget"), "intervention remaining budget")
        <= rdx004_treatment(identity.budget).label_budget_per_scope
    ):
        raise RDX004RunArtifactError("intervention remaining budget lies outside treatment")
    if float(intervention.get("wall_clock_update_seconds", -1.0)) < 0.0:
        raise RDX004RunArtifactError("intervention wall-clock duration is invalid")
    if action is InterventionAction.REPLAY_UPDATE:
        scope_ids = {str(item.get("scope_id", "")) for item in scoped}
        available_rows: list[dict[str, object]]
        if expected_replay_scoped_positions is None:
            available_rows = [
                {
                    "domain_id": str(item.get("scope_id", "")),
                    "positions": list(_positions(item.get("row_positions"), "scoped replay")),
                }
                for item in scoped
            ]
        else:
            available_rows = [
                {key: value for key, value in item.items()}
                for item in expected_replay_scoped_positions
            ]
            expected_scoped = [
                {
                    "scope_id": str(item["domain_id"]),
                    "row_positions": list(_positions(item["positions"], "expected scoped replay")),
                    "row_positions_digest": row_positions_digest(
                        _positions(item["positions"], "expected scoped replay")
                    ),
                }
                for item in available_rows
            ]
            if scoped != expected_scoped:
                raise RDX004RunArtifactError(
                    "A4 did not consume the exact historical replay memory"
                )
        if scope_ids != expected_replay_scope_ids or candidate.get(
            "historical_replay_positions_digest_available"
        ) != _digest(available_rows):
            raise RDX004RunArtifactError("A4 did not consume the exact historical replay memory")
    elif scoped or replay:
        raise RDX004RunArtifactError(f"{action.value} consumed historical replay")


def _validate_candidate_memory_identity(
    candidate: Mapping[str, Any], memory: _HistoricalMemoryEvidence
) -> None:
    if (
        _as_int(candidate.get("historical_replay_rows_available"), "replay available")
        != memory.replay_rows
        or candidate.get("historical_replay_positions_digest_available") != memory.replay_digest
        or _as_int(candidate.get("audit_rows_available"), "audit available") != memory.audit_rows
        or candidate.get("audit_positions_digest_available") != memory.audit_digest
    ):
        raise RDX004RunArtifactError("Oracle candidate historical replay/audit identity differs")


def _validate_oracle_counterfactuals(
    payload: Mapping[str, Any],
    identity: RDX004ExecutionIdentity,
    events: Sequence[Mapping[str, Any]],
    historical_memory: Mapping[int, _HistoricalMemoryEvidence],
) -> list[dict[str, Any]]:
    if set(payload) != {
        "version",
        "run_id",
        "budget",
        "smoke",
        "oracle_version",
        "information_policy",
        "horizon",
        "records",
    }:
        raise RDX004RunArtifactError("Oracle counterfactual fields differ")
    records = _records(payload.get("records"), "Oracle records")
    if (
        payload.get("version") != RDX004_RUN_ARTIFACT_VERSION
        or payload.get("run_id") != identity.run_id
        or payload.get("budget") != identity.budget.value
        or _as_bool(payload.get("smoke"), "Oracle smoke") != identity.smoke
        or payload.get("oracle_version") != OFFLINE_ORACLE_VERSION
        or payload.get("information_policy") != ORACLE_INFORMATION_POLICY
        or payload.get("horizon") != ORACLE_HORIZON
        or not records
        or (identity.smoke and len(records) != 1)
    ):
        raise RDX004RunArtifactError("Oracle counterfactual identity/semantics differs")
    required_record = {
        "prediction_index",
        "decision_id",
        "stage",
        "current_domain",
        "window_id",
        "current_evaluator_state",
        "current_truth_revealed_after_prediction",
        "query_registered_before_current_truth",
        "incoming_state_digest",
        "state_digest_after_candidates",
        "state_digest_after_selection",
        "successor_window_id",
        "horizon",
        "candidates",
        "selected_action",
        "tie_break_reason",
        "selected_optimizer_steps",
        "selected_rows_consumed",
        "selected_confirmed_success",
        "evaluator_truth_visible",
        "non_deployable_upper_bound",
        "permanent_holdout_used_for_choice",
    }
    candidate_core_fields = {
        "action",
        "feasible",
        "feasibility_reason",
        "execution_succeeded",
        "audit_admissible",
        "audit_state",
        "successor_evaluator_state",
        "confirmed_success",
        "optimizer_steps",
        "target_rows",
        "replay_rows",
        "rows_consumed",
        "incoming_model_digest",
        "deployed_model_digest",
        "deployed_threshold_digest",
        "deployed_r1_digest",
        "successor_window_id",
        "successor_row_start",
        "successor_row_stop",
        "permanent_holdout_used",
    }
    candidate_extra_fields = {
        "branch_isolated",
        "target_rows_available",
        "target_positions_digest_available",
        "target_positions_digest_consumed",
        "historical_replay_rows_available",
        "historical_replay_positions_digest_available",
        "historical_replay_positions_digest_consumed",
        "audit_rows_available",
        "audit_positions_digest_available",
        "intervention_record",
        "live_state_digest_before",
        "live_state_digest_after",
        "live_state_unchanged",
    }
    seen_decisions: set[str] = set()
    previous_prediction = -1
    for record in records:
        if set(record) != required_record:
            raise RDX004RunArtifactError("Oracle record fields differ")
        decision_id = _sha256(record.get("decision_id"), "Oracle decision ID")
        prediction_index = _as_int(record.get("prediction_index"), "Oracle prediction")
        stage = _as_int(record.get("stage"), "Oracle stage")
        domain = str(record.get("current_domain", ""))
        window_id = _as_int(record.get("window_id"), "Oracle window")
        query_registered = any(
            _as_int(event.get("prediction_index"), "event prediction") == prediction_index
            for event in events
        )
        if (
            decision_id in seen_decisions
            or prediction_index <= previous_prediction
            or stage not in {1, 2, 3}
            or domain != identity.rotation[stage]
            or _as_int(record.get("successor_window_id"), "Oracle successor window")
            != window_id + 1
            or record.get("selected_action") not in _ACTION_VALUES
            or record.get("horizon") != ORACLE_HORIZON
            or _as_bool(record.get("current_truth_revealed_after_prediction"), "truth timing")
            is not True
            or _as_bool(
                record.get("query_registered_before_current_truth"),
                "Oracle query timing",
            )
            != query_registered
            or _as_bool(record.get("evaluator_truth_visible"), "Oracle truth visibility")
            is not True
            or _as_bool(record.get("non_deployable_upper_bound"), "Oracle deployment") is not True
            or _as_bool(
                record.get("permanent_holdout_used_for_choice"),
                "Oracle holdout choice",
            )
            is not False
        ):
            raise RDX004RunArtifactError("Oracle decision identity/timing differs")
        if identity.smoke:
            stage_two_predictions = [
                _as_int(event.get("prediction_index"), "smoke event prediction")
                for event in events
                if _as_int(event.get("stage"), "smoke event stage") == 2
            ]
            if (
                stage != 2
                or domain != identity.rotation[2]
                or window_id != 1
                or not stage_two_predictions
                or prediction_index != min(stage_two_predictions) + 1
                or _as_int(record.get("successor_window_id"), "smoke successor") != 2
            ):
                raise RDX004RunArtifactError("smoke Oracle decision is outside its exact window")
        seen_decisions.add(decision_id)
        previous_prediction = prediction_index
        candidates = _records(record.get("candidates"), "Oracle candidates")
        if [candidate.get("action") for candidate in candidates] != list(_ACTION_VALUES):
            raise RDX004RunArtifactError("Oracle decision lacks exact ordered A0-A4 branches")
        available_target = {
            position
            for event in events
            if _as_int(event.get("stage"), "event stage") == stage
            and event.get("current_domain") == domain
            and _as_int(event.get("release_prediction_index"), "event release") <= prediction_index
            for position in _positions(event.get("training_positions"), "event training")
        }
        current_audit = {
            position
            for event in events
            if _as_int(event.get("stage"), "event stage") == stage
            and event.get("current_domain") == domain
            for position in _positions(event.get("audit_positions"), "event audit")
        }
        memory_evidence = historical_memory.get(stage)
        if memory_evidence is None:
            raise RDX004RunArtifactError("Oracle stage lacks validated historical memory")
        expected_replay_available = memory_evidence.replay_rows
        expected_replay_scope_ids = set(memory_evidence.scope_ids)
        typed_candidates: list[OracleCandidate] = []
        selected_candidate: dict[str, Any] | None = None
        for candidate in candidates:
            if set(candidate) != candidate_core_fields | candidate_extra_fields:
                raise RDX004RunArtifactError("Oracle candidate fields differ")
            action = str(candidate.get("action", ""))
            try:
                typed = OracleCandidate.from_mapping(
                    {field: candidate[field] for field in candidate_core_fields}
                )
            except (TypeError, ValueError) as exc:
                raise RDX004RunArtifactError("Oracle candidate semantics differ") from exc
            typed_candidates.append(typed)
            _validate_candidate_memory_identity(candidate, memory_evidence)
            if (
                _as_bool(candidate.get("branch_isolated"), "candidate branch isolation") is not True
                or _as_int(candidate.get("target_rows_available"), "target available")
                != len(available_target)
                or candidate.get("target_positions_digest_available")
                != (
                    row_positions_digest(tuple(sorted(available_target)))
                    if available_target
                    else None
                )
                or candidate.get("live_state_digest_before")
                != candidate.get("live_state_digest_after")
                or _as_bool(candidate.get("live_state_unchanged"), "candidate live-state")
                is not True
            ):
                raise RDX004RunArtifactError(
                    f"Oracle {action} candidate provenance/capacity differs"
                )
            _sha256(
                candidate.get("historical_replay_positions_digest_available"),
                "available replay digest",
            )
            _sha256(candidate.get("audit_positions_digest_available"), "available audit digest")
            _sha256(candidate.get("live_state_digest_before"), "candidate live-state digest")
            _validate_candidate_intervention(
                candidate,
                action=typed.action,
                identity=identity,
                stage=stage,
                domain=domain,
                window_id=window_id,
                available_target=tuple(sorted(available_target)),
                current_audit=current_audit,
                expected_replay_available=expected_replay_available,
                expected_replay_scope_ids=expected_replay_scope_ids,
                expected_replay_scoped_positions=memory_evidence.replay_scoped_positions,
            )
            if action == record.get("selected_action"):
                selected_candidate = candidate
        if selected_candidate is None:
            raise RDX004RunArtifactError("Oracle selected action lacks candidate evidence")
        selection = select_offline_oracle(tuple(typed_candidates))
        if (
            record.get("state_digest_after_candidates") != record.get("incoming_state_digest")
            or record.get("decision_id")
            != _digest(
                {
                    "version": RDX004_RUN_ARTIFACT_VERSION,
                    "prediction_index": prediction_index,
                    "incoming_state": record.get("incoming_state_digest"),
                    "selected_action": record.get("selected_action"),
                }
            )
            or record.get("selected_action") != selection.selected_action.value
            or record.get("tie_break_reason") != selection.tie_break_reason
            or _as_int(record.get("selected_optimizer_steps"), "selected steps")
            != _as_int(selected_candidate.get("optimizer_steps"), "candidate steps")
            or _as_int(record.get("selected_rows_consumed"), "selected rows")
            != _as_int(selected_candidate.get("rows_consumed"), "candidate rows")
            or _as_bool(record.get("selected_confirmed_success"), "selected success")
            != _as_bool(selected_candidate.get("confirmed_success"), "candidate success")
        ):
            raise RDX004RunArtifactError("Oracle selection/nonmutation evidence differs")
        for field in (
            "incoming_state_digest",
            "state_digest_after_candidates",
            "state_digest_after_selection",
        ):
            _sha256(record.get(field), f"Oracle {field}")
    return records


def _validate_branch_integrity(
    payload: Mapping[str, Any],
    identity: RDX004ExecutionIdentity,
    oracle_records: Sequence[Mapping[str, Any]],
) -> None:
    if set(payload) != {"version", "run_id", "budget", "smoke", "records"}:
        raise RDX004RunArtifactError("branch-integrity fields differ")
    records = _records(payload.get("records"), "branch integrity records")
    if (
        payload.get("version") != RDX004_RUN_ARTIFACT_VERSION
        or payload.get("run_id") != identity.run_id
        or payload.get("budget") != identity.budget.value
        or _as_bool(payload.get("smoke"), "branch smoke") != identity.smoke
        or len(records) != len(oracle_records)
    ):
        raise RDX004RunArtifactError("branch-integrity identity/count differs")
    oracle_by_id = {str(item.get("decision_id")): item for item in oracle_records}
    required = {
        "decision_id",
        "prediction_index",
        "selected_action",
        "incoming_state_digest",
        "state_digest_after_candidates",
        "state_digest_after_selection",
        "candidate_evaluation_nonmutating",
        "candidate_branch_checks",
        "nonselected_branches_nonmutating",
        "selected_deployment_matches",
    }
    seen: set[str] = set()
    for branch in records:
        if set(branch) != required:
            raise RDX004RunArtifactError("branch integrity record fields differ")
        decision_id = str(branch.get("decision_id", ""))
        oracle = oracle_by_id.get(decision_id)
        if oracle is None or decision_id in seen:
            raise RDX004RunArtifactError("branch record is duplicate or lacks Oracle decision")
        seen.add(decision_id)
        candidates = _records(oracle.get("candidates"), "Oracle candidates")
        checks = _records(branch.get("candidate_branch_checks"), "candidate branch checks")
        if len(checks) != 5:
            raise RDX004RunArtifactError("branch record lacks five candidate checks")
        if (
            _as_int(branch.get("prediction_index"), "branch prediction")
            != _as_int(oracle.get("prediction_index"), "Oracle prediction")
            or branch.get("selected_action") != oracle.get("selected_action")
            or branch.get("incoming_state_digest") != oracle.get("incoming_state_digest")
            or branch.get("state_digest_after_candidates")
            != oracle.get("state_digest_after_candidates")
            or branch.get("state_digest_after_selection")
            != oracle.get("state_digest_after_selection")
            or branch.get("incoming_state_digest") != branch.get("state_digest_after_candidates")
            or _as_bool(branch.get("candidate_evaluation_nonmutating"), "candidate nonmutation")
            is not True
            or _as_bool(
                branch.get("nonselected_branches_nonmutating"),
                "nonselected branch nonmutation",
            )
            is not True
            or _as_bool(branch.get("selected_deployment_matches"), "selected deployment")
            is not True
        ):
            raise RDX004RunArtifactError("branch nonmutation/deployment evidence differs")
        for field in (
            "incoming_state_digest",
            "state_digest_after_candidates",
            "state_digest_after_selection",
        ):
            _sha256(branch.get(field), f"branch {field}")
        for candidate, check in zip(candidates, checks, strict=True):
            if set(check) != {
                "action",
                "live_state_digest_before",
                "live_state_digest_after",
                "live_state_unchanged",
            } or (
                check.get("action") != candidate.get("action")
                or check.get("live_state_digest_before")
                != candidate.get("live_state_digest_before")
                or check.get("live_state_digest_after") != candidate.get("live_state_digest_after")
                or check.get("live_state_digest_before") != check.get("live_state_digest_after")
                or _as_bool(check.get("live_state_unchanged"), "branch unchanged") is not True
            ):
                raise RDX004RunArtifactError("candidate branch check differs")
    if seen != set(oracle_by_id):
        raise RDX004RunArtifactError("branch integrity does not cover every Oracle decision")


def _validate_intervention_log(
    rows: Sequence[Mapping[str, str]],
    identity: RDX004ExecutionIdentity,
    oracle_records: Sequence[Mapping[str, Any]],
) -> None:
    expected: list[dict[str, object]] = []
    for oracle in oracle_records:
        for candidate in _records(oracle.get("candidates"), "Oracle candidates"):
            raw = candidate.get("intervention_record")
            if raw is None:
                continue
            intervention = _mapping(raw, "candidate intervention")
            expected.append(
                {
                    "run_id": identity.run_id,
                    "budget": identity.budget.value,
                    "smoke": identity.smoke,
                    "prediction_index": oracle.get("prediction_index"),
                    "decision_id": oracle.get("decision_id"),
                    "counterfactual": True,
                    "selected": candidate.get("action") == oracle.get("selected_action"),
                    **intervention,
                }
            )
    if len(rows) != len(expected):
        raise RDX004RunArtifactError("intervention log count differs from Oracle branches")
    for row, expected_row in zip(rows, expected, strict=True):
        normalised = {
            key: "" if value is None else str(value) for key, value in expected_row.items()
        }
        if row != normalised:
            raise RDX004RunArtifactError("intervention log differs from Oracle evidence")


def _validate_window_rows(
    rows: Sequence[Mapping[str, str]],
    identity: RDX004ExecutionIdentity,
    events: Sequence[Mapping[str, Any]],
    oracle_records: Sequence[Mapping[str, Any]],
    pipeline: Mapping[str, Any],
) -> None:
    required = {
        "prediction_index",
        "seed",
        "stage",
        "current_domain",
        "window_id",
        "row_start",
        "row_stop",
        "partition_kind",
        "prediction_before_truth",
        "policy_health_evaluator_free",
        "evaluator_health_state",
        "model_digest_at_prediction",
        "model_digest_after_action",
        "oracle_decision_evaluated",
        "decision_id",
        "selected_action",
        "accepted_update_after_prediction",
        "evaluator_truth_revealed_after_action",
        "permanent_holdout_used_for_choice",
    }
    if not rows:
        raise RDX004RunArtifactError("run has no online window evidence")
    if any(required.difference(row) for row in rows):
        raise RDX004RunArtifactError("window metrics schema is incomplete")
    expected_plan = _expected_window_plan(identity, pipeline, events)
    by_index: dict[int, Mapping[str, str]] = {}
    stage_rows: dict[int, list[Mapping[str, str]]] = {}
    prior_model: str | None = None
    for row in rows:
        index = _as_int(row["prediction_index"], "window prediction index")
        stage = _as_int(row["stage"], "window stage")
        window_id = _as_int(row["window_id"], "window ID")
        incoming_model = _sha256(row["model_digest_at_prediction"], "window prediction model")
        outgoing_model = _sha256(row["model_digest_after_action"], "window post-action model")
        evaluated = _as_bool(row.get("oracle_decision_evaluated"), "Oracle evaluated")
        accepted = _as_bool(row.get("accepted_update_after_prediction"), "accepted update")
        if (
            index in by_index
            or stage not in {1, 2, 3}
            or row["current_domain"] != identity.rotation[stage]
            or _as_int(row["seed"], "window seed") != identity.seed
            or window_id < 0
            or _as_int(row["row_stop"], "window row stop")
            <= _as_int(row["row_start"], "window row start")
            or row["partition_kind"] != "online_stream"
            or row["evaluator_health_state"] not in {"SAFE", "UNCERTAIN", "HARMFUL"}
            or _as_bool(row.get("prediction_before_truth"), "prediction timing") is not True
            or _as_bool(row.get("policy_health_evaluator_free"), "policy information") is not True
            or _as_bool(
                row.get("evaluator_truth_revealed_after_action"),
                "evaluator truth timing",
            )
            is not False
            or _as_bool(row.get("permanent_holdout_used_for_choice"), "window holdout choice")
            is not False
            or (prior_model is not None and incoming_model != prior_model)
            or (not accepted and outgoing_model != incoming_model)
            or row.get("selected_action") not in _ACTION_VALUES
            or accepted != (evaluated and row.get("selected_action") != "A0_NO_OP")
            or (not evaluated and row.get("decision_id") not in {"", None})
            or (not evaluated and row.get("selected_action") != "A0_NO_OP")
        ):
            raise RDX004RunArtifactError("window chronology/state boundary differs")
        _sha256(row["model_digest_at_prediction"], "window prediction model")
        _sha256(row["model_digest_after_action"], "window post-action model")
        by_index[index] = row
        stage_rows.setdefault(stage, []).append(row)
        prior_model = outgoing_model

    expected_stages = {1, 2} if identity.smoke else {1, 2, 3}
    if set(stage_rows) != expected_stages:
        raise RDX004RunArtifactError("window stages differ from the execution plan")
    for current in stage_rows.values():
        window_ids = [_as_int(row["window_id"], "window ID") for row in current]
        predictions = [
            _as_int(row["prediction_index"], "window prediction index") for row in current
        ]
        if window_ids != list(range(len(current))) or predictions != [
            predictions[0] + window_id for window_id in window_ids
        ]:
            raise RDX004RunArtifactError("stage windows are not exactly chronological")
    if len(rows) != len(expected_plan):
        kind = "seven-window smoke" if identity.smoke else "full online-stream"
        raise RDX004RunArtifactError(f"run differs from its exact {kind} coverage")
    for row, expected in zip(rows, expected_plan, strict=True):
        if (
            _as_int(row["stage"], "window stage") != expected.stage
            or row["current_domain"] != expected.domain
            or _as_int(row["window_id"], "window ID") != expected.window_id
            or _as_int(row["prediction_index"], "window prediction") != expected.prediction_index
            or _as_int(row["row_start"], "window row start") != expected.row_start
            or _as_int(row["row_stop"], "window row stop") != expected.row_stop
            or _as_bool(row["oracle_decision_evaluated"], "Oracle evaluated")
            != expected.oracle_evaluated
        ):
            kind = "seven-window smoke" if identity.smoke else "full online-stream"
            raise RDX004RunArtifactError(f"run differs from its exact {kind} coverage")
    if identity.smoke and len(oracle_records) != 1:
        raise RDX004RunArtifactError("smoke run differs from its exact seven-window plan")
    for event in events:
        prediction = _as_int(event.get("prediction_index"), "query prediction")
        release = _as_int(event.get("release_prediction_index"), "query release")
        query_row = by_index.get(prediction)
        if (
            query_row is None
            or (_as_bool(event.get("released"), "query released") and release not in by_index)
            or _as_int(query_row["stage"], "window stage")
            != _as_int(event.get("stage"), "query stage")
            or query_row["current_domain"] != event.get("current_domain")
            or _as_int(query_row["window_id"], "window ID")
            != _as_int(event.get("selection_window_id"), "selection window")
        ):
            raise RDX004RunArtifactError("query/release lacks exact online window evidence")
    for oracle in oracle_records:
        prediction = _as_int(oracle.get("prediction_index"), "Oracle prediction")
        oracle_row = by_index.get(prediction)
        if (
            oracle_row is None
            or _as_bool(oracle_row.get("oracle_decision_evaluated"), "Oracle evaluated") is not True
            or oracle_row.get("decision_id") != oracle.get("decision_id")
            or oracle_row.get("selected_action") != oracle.get("selected_action")
            or _as_bool(
                oracle_row.get("permanent_holdout_used_for_choice"),
                "window holdout choice",
            )
            is not False
        ):
            raise RDX004RunArtifactError("Oracle decision lacks online prediction evidence")
    evaluated_rows = [
        row for row in rows if _as_bool(row.get("oracle_decision_evaluated"), "Oracle evaluated")
    ]
    if len(evaluated_rows) != len(oracle_records):
        raise RDX004RunArtifactError("window Oracle-decision count differs")


def _validate_oracle_window_bindings(
    oracle_records: Sequence[Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, str]],
    identity: RDX004ExecutionIdentity,
    events: Sequence[Mapping[str, Any]],
    pipeline: Mapping[str, Any],
) -> None:
    expected = [
        item for item in _expected_window_plan(identity, pipeline, events) if item.oracle_evaluated
    ]
    if len(oracle_records) != len(expected):
        raise RDX004RunArtifactError("Oracle coverage differs from the exact one-step horizon")
    rows_by_prediction = {
        _as_int(row.get("prediction_index"), "window prediction"): row for row in window_rows
    }
    for record, route in zip(oracle_records, expected, strict=True):
        row = rows_by_prediction.get(route.prediction_index)
        if row is None:
            raise RDX004RunArtifactError("Oracle decision lacks its current window")
        candidates = _records(record.get("candidates"), "Oracle candidates")
        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate.get("action") == record.get("selected_action")
            ),
            None,
        )
        if (
            _as_int(record.get("prediction_index"), "Oracle prediction") != route.prediction_index
            or _as_int(record.get("stage"), "Oracle stage") != route.stage
            or record.get("current_domain") != route.domain
            or _as_int(record.get("window_id"), "Oracle window") != route.window_id
            or _as_int(record.get("successor_window_id"), "Oracle successor") != route.window_id + 1
            or record.get("current_evaluator_state") != row.get("evaluator_health_state")
            or selected is None
            or selected.get("deployed_model_digest") != row.get("model_digest_after_action")
        ):
            raise RDX004RunArtifactError("Oracle current-window binding differs")
        for candidate in candidates:
            if (
                candidate.get("incoming_model_digest") != row.get("model_digest_at_prediction")
                or _as_int(candidate.get("successor_window_id"), "candidate successor")
                != route.window_id + 1
                or _as_int(candidate.get("successor_row_start"), "successor row start")
                != route.successor_row_start
                or _as_int(candidate.get("successor_row_stop"), "successor row stop")
                != route.successor_row_stop
            ):
                raise RDX004RunArtifactError("Oracle candidate model/successor binding differs")


def _validate_holdout_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    identity: RDX004ExecutionIdentity,
    window_rows: Sequence[Mapping[str, str]],
) -> None:
    required = {
        "event_index",
        "event",
        "stage",
        "prediction_index",
        "holdout_dataset_id",
        "partition_kind",
        "evaluation_only",
        "used_for_query",
        "used_for_training",
        "used_for_adaptation",
        "used_for_action_choice",
        "used_for_oracle_choice",
        "model_digest",
    }
    if not rows:
        raise RDX004RunArtifactError("run lacks holdout evaluation evidence")
    if any(required.difference(row) for row in rows):
        raise RDX004RunArtifactError("holdout metrics schema is incomplete")
    windows_by_stage: dict[int, list[Mapping[str, str]]] = {}
    for window in window_rows:
        windows_by_stage.setdefault(_as_int(window.get("stage"), "window stage"), []).append(window)
    expected_calls: list[tuple[str, int, str, str, tuple[str, ...]]] = [
        (
            "source_initial",
            0,
            "",
            window_rows[0]["model_digest_at_prediction"],
            identity.rotation[:1],
        )
    ]
    for stage in sorted(windows_by_stage):
        stage_windows = windows_by_stage[stage]
        for window in stage_windows:
            if _as_bool(window.get("accepted_update_after_prediction"), "window accepted update"):
                expected_calls.append(
                    (
                        "post_accept",
                        stage,
                        window["prediction_index"],
                        window["model_digest_after_action"],
                        identity.rotation[: stage + 1],
                    )
                )
        final_window = stage_windows[-1]
        expected_calls.append(
            (
                "domain_end",
                stage,
                final_window["prediction_index"],
                final_window["model_digest_after_action"],
                identity.rotation[: stage + 1],
            )
        )
    final_stage = max(windows_by_stage)
    final_window = windows_by_stage[final_stage][-1]
    expected_calls.append(
        (
            "final",
            final_stage,
            final_window["prediction_index"],
            final_window["model_digest_after_action"],
            identity.rotation[: final_stage + 1],
        )
    )
    expected_rows = [
        (
            event_index,
            event,
            stage,
            prediction,
            domain,
            model_digest,
        )
        for event_index, (event, stage, prediction, model_digest, domains) in enumerate(
            expected_calls, start=1
        )
        for domain in domains
    ]
    actual_rows: list[tuple[int, str, int, str, str, str]] = []
    for row in rows:
        if (
            row["partition_kind"] != "permanent_holdout"
            or _as_bool(row["evaluation_only"], "holdout evaluation-only") is not True
            or _as_bool(row["used_for_query"], "holdout query use") is not False
            or _as_bool(row["used_for_training"], "holdout training use") is not False
            or _as_bool(row["used_for_adaptation"], "holdout adaptation use") is not False
            or _as_bool(row["used_for_action_choice"], "holdout action use") is not False
            or _as_bool(row["used_for_oracle_choice"], "holdout Oracle use") is not False
        ):
            raise RDX004RunArtifactError("permanent holdout was not evaluation-only")
        model_digest = _sha256(row["model_digest"], "holdout model")
        actual_rows.append(
            (
                _as_int(row["event_index"], "holdout event index"),
                row["event"],
                _as_int(row["stage"], "holdout stage"),
                row["prediction_index"],
                row["holdout_dataset_id"],
                model_digest,
            )
        )
    if actual_rows != expected_rows:
        raise RDX004RunArtifactError("holdout event/domain/model coverage differs")


def _walk_information_boundary(value: object, *, context: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if key == "partition_kind" and item == "permanent_holdout":
                raise RDX004RunArtifactError(
                    f"permanent holdout entered mutable evidence: {context}"
                )
            if "permanent_holdout_used" in key and _as_bool(item, key):
                raise RDX004RunArtifactError(f"permanent holdout use recorded: {context}")
            _walk_information_boundary(item, context=context)
    elif isinstance(value, list):
        for item in value:
            _walk_information_boundary(item, context=context)


def _validate_reference_history(
    payload: Mapping[str, Any],
    identity: RDX004ExecutionIdentity,
    starting_state: Mapping[str, Any],
) -> None:
    if set(payload) != {"version", "run_id", "budget", "smoke", "states"}:
        raise RDX004RunArtifactError("reference history fields differ")
    states = _records(payload.get("states"), "reference states")
    if (
        payload.get("version") != RDX004_RUN_ARTIFACT_VERSION
        or payload.get("run_id") != identity.run_id
        or payload.get("budget") != identity.budget.value
        or _as_bool(payload.get("smoke"), "reference smoke") != identity.smoke
        or not states
    ):
        raise RDX004RunArtifactError("reference-state history identity differs")
    first = states[0]
    fixed_data = _mapping(first.get("fixed_data"), "initial reference fixed data")
    if (
        first.get("detector_model_digest") != starting_state.get("initial_model_digest")
        or first.get("detector_threshold_digest") != starting_state.get("threshold_digest")
        or fixed_data.get("preprocessor_digest") != starting_state.get("preprocessor_digest")
        or fixed_data.get("source_scope") != starting_state.get("source_domain")
    ):
        raise RDX004RunArtifactError("initial reference state differs from paired Study-1")
    fixed_digest = fixed_data.get("fixed_data_digest")
    _sha256(fixed_digest, "fixed R1 data")
    for state in states:
        current_fixed = _mapping(state.get("fixed_data"), "reference fixed data")
        if current_fixed.get("fixed_data_digest") != fixed_digest or current_fixed.get(
            "preprocessor_digest"
        ) != starting_state.get("preprocessor_digest"):
            raise RDX004RunArtifactError("R1 refresh changed fixed source evidence")
        for field in (
            "detector_model_digest",
            "detector_threshold_digest",
            "state_digest",
        ):
            _sha256(state.get(field), f"reference {field}")
    _walk_information_boundary(payload, context="reference_state_history")


def _validate_resources_and_summary(
    resources: Mapping[str, Any],
    summary: Mapping[str, Any],
    identity: RDX004ExecutionIdentity,
    preflight: _PreflightReference,
    events: Sequence[Mapping[str, Any]],
    closures: Sequence[Mapping[str, Any]],
    oracle_records: Sequence[Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, str]],
    holdout_rows: Sequence[Mapping[str, str]],
    memory_payload: Mapping[str, Any],
) -> None:
    selected = sum(len(_positions(event["query_positions"], "query")) for event in events)
    training = sum(
        len(_positions(event["training_positions"], "training"))
        for event in events
        if _as_bool(event.get("released"), "released")
    )
    audit = sum(
        len(_positions(event["audit_positions"], "audit"))
        for event in events
        if _as_bool(event.get("released"), "released")
    )
    labels_released = sum(
        len(_positions(event["query_positions"], "query"))
        for event in events
        if _as_bool(event.get("released"), "released")
    )
    candidates = [
        candidate
        for record in oracle_records
        for candidate in _records(record.get("candidates"), "Oracle candidates")
    ]
    attempted = [item for item in candidates if item.get("intervention_record") is not None]
    selected_candidates = [
        next(
            item
            for item in _records(record.get("candidates"), "Oracle candidates")
            if item.get("action") == record.get("selected_action")
        )
        for record in oracle_records
    ]
    unsafe = sum(row.get("evaluator_health_state") == "HARMFUL" for row in window_rows)
    final_memory = _mapping(memory_payload.get("final_memory"), "final memory")
    expected_resources = {
        "version": RDX004_RUN_ARTIFACT_VERSION,
        "run_id": identity.run_id,
        "budget": identity.budget.value,
        "smoke": identity.smoke,
        "labels_requested": selected,
        "labels_released": labels_released,
        "query_events": len(events),
        "released_query_events": sum(
            _as_bool(event.get("released"), "released") for event in events
        ),
        "training_rows_allocated": training,
        "audit_rows_escrowed": audit,
        "scope_closures": len(closures),
        "oracle_decisions": len(oracle_records),
        "counterfactual_action_attempts": len(attempted),
        "counterfactual_optimizer_steps": sum(
            _as_int(item.get("optimizer_steps"), "candidate steps") for item in attempted
        ),
        "selected_optimizer_steps": sum(
            _as_int(item.get("optimizer_steps"), "selected steps") for item in selected_candidates
        ),
        "selected_model_changing_actions": sum(
            item.get("action") != "A0_NO_OP" for item in selected_candidates
        ),
        "accepted_updates": sum(
            _as_bool(row.get("accepted_update_after_prediction"), "accepted update")
            for row in window_rows
        ),
        "unsafe_exposure_windows": unsafe,
        "operating_envelope_compliance": (
            None if not window_rows else 1.0 - unsafe / len(window_rows)
        ),
        "final_replay_rows": _as_int(final_memory.get("replay_size"), "replay size"),
        "final_audit_rows": _as_int(final_memory.get("audit_size"), "audit size"),
        "memory_bytes": _as_int(final_memory.get("total_bytes"), "memory bytes"),
        "holdout_evaluation_rows": len(holdout_rows),
        "permanent_holdout_rows_used": 0,
    }
    if resources != expected_resources:
        raise RDX004RunArtifactError("resource metrics differ from artifact recomputation")
    required_summary = {
        "version",
        "status",
        "run_id",
        "budget",
        "sequence",
        "rotation",
        "seed",
        "method",
        "smoke",
        "confirmatory_eligible",
        "non_scientific_smoke",
        "window_count",
        "query_event_count",
        "released_query_event_count",
        "scope_closure_count",
        "oracle_decision_count",
        "source_state_reused",
        "source_detector_retrained",
        "preprocessor_unchanged",
        "threshold_policy_frozen",
        "permanent_holdout_used",
        "permanent_holdout_used_for_choice",
        "branch_isolation_verified",
        "preflight_bundle_digest",
        "protocol_sha256",
        "final_model_digest",
    }
    if set(summary) != required_summary:
        raise RDX004RunArtifactError("run summary fields differ")
    if (
        summary.get("version") != RDX004_RUN_ARTIFACT_VERSION
        or summary.get("status") != "complete"
        or summary.get("run_id") != identity.run_id
        or summary.get("budget") != identity.budget.value
        or summary.get("sequence") != list(identity.rotation)
        or summary.get("rotation") != list(identity.rotation)
        or _as_int(summary.get("seed"), "summary seed") != identity.seed
        or summary.get("method") != RDX004_METHOD
        or _as_bool(summary.get("smoke"), "summary smoke") != identity.smoke
        or _as_bool(summary.get("confirmatory_eligible"), "summary confirmatory eligibility")
        != identity.confirmatory_eligible
        or _as_bool(summary.get("non_scientific_smoke"), "summary smoke status") != identity.smoke
        or _as_int(summary.get("window_count"), "summary window count") != len(window_rows)
        or _as_int(summary.get("query_event_count"), "summary query count") != len(events)
        or _as_int(summary.get("released_query_event_count"), "summary releases")
        != expected_resources["released_query_events"]
        or _as_int(summary.get("scope_closure_count"), "summary closure count") != len(closures)
        or _as_int(summary.get("oracle_decision_count"), "summary Oracle count")
        != len(oracle_records)
        or _as_bool(summary.get("source_state_reused"), "source-state reuse") is not True
        or _as_bool(summary.get("source_detector_retrained"), "source retraining") is not False
        or _as_bool(summary.get("preprocessor_unchanged"), "preprocessor mutation") is not True
        or _as_bool(summary.get("threshold_policy_frozen"), "threshold policy") is not True
        or _as_bool(summary.get("permanent_holdout_used"), "summary holdout use") is not False
        or _as_bool(summary.get("permanent_holdout_used_for_choice"), "summary holdout choice")
        is not False
        or _as_bool(summary.get("branch_isolation_verified"), "branch isolation") is not True
        or summary.get("preflight_bundle_digest") != preflight.bundle_digest
        or summary.get("protocol_sha256") != RDX004_PROTOCOL_SHA256
        or summary.get("final_model_digest") != window_rows[-1].get("model_digest_after_action")
    ):
        raise RDX004RunArtifactError("run summary differs from validated evidence")


def _validate_rdx004_run_with_authority(
    run_dir: str | Path,
    *,
    expected_smoke: bool,
    preflight_dir: str | Path,
    authority: _PreflightAuthority | None,
) -> ValidatedRDX004Run:
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise RDX004RunArtifactError(f"RDX-004 run directory is absent: {root}")
    manifest, _bundle_digest = _seal_manifest_identity(
        root,
        expected_files=RDX004_RUN_OUTPUT_FILES,
        manifest_name=MANIFEST_FILENAME,
        expected_version=RDX004_RUN_ARTIFACT_VERSION,
    )
    payloads = {
        name: _load_json(root / name) for name in _JSON_FILENAMES.difference({MANIFEST_FILENAME})
    }
    config = _load_yaml(root / CONFIG_FILENAME)
    _intervention_columns, intervention_rows = _read_csv(root / INTERVENTION_FILENAME)
    _window_columns, window_rows = _read_csv(root / WINDOW_FILENAME)
    _holdout_columns, holdout_rows = _read_csv(root / HOLDOUT_FILENAME)

    contract = payloads[EXECUTION_CONTRACT_FILENAME]
    identity = _identity_from_contract(contract, expected_smoke=expected_smoke)
    preflight_record = _mapping(contract.get("preflight"), "execution preflight identity")
    declared_bundle_digest = _sha256(
        preflight_record.get("bundle_digest"), "declared preflight bundle"
    )
    preflight = (
        _load_preflight_reference(
            preflight_dir,
            identity,
            declared_bundle_digest=declared_bundle_digest,
        )
        if authority is None
        else _preflight_reference_from_authority(
            authority,
            identity,
            declared_bundle_digest=declared_bundle_digest,
        )
    )
    _validate_execution_contract(contract, identity, preflight)
    _validate_resolved_config(config, identity, contract, preflight)

    events, closures = _validate_query_evidence(
        payloads[QUERY_EVIDENCE_FILENAME], identity, preflight
    )
    memory_payload = payloads[MEMORY_FILENAME]
    historical_memory = _validate_memory_evidence(
        memory_payload, identity, preflight, events, closures
    )
    oracle_records = _validate_oracle_counterfactuals(
        payloads[ORACLE_COUNTERFACTUALS_FILENAME],
        identity,
        events,
        historical_memory,
    )
    _validate_branch_integrity(payloads[BRANCH_INTEGRITY_FILENAME], identity, oracle_records)
    _validate_intervention_log(intervention_rows, identity, oracle_records)
    pipeline = _mapping(contract.get("pipeline"), "execution pipeline")
    _validate_window_rows(window_rows, identity, events, oracle_records, pipeline)
    _validate_oracle_window_bindings(
        oracle_records,
        window_rows,
        identity,
        events,
        pipeline,
    )
    _validate_holdout_rows(holdout_rows, identity=identity, window_rows=window_rows)
    starting_state = _mapping(contract.get("starting_state"), "execution starting state")
    _validate_reference_history(payloads[REFERENCE_FILENAME], identity, starting_state)

    for name, value in (
        (QUERY_EVIDENCE_FILENAME, payloads[QUERY_EVIDENCE_FILENAME]),
        (MEMORY_FILENAME, memory_payload),
        (ORACLE_COUNTERFACTUALS_FILENAME, payloads[ORACLE_COUNTERFACTUALS_FILENAME]),
        (BRANCH_INTEGRITY_FILENAME, payloads[BRANCH_INTEGRITY_FILENAME]),
    ):
        _walk_information_boundary(value, context=name)
    _validate_resources_and_summary(
        payloads[RESOURCE_FILENAME],
        payloads[SUMMARY_FILENAME],
        identity,
        preflight,
        events,
        closures,
        oracle_records,
        window_rows,
        holdout_rows,
        memory_payload,
    )
    return ValidatedRDX004Run(
        path=root,
        run_id=identity.run_id,
        expected_confirmatory_run_id=identity.expected_confirmatory_run_id,
        budget=identity.budget,
        rotation=identity.rotation,
        seed=identity.seed,
        smoke=identity.smoke,
        paired_b100_experiment_id=identity.paired_b100_experiment_id,
        paired_study1_experiment_id=identity.paired_study1_experiment_id,
        preflight_bundle_digest=preflight.bundle_digest,
        artifact_bundle_digest=_sha256(manifest.get("bundle_digest"), "RDX run bundle digest"),
        query_event_count=len(events),
        scope_closure_count=len(closures),
        oracle_decision_count=len(oracle_records),
        summary=dict(payloads[SUMMARY_FILENAME]),
    )


def validate_rdx004_run(
    run_dir: str | Path,
    *,
    expected_smoke: bool = False,
    preflight_dir: str | Path = Path("rdx/training-evidence-preflight-v1"),
) -> ValidatedRDX004Run:
    """Validate one complete sealed RDX-004 run without executing the method.

    Confirmatory validation is the default.  A smoke artifact is accepted only
    when the caller explicitly sets ``expected_smoke=True``; its bounded plan
    remains non-scientific and cannot be mistaken for a roster cell.  Each call
    independently validates its preflight authority, matching the original
    standalone fail-closed behavior.
    """

    return _validate_rdx004_run_with_authority(
        run_dir,
        expected_smoke=expected_smoke,
        preflight_dir=preflight_dir,
        authority=None,
    )


def validate_rdx004_runs(
    run_dirs: Sequence[str | Path],
    *,
    expected_smoke: bool = False,
    preflight_dir: str | Path = Path("rdx/training-evidence-preflight-v1"),
) -> tuple[ValidatedRDX004Run, ...]:
    """Validate a batch against one scoped common authority and mutation snapshot.

    The preflight is independently source-rederived once for this call.  Every
    run still receives the complete standalone semantic validation, while paired
    B100 validation is reused only inside this batch.  Full dependency file sets
    and content digests are checked again before a successful return.
    """

    paths = tuple(Path(path).resolve() for path in run_dirs)
    if not paths:
        raise RDX004RunArtifactError("RDX-004 batch validation requires at least one run")
    if len(paths) != len(set(paths)):
        raise RDX004RunArtifactError("RDX-004 batch validation contains duplicate run paths")
    authority = _load_preflight_authority(preflight_dir)
    snapshot = _capture_dependency_snapshot(authority, paths)
    validated = tuple(
        _validate_rdx004_run_with_authority(
            path,
            expected_smoke=expected_smoke,
            preflight_dir=preflight_dir,
            authority=authority,
        )
        for path in paths
    )
    _assert_dependency_snapshot_unchanged(snapshot)
    return validated


__all__ = [
    "BRANCH_INTEGRITY_FILENAME",
    "CONFIG_FILENAME",
    "EXECUTION_CONTRACT_FILENAME",
    "HOLDOUT_FILENAME",
    "INTERVENTION_FILENAME",
    "MANIFEST_FILENAME",
    "MEMORY_FILENAME",
    "ORACLE_COUNTERFACTUALS_FILENAME",
    "QUERY_EVIDENCE_FILENAME",
    "RDX004_RUN_ARTIFACT_VERSION",
    "RDX004_RUN_OUTPUT_FILES",
    "REFERENCE_FILENAME",
    "RESOURCE_FILENAME",
    "SUMMARY_FILENAME",
    "WINDOW_FILENAME",
    "RDX004RunArtifactError",
    "ValidatedRDX004Run",
    "validate_rdx004_run",
    "validate_rdx004_runs",
]
