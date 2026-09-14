from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from danids.attacks import (
    FROZEN_STUDY5_DATASET_FINGERPRINTS,
    ChronologicalLabelSupport,
    ExactNativeInventory,
    ExactNativeLabelObservation,
    HypothesisStatus,
    MappingStatus,
    NativeAttackKey,
    Study5ContractError,
    inspect_materialized_inventory,
    load_coherent_study5_manifests,
    load_study5_contract,
    scan_configured_dataset_inventory,
    validate_exact_inventory,
)
from danids.attacks.contract import canonical_contract_sha256
from danids.cli import main
from danids.config.experiment import ExperimentConfig
from danids.data.loading import DataLoadingError
from danids.data.manifests import fingerprint_file, generate_split_manifest
from danids.data.materialized import (
    MATERIALIZER_VERSION,
    materialize_dataset,
    open_existing_materialized_dataset,
)
from danids.data.registry import DatasetRegistry, DatasetSpec
from danids.data.schema import FeatureContract

CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "study5" / "attack_ontology_v1.yaml"
)


def _contract_payload() -> dict[str, Any]:
    raw = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _write_contract(tmp_path: Path, raw: dict[str, Any], *, reseal: bool = True) -> Path:
    payload = copy.deepcopy(raw)
    if reseal:
        payload["contract_sha256"] = canonical_contract_sha256(payload)
    path = tmp_path / "attack-ontology.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _native_entry(raw: dict[str, Any], dataset_id: str, label: str) -> dict[str, Any]:
    matches = [
        item
        for item in raw["native_attacks"]
        if item["dataset_id"] == dataset_id and item["exact_native_label"] == label
    ]
    assert len(matches) == 1
    return matches[0]


def test_frozen_contract_has_complete_exact_mapping_inventory() -> None:
    contract = load_study5_contract(CONTRACT_PATH)

    assert contract.dataset_fingerprints == FROZEN_STUDY5_DATASET_FINGERPRINTS
    assert len(contract.native_label_supports) == 40
    assert len(contract.native_attacks) == 36
    assert (
        sum(item.mapping_status is MappingStatus.MAPPED for item in contract.native_attacks) == 24
    )
    assert (
        sum(item.mapping_status is MappingStatus.UNMAPPED for item in contract.native_attacks) == 12
    )
    assert all(item.key.exact_native_label != "Benign" for item in contract.native_attacks)
    assert contract.support_for("U", "Worms") == ChronologicalLabelSupport(
        total=158,
        initial_train=50,
        source_validation=60,
        later_online=110,
        permanent_holdout=48,
    )

    raw_c_label = contract.mapping_for("C", "Infilteration")
    assert raw_c_label.mapping_status is MappingStatus.UNMAPPED
    assert raw_c_label.display_native_label == "Infiltration"
    with pytest.raises(Study5ContractError, match="unknown exact native attack key"):
        contract.mapping_for("C", "Infiltration")


def test_exact_mapping_lookup_preserves_case_and_dataset_qualification() -> None:
    contract = load_study5_contract(CONTRACT_PATH)

    assert contract.mapping_for("U", "DoS").semantic_family == "Availability / Impact"
    assert contract.mapping_for("T", "dos").semantic_family == "Availability / Impact"
    assert contract.mapping_for("U", "Backdoor").mapping_status is MappingStatus.UNMAPPED
    assert contract.mapping_for("T", "Backdoor").mapping_status is MappingStatus.UNMAPPED
    with pytest.raises(Study5ContractError, match="normalization is forbidden"):
        contract.mapping_for("T", "backdoor")
    with pytest.raises(Study5ContractError, match="unknown exact native attack key"):
        contract.mapping_for("B", "Backdoor")


def test_frozen_estimand_helpers_apply_exact_boundaries() -> None:
    contract = load_study5_contract(CONTRACT_PATH)

    assert not contract.estimands.support.is_supported(49)
    assert contract.estimands.support.is_supported(50)
    hidden = contract.estimands.hidden_family_failure
    assert hidden.occurs(0.10, (0.01, 0.1000001))
    assert not hidden.occurs(0.1000001, (0.50,))
    assert not hidden.occurs(0.10, (0.10,))
    assert contract.hypothesis_status("H6") is HypothesisStatus.NOT_CURRENTLY_TESTABLE
    assert (
        contract.hypothesis_status("H7")
        is HypothesisStatus.PARTIALLY_TESTABLE_EXISTING_SINGLE_ORDER
    )
    assert contract.hypothesis_status("H8") is HypothesisStatus.NOT_CURRENTLY_TESTABLE


@pytest.mark.parametrize(
    ("dataset_id", "label", "replacement"),
    [
        ("C", "Infilteration", "Infiltration"),
        ("T", "Backdoor", "backdoor"),
    ],
)
def test_contract_rejects_spelling_or_case_changes(
    tmp_path: Path, dataset_id: str, label: str, replacement: str
) -> None:
    raw = _contract_payload()
    _native_entry(raw, dataset_id, label)["exact_native_label"] = replacement

    with pytest.raises(Study5ContractError, match="unknown exact native attack mapping key"):
        load_study5_contract(_write_contract(tmp_path, raw))


def test_contract_rejects_dataset_requalification(tmp_path: Path) -> None:
    raw = _contract_payload()
    _native_entry(raw, "C", "Bot")["dataset_id"] = "U"

    with pytest.raises(Study5ContractError, match="unknown exact native attack mapping key"):
        load_study5_contract(_write_contract(tmp_path, raw))


def test_contract_rejects_dataset_fingerprint_change(tmp_path: Path) -> None:
    raw = _contract_payload()
    raw["datasets"]["U"]["sha256"] = "0" * 64

    with pytest.raises(Study5ContractError, match=r"datasets\.U\.sha256"):
        load_study5_contract(_write_contract(tmp_path, raw))


def test_contract_rejects_mapping_change(tmp_path: Path) -> None:
    raw = _contract_payload()
    item = _native_entry(raw, "U", "DoS")
    item["mapping_status"] = "UNMAPPED"
    item["semantic_family"] = None

    with pytest.raises(Study5ContractError, match="differs from the primary freeze"):
        load_study5_contract(_write_contract(tmp_path, raw))


def test_contract_rejects_support_change(tmp_path: Path) -> None:
    raw = _contract_payload()
    raw["estimands"]["support"]["minimum_evaluation_support"] = 49

    with pytest.raises(Study5ContractError, match="supported-family definition"):
        load_study5_contract(_write_contract(tmp_path, raw))


def test_contract_rejects_exact_label_support_change(tmp_path: Path) -> None:
    raw = _contract_payload()
    support = _native_entry(raw, "U", "Worms")["support"]
    support["initial_train"] -= 1
    support["later_online"] -= 1
    support["total"] -= 1

    with pytest.raises(Study5ContractError, match=r"support for \(U, Worms\)"):
        load_study5_contract(_write_contract(tmp_path, raw))


def test_contract_rejects_novelty_change(tmp_path: Path) -> None:
    raw = _contract_payload()
    raw["estimands"]["novelty"]["assignment"] = "PER_WINDOW"

    with pytest.raises(Study5ContractError, match="domain-entry novelty definition"):
        load_study5_contract(_write_contract(tmp_path, raw))


def test_contract_rejects_estimand_change(tmp_path: Path) -> None:
    raw = _contract_payload()
    raw["estimands"]["hidden_family_failure"]["aggregate_loss_limit"] = 0.11

    with pytest.raises(Study5ContractError, match="hidden-family-failure definition"):
        load_study5_contract(_write_contract(tmp_path, raw))


def test_contract_rejects_declared_hash_corruption(tmp_path: Path) -> None:
    raw = _contract_payload()
    raw["contract_sha256"] = "0" * 64

    with pytest.raises(Study5ContractError, match="does not match the canonical YAML payload"):
        load_study5_contract(_write_contract(tmp_path, raw, reseal=False))


def test_contract_digest_prevents_unreviewed_display_metadata_change(tmp_path: Path) -> None:
    raw = _contract_payload()
    _native_entry(raw, "C", "Infilteration")["display_native_label"] = "Changed display"

    with pytest.raises(Study5ContractError, match="contract digest differs from the freeze"):
        load_study5_contract(_write_contract(tmp_path, raw))


def test_contract_rejects_duplicate_exact_native_key(tmp_path: Path) -> None:
    raw = _contract_payload()
    raw["native_attacks"].append(copy.deepcopy(_native_entry(raw, "U", "DoS")))

    with pytest.raises(Study5ContractError, match="duplicate exact native attack mapping key"):
        load_study5_contract(_write_contract(tmp_path, raw))


def test_contract_rejects_missing_exact_native_key(tmp_path: Path) -> None:
    raw = _contract_payload()
    raw["native_attacks"].remove(_native_entry(raw, "U", "DoS"))

    with pytest.raises(Study5ContractError, match="missing exact keys"):
        load_study5_contract(_write_contract(tmp_path, raw))


def test_contract_rejects_unknown_exact_native_key(tmp_path: Path) -> None:
    raw = _contract_payload()
    unknown = copy.deepcopy(_native_entry(raw, "U", "Analysis"))
    unknown["exact_native_label"] = "UnknownAttack"
    raw["native_attacks"].append(unknown)

    with pytest.raises(Study5ContractError, match="unknown exact native attack mapping key"):
        load_study5_contract(_write_contract(tmp_path, raw))


def _write_small_materialized_cache(
    tmp_path: Path,
    *,
    binary_labels: np.ndarray | None = None,
) -> Path:
    path = tmp_path / "U-cache"
    path.mkdir()
    attack_codes = np.asarray([0, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int32)
    binary = (
        np.asarray([0, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int8)
        if binary_labels is None
        else binary_labels
    )
    np.save(path / "attack.npy", attack_codes)
    np.save(path / "binary.npy", binary)
    metadata = {
        "dataset_id": "U",
        "row_count": 10,
        "attack_labels": ["Benign", "DoS"],
        "source_sha256": "a" * 64,
        "materializer_version": MATERIALIZER_VERSION,
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_materialized_inventory_uses_frozen_chronological_slices_read_only(
    tmp_path: Path,
) -> None:
    path = _write_small_materialized_cache(tmp_path)
    before = {
        item.name: (item.read_bytes(), item.stat().st_mtime_ns)
        for item in path.iterdir()
        if item.is_file()
    }

    inventory = inspect_materialized_inventory(path, chunk_rows=3)

    assert inventory.row_count == 10
    assert inventory.source_kind == "MATERIALIZED_NUMPY_MEMMAP"
    assert inventory.labels[NativeAttackKey("U", "Benign")].chronological_support == (
        ChronologicalLabelSupport(5, 3, 1, 4, 1)
    )
    assert inventory.labels[NativeAttackKey("U", "DoS")].chronological_support == (
        ChronologicalLabelSupport(5, 3, 1, 4, 1)
    )
    after = {
        item.name: (item.read_bytes(), item.stat().st_mtime_ns)
        for item in path.iterdir()
        if item.is_file()
    }
    assert after == before


def test_materialized_inventory_rejects_native_binary_mismatch(tmp_path: Path) -> None:
    binary = np.asarray([0, 0, 0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int8)
    path = _write_small_materialized_cache(tmp_path, binary_labels=binary)

    with pytest.raises(Study5ContractError, match="invalid binary pairing"):
        inspect_materialized_inventory(path, chunk_rows=4)


def test_raw_inventory_preserves_exact_case_and_is_read_only(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "Label": [0, 1, 1],
            "Attack": ["Benign", "Backdoor", "backdoor"],
        }
    ).to_csv(path, index=False)
    spec = DatasetSpec(dataset_id="U", name="test", path=path)
    source = fingerprint_file(path)
    before = (path.read_bytes(), path.stat().st_mtime_ns)

    inventory = scan_configured_dataset_inventory(spec, source.sha256, chunk_rows=1)

    assert set(inventory.labels) == {
        NativeAttackKey("U", "Benign"),
        NativeAttackKey("U", "Backdoor"),
        NativeAttackKey("U", "backdoor"),
    }
    assert all(item.chronological_support is None for item in inventory.labels.values())
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_raw_inventory_rejects_normalized_benign_identity(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    pd.DataFrame({"Label": [0], "Attack": ["benign"]}).to_csv(path, index=False)
    spec = DatasetSpec(dataset_id="U", name="test", path=path)

    with pytest.raises(Study5ContractError, match="invalid binary pairing"):
        scan_configured_dataset_inventory(
            spec,
            fingerprint_file(path).sha256,
            chunk_rows=1,
        )


def _inventory_from_contract(
    dataset_id: str, *, chronological: bool = True
) -> ExactNativeInventory:
    contract = load_study5_contract(CONTRACT_PATH)
    expected = {
        key: support
        for key, support in contract.native_label_supports.items()
        if key.dataset_id == dataset_id
    }
    observations = {
        key: ExactNativeLabelObservation(
            key=key,
            binary_label=contract.binary_target_for(
                key.dataset_id,
                key.exact_native_label,
            ),
            total=support.total,
            chronological_support=support if chronological else None,
        )
        for key, support in expected.items()
    }
    return ExactNativeInventory(
        dataset_id=dataset_id,
        source_sha256=contract.dataset(dataset_id).sha256,
        row_count=sum(support.total for support in expected.values()),
        source_kind="TEST",
        labels=observations,
    )


def test_observed_inventory_rejects_missing_or_unknown_exact_labels() -> None:
    contract = load_study5_contract(CONTRACT_PATH)
    valid = _inventory_from_contract("C")
    missing = dict(valid.labels)
    missing.pop(NativeAttackKey("C", "Infilteration"))
    with pytest.raises(Study5ContractError, match="missing=Infilteration"):
        validate_exact_inventory(
            contract,
            replace(valid, labels=missing),
            require_chronological_support=True,
        )

    unknown = dict(valid.labels)
    original = unknown.pop(NativeAttackKey("C", "Infilteration"))
    changed_key = NativeAttackKey("C", "Infiltration")
    unknown[changed_key] = replace(original, key=changed_key)
    with pytest.raises(Study5ContractError, match="unknown=Infiltration"):
        validate_exact_inventory(
            contract,
            replace(valid, labels=unknown),
            require_chronological_support=True,
        )


def test_observed_inventory_rejects_chronological_support_change() -> None:
    contract = load_study5_contract(CONTRACT_PATH)
    valid = _inventory_from_contract("U")
    labels = dict(valid.labels)
    key = NativeAttackKey("U", "Worms")
    observation = labels[key]
    assert observation.chronological_support is not None
    changed = replace(
        observation.chronological_support,
        initial_train=observation.chronological_support.initial_train - 1,
    )
    labels[key] = replace(observation, chronological_support=changed)

    with pytest.raises(Study5ContractError, match="chronological support differs"):
        validate_exact_inventory(
            contract,
            replace(valid, labels=labels),
            require_chronological_support=True,
        )


def test_manifest_identity_validation_accepts_role_duplicates_and_rejects_drift(
    tmp_path: Path,
    registry: DatasetRegistry,
    contract: FeatureContract,
    experiment: ExperimentConfig,
) -> None:
    manifest_dir = tmp_path / "manifests"
    manifests = {}
    for dataset_id in ("U", "T", "B", "C"):
        manifest = generate_split_manifest(
            registry[dataset_id],
            contract,
            role="later",
            split_version=experiment.split_version,
            seed=experiment.seed,
            splits=experiment.splits,
        )
        manifest.write(manifest_dir / f"later-{dataset_id}.json")
        manifests[dataset_id] = manifest
    initial_u = generate_split_manifest(
        registry["U"],
        contract,
        role="initial",
        split_version=experiment.split_version,
        seed=experiment.seed,
        splits=experiment.splits,
    )
    initial_u.write(manifest_dir / "initial-U.json")
    frozen = load_study5_contract(CONTRACT_PATH)
    synthetic_contract = replace(
        frozen,
        datasets=tuple(
            replace(
                dataset,
                sha256=manifests[dataset.dataset_id].source.sha256,
            )
            for dataset in frozen.datasets
        ),
    )

    selected = load_coherent_study5_manifests(
        synthetic_contract,
        registry,
        manifest_dir,
    )
    assert set(selected) == {"U", "T", "B", "C"}

    changed = initial_u.to_dict()
    changed["feature_contract_version"] = "different-feature-contract"
    (manifest_dir / "changed-U.json").write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(Study5ContractError, match="full chronological representation"):
        load_coherent_study5_manifests(
            synthetic_contract,
            registry,
            manifest_dir,
        )


def test_open_existing_materialized_dataset_never_creates_a_missing_cache(
    tmp_path: Path,
    registry: DatasetRegistry,
    contract: FeatureContract,
    experiment: ExperimentConfig,
) -> None:
    manifest = generate_split_manifest(
        registry["U"],
        contract,
        role="later",
        split_version=experiment.split_version,
        seed=experiment.seed,
        splits=experiment.splits,
    )
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    with pytest.raises(DataLoadingError, match="existing materialized cache is missing"):
        open_existing_materialized_dataset(cache_root, manifest)
    assert not tuple(cache_root.iterdir())

    materialized = materialize_dataset(
        registry["U"],
        contract,
        manifest,
        cache_root=cache_root,
        chunk_rows=7,
    )
    before = {
        item.relative_to(cache_root): (item.read_bytes(), item.stat().st_mtime_ns)
        for item in cache_root.rglob("*")
        if item.is_file()
    }
    opened = open_existing_materialized_dataset(cache_root, manifest)
    assert opened.path == materialized.path
    after = {
        item.relative_to(cache_root): (item.read_bytes(), item.stat().st_mtime_ns)
        for item in cache_root.rglob("*")
        if item.is_file()
    }
    assert after == before


def test_validate_study5_contract_cli_help(capsys: object) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["validate-study5-contract", "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "--contract" in output
    assert "--datasets-config" in output
    assert "--manifest-dir" in output
    assert "--materialization-root" in output
    assert "--chunk-rows" in output
