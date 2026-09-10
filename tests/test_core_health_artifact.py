from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from danids.policy.health_artifact import (
    HEALTH_MODEL_FILENAME,
    HEALTH_MODEL_MANIFEST_FILENAME,
    POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
    POLICY_HEALTH_FEATURES,
    TAU_HARMFUL,
    TAU_SAFE,
    PolicyHealthVector,
    PredictedHealthState,
    build_health_model_artifact,
    classify_harm_probability,
    validate_health_model_artifact,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _dataset(path: Path) -> Path:
    rng = np.random.default_rng(81)
    rows: list[dict[str, object]] = []
    for domain_number, domain in enumerate(("U", "T", "C", "B")):
        for window_id in range(30):
            state = ("SAFE", "HARMFUL", "HARMFUL", "UNCERTAIN")[window_id % 4]
            if domain == "U" and window_id == 0:
                row_start, row_stop, partition = 0, 100, "validation"
            elif domain == "U" and window_id == 1:
                # A different role/key sharing raw rows must remain in one component.
                row_start, row_stop, partition = 50, 150, "online_stream"
            else:
                row_start = 1000 + window_id * 200
                row_stop = row_start + 100
                partition = "validation" if window_id % 5 == 0 else "online_stream"
            centre = {"SAFE": -1.0, "UNCERTAIN": 0.0, "HARMFUL": 1.0}[state]
            features = {
                name: float(centre + domain_number * 0.03 + rng.normal(0.0, 0.15))
                for name in POLICY_HEALTH_FEATURES
            }
            rows.append(
                {
                    "current_domain": domain,
                    "partition_kind": partition,
                    "row_start": row_start,
                    "row_stop": row_stop,
                    "window_id": window_id,
                    "health_state": state,
                    **features,
                }
            )
    frame = pd.DataFrame(rows)
    frame.loc[3, POLICY_HEALTH_FEATURES[0]] = np.nan
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def health_dataset(tmp_path: Path) -> Path:
    return _dataset(tmp_path / "study3_health_dataset.csv")


@pytest.fixture
def health_artifact(tmp_path: Path, health_dataset: Path) -> tuple[Path, Path, str]:
    dataset_sha = _sha256(health_dataset)
    output = tmp_path / "health-model"
    build_health_model_artifact(
        health_dataset,
        output,
        expected_dataset_sha256=dataset_sha,
    )
    return output, health_dataset, dataset_sha


def _rewrite_manifest(path: Path, mutate: object) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert callable(mutate)
    mutate(manifest)
    without_identity = dict(manifest)
    without_identity.pop("artifact_identity_sha256")
    manifest["artifact_identity_sha256"] = _canonical_digest(without_identity)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_exact_policy_vector_allowlist_order_missing_values_and_boundaries() -> None:
    reverse = {
        name: float(index) for index, name in reversed(tuple(enumerate(POLICY_HEALTH_FEATURES)))
    }
    vector = PolicyHealthVector.from_mapping(reverse)
    assert tuple(vector.as_mapping()) == POLICY_HEALTH_FEATURES
    assert vector.values == tuple(float(index) for index in range(28))
    assert list(vector.as_frame().columns) == list(POLICY_HEALTH_FEATURES)

    with pytest.raises(ValueError, match="unexpected"):
        PolicyHealthVector.from_mapping({**reverse, "health_state": "HARMFUL"})
    missing = dict(reverse)
    missing.pop(POLICY_HEALTH_FEATURES[0])
    with pytest.raises(ValueError, match="missing"):
        PolicyHealthVector.from_mapping(missing)
    with pytest.raises(TypeError, match="not numeric"):
        PolicyHealthVector.from_mapping({**reverse, POLICY_HEALTH_FEATURES[0]: True})
    PolicyHealthVector.from_mapping({**reverse, POLICY_HEALTH_FEATURES[0]: float("nan")})
    with pytest.raises(ValueError, match="infinite"):
        PolicyHealthVector.from_mapping({**reverse, POLICY_HEALTH_FEATURES[0]: float("inf")})

    assert classify_harm_probability(TAU_SAFE) is PredictedHealthState.SAFE
    assert classify_harm_probability(np.nextafter(TAU_SAFE, np.inf)) is (
        PredictedHealthState.UNCERTAIN
    )
    assert classify_harm_probability(np.nextafter(TAU_HARMFUL, -np.inf)) is (
        PredictedHealthState.UNCERTAIN
    )
    assert classify_harm_probability(TAU_HARMFUL) is PredictedHealthState.HARMFUL


def test_artifact_round_trip_is_write_once_and_records_overlap_safe_provenance(
    health_artifact: tuple[Path, Path, str],
) -> None:
    output, dataset, dataset_sha = health_artifact
    frozen = validate_health_model_artifact(
        output,
        study3_dataset_path=dataset,
        expected_dataset_sha256=dataset_sha,
    )
    manifest = frozen.manifest
    assert manifest["feature_contract"] == {
        "name": "task005_combined_unlabelled",
        "ordered_features": list(POLICY_HEALTH_FEATURES),
        "feature_count": 28,
        "sha256": POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
    }
    assert manifest["dataset"]["canonical_csv_sha256"] == dataset_sha
    assert manifest["dataset"]["expected_canonical_csv_sha256"] == dataset_sha
    assert manifest["split"]["cross_split_component_overlap_count"] == 0
    assert manifest["split"]["cross_split_physical_window_overlap_count"] == 0
    assert manifest["split"]["cross_split_raw_interval_overlap_count"] == 0
    assert manifest["split"]["overlap_component_count"] < manifest["split"]["physical_window_count"]
    assert manifest["model"]["refit_after_calibration"] is False
    assert manifest["calibration"]["tau_safe"] == TAU_SAFE
    assert manifest["calibration"]["tau_harmful"] == TAU_HARMFUL
    assert len(manifest["code_commit_sha"]) == 40
    assert len(frozen.artifact_identity) == 64
    vector = PolicyHealthVector.from_mapping({name: 0.0 for name in POLICY_HEALTH_FEATURES})
    assert 0.0 <= frozen.decide(vector).harm_probability <= 1.0
    missing_vector = PolicyHealthVector.from_mapping(
        {name: float("nan") for name in POLICY_HEALTH_FEATURES}
    )
    assert 0.0 <= frozen.decide(missing_vector).harm_probability <= 1.0
    with pytest.raises(FileExistsError, match="already exists"):
        build_health_model_artifact(
            dataset,
            output,
            expected_dataset_sha256=dataset_sha,
        )
    with pytest.raises(ValueError, match="canonical dataset digest"):
        validate_health_model_artifact(output, study3_dataset_path=dataset)


def test_serialized_model_corruption_is_rejected(
    health_artifact: tuple[Path, Path, str],
) -> None:
    output, _, dataset_sha = health_artifact
    model_path = output / HEALTH_MODEL_FILENAME
    model_path.write_bytes(model_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="size differs"):
        validate_health_model_artifact(output, expected_dataset_sha256=dataset_sha)


@pytest.mark.parametrize("corruption", ["feature", "threshold", "split"])
def test_coherently_rehashed_manifest_corruption_is_rejected(
    health_artifact: tuple[Path, Path, str], corruption: str
) -> None:
    output, dataset, dataset_sha = health_artifact
    manifest_path = output / HEALTH_MODEL_MANIFEST_FILENAME

    def mutate(manifest: dict[str, object]) -> None:
        if corruption == "feature":
            contract = manifest["feature_contract"]
            assert isinstance(contract, dict)
            contract["ordered_features"] = list(reversed(POLICY_HEALTH_FEATURES))
        elif corruption == "threshold":
            calibration = manifest["calibration"]
            assert isinstance(calibration, dict)
            calibration["tau_safe"] = TAU_SAFE - 0.01
        else:
            split = manifest["split"]
            assert isinstance(split, dict)
            split["fit_component_digest"] = "0" * 64

    _rewrite_manifest(manifest_path, mutate)
    match = {
        "feature": "feature contract differs",
        "threshold": "calibration differs",
        "split": "overlap-component provenance differs",
    }[corruption]
    with pytest.raises(ValueError, match=match):
        validate_health_model_artifact(
            output,
            study3_dataset_path=dataset,
            expected_dataset_sha256=dataset_sha,
        )


def test_extra_artifact_file_and_changed_canonical_dataset_are_rejected(
    health_artifact: tuple[Path, Path, str],
) -> None:
    output, dataset, dataset_sha = health_artifact
    (output / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="extra files"):
        validate_health_model_artifact(output, expected_dataset_sha256=dataset_sha)
    (output / "unexpected.txt").unlink()
    dataset.write_text(dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset digest differs"):
        validate_health_model_artifact(
            output,
            study3_dataset_path=dataset,
            expected_dataset_sha256=dataset_sha,
        )


def test_artifact_generation_is_deterministic(health_dataset: Path, tmp_path: Path) -> None:
    dataset_sha = _sha256(health_dataset)
    first = build_health_model_artifact(
        health_dataset,
        tmp_path / "first",
        expected_dataset_sha256=dataset_sha,
    )
    second = build_health_model_artifact(
        health_dataset,
        tmp_path / "second",
        expected_dataset_sha256=dataset_sha,
    )
    assert (first / HEALTH_MODEL_FILENAME).read_bytes() == (
        second / HEALTH_MODEL_FILENAME
    ).read_bytes()
    assert (first / HEALTH_MODEL_MANIFEST_FILENAME).read_bytes() == (
        second / HEALTH_MODEL_MANIFEST_FILENAME
    ).read_bytes()


@pytest.mark.parametrize("mutation", ["missing", "additional"])
def test_artifact_builder_rejects_changed_health_feature_contract(
    health_dataset: Path, tmp_path: Path, mutation: str
) -> None:
    frame = pd.read_csv(health_dataset)
    if mutation == "missing":
        frame = frame.drop(columns=POLICY_HEALTH_FEATURES[-1])
    else:
        frame["model_unexpected_signal"] = 0.0
    changed = tmp_path / f"changed-{mutation}.csv"
    frame.to_csv(changed, index=False)
    with pytest.raises(ValueError, match="frozen ordered Study-4 policy contract"):
        build_health_model_artifact(
            changed,
            tmp_path / f"artifact-{mutation}",
            expected_dataset_sha256=_sha256(changed),
        )
