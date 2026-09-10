"""Frozen Study-4 health model, strict policy vector, and artifact validation.

The Study-4 controller consumes a deliberately narrow capability: an ordered vector
of the 28 TASK-005 ``combined_unlabelled`` signals.  This module owns that contract,
the prospectively frozen gradient-boosting model, and its write-once provenance.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import pickle
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingClassifier  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedGroupKFold  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from danids.config.core import CORE_HEALTH_FEATURES, CORE_TAU_HARMFUL, CORE_TAU_SAFE
from danids.config.health import HealthPredictorConfig
from danids.health.predictors import feature_columns, make_predictor

HEALTH_MODEL_ARTIFACT_VERSION = "task006-frozen-health-model-v1"
HEALTH_MODEL_FILENAME = "health_model.pkl"
HEALTH_MODEL_MANIFEST_FILENAME = "health_model_manifest.json"
PHYSICAL_WINDOW_KEY_VERSION = "task006-physical-window-key-v1"
OVERLAP_GROUPING_VERSION = "task006-overlap-component-v1"
FIT_CALIBRATION_SPLIT_VERSION = "task006-overlap-component-sgkf-v1"

FROZEN_STUDY3_DATASET_SHA256 = "e261d1b58e81a5d25c4b26ca5b0fa543a22146dd9ef1c73b0db87c01a6283968"
FROZEN_CORE_DATASET_FINGERPRINTS: dict[str, str] = {
    "U": "4ebb97bd74412d566137d95a6fc3ffd8f374f1cf8cfe204d007848e7a668f9b5",
    "T": "53ec8f468a43ede9b1536fabc0390af2fa33ab4312b23ce4d864f186a4651f78",
    "C": "242a6971cc801eae621b1fc4d966db2cd0af9cc866805f36a6fc5d0058dfbb74",
    "B": "8bde1f6f1c8bc59dcb49828fb5b9d65c0b63e06d92b2b9f15b37159e923009ea",
}

POLICY_HEALTH_FEATURES = CORE_HEALTH_FEATURES
TAU_SAFE = CORE_TAU_SAFE
TAU_HARMFUL = CORE_TAU_HARMFUL

_FROZEN_SPLIT_IDENTITY: dict[str, object] = {
    "overlap_component_count": 806,
    "physical_window_count": 1342,
    "fit_component_count": 685,
    "calibration_component_count": 121,
    "fit_physical_window_count": 1000,
    "calibration_physical_window_count": 342,
    "fit_row_count": 8046,
    "calibration_row_count": 2412,
    "fit_component_digest": "d40a2fc5d6a5e7ffdf269543639baac709df01e3438af19f94b0173e1c2766fe",
    "calibration_component_digest": (
        "f7b324d250a90a135ca7f1aa70d18c8c06faa1dd0373bd262c3ea0d4ade8da50"
    ),
    "fit_physical_window_digest": (
        "ba7f326285166687f29487540178503862d80d33eee6d824d7e041a32b1463a0"
    ),
    "calibration_physical_window_digest": (
        "4f60c17fe056c8d660c8aedea78fd187dfb317784d68423cec0f4eb4adf98087"
    ),
}

_FROZEN_STRATEGY_C_DIAGNOSTICS: dict[str, object] = {
    "calibration_support_by_domain": {
        "B": {"SAFE": 0, "HARMFUL": 313},
        "C": {"SAFE": 1, "HARMFUL": 452},
        "T": {"SAFE": 227, "HARMFUL": 1121},
        "U": {"SAFE": 0, "HARMFUL": 39},
    },
    "observed_tau_safe": TAU_SAFE,
    "observed_tau_harmful": TAU_HARMFUL,
    "safe_adjacent_boundary": 0.9018519135061516,
    "harmful_adjacent_boundary": 0.9962047794362529,
    "safe_error_allowance": 15,
    "harmful_error_allowance": 11,
    "determining_safe_domain": "B",
    "determining_harmful_domain": "T",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


POLICY_HEALTH_FEATURE_CONTRACT_DIGEST = _canonical_digest(list(POLICY_HEALTH_FEATURES))


class PredictedHealthState(StrEnum):
    """Policy-visible health state; deliberately distinct from evaluator truth."""

    SAFE = "PREDICTED_SAFE"
    UNCERTAIN = "PREDICTED_UNCERTAIN"
    HARMFUL = "PREDICTED_HARMFUL"


@dataclass(frozen=True, slots=True)
class PolicyHealthVector:
    """Exact ordered policy health vector with no evaluator or domain metadata."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(POLICY_HEALTH_FEATURES):
            raise ValueError(f"policy health vector requires {len(POLICY_HEALTH_FEATURES)} values")
        if any(math.isinf(value) for value in self.values):
            raise ValueError("policy health vector contains an infinite value")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> PolicyHealthVector:
        """Build the canonical vector after enforcing an exact feature allowlist."""

        expected = set(POLICY_HEALTH_FEATURES)
        observed = set(raw)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"policy health feature contract differs: missing={missing}, unexpected={extra}"
            )
        values: list[float] = []
        for name in POLICY_HEALTH_FEATURES:
            raw_value = raw[name]
            if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
                raw_value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"policy health feature {name} is not numeric")
            values.append(float(raw_value))
        return cls(tuple(values))

    @classmethod
    def from_ordered(cls, columns: Sequence[str], values: Sequence[float]) -> PolicyHealthVector:
        """Build from an already ordered vector, rejecting any order drift."""

        if tuple(columns) != POLICY_HEALTH_FEATURES:
            raise ValueError("ordered policy health feature contract differs")
        return cls(tuple(float(value) for value in values))

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.values], columns=list(POLICY_HEALTH_FEATURES))

    def as_mapping(self) -> dict[str, float]:
        """Return insertion-ordered canonical features for persistence/logging."""

        return dict(zip(POLICY_HEALTH_FEATURES, self.values, strict=True))


@dataclass(frozen=True, slots=True)
class HealthDecision:
    harm_probability: float
    predicted_state: PredictedHealthState


@dataclass(frozen=True, slots=True)
class FrozenHealthModel:
    """Validated deployed health model and its immutable identity."""

    model: Pipeline
    artifact_identity: str
    serialized_model_sha256: str
    manifest: Mapping[str, Any]

    def score(self, vector: PolicyHealthVector) -> float:
        probability = float(self.model.predict_proba(vector.as_frame())[0, 1])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("health model returned an invalid harm probability")
        return probability

    def decide(self, vector: PolicyHealthVector) -> HealthDecision:
        probability = self.score(vector)
        return HealthDecision(probability, classify_harm_probability(probability))


def classify_harm_probability(probability: float) -> PredictedHealthState:
    """Apply frozen Strategy-C thresholds with inclusive boundary semantics."""

    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("harm probability must be finite and lie in [0, 1]")
    if probability <= TAU_SAFE:
        return PredictedHealthState.SAFE
    if probability >= TAU_HARMFUL:
        return PredictedHealthState.HARMFUL
    return PredictedHealthState.UNCERTAIN


@dataclass(frozen=True, slots=True)
class _PreparedSplit:
    frame: pd.DataFrame
    fit_indices: NDArray[np.int64]
    calibration_indices: NDArray[np.int64]
    provenance: dict[str, Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_lines(values: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_git_sha(value: object) -> bool:
    if not isinstance(value, str) or len(value) not in (40, 64):
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_fingerprints(fingerprints: Mapping[str, str]) -> dict[str, str]:
    if set(fingerprints) != {"U", "T", "C", "B"}:
        raise ValueError("dataset fingerprints must contain exactly U, T, C, and B")
    result = {domain: str(fingerprints[domain]) for domain in ("U", "T", "C", "B")}
    if not all(_is_sha256(value) for value in result.values()):
        raise ValueError("every dataset fingerprint must be a SHA-256 string")
    if len(set(result.values())) != 4:
        raise ValueError("core dataset fingerprints must be distinct")
    return result


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"Study-3 canonical dataset does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        try:
            header = next(csv.reader(handle))
        except StopIteration as error:
            raise ValueError("Study-3 canonical dataset is empty") from error
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise ValueError(f"Study-3 canonical dataset has duplicate columns: {duplicates}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Study-3 canonical dataset is empty")
    return frame


def _physical_window_key(row: Any, fingerprints: Mapping[str, str]) -> str:
    domain = str(row.current_domain)
    if domain not in fingerprints:
        raise ValueError(f"health row has unknown current domain: {domain}")
    payload = {
        "dataset_fingerprint": fingerprints[domain],
        "partition_kind": str(row.partition_kind),
        "row_start": int(row.row_start),
        "row_stop": int(row.row_stop),
        "window_id": int(row.window_id),
    }
    return _canonical_digest(payload)


def _overlap_components(unique: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}

    def finish(keys: list[str]) -> None:
        if not keys:
            return
        component_digest = _digest_lines(set(keys))
        result.update({key: component_digest for key in keys})

    for _, domain_frame in unique.groupby("current_domain", sort=True):
        ordered = domain_frame.sort_values(
            ["row_start", "row_stop", "physical_window_key"], kind="mergesort"
        )
        component: list[str] = []
        component_stop: int | None = None
        records = ordered[["row_start", "row_stop", "physical_window_key"]]
        for start_raw, stop_raw, key_raw in records.itertuples(index=False, name=None):
            start, stop = int(start_raw), int(stop_raw)
            key = str(key_raw)
            if component_stop is None or start >= component_stop:
                finish(component)
                component = [key]
                component_stop = stop
            else:
                component.append(key)
                component_stop = max(component_stop, stop)
        finish(component)
    return result


def _partition_summary(part: pd.DataFrame) -> dict[str, Any]:
    return {
        "overlap_component_count": int(part["overlap_component"].nunique()),
        "physical_window_count": int(part["physical_window_key"].nunique()),
        "row_count": len(part),
        "confident_row_count": int(part["health_state"].isin(("SAFE", "HARMFUL")).sum()),
        "state_counts": {
            state: int((part["health_state"] == state).sum())
            for state in ("SAFE", "HARMFUL", "UNCERTAIN")
        },
        "domain_counts": {
            str(domain): int(count)
            for domain, count in (
                part["current_domain"].value_counts(sort=False).sort_index().items()
            )
        },
    }


def _prepare_overlap_component_split(
    source: pd.DataFrame, fingerprints: Mapping[str, str]
) -> _PreparedSplit:
    required = {
        "current_domain",
        "partition_kind",
        "row_start",
        "row_stop",
        "window_id",
        "health_state",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"Study-3 canonical dataset lacks grouping columns: {missing}")
    if set(source["current_domain"].astype(str)) != {"U", "T", "C", "B"}:
        raise ValueError("Study-3 canonical dataset must contain exactly the four core domains")
    states = set(source["health_state"].astype(str))
    if not states.issubset({"SAFE", "HARMFUL", "UNCERTAIN"}) or not {
        "SAFE",
        "HARMFUL",
    }.issubset(states):
        raise ValueError("Study-3 health states are invalid or lack a confident class")
    if not set(source["partition_kind"].astype(str)).issubset({"validation", "online_stream"}):
        raise ValueError("Study-3 health rows contain an unsupported partition role")
    frame = source.copy()
    for name in ("row_start", "row_stop", "window_id"):
        values = pd.to_numeric(frame[name], errors="raise")
        if values.isna().any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Study-3 grouping column {name} must contain integers")
        frame[name] = values.astype(np.int64)
    if (frame["row_start"] < 0).any() or (frame["row_stop"] <= frame["row_start"]).any():
        raise ValueError("Study-3 health rows contain invalid raw-row intervals")
    if (frame["window_id"] < 0).any():
        raise ValueError("Study-3 health rows contain a negative window ID")
    frame["current_domain"] = frame["current_domain"].astype(str)
    frame["partition_kind"] = frame["partition_kind"].astype(str)
    frame["health_state"] = frame["health_state"].astype(str)
    frame["physical_window_key"] = [
        _physical_window_key(row, fingerprints) for row in frame.itertuples(index=False)
    ]
    identity_columns = [
        "current_domain",
        "partition_kind",
        "row_start",
        "row_stop",
        "window_id",
    ]
    maximum_identity_variants = int(
        frame.groupby("physical_window_key")[identity_columns].nunique().to_numpy().max()
    )
    if maximum_identity_variants != 1:
        raise ValueError("physical-window key has inconsistent grouping metadata")
    unique = frame.drop_duplicates("physical_window_key")[
        ["physical_window_key", "current_domain", "row_start", "row_stop"]
    ].reset_index(drop=True)
    component_map = _overlap_components(unique)
    frame["overlap_component"] = frame["physical_window_key"].map(component_map)
    if frame["overlap_component"].isna().any():
        raise RuntimeError("failed to assign an overlap component")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        fit_raw, calibration_raw = next(
            splitter.split(
                frame,
                frame["current_domain"] + "|" + frame["health_state"],
                groups=frame["overlap_component"],
            )
        )
    except ValueError as error:
        raise ValueError("Study-3 dataset cannot satisfy the frozen grouped split") from error
    fit_indices = np.asarray(fit_raw, dtype=np.int64)
    calibration_indices = np.asarray(calibration_raw, dtype=np.int64)
    fit = frame.iloc[fit_indices]
    calibration = frame.iloc[calibration_indices]
    fit_components = set(fit["overlap_component"].astype(str))
    calibration_components = set(calibration["overlap_component"].astype(str))
    if fit_components.intersection(calibration_components):
        raise RuntimeError("fit and calibration overlap components are not disjoint")
    fit_windows = set(fit["physical_window_key"].astype(str))
    calibration_windows = set(calibration["physical_window_key"].astype(str))
    if fit_windows.intersection(calibration_windows):
        raise RuntimeError("fit and calibration physical windows are not disjoint")
    component_sizes = (
        unique.assign(overlap_component=unique["physical_window_key"].map(component_map))
        .groupby("overlap_component")
        .size()
    )
    provenance = {
        "version": FIT_CALIBRATION_SPLIT_VERSION,
        "physical_window_key_version": PHYSICAL_WINDOW_KEY_VERSION,
        "overlap_grouping_version": OVERLAP_GROUPING_VERSION,
        "overlap_definition": (
            "transitive connected components of same-dataset half-open raw-row "
            "intervals with positive overlap"
        ),
        "splitter": {
            "class": "StratifiedGroupKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "selected_fold_index": 0,
            "strata": "current_domain|health_state",
            "uncertain_role": "split_assignment_only_excluded_from_fit_and_calibration_labels",
        },
        "overlap_component_count": int(frame["overlap_component"].nunique()),
        "physical_window_count": int(frame["physical_window_key"].nunique()),
        "component_physical_window_size_distribution": {
            str(size): int(count)
            for size, count in component_sizes.value_counts().sort_index().items()
        },
        "fit": _partition_summary(fit),
        "calibration": _partition_summary(calibration),
        "fit_component_digest": _digest_lines(fit_components),
        "calibration_component_digest": _digest_lines(calibration_components),
        "fit_physical_window_digest": _digest_lines(fit_windows),
        "calibration_physical_window_digest": _digest_lines(calibration_windows),
        "cross_split_component_overlap_count": 0,
        "cross_split_physical_window_overlap_count": 0,
        "cross_split_raw_interval_overlap_count": 0,
    }
    return _PreparedSplit(frame, fit_indices, calibration_indices, provenance)


def _tie_aware_threshold(
    scores: NDArray[np.float64], error_rate: float, *, false_safe: bool
) -> tuple[float, float, int]:
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    if not false_safe:
        ordered = ordered[::-1]
    allowance = math.floor(error_rate * len(ordered))
    if not len(ordered) or allowance >= len(ordered):
        raise ValueError("threshold derivation lacks supported calibration scores")
    boundary = float(ordered[allowance])
    direction = -np.inf if false_safe else np.inf
    return float(np.nextafter(boundary, direction)), boundary, allowance


def _strategy_c_diagnostics(
    calibration: pd.DataFrame, probabilities: NDArray[np.float64]
) -> dict[str, Any]:
    work = calibration.assign(_harm_probability=np.asarray(probabilities, dtype=np.float64))
    supported: dict[str, dict[str, int]] = {}
    safe_candidates: list[tuple[str, float, float, int]] = []
    harmful_candidates: list[tuple[str, float, float, int]] = []
    for domain, part in work.groupby("current_domain", sort=True):
        harmful = part.loc[part["health_state"] == "HARMFUL", "_harm_probability"].to_numpy()
        safe = part.loc[part["health_state"] == "SAFE", "_harm_probability"].to_numpy()
        supported[str(domain)] = {"SAFE": len(safe), "HARMFUL": len(harmful)}
        if len(harmful):
            threshold, boundary, allowance = _tie_aware_threshold(harmful, 0.05, false_safe=True)
            safe_candidates.append((str(domain), threshold, boundary, allowance))
        if len(safe):
            threshold, boundary, allowance = _tie_aware_threshold(safe, 0.05, false_safe=False)
            harmful_candidates.append((str(domain), threshold, boundary, allowance))
    if not safe_candidates or not harmful_candidates:
        raise ValueError("Strategy-C derivation lacks supported SAFE or HARMFUL calibration data")
    safe_choice = min(safe_candidates, key=lambda item: item[1])
    harmful_choice = max(harmful_candidates, key=lambda item: item[1])
    return {
        "calibration_support_by_domain": supported,
        "observed_tau_safe": safe_choice[1],
        "observed_tau_harmful": harmful_choice[1],
        "safe_adjacent_boundary": safe_choice[2],
        "harmful_adjacent_boundary": harmful_choice[2],
        "safe_error_allowance": safe_choice[3],
        "harmful_error_allowance": harmful_choice[3],
        "determining_safe_domain": safe_choice[0],
        "determining_harmful_domain": harmful_choice[0],
    }


def _software_provenance() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "platform": platform.platform(),
    }


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or not _is_git_sha(value):
        raise RuntimeError("cannot resolve Git commit for health-model provenance")
    return value


def _model_contract() -> dict[str, Any]:
    return {
        "pipeline_steps": ["imputer", "scaler", "predictor"],
        "imputer": {
            "class": "SimpleImputer",
            "strategy": "median",
            "add_indicator": True,
            "keep_empty_features": True,
        },
        "scaler": {"class": "StandardScaler"},
        "predictor": {
            "class": "GradientBoostingClassifier",
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 2,
            "random_state": 42,
        },
        "feature_set": "combined_unlabelled",
        "fit_labels": ["SAFE", "HARMFUL"],
        "excluded_fit_label": "UNCERTAIN",
    }


def _calibration_contract(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "strategy": "C_worst_domain_05",
        "status": "closed_benchmark_heuristic",
        "error_rate": 0.05,
        "tau_safe": TAU_SAFE,
        "tau_harmful": TAU_HARMFUL,
        "safe_semantics": "PREDICTED_SAFE iff p_harm <= tau_safe",
        "harmful_semantics": "PREDICTED_HARMFUL iff p_harm >= tau_harmful",
        "uncertain_semantics": "PREDICTED_UNCERTAIN otherwise",
        "tie_handling": "inclusive decisions; adjacent disallowed tie excluded with nextafter",
        "selection_reason": (
            "only tested direct threshold strategy satisfying the supported per-domain "
            "5% HARMFUL-to-SAFE criterion"
        ),
        "limitations": [
            "not a universal probability calibration guarantee",
            "SAFE calibration support is sparse or nonrepresentative for some domains",
            "unsupported domain-wise SAFE error rates are not claimed",
            "PREDICTED_SAFE is not certified operational safety",
        ],
        "derivation_diagnostics": dict(diagnostics),
    }


def build_health_model_artifact(
    study3_dataset_path: str | Path,
    output_dir: str | Path,
    *,
    dataset_fingerprints: Mapping[str, str] = FROZEN_CORE_DATASET_FINGERPRINTS,
    expected_dataset_sha256: str = FROZEN_STUDY3_DATASET_SHA256,
) -> Path:
    """Fit and persist the frozen health model without refitting after calibration."""

    dataset_path = Path(study3_dataset_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"health-model artifact path already exists: {output}")
    fingerprints = _validate_fingerprints(dataset_fingerprints)
    if not _is_sha256(expected_dataset_sha256):
        raise ValueError("expected Study-3 canonical dataset digest is invalid")
    dataset_sha256 = _file_sha256(dataset_path)
    if dataset_sha256 != expected_dataset_sha256:
        raise ValueError("Study-3 canonical dataset differs from the expected frozen digest")
    frame = _load_dataset(dataset_path)
    resolved_features = feature_columns(frame, "combined_unlabelled")
    if resolved_features != POLICY_HEALTH_FEATURES:
        raise ValueError(
            "Study-3 combined_unlabelled columns differ from the frozen ordered "
            "Study-4 policy contract"
        )
    matrix = frame.loc[:, list(POLICY_HEALTH_FEATURES)].to_numpy(dtype=np.float64)
    if np.isinf(matrix).any():
        raise ValueError("Study-3 health feature matrix contains infinity")
    prepared = _prepare_overlap_component_split(frame, fingerprints)
    fit = prepared.frame.iloc[prepared.fit_indices]
    fit = fit.loc[fit["health_state"].isin(("SAFE", "HARMFUL"))]
    calibration = prepared.frame.iloc[prepared.calibration_indices]
    calibration = calibration.loc[calibration["health_state"].isin(("SAFE", "HARMFUL"))]
    labels = (fit["health_state"] == "HARMFUL").to_numpy(dtype=np.int8)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("health-model fit partition must contain SAFE and HARMFUL rows")
    config = HealthPredictorConfig()
    config.validate()
    model = make_predictor("gradient_boosting", config, seed=42)
    model.fit(fit.loc[:, list(POLICY_HEALTH_FEATURES)], labels)
    calibration_probabilities = np.asarray(
        model.predict_proba(calibration.loc[:, list(POLICY_HEALTH_FEATURES)])[:, 1],
        dtype=np.float64,
    )
    diagnostics = _strategy_c_diagnostics(calibration, calibration_probabilities)
    if expected_dataset_sha256 == FROZEN_STUDY3_DATASET_SHA256 and (
        diagnostics["observed_tau_safe"] != TAU_SAFE
        or diagnostics["observed_tau_harmful"] != TAU_HARMFUL
    ):
        raise ValueError("frozen Study-3 dataset no longer reproduces Strategy-C thresholds")
    model_bytes = pickle.dumps(model, protocol=5)
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    manifest: dict[str, Any] = {
        "version": HEALTH_MODEL_ARTIFACT_VERSION,
        "code_commit_sha": _git_commit(),
        "dataset": {
            "canonical_csv_filename": dataset_path.name,
            "canonical_csv_sha256": dataset_sha256,
            "expected_canonical_csv_sha256": expected_dataset_sha256,
            "row_count": len(frame),
            "column_count": len(frame.columns),
            "dataset_fingerprints": fingerprints,
        },
        "feature_contract": {
            "name": "task005_combined_unlabelled",
            "ordered_features": list(POLICY_HEALTH_FEATURES),
            "feature_count": len(POLICY_HEALTH_FEATURES),
            "sha256": POLICY_HEALTH_FEATURE_CONTRACT_DIGEST,
        },
        "split": prepared.provenance,
        "model": {
            "filename": HEALTH_MODEL_FILENAME,
            "serialization": "python_pickle_protocol_5",
            "serialized_sha256": model_sha256,
            "serialized_size_bytes": len(model_bytes),
            "contract": _model_contract(),
            "fit_row_count": len(fit),
            "fit_state_counts": {
                "SAFE": int((fit["health_state"] == "SAFE").sum()),
                "HARMFUL": int((fit["health_state"] == "HARMFUL").sum()),
            },
            "refit_after_calibration": False,
        },
        "calibration": _calibration_contract(diagnostics),
        "software": _software_provenance(),
    }
    manifest["artifact_identity_sha256"] = _canonical_digest(manifest)
    output.mkdir(parents=True, exist_ok=False)
    with (output / HEALTH_MODEL_FILENAME).open("xb") as handle:
        handle.write(model_bytes)
    with (output / HEALTH_MODEL_MANIFEST_FILENAME).open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    validate_health_model_artifact(
        output,
        study3_dataset_path=dataset_path,
        expected_dataset_sha256=expected_dataset_sha256,
    )
    return output


def _require_exact_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    observed = set(raw)
    if observed != expected:
        raise ValueError(
            f"{context} keys differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _as_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _validate_partition_summary(raw: object, context: str) -> Mapping[str, Any]:
    part = _as_mapping(raw, context)
    _require_exact_keys(
        part,
        {
            "overlap_component_count",
            "physical_window_count",
            "row_count",
            "confident_row_count",
            "state_counts",
            "domain_counts",
        },
        context,
    )
    states = _as_mapping(part["state_counts"], f"{context} state counts")
    domains = _as_mapping(part["domain_counts"], f"{context} domain counts")
    _require_exact_keys(states, {"SAFE", "HARMFUL", "UNCERTAIN"}, f"{context} states")
    _require_exact_keys(domains, {"U", "T", "C", "B"}, f"{context} domains")
    counts = [part["overlap_component_count"], part["physical_window_count"], part["row_count"]]
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in counts):
        raise ValueError(f"{context} contains an invalid positive count")
    state_values = list(states.values())
    domain_values = list(domains.values())
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (*state_values, *domain_values)
    ):
        raise ValueError(f"{context} contains an invalid class/domain count")
    if sum(state_values) != part["row_count"] or sum(domain_values) != part["row_count"]:
        raise ValueError(f"{context} row-count accounting differs")
    if part["confident_row_count"] != states["SAFE"] + states["HARMFUL"]:
        raise ValueError(f"{context} confident-row accounting differs")
    return part


def _validate_split_contract(raw: object, *, dataset_row_count: int, frozen: bool) -> None:
    split = _as_mapping(raw, "fit/calibration split")
    _require_exact_keys(
        split,
        {
            "version",
            "physical_window_key_version",
            "overlap_grouping_version",
            "overlap_definition",
            "splitter",
            "overlap_component_count",
            "physical_window_count",
            "component_physical_window_size_distribution",
            "fit",
            "calibration",
            "fit_component_digest",
            "calibration_component_digest",
            "fit_physical_window_digest",
            "calibration_physical_window_digest",
            "cross_split_component_overlap_count",
            "cross_split_physical_window_overlap_count",
            "cross_split_raw_interval_overlap_count",
        },
        "fit/calibration split",
    )
    if (
        split["version"] != FIT_CALIBRATION_SPLIT_VERSION
        or split["physical_window_key_version"] != PHYSICAL_WINDOW_KEY_VERSION
        or split["overlap_grouping_version"] != OVERLAP_GROUPING_VERSION
        or split["overlap_definition"]
        != (
            "transitive connected components of same-dataset half-open raw-row "
            "intervals with positive overlap"
        )
    ):
        raise ValueError("fit/calibration grouping contract differs")
    expected_splitter = {
        "class": "StratifiedGroupKFold",
        "n_splits": 5,
        "shuffle": True,
        "random_state": 42,
        "selected_fold_index": 0,
        "strata": "current_domain|health_state",
        "uncertain_role": "split_assignment_only_excluded_from_fit_and_calibration_labels",
    }
    if split["splitter"] != expected_splitter:
        raise ValueError("fit/calibration splitter contract differs")
    fit = _validate_partition_summary(split["fit"], "fit partition")
    calibration = _validate_partition_summary(split["calibration"], "calibration partition")
    if (
        fit["row_count"] + calibration["row_count"] != dataset_row_count
        or fit["overlap_component_count"] + calibration["overlap_component_count"]
        != split["overlap_component_count"]
        or fit["physical_window_count"] + calibration["physical_window_count"]
        != split["physical_window_count"]
    ):
        raise ValueError("fit/calibration split count accounting differs")
    if any(
        split[name] != 0
        for name in (
            "cross_split_component_overlap_count",
            "cross_split_physical_window_overlap_count",
            "cross_split_raw_interval_overlap_count",
        )
    ):
        raise ValueError("fit/calibration split contains leakage overlap")
    for name in (
        "fit_component_digest",
        "calibration_component_digest",
        "fit_physical_window_digest",
        "calibration_physical_window_digest",
    ):
        if not _is_sha256(split[name]):
            raise ValueError(f"fit/calibration split has invalid {name}")
    distribution = _as_mapping(
        split["component_physical_window_size_distribution"], "component-size distribution"
    )
    try:
        size_counts = {int(size): int(count) for size, count in distribution.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("component-size distribution is invalid") from error
    if any(size <= 0 or count <= 0 for size, count in size_counts.items()):
        raise ValueError("component-size distribution contains non-positive values")
    if (
        sum(size_counts.values()) != split["overlap_component_count"]
        or sum(size * count for size, count in size_counts.items())
        != split["physical_window_count"]
    ):
        raise ValueError("component-size distribution accounting differs")
    if frozen:
        observed_identity = {
            "overlap_component_count": split["overlap_component_count"],
            "physical_window_count": split["physical_window_count"],
            "fit_component_count": fit["overlap_component_count"],
            "calibration_component_count": calibration["overlap_component_count"],
            "fit_physical_window_count": fit["physical_window_count"],
            "calibration_physical_window_count": calibration["physical_window_count"],
            "fit_row_count": fit["row_count"],
            "calibration_row_count": calibration["row_count"],
            "fit_component_digest": split["fit_component_digest"],
            "calibration_component_digest": split["calibration_component_digest"],
            "fit_physical_window_digest": split["fit_physical_window_digest"],
            "calibration_physical_window_digest": split["calibration_physical_window_digest"],
        }
        if observed_identity != _FROZEN_SPLIT_IDENTITY:
            raise ValueError("frozen fit/calibration split identity differs")


def _validate_model(model: object) -> Pipeline:
    if not isinstance(model, Pipeline) or [name for name, _ in model.steps] != [
        "imputer",
        "scaler",
        "predictor",
    ]:
        raise ValueError("serialized health model is not the frozen pipeline")
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    predictor = model.named_steps["predictor"]
    if not isinstance(imputer, SimpleImputer) or (
        imputer.strategy,
        imputer.add_indicator,
        imputer.keep_empty_features,
    ) != ("median", True, True):
        raise ValueError("serialized health-model imputer differs from the frozen contract")
    if not isinstance(scaler, StandardScaler):
        raise ValueError("serialized health-model scaler differs from the frozen contract")
    if not isinstance(predictor, GradientBoostingClassifier) or (
        predictor.n_estimators,
        predictor.learning_rate,
        predictor.max_depth,
        predictor.random_state,
    ) != (100, 0.05, 2, 42):
        raise ValueError("serialized health predictor differs from the frozen GB contract")
    if tuple(str(value) for value in getattr(model, "feature_names_in_", ())) != (
        POLICY_HEALTH_FEATURES
    ):
        raise ValueError("serialized health model feature order differs")
    if int(getattr(model, "n_features_in_", -1)) != len(POLICY_HEALTH_FEATURES):
        raise ValueError("serialized health model feature count differs")
    if not np.array_equal(getattr(predictor, "classes_", np.empty(0)), np.asarray([0, 1])):
        raise ValueError("serialized health model class contract differs")
    return model


def validate_health_model_artifact(
    artifact_dir: str | Path,
    *,
    study3_dataset_path: str | Path | None = None,
    expected_dataset_sha256: str = FROZEN_STUDY3_DATASET_SHA256,
    require_software_match: bool = True,
) -> FrozenHealthModel:
    """Strictly validate provenance and return the only policy-facing model handle."""

    root = Path(artifact_dir).resolve()
    expected_files = {HEALTH_MODEL_FILENAME, HEALTH_MODEL_MANIFEST_FILENAME}
    if not root.is_dir() or {path.name for path in root.iterdir()} != expected_files:
        raise ValueError("health-model artifact directory is incomplete or contains extra files")
    manifest_path = root / HEALTH_MODEL_MANIFEST_FILENAME
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("health-model manifest is invalid JSON") from error
    manifest = _as_mapping(raw_manifest, "health-model manifest")
    _require_exact_keys(
        manifest,
        {
            "version",
            "code_commit_sha",
            "dataset",
            "feature_contract",
            "split",
            "model",
            "calibration",
            "software",
            "artifact_identity_sha256",
        },
        "health-model manifest",
    )
    if manifest["version"] != HEALTH_MODEL_ARTIFACT_VERSION:
        raise ValueError("unsupported health-model artifact version")
    if not _is_git_sha(manifest["code_commit_sha"]):
        raise ValueError("health-model Git commit provenance is invalid")
    identity = manifest["artifact_identity_sha256"]
    without_identity = dict(manifest)
    del without_identity["artifact_identity_sha256"]
    if not _is_sha256(identity) or identity != _canonical_digest(without_identity):
        raise ValueError("health-model artifact identity digest differs")
    dataset = _as_mapping(manifest["dataset"], "health-model dataset provenance")
    _require_exact_keys(
        dataset,
        {
            "canonical_csv_filename",
            "canonical_csv_sha256",
            "expected_canonical_csv_sha256",
            "row_count",
            "column_count",
            "dataset_fingerprints",
        },
        "health-model dataset provenance",
    )
    if (
        not _is_sha256(expected_dataset_sha256)
        or not _is_sha256(dataset["canonical_csv_sha256"])
        or dataset["expected_canonical_csv_sha256"] != expected_dataset_sha256
        or dataset["canonical_csv_sha256"] != expected_dataset_sha256
    ):
        raise ValueError("health-model canonical dataset digest is invalid")
    fingerprints = _validate_fingerprints(
        _as_mapping(dataset["dataset_fingerprints"], "dataset fingerprints")
    )
    if expected_dataset_sha256 == FROZEN_STUDY3_DATASET_SHA256 and (
        fingerprints != FROZEN_CORE_DATASET_FINGERPRINTS
    ):
        raise ValueError("frozen core dataset fingerprints differ")
    if (
        not isinstance(dataset["row_count"], int)
        or isinstance(dataset["row_count"], bool)
        or dataset["row_count"] <= 0
        or not isinstance(dataset["column_count"], int)
        or isinstance(dataset["column_count"], bool)
        or dataset["column_count"] <= 0
        or not isinstance(dataset["canonical_csv_filename"], str)
        or not dataset["canonical_csv_filename"]
    ):
        raise ValueError("health-model canonical dataset shape/name provenance is invalid")
    features = _as_mapping(manifest["feature_contract"], "health feature contract")
    _require_exact_keys(
        features,
        {"name", "ordered_features", "feature_count", "sha256"},
        "health feature contract",
    )
    if (
        features["name"] != "task005_combined_unlabelled"
        or features["ordered_features"] != list(POLICY_HEALTH_FEATURES)
        or features["feature_count"] != len(POLICY_HEALTH_FEATURES)
        or features["sha256"] != POLICY_HEALTH_FEATURE_CONTRACT_DIGEST
    ):
        raise ValueError("health feature contract differs from frozen Study-4 allowlist")
    _validate_split_contract(
        manifest["split"],
        dataset_row_count=int(dataset["row_count"]),
        frozen=expected_dataset_sha256 == FROZEN_STUDY3_DATASET_SHA256,
    )
    model_manifest = _as_mapping(manifest["model"], "health model provenance")
    _require_exact_keys(
        model_manifest,
        {
            "filename",
            "serialization",
            "serialized_sha256",
            "serialized_size_bytes",
            "contract",
            "fit_row_count",
            "fit_state_counts",
            "refit_after_calibration",
        },
        "health model provenance",
    )
    if (
        model_manifest["filename"] != HEALTH_MODEL_FILENAME
        or model_manifest["serialization"] != "python_pickle_protocol_5"
        or model_manifest["contract"] != _model_contract()
        or model_manifest["refit_after_calibration"] is not False
        or not _is_sha256(model_manifest["serialized_sha256"])
    ):
        raise ValueError("health model provenance differs from the frozen contract")
    fit_state_counts = _as_mapping(model_manifest["fit_state_counts"], "model fit states")
    _require_exact_keys(fit_state_counts, {"SAFE", "HARMFUL"}, "model fit states")
    if (
        not isinstance(model_manifest["fit_row_count"], int)
        or isinstance(model_manifest["fit_row_count"], bool)
        or model_manifest["fit_row_count"] <= 0
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in fit_state_counts.values()
        )
        or sum(fit_state_counts.values()) != model_manifest["fit_row_count"]
    ):
        raise ValueError("health model fit-state accounting differs")
    split_fit = _as_mapping(
        _as_mapping(manifest["split"], "fit/calibration split")["fit"], "fit partition"
    )
    split_fit_states = _as_mapping(split_fit["state_counts"], "fit partition states")
    if (
        model_manifest["fit_row_count"] != split_fit["confident_row_count"]
        or fit_state_counts["SAFE"] != split_fit_states["SAFE"]
        or fit_state_counts["HARMFUL"] != split_fit_states["HARMFUL"]
    ):
        raise ValueError("health model fit support differs from split provenance")
    if (
        not isinstance(model_manifest["serialized_size_bytes"], int)
        or isinstance(model_manifest["serialized_size_bytes"], bool)
        or model_manifest["serialized_size_bytes"] <= 0
    ):
        raise ValueError("serialized health model size provenance is invalid")
    model_path = root / HEALTH_MODEL_FILENAME
    model_bytes = model_path.read_bytes()
    if len(model_bytes) != model_manifest["serialized_size_bytes"]:
        raise ValueError("serialized health model size differs")
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    if model_sha256 != model_manifest["serialized_sha256"]:
        raise ValueError("serialized health model digest differs")
    calibration = _as_mapping(manifest["calibration"], "health calibration")
    _require_exact_keys(
        calibration,
        {
            "strategy",
            "status",
            "error_rate",
            "tau_safe",
            "tau_harmful",
            "safe_semantics",
            "harmful_semantics",
            "uncertain_semantics",
            "tie_handling",
            "selection_reason",
            "limitations",
            "derivation_diagnostics",
        },
        "health calibration",
    )
    expected_calibration = _calibration_contract(
        _as_mapping(calibration["derivation_diagnostics"], "threshold diagnostics")
    )
    if calibration != expected_calibration:
        raise ValueError("health calibration differs from frozen Strategy C")
    if expected_dataset_sha256 == FROZEN_STUDY3_DATASET_SHA256 and (
        calibration["derivation_diagnostics"] != _FROZEN_STRATEGY_C_DIAGNOSTICS
    ):
        raise ValueError("frozen Strategy-C derivation diagnostics differ")
    software = _as_mapping(manifest["software"], "software provenance")
    _require_exact_keys(
        software,
        {"python", "python_major_minor", "numpy", "pandas", "scikit_learn", "platform"},
        "software provenance",
    )
    if not all(isinstance(value, str) and value for value in software.values()):
        raise ValueError("software provenance contains an empty version")
    if require_software_match:
        current = _software_provenance()
        for name in ("python_major_minor", "numpy", "pandas", "scikit_learn"):
            if software[name] != current[name]:
                raise ValueError(f"health-model software version differs for {name}")
    try:
        raw_model = pickle.loads(model_bytes)
    except Exception as error:
        raise ValueError("serialized health model cannot be loaded") from error
    model = _validate_model(raw_model)
    if study3_dataset_path is not None:
        dataset_path = Path(study3_dataset_path).resolve()
        if dataset_path.name != dataset["canonical_csv_filename"]:
            raise ValueError("canonical Study-3 dataset filename differs from provenance")
        if _file_sha256(dataset_path) != dataset["canonical_csv_sha256"]:
            raise ValueError("canonical Study-3 dataset digest differs from provenance")
        source = _load_dataset(dataset_path)
        if len(source) != dataset["row_count"] or len(source.columns) != dataset["column_count"]:
            raise ValueError("canonical Study-3 dataset shape differs from provenance")
        if feature_columns(source, "combined_unlabelled") != POLICY_HEALTH_FEATURES:
            raise ValueError("canonical Study-3 feature contract differs")
        prepared = _prepare_overlap_component_split(source, fingerprints)
        if prepared.provenance != manifest["split"]:
            raise ValueError("fit/calibration overlap-component provenance differs")
        fit = prepared.frame.iloc[prepared.fit_indices]
        fit = fit.loc[fit["health_state"].isin(("SAFE", "HARMFUL"))]
        expected_counts = {
            "SAFE": int((fit["health_state"] == "SAFE").sum()),
            "HARMFUL": int((fit["health_state"] == "HARMFUL").sum()),
        }
        if (
            len(fit) != model_manifest["fit_row_count"]
            or expected_counts != model_manifest["fit_state_counts"]
        ):
            raise ValueError("health-model fit support differs from provenance")
        calibration_frame = prepared.frame.iloc[prepared.calibration_indices]
        calibration_frame = calibration_frame.loc[
            calibration_frame["health_state"].isin(("SAFE", "HARMFUL"))
        ]
        probabilities = np.asarray(
            model.predict_proba(calibration_frame.loc[:, list(POLICY_HEALTH_FEATURES)])[:, 1],
            dtype=np.float64,
        )
        diagnostics = _strategy_c_diagnostics(calibration_frame, probabilities)
        if diagnostics != calibration["derivation_diagnostics"]:
            raise ValueError("persisted Strategy-C derivation differs from model recomputation")
    return FrozenHealthModel(model, str(identity), model_sha256, manifest)


__all__ = [
    "FROZEN_CORE_DATASET_FINGERPRINTS",
    "FROZEN_STUDY3_DATASET_SHA256",
    "HEALTH_MODEL_ARTIFACT_VERSION",
    "POLICY_HEALTH_FEATURES",
    "POLICY_HEALTH_FEATURE_CONTRACT_DIGEST",
    "TAU_HARMFUL",
    "TAU_SAFE",
    "FrozenHealthModel",
    "HealthDecision",
    "PolicyHealthVector",
    "PredictedHealthState",
    "build_health_model_artifact",
    "classify_harm_probability",
    "validate_health_model_artifact",
]
