from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from danids.config.experiment import ExperimentConfig, load_experiment_config
from danids.data.registry import DatasetRegistry
from danids.data.schema import FeatureContract, discover_core_feature_contract


@pytest.fixture
def benchmark_files(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    permutation = [
        8,
        1,
        19,
        3,
        24,
        0,
        17,
        2,
        14,
        4,
        23,
        6,
        11,
        5,
        22,
        7,
        9,
        10,
        12,
        13,
        15,
        16,
        18,
        20,
        21,
    ]
    for offset, dataset_id in enumerate(("U", "T", "B", "C")):
        chronological: list[dict[str, Any]] = []
        for timestamp in range(25):
            chronological.append(
                {
                    "FLOW_START_MILLISECONDS": 1_000 + timestamp,
                    "IPV4_SRC_ADDR": f"10.{offset}.0.{timestamp}",
                    "IPV4_DST_ADDR": f"192.168.{offset}.{timestamp}",
                    "FLOW_ID": f"{dataset_id}-{timestamp}",
                    "F1": float(timestamp),
                    "F2": (
                        "bad"
                        if timestamp == 2
                        else float("inf")
                        if timestamp == 3
                        else float(timestamp * 2)
                    ),
                    f"{dataset_id}_ONLY": timestamp,
                    "Label": timestamp % 2,
                    "Attack": "Benign" if timestamp % 2 == 0 else f"native-{dataset_id}",
                }
            )
        frame = pd.DataFrame([chronological[index] for index in permutation])
        path = tmp_path / f"{dataset_id}.csv"
        frame.to_csv(path, index=False)
        paths[dataset_id] = path
    return paths


@pytest.fixture
def datasets_config(tmp_path: Path, benchmark_files: dict[str, Path]) -> Path:
    path = tmp_path / "datasets.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "datasets": {key: {"path": str(value)} for key, value in benchmark_files.items()},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def experiment_config(tmp_path: Path) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "E1_SYNTHETIC_B-U-C-T_s42",
                "study": "E1",
                "seed": 42,
                "datasets": {"sequence": ["B", "U", "C", "T"], "split_version": "test-v1"},
                "stream": {"window_size": 6, "boundary_mode": "task_free"},
                "splits": {
                    "initial": {"train": 0.6, "validation": 0.2, "holdout": 0.2},
                    "later": {"online": 0.8, "holdout": 0.2},
                },
                "operating_envelope": {"target_fpr": 0.001},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def registry(datasets_config: Path) -> DatasetRegistry:
    return DatasetRegistry.from_yaml(datasets_config)


@pytest.fixture
def contract(registry: DatasetRegistry) -> FeatureContract:
    return discover_core_feature_contract(registry)


@pytest.fixture
def experiment(experiment_config: Path) -> ExperimentConfig:
    return load_experiment_config(experiment_config)
