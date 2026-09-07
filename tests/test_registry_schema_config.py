from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from danids.config.experiment import ExperimentConfigError, load_experiment_config
from danids.data.registry import CORE_DATASET_IDS, DatasetConfigError, DatasetRegistry
from danids.data.schema import (
    SchemaError,
    discover_core_feature_contract,
    is_identifier_or_absolute_time,
)


def test_registry_defines_core_ids_and_resolves_paths(
    registry: DatasetRegistry, benchmark_files: dict[str, Path]
) -> None:
    assert tuple(registry) == CORE_DATASET_IDS
    assert registry["U"].name == "NF-UNSW-NB15-v3"
    assert registry["U"].path == benchmark_files["U"].resolve()


def test_environment_path_configuration(
    tmp_path: Path, benchmark_files: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_DANIDS_U", str(benchmark_files["U"]))
    config = tmp_path / "one.yaml"
    config.write_text("datasets:\n  U:\n    path_env: TEST_DANIDS_U\n", encoding="utf-8")
    assert DatasetRegistry.from_yaml(config)["U"].path == benchmark_files["U"].resolve()


def test_unresolved_environment_path_fails(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("datasets:\n  U:\n    path_env: DEFINITELY_NOT_SET\n", encoding="utf-8")
    with pytest.raises(DatasetConfigError, match="not set"):
        DatasetRegistry.from_yaml(config)


def test_common_contract_uses_only_shared_features_and_excludes_sensitive_columns(
    registry: DatasetRegistry,
) -> None:
    contract = discover_core_feature_contract(registry)
    assert contract.feature_columns == ("F1", "F2")
    forbidden = {
        "Label",
        "Attack",
        "FLOW_START_MILLISECONDS",
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
        "FLOW_ID",
    }
    assert forbidden.isdisjoint(contract.feature_columns)


def test_identifier_guard_does_not_drop_ordinary_ip_packet_features() -> None:
    assert is_identifier_or_absolute_time("SRC_IP")
    assert is_identifier_or_absolute_time("FLOW_END_MILLISECONDS")
    assert not is_identifier_or_absolute_time("MIN_IP_PKT_LEN")
    assert not is_identifier_or_absolute_time("FLOW_DURATION_MILLISECONDS")


def test_missing_required_schema_fails_loudly(
    tmp_path: Path, benchmark_files: dict[str, Path]
) -> None:
    broken = tmp_path / "broken.csv"
    broken.write_text("FLOW_START_MILLISECONDS,F1,Label,Attack\n1,2,0,Benign\n", encoding="utf-8")
    entries = {key: {"path": str(value)} for key, value in benchmark_files.items()}
    entries["T"] = {"path": str(broken)}
    config = tmp_path / "broken.yaml"
    config.write_text(yaml.safe_dump({"datasets": entries}), encoding="utf-8")
    with pytest.raises(SchemaError, match="missing required columns"):
        discover_core_feature_contract(DatasetRegistry.from_yaml(config))


def test_experiment_accepts_arbitrary_core_sequence(experiment_config: Path) -> None:
    config = load_experiment_config(experiment_config)
    assert config.sequence == ("B", "U", "C", "T")
    assert config.window_size == 6


def test_experiment_rejects_methodological_split_change(
    tmp_path: Path, experiment_config: Path
) -> None:
    raw = yaml.safe_load(experiment_config.read_text(encoding="utf-8"))
    raw["splits"]["initial"]["train"] = 0.5
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ExperimentConfigError, match="60/20/20"):
        load_experiment_config(changed)
