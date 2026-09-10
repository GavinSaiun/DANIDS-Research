from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from danids.cli import build_parser
from danids.config.core import (
    CORE_HEALTH_FEATURES,
    CORE_TAU_HARMFUL,
    CORE_TAU_SAFE,
    load_study4_core_config,
)
from danids.config.experiment import ExperimentConfigError

CONFIG = Path("configs/experiments/task006_core_u-t-c-b.yaml")


def _mutated_config(tmp_path: Path, *keys: str, value: object) -> Path:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    target = raw
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    output = tmp_path / "mutated.yaml"
    output.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return output


def test_frozen_core_config_loads_exact_contract() -> None:
    config = load_study4_core_config(CONFIG)

    assert config.health.feature_contract == CORE_HEALTH_FEATURES
    assert config.health.tau_safe == CORE_TAU_SAFE
    assert config.health.tau_harmful == CORE_TAU_HARMFUL
    assert config.supervision.query_batch_size == 25
    assert config.supervision.maximum_query_events == 4
    assert config.allocation.replay_per_release == 20
    assert config.allocation.audit_per_release == 5
    assert config.actions.normal_escalation == ("A2_HEAD_UPDATE", "A4_REPLAY_UPDATE")
    assert config.actions.a3_core_selectable is False


@pytest.mark.parametrize(
    "command",
    [
        "validate-core-config-study4",
        "build-health-model-study4",
        "validate-health-model-study4",
    ],
)
def test_core_cli_commands_are_registered(command: str) -> None:
    help_text = build_parser().format_help()
    assert command in help_text


@pytest.mark.parametrize(
    ("keys", "value"),
    [
        (("health", "tau_safe"), 0.5),
        (("health", "feature_contract"), list(reversed(CORE_HEALTH_FEATURES))),
        (("supervision", "query_batch_size"), 20),
        (("allocation", "audit_per_release"), 4),
        (("actions", "a3_core_selectable"), True),
        (("information_boundary", "domain_identity_visible"), True),
    ],
)
def test_core_config_rejects_scientific_contract_changes(
    tmp_path: Path, keys: tuple[str, ...], value: object
) -> None:
    with pytest.raises(ExperimentConfigError):
        load_study4_core_config(_mutated_config(tmp_path, *keys, value=value))


def test_core_config_rejects_unexpected_keys(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["health"]["offline_health_state"] = "HARMFUL"
    output = tmp_path / "unexpected.yaml"
    output.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ExperimentConfigError, match="unexpected"):
        load_study4_core_config(output)


def test_core_config_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    output = tmp_path / "duplicate.yaml"
    output.write_text(
        CONFIG.read_text(encoding="utf-8") + "\nseed: 44\n",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentConfigError, match="duplicate YAML key"):
        load_study4_core_config(output)
