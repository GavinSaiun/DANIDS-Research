"""Execution-authorization tests for the separate RDX-006 contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from danids.config.rdx_training_evidence import (
    RDX004_PROTOCOL_SHA256,
    RDX004Budget,
    RDX004ConfigError,
    build_rdx004_run_roster,
    load_rdx004_training_evidence_config,
)
from danids.config.rdx_training_execution import (
    RDX004_SMOKE_RUN_NAMESPACE,
    RDX006_EXECUTION_IMPLEMENTATION_VERSION,
    RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST,
    RDX006ExecutionConfigError,
    load_rdx006_execution_config,
)

EXECUTION_CONFIG_PATH = Path("configs/experiments/rdx/rdx006_execution_v1.yaml")
PREFLIGHT_CONFIG_PATH = Path("configs/experiments/rdx/rdx004_training_evidence_v1.yaml")
ROTATION = ("U", "T", "C", "B")


def _authorize(*, budget: RDX004Budget, smoke: bool = False):  # type: ignore[no-untyped-def]
    config = load_rdx006_execution_config(EXECUTION_CONFIG_PATH)
    return config.authorize(
        execute=True,
        budget=budget,
        rotation=ROTATION,
        seed=42,
        smoke=smoke,
        preflight_bundle_digest=RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST,
        preflight_verdict="GO",
        paired_b100_experiment_id="E4_OFFLINE_ORACLE_U-T-C-B_s42",
        paired_study1_experiment_id="E1_STATIC_MLP_U-T-C-B_s42",
    )


def test_execution_contract_is_separate_and_digest_bound() -> None:
    execution = load_rdx006_execution_config(EXECUTION_CONFIG_PATH)
    preflight = load_rdx004_training_evidence_config(PREFLIGHT_CONFIG_PATH)

    assert execution.execution_authorized is True
    assert execution.implementation_version == RDX006_EXECUTION_IMPLEMENTATION_VERSION
    assert execution.protocol_sha256 == RDX004_PROTOCOL_SHA256
    assert execution.required_preflight_bundle_digest == RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST
    assert execution.explicit_execute_flag == "--execute"
    assert preflight.execution_authorized is False


def test_programmatic_execution_config_cannot_bypass_frozen_metadata() -> None:
    config = load_rdx006_execution_config(EXECUTION_CONFIG_PATH)
    with pytest.raises(RDX006ExecutionConfigError, match="protocol_sha256 differs"):
        replace(config, protocol_sha256="0" * 64)


def test_full_and_smoke_identities_are_unambiguous() -> None:
    full = _authorize(budget=RDX004Budget.B400)
    smoke = _authorize(budget=RDX004Budget.B1600, smoke=True)
    preflight = load_rdx004_training_evidence_config(PREFLIGHT_CONFIG_PATH)
    roster_ids = {identity.run_id for identity in build_rdx004_run_roster(preflight)}

    assert full.run_id == "RDX004_OFFLINE_ORACLE_B400_U-T-C-B_s42"
    assert full.output_namespace == "runs/rdx004-training-evidence-v1"
    assert full.confirmatory_eligible is True
    assert smoke.run_id == "RDX004_SMOKE_OFFLINE_ORACLE_B1600_U-T-C-B_s42"
    assert "SMOKE" in smoke.run_id
    assert smoke.output_namespace == RDX004_SMOKE_RUN_NAMESPACE
    assert smoke.confirmatory_eligible is False
    assert smoke.run_id not in roster_ids
    assert smoke.expected_confirmatory_run_id in roster_ids
    assert smoke.to_dict()["smoke"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("execute", False, "explicit --execute"),
        ("preflight_bundle_digest", "0" * 64, "preflight bundle digest"),
        ("preflight_verdict", "NO-GO", "verdict must be GO"),
        ("paired_b100_experiment_id", "wrong", "paired B100 identity"),
        ("paired_study1_experiment_id", "wrong", "Study-1 starting-state identity"),
    ),
)
def test_execution_authorization_fails_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    config = load_rdx006_execution_config(EXECUTION_CONFIG_PATH)
    arguments: dict[str, object] = {
        "execute": True,
        "budget": RDX004Budget.B400,
        "rotation": ROTATION,
        "seed": 42,
        "smoke": False,
        "preflight_bundle_digest": RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST,
        "preflight_verdict": "GO",
        "paired_b100_experiment_id": "E4_OFFLINE_ORACLE_U-T-C-B_s42",
        "paired_study1_experiment_id": "E1_STATIC_MLP_U-T-C-B_s42",
    }
    arguments[field] = value
    with pytest.raises(RDX006ExecutionConfigError, match=message):
        config.authorize(**arguments)  # type: ignore[arg-type]


def test_b100_unknown_rotation_and_unknown_seed_cannot_be_authorized() -> None:
    config = load_rdx006_execution_config(EXECUTION_CONFIG_PATH)
    common = {
        "execute": True,
        "smoke": False,
        "preflight_bundle_digest": RDX006_REQUIRED_PREFLIGHT_BUNDLE_DIGEST,
        "preflight_verdict": "GO",
        "paired_b100_experiment_id": "E4_OFFLINE_ORACLE_U-T-C-B_s42",
        "paired_study1_experiment_id": "E1_STATIC_MLP_U-T-C-B_s42",
    }
    with pytest.raises(RDX004ConfigError, match="B400 or B1600"):
        config.authorize(
            budget=RDX004Budget.B100,
            rotation=ROTATION,
            seed=42,
            **common,
        )
    with pytest.raises(RDX004ConfigError, match="frozen U/T/C/B rotation"):
        config.authorize(
            budget=RDX004Budget.B400,
            rotation=("U", "C", "T", "B"),
            seed=42,
            **common,
        )
    with pytest.raises(RDX004ConfigError, match="42, 43, or 44"):
        config.authorize(
            budget=RDX004Budget.B400,
            rotation=ROTATION,
            seed=45,
            **common,
        )
