from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from danids.config.study4 import Study4Method, load_study4_execution_config
from danids.evaluation import policy_qualification as qualification
from danids.evaluation.policy_qualification import (
    DISABLED_STATUS,
    PolicyQualificationError,
    compute_qualification_upper_bound,
    evaluate_policy_qualification,
    minimum_successes_for_wilson,
    wilson_lower_bound,
)
from danids.experiments.study4 import run_study4_experiment


def _calibration_frame(
    component_count: int,
    success_count: int,
    *,
    model_changing_components: int | None = None,
    duplicate_success_actions: bool = False,
) -> pd.DataFrame:
    model_count = (
        component_count if model_changing_components is None else model_changing_components
    )
    rows: list[dict[str, Any]] = []
    for index in range(component_count):
        success = index < success_count
        rows.append(
            {
                "physical_component": f"component-{index:03d}",
                "policy_partition": "calibration",
                "action": "A0_NO_OP",
                "feasible": True,
                "binary_fit_target": int(success),
                "current_domain": "T" if success else "C",
            }
        )
        if index < model_count:
            rows.append(
                {
                    "physical_component": f"component-{index:03d}",
                    "policy_partition": "calibration",
                    "action": "A2_HEAD_UPDATE",
                    "feasible": True,
                    "binary_fit_target": int(success and duplicate_success_actions),
                    "current_domain": "T" if success else "C",
                }
            )
    return pd.DataFrame(rows)


def _validated_evaluation(tmp_path: Path, frame: pd.DataFrame) -> SimpleNamespace:
    source = tmp_path / "evaluation"
    source.mkdir()
    (source / "evaluation_contract.json").write_text("{}\n", encoding="utf-8")
    return SimpleNamespace(
        path=source,
        contract={
            "canonical_dataset_sha256": "a" * 64,
            "scientific_contract_digest": "b" * 64,
        },
        trials=frame,
    )


def test_frozen_wilson_requirement_is_exact() -> None:
    assert minimum_successes_for_wilson(100) == 96
    assert wilson_lower_bound(95, 100) < 0.90
    assert wilson_lower_bound(96, 100) >= 0.90
    assert (
        compute_qualification_upper_bound(
            _calibration_frame(100, 95, duplicate_success_actions=True)
        ).oracle_qualification_possible
        is False
    )


def test_current_style_support_fails_closed_without_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _calibration_frame(103, 10, duplicate_success_actions=True)
    validated = _validated_evaluation(tmp_path, frame)
    calls = 0

    def validate(_: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return validated

    monkeypatch.setattr(qualification, "validate_policy_development_evaluation", validate)
    output = evaluate_policy_qualification(validated.path, tmp_path / "qualification")
    payload = qualification.validate_policy_qualification_artifact(output)

    assert calls == 1
    assert payload["status"] == DISABLED_STATUS
    assert payload["calibration_component_count"] == 103
    assert payload["successful_component_union_count"] == 10
    assert payload["successful_model_changing_component_union_count"] == 10
    assert payload["maximum_successes_at_minimum_recommendations"] == 10
    assert payload["policy_enabled"] is False
    assert payload["action_models_fitted"] is False
    assert payload["tau_success_selected"] is False
    assert not list(output.glob("*model*"))


def test_distinct_recommendation_and_model_changing_support_are_required() -> None:
    fewer_recommendations = compute_qualification_upper_bound(
        _calibration_frame(99, 99, duplicate_success_actions=True)
    )
    assert fewer_recommendations.oracle_qualification_possible is False
    assert fewer_recommendations.maximum_successes_at_minimum_recommendations is None

    fewer_model_changes = compute_qualification_upper_bound(
        _calibration_frame(
            100,
            100,
            model_changing_components=24,
            duplicate_success_actions=True,
        )
    )
    assert fewer_model_changes.model_changing_recommendation_component_count == 24
    assert fewer_model_changes.oracle_qualification_possible is False


def test_sibling_action_successes_count_once_per_physical_component() -> None:
    bound = compute_qualification_upper_bound(
        _calibration_frame(100, 96, duplicate_success_actions=True)
    )
    assert bound.successful_component_union_count == 96
    assert bound.successful_model_changing_component_union_count == 96
    assert bound.maximum_successes_at_minimum_recommendations == 96
    assert bound.oracle_qualification_possible is True


def test_upper_bound_rejects_fit_calibration_leakage() -> None:
    frame = _calibration_frame(100, 96, duplicate_success_actions=True)
    frame.loc[0, "policy_partition"] = "fit"
    with pytest.raises(PolicyQualificationError, match="only the frozen calibration"):
        compute_qualification_upper_bound(frame)


def test_disabled_qualification_blocks_policy_before_e4_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _calibration_frame(103, 10, duplicate_success_actions=True)
    validated = _validated_evaluation(tmp_path, frame)
    monkeypatch.setattr(
        qualification,
        "validate_policy_development_evaluation",
        lambda _: validated,
    )
    artifact = evaluate_policy_qualification(validated.path, tmp_path / "qualification")
    root = Path(__file__).resolve().parents[1]
    core = load_study4_execution_config(
        root / "configs" / "experiments" / "task006_e4_core_u-t-c-b.yaml"
    )
    policy = replace(core, method=Study4Method.DANIDS_POLICY)

    with pytest.raises(PolicyQualificationError, match=DISABLED_STATUS):
        run_study4_experiment(
            cast(Any, object()),
            policy,
            initial_run="unused",
            health_artifact_dir="unused",
            manifest_dir="unused",
            output_root="unused",
            policy_qualification_dir=artifact,
        )


def test_qualification_artifact_tampering_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _calibration_frame(103, 10, duplicate_success_actions=True)
    validated = _validated_evaluation(tmp_path, frame)
    monkeypatch.setattr(
        qualification,
        "validate_policy_development_evaluation",
        lambda _: validated,
    )
    artifact = evaluate_policy_qualification(validated.path, tmp_path / "qualification")
    path = artifact / "policy_qualification.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["policy_enabled"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PolicyQualificationError, match="digest differs"):
        qualification.validate_policy_qualification_artifact(artifact)
