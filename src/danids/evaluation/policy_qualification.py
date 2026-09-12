from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isclose, sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from danids.adaptation.actions import InterventionAction
from danids.evaluation.policy_development import validate_policy_development_evaluation
from danids.policy.development import expected_actions

POLICY_QUALIFICATION_VERSION = "task006-policy-qualification-v1"
DISABLED_STATUS = "DISABLED_INSUFFICIENT_CALIBRATION_SUCCESS_SUPPORT"
BOUND_PASSED_STATUS = "ORACLE_BOUND_PERMITS_MODEL_QUALIFICATION"
ELIGIBLE_ACTIONS = (
    InterventionAction.NO_OP.value,
    InterventionAction.HEAD_UPDATE.value,
    InterventionAction.FULL_FINE_TUNE.value,
    InterventionAction.REPLAY_UPDATE.value,
)
MODEL_CHANGING_ACTIONS = (
    InterventionAction.HEAD_UPDATE.value,
    InterventionAction.FULL_FINE_TUNE.value,
    InterventionAction.REPLAY_UPDATE.value,
)
MINIMUM_RECOMMENDATIONS = 100
MINIMUM_MODEL_CHANGING_SELECTIONS = 25
REQUIRED_WILSON_LOWER_BOUND = 0.90
WILSON_CONFIDENCE = 0.95
QUALIFICATION_FILES = {
    "policy_qualification.json",
    "policy_qualification_summary.md",
    "policy_qualification_manifest.json",
}


class PolicyQualificationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class QualificationUpperBound:
    calibration_component_count: int
    eligible_recommendation_component_count: int
    model_changing_recommendation_component_count: int
    successful_component_union_count: int
    successful_model_changing_component_union_count: int
    maximum_successes_at_minimum_recommendations: int | None
    maximum_empirical_success_rate_at_minimum_recommendations: float | None
    wilson_lower_at_minimum_recommendations: float | None
    maximum_wilson_lower_bound: float | None
    oracle_qualification_possible: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool(value: object, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).casefold()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise PolicyQualificationError(f"{name} must be boolean")


def _is_confirmed_success(value: object) -> bool:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(float(value))):
        return False
    try:
        return float(str(value)) == 1.0
    except ValueError as exc:
        raise PolicyQualificationError("binary_fit_target must be binary or missing") from exc


def wilson_lower_bound(
    successes: int,
    trials: int,
    *,
    confidence: float = WILSON_CONFIDENCE,
) -> float:
    """Return the deterministic two-sided Wilson interval lower endpoint."""

    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= trials and trials > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Wilson confidence must lie strictly between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2.0 * trials)
    radius = z * sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
    return (centre - radius) / denominator


def minimum_successes_for_wilson(
    trials: int,
    *,
    required_lower_bound: float = REQUIRED_WILSON_LOWER_BOUND,
) -> int:
    if not 0.0 <= required_lower_bound <= 1.0:
        raise ValueError("required Wilson lower bound must lie in [0, 1]")
    for successes in range(trials + 1):
        if wilson_lower_bound(successes, trials) >= required_lower_bound:
            return successes
    raise ValueError("the requested Wilson lower bound is unattainable")


def _component_options(calibration: pd.DataFrame) -> list[set[tuple[int, int]]]:
    result: list[set[tuple[int, int]]] = []
    for _, rows in calibration.groupby("physical_component", sort=True):
        options: set[tuple[int, int]] = set()
        for row in rows.itertuples(index=False):
            action = str(row.action)
            if action not in ELIGIBLE_ACTIONS or not _bool(row.feasible, "feasible"):
                continue
            options.add(
                (
                    int(action in MODEL_CHANGING_ACTIONS),
                    int(_is_confirmed_success(row.binary_fit_target)),
                )
            )
        result.append(options)
    return result


def compute_qualification_upper_bound(calibration: pd.DataFrame) -> QualificationUpperBound:
    """Solve the distinct-component oracle action-selection problem exactly."""

    required = {
        "physical_component",
        "policy_partition",
        "action",
        "feasible",
        "binary_fit_target",
        "current_domain",
    }
    missing = sorted(required.difference(calibration.columns))
    if missing:
        raise PolicyQualificationError(f"qualification data lacks fields: {missing}")
    if set(calibration["policy_partition"].astype(str)) != {"calibration"}:
        raise PolicyQualificationError(
            "qualification upper bound accepts only the frozen calibration partition"
        )
    options = _component_options(calibration)
    eligible_count = sum(bool(value) for value in options)
    model_count = sum(any(option[0] for option in value) for value in options)
    successful = {
        str(row.physical_component)
        for row in calibration.itertuples(index=False)
        if str(row.action) in ELIGIBLE_ACTIONS
        and _bool(row.feasible, "feasible")
        and _is_confirmed_success(row.binary_fit_target)
    }
    successful_model = {
        str(row.physical_component)
        for row in calibration.itertuples(index=False)
        if str(row.action) in MODEL_CHANGING_ACTIONS
        and _bool(row.feasible, "feasible")
        and _is_confirmed_success(row.binary_fit_target)
    }

    # dp[recommendations][capped model-changing selections] = maximum successes.
    impossible = -1
    cap = MINIMUM_MODEL_CHANGING_SELECTIONS
    dp = [[impossible] * (cap + 1) for _ in range(len(options) + 1)]
    dp[0][0] = 0
    processed = 0
    for component_options in options:
        next_dp = [row.copy() for row in dp]
        for recommendations in range(processed + 1):
            for model_changing in range(cap + 1):
                successes = dp[recommendations][model_changing]
                if successes < 0:
                    continue
                for is_model_changing, is_success in component_options:
                    next_model = min(cap, model_changing + is_model_changing)
                    next_dp[recommendations + 1][next_model] = max(
                        next_dp[recommendations + 1][next_model], successes + is_success
                    )
        dp = next_dp
        processed += 1

    at_minimum = dp[MINIMUM_RECOMMENDATIONS][cap] if len(options) >= 100 else impossible
    minimum_successes = None if at_minimum < 0 else at_minimum
    minimum_rate = (
        None if minimum_successes is None else minimum_successes / MINIMUM_RECOMMENDATIONS
    )
    minimum_wilson = (
        None
        if minimum_successes is None
        else wilson_lower_bound(minimum_successes, MINIMUM_RECOMMENDATIONS)
    )
    attainable: list[float] = []
    for recommendations in range(MINIMUM_RECOMMENDATIONS, len(options) + 1):
        successes = dp[recommendations][cap]
        if successes >= 0:
            attainable.append(wilson_lower_bound(successes, recommendations))
    maximum_wilson = max(attainable) if attainable else None
    qualifies = (
        eligible_count >= MINIMUM_RECOMMENDATIONS
        and model_count >= MINIMUM_MODEL_CHANGING_SELECTIONS
        and maximum_wilson is not None
        and maximum_wilson >= REQUIRED_WILSON_LOWER_BOUND
    )
    return QualificationUpperBound(
        calibration_component_count=int(calibration["physical_component"].nunique()),
        eligible_recommendation_component_count=eligible_count,
        model_changing_recommendation_component_count=model_count,
        successful_component_union_count=len(successful),
        successful_model_changing_component_union_count=len(successful_model),
        maximum_successes_at_minimum_recommendations=minimum_successes,
        maximum_empirical_success_rate_at_minimum_recommendations=minimum_rate,
        wilson_lower_at_minimum_recommendations=minimum_wilson,
        maximum_wilson_lower_bound=maximum_wilson,
        oracle_qualification_possible=qualifies,
    )


def _support_by_action_class(calibration: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    targets = pd.to_numeric(calibration["binary_fit_target"], errors="coerce")
    for action in expected_actions():
        action_rows = calibration.loc[calibration["action"] == action]
        action_targets = targets.loc[action_rows.index]
        result[action] = {
            str(target): {
                "trial_count": int((action_targets == target).sum()),
                "distinct_physical_components": int(
                    action_rows.loc[action_targets == target, "physical_component"].nunique()
                ),
            }
            for target in (0, 1)
        }
    return result


def _reason(bound: QualificationUpperBound, minimum_successes: int) -> str:
    failures: list[str] = []
    if bound.eligible_recommendation_component_count < MINIMUM_RECOMMENDATIONS:
        failures.append(
            f"only {bound.eligible_recommendation_component_count} distinct eligible calibration "
            f"components are available; {MINIMUM_RECOMMENDATIONS} are required"
        )
    if bound.model_changing_recommendation_component_count < MINIMUM_MODEL_CHANGING_SELECTIONS:
        failures.append(
            f"only {bound.model_changing_recommendation_component_count} distinct calibration "
            "components permit a model-changing selection; "
            f"{MINIMUM_MODEL_CHANGING_SELECTIONS} are required"
        )
    if (
        bound.maximum_successes_at_minimum_recommendations is not None
        and bound.maximum_successes_at_minimum_recommendations < minimum_successes
    ):
        failures.append(
            f"the oracle can confirm at most "
            f"{bound.maximum_successes_at_minimum_recommendations}/100 successes; "
            f"at least {minimum_successes}/100 are required by the frozen Wilson rule"
        )
    if not failures:
        failures.append("the oracle Wilson lower bound cannot reach the frozen requirement")
    return "; ".join(failures)


def evaluate_policy_qualification(
    evaluation_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Validate an aggregate and evaluate the model-independent Policy upper bound."""

    validated = validate_policy_development_evaluation(evaluation_dir)
    calibration = validated.trials.loc[validated.trials["policy_partition"] == "calibration"].copy()
    bound = compute_qualification_upper_bound(calibration)
    minimum_successes = minimum_successes_for_wilson(MINIMUM_RECOMMENDATIONS)
    success_rows = calibration.loc[
        calibration["action"].isin(ELIGIBLE_ACTIONS)
        & (pd.to_numeric(calibration["binary_fit_target"], errors="coerce") == 1)
    ]
    status = BOUND_PASSED_STATUS if bound.oracle_qualification_possible else DISABLED_STATUS
    reason = (
        "The oracle upper bound permits the separately specified action-model qualification stage."
        if bound.oracle_qualification_possible
        else _reason(bound, minimum_successes)
    )
    payload: dict[str, Any] = {
        "artifact_version": POLICY_QUALIFICATION_VERSION,
        "status": status,
        "source_evaluation_contract_sha256": _sha256(validated.path / "evaluation_contract.json"),
        "source_canonical_dataset_sha256": validated.contract["canonical_dataset_sha256"],
        "source_scientific_contract_digest": validated.contract["scientific_contract_digest"],
        "eligible_actions": list(ELIGIBLE_ACTIONS),
        "model_changing_actions": list(MODEL_CHANGING_ACTIONS),
        "calibration_binary_support_by_action_class": _support_by_action_class(calibration),
        **asdict(bound),
        "minimum_recommendation_requirement": MINIMUM_RECOMMENDATIONS,
        "minimum_model_changing_selection_requirement": MINIMUM_MODEL_CHANGING_SELECTIONS,
        "wilson_confidence": WILSON_CONFIDENCE,
        "required_wilson_lower_bound": REQUIRED_WILSON_LOWER_BOUND,
        "minimum_successes_required_at_n_100": minimum_successes,
        "maximum_achievable_confirmed_success_count": bound.successful_component_union_count,
        "domains_containing_calibration_success": sorted(
            success_rows["current_domain"].astype(str).unique().tolist()
        ),
        "policy_enabled": False,
        "action_models_fitted": False,
        "tau_success_selected": False,
        "fail_closed_reason": None if bound.oracle_qualification_possible else reason,
        "next_stage_reason": reason if bound.oracle_qualification_possible else None,
    }
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Policy qualification artifact: {output}")
    output.mkdir(parents=True, exist_ok=False)
    json_path = output / "policy_qualification.json"
    summary_path = output / "policy_qualification_summary.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_lines = [
        "# DANIDS-Policy qualification",
        "",
        f"Status: `{status}`",
        "",
        f"Policy enabled: **{str(payload['policy_enabled']).lower()}**",
        f"Calibration physical components: {bound.calibration_component_count}",
        f"Components with an eligible SUCCESS: {bound.successful_component_union_count}",
        "Components with a model-changing SUCCESS: "
        f"{bound.successful_model_changing_component_union_count}",
        "Maximum confirmed successes at 100 distinct recommendations: "
        f"{bound.maximum_successes_at_minimum_recommendations}",
        f"Wilson 95% lower bound at that maximum: {bound.wilson_lower_at_minimum_recommendations}",
        "",
        f"Reason: {reason}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    manifest = {
        "version": POLICY_QUALIFICATION_VERSION,
        "files": {
            json_path.name: _sha256(json_path),
            summary_path.name: _sha256(summary_path),
        },
    }
    (output / "policy_qualification_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_policy_qualification_artifact(output)
    return output


def validate_policy_qualification_artifact(artifact_dir: str | Path) -> dict[str, Any]:
    """Validate the write-once qualification artifact used by the E4 launch gate."""

    root = Path(artifact_dir).resolve()
    observed = {item.name for item in root.iterdir() if item.is_file()}
    if observed != QUALIFICATION_FILES:
        raise PolicyQualificationError(
            "Policy qualification file set differs: "
            f"missing={sorted(QUALIFICATION_FILES - observed)}, "
            f"unexpected={sorted(observed - QUALIFICATION_FILES)}"
        )
    try:
        manifest = json.loads(
            (root / "policy_qualification_manifest.json").read_text(encoding="utf-8")
        )
        payload = json.loads((root / "policy_qualification.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PolicyQualificationError("Policy qualification JSON is malformed") from exc
    expected_files = QUALIFICATION_FILES - {"policy_qualification_manifest.json"}
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"version", "files"}
        or manifest["version"] != POLICY_QUALIFICATION_VERSION
        or not isinstance(manifest["files"], dict)
        or set(manifest["files"]) != expected_files
    ):
        raise PolicyQualificationError("Policy qualification manifest differs")
    for name, digest in manifest["files"].items():
        if digest != _sha256(root / name):
            raise PolicyQualificationError(f"Policy qualification artifact digest differs: {name}")
    if not isinstance(payload, dict) or payload.get("artifact_version") != (
        POLICY_QUALIFICATION_VERSION
    ):
        raise PolicyQualificationError("Policy qualification payload version differs")
    frozen_fields = {
        "eligible_actions": list(ELIGIBLE_ACTIONS),
        "model_changing_actions": list(MODEL_CHANGING_ACTIONS),
        "minimum_recommendation_requirement": MINIMUM_RECOMMENDATIONS,
        "minimum_model_changing_selection_requirement": MINIMUM_MODEL_CHANGING_SELECTIONS,
        "wilson_confidence": WILSON_CONFIDENCE,
        "required_wilson_lower_bound": REQUIRED_WILSON_LOWER_BOUND,
    }
    if any(payload.get(key) != value for key, value in frozen_fields.items()):
        raise PolicyQualificationError("Policy qualification frozen contract differs")
    for name in (
        "source_evaluation_contract_sha256",
        "source_canonical_dataset_sha256",
        "source_scientific_contract_digest",
    ):
        value = payload.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise PolicyQualificationError(f"Policy qualification {name} is malformed")
    if payload.get("policy_enabled") is not False:
        raise PolicyQualificationError(
            "this qualification-stage artifact cannot enable DANIDS-Policy"
        )
    if (
        payload.get("action_models_fitted") is not False
        or payload.get("tau_success_selected") is not False
    ):
        raise PolicyQualificationError("Policy qualification falsely reports fitted Policy state")
    expected_minimum = minimum_successes_for_wilson(
        int(payload["minimum_recommendation_requirement"]),
        required_lower_bound=float(payload["required_wilson_lower_bound"]),
    )
    if int(payload["minimum_successes_required_at_n_100"]) != expected_minimum:
        raise PolicyQualificationError("Policy qualification Wilson requirement differs")
    possible = payload.get("oracle_qualification_possible")
    if not isinstance(possible, bool):
        raise PolicyQualificationError("Policy qualification oracle result is malformed")
    expected_status = BOUND_PASSED_STATUS if possible else DISABLED_STATUS
    if payload.get("status") != expected_status:
        raise PolicyQualificationError("Policy qualification status differs from oracle bound")
    maximum = payload.get("maximum_successes_at_minimum_recommendations")
    if maximum is not None:
        maximum = int(maximum)
        if not 0 <= maximum <= MINIMUM_RECOMMENDATIONS:
            raise PolicyQualificationError("Policy qualification success maximum is malformed")
        expected_rate = maximum / MINIMUM_RECOMMENDATIONS
        expected_lower = wilson_lower_bound(maximum, MINIMUM_RECOMMENDATIONS)
        if not isclose(
            float(payload["maximum_empirical_success_rate_at_minimum_recommendations"]),
            expected_rate,
            rel_tol=0.0,
            abs_tol=1e-15,
        ) or not isclose(
            float(payload["wilson_lower_at_minimum_recommendations"]),
            expected_lower,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise PolicyQualificationError("Policy qualification Wilson result differs")
    if not possible and not isinstance(payload.get("fail_closed_reason"), str):
        raise PolicyQualificationError("disabled Policy qualification lacks a reason")
    return payload


def require_enabled_policy_qualification(artifact_dir: str | Path | None) -> None:
    """Fail before E4 setup unless a validated artifact explicitly enables Policy."""

    if artifact_dir is None:
        raise PolicyQualificationError(
            "DANIDS_POLICY requires an explicit validated Policy qualification artifact"
        )
    payload = validate_policy_qualification_artifact(artifact_dir)
    if payload["policy_enabled"] is not True:
        raise PolicyQualificationError(
            f"DANIDS_POLICY is disabled by qualification status {payload['status']}: "
            f"{payload['fail_closed_reason']}"
        )


__all__ = [
    "BOUND_PASSED_STATUS",
    "DISABLED_STATUS",
    "ELIGIBLE_ACTIONS",
    "MINIMUM_MODEL_CHANGING_SELECTIONS",
    "MINIMUM_RECOMMENDATIONS",
    "MODEL_CHANGING_ACTIONS",
    "POLICY_QUALIFICATION_VERSION",
    "PolicyQualificationError",
    "QualificationUpperBound",
    "compute_qualification_upper_bound",
    "evaluate_policy_qualification",
    "minimum_successes_for_wilson",
    "require_enabled_policy_qualification",
    "validate_policy_qualification_artifact",
    "wilson_lower_bound",
]
