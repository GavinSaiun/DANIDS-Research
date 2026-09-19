from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from danids import cli
from danids.cli import build_parser
from danids.evaluation import rdx_training_analysis
from danids.evaluation.rdx_training_analysis import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    RdxTrainingAnalysisError,
    compute_primary_outcome,
    evaluate_decision_gate,
    stratified_bootstrap_interval,
)

_ACTIONS = (
    "A0_NO_OP",
    "A1_RECALIBRATE",
    "A2_HEAD_UPDATE",
    "A3_FULL_FINE_TUNE",
    "A4_REPLAY_UPDATE",
)
_ROTATIONS = ("U-T-C-B", "T-C-B-U", "C-B-U-T", "B-U-T-C")
_SEEDS = (42, 43, 44)


def _candidate(
    action: str,
    *,
    success: bool = False,
    successor: str = "HARMFUL",
    feasible: bool = True,
    execution_succeeded: bool = True,
    audit_admissible: bool = True,
) -> dict[str, object]:
    if success:
        successor = "SAFE"
        feasible = True
        execution_succeeded = True
        audit_admissible = True
    if not feasible:
        execution_succeeded = False
        audit_admissible = False
    return {
        "action": action,
        "feasible": feasible,
        "feasibility_reason": "eligible" if feasible else "insufficient_support",
        "execution_succeeded": execution_succeeded,
        "audit_admissible": audit_admissible,
        "audit_state": "SAFE" if audit_admissible else "NOT_ADMISSIBLE",
        "successor_evaluator_state": successor,
        "confirmed_success": success,
        "optimizer_steps": 0,
        "target_rows": 0,
        "replay_rows": 0,
        "rows_consumed": 0,
        "incoming_model_digest": "1" * 64,
        "deployed_model_digest": "2" * 64,
        "deployed_threshold_digest": "3" * 64,
        "deployed_r1_digest": "4" * 64,
        "successor_window_id": 1,
        "successor_row_start": 50_000,
        "successor_row_stop": 100_000,
        "permanent_holdout_used": False,
    }


def _decision(
    successful_actions: Sequence[str] = (),
    *,
    state: str = "HARMFUL",
    uncertain_actions: Sequence[str] = (),
) -> dict[str, object]:
    successes = set(successful_actions)
    uncertain = set(uncertain_actions)
    return {
        "current_evaluator_state": state,
        "candidates": [
            _candidate(
                action,
                success=action in successes,
                successor="UNCERTAIN" if action in uncertain else "HARMFUL",
            )
            for action in _ACTIONS
        ],
    }


def test_primary_outcomes_use_capability_sets_and_keep_all_harmful_decisions() -> None:
    outcome = compute_primary_outcome(
        [
            _decision(["A0_NO_OP"]),
            _decision(["A1_RECALIBRATE"]),
            _decision(["A2_HEAD_UPDATE"]),
            _decision(["A3_FULL_FINE_TUNE"]),
            _decision(["A4_REPLAY_UPDATE"]),
            _decision(uncertain_actions=_ACTIONS),
            _decision(_ACTIONS, state="SAFE"),
        ]
    )

    assert outcome.harmful_denominator == 6
    assert outcome.p1_numerator == 5
    assert outcome.p2_numerator == 3
    assert outcome.a3_exclusive_numerator == 1
    assert outcome.p1 == Fraction(5, 6)
    assert outcome.p2 == Fraction(1, 2)
    assert outcome.a3_exclusive == Fraction(1, 6)


def test_a3_exclusive_requires_no_successful_sibling_action() -> None:
    outcome = compute_primary_outcome(
        [
            _decision(["A3_FULL_FINE_TUNE"]),
            _decision(["A0_NO_OP", "A3_FULL_FINE_TUNE"]),
            _decision(["A2_HEAD_UPDATE", "A3_FULL_FINE_TUNE"]),
        ]
    )

    assert outcome.p1 == Fraction(1, 1)
    assert outcome.p2 == Fraction(1, 3)
    assert outcome.a3_exclusive_numerator == 1
    assert outcome.a3_exclusive == Fraction(1, 3)


def test_primary_outcome_zero_harmful_denominator_is_unavailable_not_zero() -> None:
    outcome = compute_primary_outcome([_decision(_ACTIONS, state="SAFE")])

    assert outcome.harmful_denominator == 0
    assert outcome.p1_numerator == 0
    assert outcome.p2_numerator == 0
    assert outcome.a3_exclusive_numerator == 0
    assert outcome.p1 is None
    assert outcome.p2 is None
    assert outcome.a3_exclusive is None


def test_scope_token_rows_ignore_bare_memory_allocation_scopes() -> None:
    query_scopes = [
        {
            "stage": stage,
            "current_domain": domain,
            "opaque_scope_token": f"scope-{stage}",
        }
        for stage, domain in enumerate(("T", "C", "B"), start=1)
    ]
    item = SimpleNamespace(
        memory={"scopes": [{"scope_id": f"scope-{stage}"} for stage in range(1, 4)]},
        query={"scopes": query_scopes},
        run=SimpleNamespace(run_id="E4_OFFLINE_ORACLE_U-T-C-B_s42"),
    )

    assert rdx_training_analysis._scope_token_rows(item) == query_scopes


def test_bootstrap_uses_frozen_stratified_contract_and_is_order_independent() -> None:
    differences: dict[str, tuple[Fraction, ...]] = {
        "U-T-C-B": (Fraction(1, 10),) * 3,
        "T-C-B-U": (Fraction(1, 5),) * 3,
        "C-B-U-T": (Fraction(3, 10),) * 3,
        "B-U-T-C": (Fraction(2, 5),) * 3,
    }

    forward = stratified_bootstrap_interval(differences)
    reverse = stratified_bootstrap_interval(dict(reversed(tuple(differences.items()))))

    assert forward == reverse
    assert forward["seed"] == BOOTSTRAP_SEED == 855_488_261_130_030_661
    assert forward["replicates"] == BOOTSTRAP_REPLICATES == 10_000
    assert forward["mean_lower"] == pytest.approx(0.25)
    assert forward["mean_upper"] == pytest.approx(0.25)
    assert forward["median_lower"] == pytest.approx(0.25)
    assert forward["median_upper"] == pytest.approx(0.25)


def test_bootstrap_matches_frozen_nonconstant_pcg64_reference() -> None:
    differences = {
        "U-T-C-B": (Fraction(0), Fraction(1, 10), Fraction(2, 10)),
        "T-C-B-U": (Fraction(-1, 10), Fraction(0), Fraction(1, 10)),
        "C-B-U-T": (Fraction(1, 20), Fraction(3, 20), Fraction(1, 4)),
        "B-U-T-C": (Fraction(-1, 5), Fraction(1, 5), Fraction(3, 5)),
    }

    result = stratified_bootstrap_interval(differences)

    assert result["mean_lower"] == pytest.approx(0.0125)
    assert result["mean_upper"] == pytest.approx(0.2125)
    assert result["median_lower"] == pytest.approx(0.0)
    assert result["median_upper"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "differences",
    [
        {rotation: (0.0, 0.0, 0.0) for rotation in _ROTATIONS[:-1]},
        {
            **{rotation: (0.0, 0.0, 0.0) for rotation in _ROTATIONS},
            _ROTATIONS[0]: (0.0, 0.0),
        },
    ],
)
def test_bootstrap_rejects_noncanonical_rotation_seed_units(
    differences: Mapping[str, Sequence[float]],
) -> None:
    with pytest.raises(RdxTrainingAnalysisError):
        stratified_bootstrap_interval(differences)


def _primary_rows(
    *,
    b400_p1: Mapping[str, Fraction] | None = None,
    b400_p2: Mapping[str, Fraction] | None = None,
    b400_a3: Mapping[str, Fraction] | None = None,
    b1600_p1: Mapping[str, Fraction] | None = None,
    b1600_p2: Mapping[str, Fraction] | None = None,
    b1600_a3: Mapping[str, Fraction] | None = None,
) -> list[dict[str, object]]:
    zero = {rotation: Fraction(0) for rotation in _ROTATIONS}
    values = {
        "B100": (zero, zero, zero),
        "B400": (b400_p1 or zero, b400_p2 or zero, b400_a3 or zero),
        "B1600": (b1600_p1 or zero, b1600_p2 or zero, b1600_a3 or zero),
    }
    rows: list[dict[str, object]] = []
    for budget in ("B100", "B400", "B1600"):
        p1, p2, a3 = values[budget]
        for sequence in _ROTATIONS:
            for seed in _SEEDS:
                rows.append(
                    {
                        "budget": budget,
                        "sequence": sequence,
                        "seed": seed,
                        "p1_exact": str(p1[sequence]),
                        "p2_exact": str(p2[sequence]),
                        "a3_exclusive_exact": str(a3[sequence]),
                    }
                )
    return rows


def _constant(value: Fraction) -> dict[str, Fraction]:
    return {rotation: value for rotation in _ROTATIONS}


def test_incomplete_treatment_does_not_publish_partial_aggregate() -> None:
    primary = _primary_rows()
    primary[0] = {
        **primary[0],
        "harmful_denominator": 0,
        "p1_exact": "",
        "p2_exact": "",
        "a3_exclusive_exact": "",
    }
    for row in primary[1:]:
        row["harmful_denominator"] = 1

    summaries = rdx_training_analysis._treatment_summaries(primary)
    b100_p1 = next(row for row in summaries if row["budget"] == "B100" and row["outcome"] == "P1")

    assert b100_p1["unavailable_units"] == 1
    assert b100_p1["mean_unit_rate"] is None
    assert b100_p1["median_unit_rate"] is None
    assert b100_p1["minimum_unit_rate"] is None
    assert b100_p1["maximum_unit_rate"] is None


def test_a3_exclusive_has_paired_summary_without_unregistered_bootstrap_interval() -> None:
    primary = _primary_rows()
    for row in primary:
        row["harmful_denominator"] = 1
    paired = rdx_training_analysis._paired_rows(primary)

    summaries = rdx_training_analysis._bootstrap_rows(paired)

    assert len(summaries) == 9
    assert {row["outcome"] for row in summaries} == {"P1", "P2", "A3_EXCLUSIVE"}
    a3_rows = [row for row in summaries if row["outcome"] == "A3_EXCLUSIVE"]
    assert all(row["mean_difference"] == 0.0 for row in a3_rows)
    assert all(row["median_difference"] == 0.0 for row in a3_rows)
    assert all(row["mean_ci_lower"] is None for row in a3_rows)
    assert all(row["mean_ci_upper"] is None for row in a3_rows)
    assert all(row["median_ci_lower"] is None for row in a3_rows)
    assert all(row["median_ci_upper"] is None for row in a3_rows)
    assert all(row["bootstrap_replicates"] == 0 for row in a3_rows)


def test_scheduled_target_doses_come_from_release_provenance() -> None:
    scopes = []
    for stage, domain in enumerate(("T", "C", "B"), start=1):
        scopes.append(
            {
                "stage": stage,
                "current_domain": domain,
                "allocation": {
                    "releases": [
                        {
                            "release_index": release,
                            "replay_positions": list(
                                range(
                                    stage * 10_000 + release * 20,
                                    stage * 10_000 + (release + 1) * 20,
                                )
                            ),
                        }
                        for release in range(4)
                    ]
                },
            }
        )
    item = SimpleNamespace(
        query={"scopes": scopes},
        run=SimpleNamespace(
            budget=rdx_training_analysis.RDX004Budget.B100,
            sequence=("U", "T", "C", "B"),
            run_id="E4_OFFLINE_ORACLE_U-T-C-B_s42",
        ),
    )

    assert rdx_training_analysis._scheduled_target_doses(item) == {
        "1:T": [20, 40, 60, 80],
        "2:C": [20, 40, 60, 80],
        "3:B": [20, 40, 60, 80],
    }


def test_accepted_model_changing_updates_excludes_a1_and_a0() -> None:
    records = []
    for action in ("A0_NO_OP", "A1_RECALIBRATE", "A2_HEAD_UPDATE", "A3_FULL_FINE_TUNE"):
        record = _decision()
        record["selected_action"] = action
        records.append(record)

    assert rdx_training_analysis._accepted_model_changing_updates(records) == 2


def test_gate_core_followup_includes_exact_two_point_boundary() -> None:
    rows = _primary_rows(b400_p2=_constant(Fraction(1, 50)))
    result = evaluate_decision_gate(rows)

    assert len(rows) == 36
    assert result["status"] == "CORE_FOLLOWUP_JUSTIFIED"
    assert result["qualifying_budgets"] == ["B400"]


def test_gate_action_space_followup_requires_material_a3_exclusive_same_budget() -> None:
    material = _constant(Fraction(1, 50))
    result = evaluate_decision_gate(_primary_rows(b400_p1=material, b400_a3=material))

    assert result["status"] == "ACTION_SPACE_FOLLOWUP_JUSTIFIED"
    assert result["qualifying_budgets"] == ["B400"]


def test_gate_reports_no_material_recoverability_expansion() -> None:
    below = _constant(Fraction(19, 1000))
    result = evaluate_decision_gate(_primary_rows(b400_p1=below, b1600_p2=below))

    assert result["status"] == "TRAINING_EVIDENCE_INCREASE_NOT_MATERIALLY_RECOVERABILITY_EXPANDING"
    assert result["qualifying_budgets"] == []


def test_gate_preserves_mixed_or_order_dependent_status() -> None:
    mixed = {
        _ROTATIONS[0]: Fraction(1, 25),
        _ROTATIONS[1]: Fraction(1, 25),
        _ROTATIONS[2]: Fraction(0),
        _ROTATIONS[3]: Fraction(0),
    }
    result = evaluate_decision_gate(_primary_rows(b400_p2=mixed))

    assert result["status"] == "NO_FROZEN_FOLLOWUP_GATE_REACHED_MIXED_OR_ORDER_DEPENDENT"
    assert result["qualifying_budgets"] == []


def test_gate_incomplete_precedes_an_otherwise_qualifying_core_result() -> None:
    rows = _primary_rows(b400_p2=_constant(Fraction(1, 50)))
    rows[0] = {
        **rows[0],
        "p1_exact": "",
        "p2_exact": "",
        "a3_exclusive_exact": "",
    }

    result = evaluate_decision_gate(rows)

    assert result["status"] == "NOT_ASSESSED_INCOMPLETE"
    assert result["qualifying_budgets"] == []


def test_gate_core_status_has_precedence_over_action_space_status() -> None:
    material = _constant(Fraction(1, 50))
    result = evaluate_decision_gate(
        _primary_rows(
            b400_p2=material,
            b1600_p1=material,
            b1600_a3=material,
        )
    )

    assert result["status"] == "CORE_FOLLOWUP_JUSTIFIED"
    assert result["qualifying_budgets"] == ["B400"]


def test_rdx007_cli_contract() -> None:
    args = build_parser().parse_args(
        [
            "analyze-rdx004-training-evidence",
            "--study4-evaluation-dir",
            "study4/e4-confirmatory-final",
            "--preflight-dir",
            "rdx/training-evidence-preflight-v1",
            "--run-root",
            "runs/rdx004-training-evidence-v1",
            "--output-dir",
            "rdx/training-evidence-analysis-v1",
        ]
    )

    assert args.command == "analyze-rdx004-training-evidence"
    assert args.study4_evaluation_dir == Path("study4/e4-confirmatory-final")
    assert args.preflight_dir == Path("rdx/training-evidence-preflight-v1")
    assert args.run_root == Path("runs/rdx004-training-evidence-v1")
    assert args.output_dir == Path("rdx/training-evidence-analysis-v1")


def test_rdx007_cli_reports_analysis_errors_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_analysis(*_args: object, **_kwargs: object) -> Path:
        raise RdxTrainingAnalysisError("analysis contract differs")

    monkeypatch.setattr(cli, "analyze_rdx004_training_evidence", fail_analysis)

    status = cli.main(
        [
            "analyze-rdx004-training-evidence",
            "--study4-evaluation-dir",
            "study4/e4-confirmatory-final",
            "--preflight-dir",
            "rdx/training-evidence-preflight-v1",
            "--run-root",
            "runs/rdx004-training-evidence-v1",
            "--output-dir",
            "rdx/training-evidence-analysis-v1",
        ]
    )

    captured = capsys.readouterr()
    assert status == 2
    assert captured.out == ""
    assert captured.err == "error: analysis contract differs\n"


def test_write_bundle_does_not_publish_before_independent_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs = SimpleNamespace()
    output = tmp_path / "analysis"
    monkeypatch.setattr(
        rdx_training_analysis,
        "_ensure_output_is_separate",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        rdx_training_analysis,
        "_expected_texts",
        lambda _inputs: ({"result.txt": "result\n"}, {}, {}),
    )
    monkeypatch.setattr(
        rdx_training_analysis,
        "_manifest_payload",
        lambda _inputs, _root: {"version": "test"},
    )
    monkeypatch.setattr(
        rdx_training_analysis,
        "_validate_output_with_inputs",
        lambda *_args, **_kwargs: None,
    )

    def fail_validation(_output: Path) -> None:
        raise RdxTrainingAnalysisError("independent validation failed")

    monkeypatch.setattr(
        rdx_training_analysis,
        "validate_rdx_training_analysis",
        fail_validation,
    )

    with pytest.raises(RdxTrainingAnalysisError, match="independent validation failed"):
        rdx_training_analysis._write_bundle(inputs, output)

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []
