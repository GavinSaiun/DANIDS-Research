import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from danids.adaptation.actions import (
    InterventionAction,
    replay_flat_positions_digest,
    replay_scoped_positions_digest,
)
from danids.attacks import load_study5_contract
from danids.continual.supervision import row_positions_digest
from danids.evaluation.study5 import (
    Study5ThreatAuditError,
    _validate_study4_learned_state_metadata,
)
from danids.evaluation.study5_sources import (
    Study5SourceError,
    _accepted_s4_intervention_by_prediction,
    _canonical_native_group,
    _native_counts,
    _resolve_s4_learned_states,
    _validate_combined_rows,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = load_study5_contract(ROOT / "configs/study5/attack_ontology_v1.yaml")


def _native_group(label: str = "DoS") -> list[dict[str, object]]:
    return _canonical_native_group(
        contract=CONTRACT,
        native_rows=(
            {
                "native_attack_label": label,
                "support": "50",
                "recall": "0.6",
                "macro_native_attack_recall": "0.6",
                "worst_native_attack_recall": "0.6",
            },
        ),
        binary_row={"attack_count": "50", "tp": "30", "fn": "20", "tpr": "0.6"},
        source_study="S1",
        analysis_role="STUDY1_STATIC_MATRIX",
        experiment_id="E1_STATIC_MLP_U-T-C-B_s42",
        method="static",
        sequence=("U", "T", "C", "B"),
        seed=42,
        stratum="ONLINE_STREAM",
        lifecycle_event="window_prediction",
        stage=2,
        event_index=None,
        prediction_index=None,
        window_id=0,
        evaluated_domain="U",
        row_start=0,
        row_stop=50,
        timestamp_start=None,
        timestamp_end=None,
        model_digest="model",
        operating_envelope_state=None,
        fingerprint=CONTRACT.dataset_fingerprints["U"],
    )


def test_native_group_rejects_unknown_exact_native_label() -> None:
    with pytest.raises(Study5SourceError, match="unknown exact native label"):
        _native_group("dos")


def test_native_count_reconstruction_rejects_corrupt_recall() -> None:
    with pytest.raises(Study5SourceError, match="cannot be reconstructed exactly"):
        _native_counts({"support": "50", "recall": "0.611"}, "corrupt")


def test_permanent_holdout_rejects_redistributed_native_support() -> None:
    attacks = [
        item
        for item in CONTRACT.native_attacks
        if item.key.dataset_id == "U" and item.support.permanent_holdout > 0
    ]
    supports = {item.key.exact_native_label: item.support.permanent_holdout for item in attacks}
    supports[attacks[0].key.exact_native_label] += 1
    supports[attacks[1].key.exact_native_label] -= 1
    rows = tuple(
        {
            "native_attack_label": label,
            "support": str(support),
            "recall": "1.0",
            "macro_native_attack_recall": "1.0",
            "worst_native_attack_recall": "1.0",
        }
        for label, support in supports.items()
    )
    attack_count = sum(supports.values())
    with pytest.raises(Study5SourceError, match="differs from TASK-007"):
        _canonical_native_group(
            contract=CONTRACT,
            native_rows=rows,
            binary_row={
                "attack_count": attack_count,
                "tp": attack_count,
                "fn": 0,
                "tpr": 1.0,
            },
            source_study="S1",
            analysis_role="STUDY1_STATIC_MATRIX",
            experiment_id="E1_STATIC_MLP_U-T-C-B_s42",
            method="static",
            sequence=("U", "T", "C", "B"),
            seed=42,
            stratum="PERMANENT_HOLDOUT",
            lifecycle_event="source_initial",
            stage=1,
            event_index=None,
            prediction_index=None,
            window_id=None,
            evaluated_domain="U",
            row_start=2_000_000,
            row_stop=2_500_000,
            timestamp_start=None,
            timestamp_end=None,
            model_digest="model",
            operating_envelope_state=None,
            fingerprint=CONTRACT.dataset_fingerprints["U"],
        )


def test_combined_rows_reject_paired_physical_support_mismatch() -> None:
    first = _native_group()[0]
    second = dict(first)
    second["experiment_id"] = "E1_STATIC_MLP_U-T-C-B_s43"
    second["evaluation_slice_digest"] = "b" * 64
    second["support"] = 49
    with pytest.raises(Study5SourceError, match="different family support"):
        _validate_combined_rows((first, second))


def test_study4_learned_state_uses_final_post_accept_and_own_domain_end() -> None:
    run = SimpleNamespace(
        path=Path("synthetic"),
        experiment_id="E4_ALWAYS_ADAPT_U-T-C-B_s42",
        sequence=("U", "T", "C", "B"),
        holdouts=(
            {
                "event": "source_initial",
                "holdout_dataset_id": "U",
                "stage": 0,
                "event_index": 1,
                "prediction_index": None,
            },
            {
                "event": "post_accept",
                "holdout_dataset_id": "T",
                "stage": 1,
                "event_index": 3,
                "prediction_index": 10,
            },
            {
                "event": "post_accept",
                "holdout_dataset_id": "T",
                "stage": 1,
                "event_index": 5,
                "prediction_index": 20,
            },
            {
                "event": "domain_end",
                "holdout_dataset_id": "C",
                "stage": 2,
                "event_index": 6,
                "prediction_index": 30,
            },
            {
                "event": "domain_end",
                "holdout_dataset_id": "C",
                "stage": 3,
                "event_index": 7,
                "prediction_index": 40,
            },
            {
                "event": "domain_end",
                "holdout_dataset_id": "B",
                "stage": 3,
                "event_index": 8,
                "prediction_index": 50,
            },
        ),
    )
    accepted = {
        10: {"_study5_update_evidence_domains": ("T",)},
        20: {"_study5_update_evidence_domains": ("T",)},
    }

    anchors, availability = _resolve_s4_learned_states(run, accepted)

    assert anchors["T"] == ("post_accept", 5, 20)
    statuses = {item.dataset_id: item.status for item in availability}
    assert statuses == {
        "U": "LEARNED",
        "T": "LEARNED",
        "C": "NOT_LEARNED_NO_UPDATE",
        "B": "NOT_LEARNED_NO_UPDATE",
    }


def test_study4_accepted_replay_attributes_all_released_evidence_domains(
    tmp_path: Path,
) -> None:
    target = (1, 2)
    replay_scopes = [
        {
            "scope_id": "U",
            "row_positions": [5],
            "row_positions_digest": row_positions_digest([5]),
        },
        {
            "scope_id": "token-t",
            "row_positions": [10, 11],
            "row_positions_digest": row_positions_digest([10, 11]),
        },
    ]
    replay_flat = (5, 10, 11)
    query_log = {
        "scopes": [
            {
                "stage": 1,
                "current_domain": "T",
                "opaque_scope_token": "token-t",
                "allocation": {
                    "scope_id": "token-t",
                    "historical": True,
                    "historical_activation_window": 10,
                    "replay_positions": [10, 11],
                    "releases": [{"release_window": 2, "replay_positions": [10, 11]}],
                },
            },
            {
                "stage": 2,
                "current_domain": "C",
                "opaque_scope_token": "token-c",
                "allocation": {
                    "scope_id": "token-c",
                    "historical": False,
                    "historical_activation_window": None,
                    "replay_positions": [1, 2],
                    "releases": [{"release_window": 11, "replay_positions": [1, 2]}],
                },
            },
        ]
    }
    (tmp_path / "query_log.json").write_text(json.dumps(query_log), encoding="utf-8")
    run = SimpleNamespace(
        path=tmp_path,
        sequence=("U", "T", "C", "B"),
        windows=(
            {
                "prediction_index": 11,
                "stage": 2,
                "current_domain": "C",
                "window_id": 0,
            },
        ),
        interventions=(
            {
                "accepted": True,
                "prediction_index": 11,
                "domain_stage": 2,
                "current_domain": "C",
                "window_id": 0,
                "labels_available": 2,
                "action_attempted": InterventionAction.REPLAY_UPDATE.value,
                "target_rows": 2,
                "target_row_positions": json.dumps(target),
                "target_row_positions_digest": row_positions_digest(target),
                "calibration_rows": 0,
                "calibration_row_positions": "[]",
                "calibration_row_positions_digest": "",
                "replay_rows": 3,
                "replay_row_positions": json.dumps(replay_flat),
                "replay_row_positions_digest": replay_flat_positions_digest(replay_flat),
                "replay_scoped_row_positions": json.dumps(replay_scopes),
                "replay_scoped_row_positions_digest": replay_scoped_positions_digest(replay_scopes),
            },
        ),
    )

    accepted = _accepted_s4_intervention_by_prediction(run)

    assert accepted[11]["_study5_update_evidence_domains"] == ("U", "T", "C")


def test_study4_learned_metadata_rejects_conflicting_native_status() -> None:
    experiment_id = "E4_STATIC_U-T-C-B_s42"
    records = [
        {
            "experiment_id": experiment_id,
            "dataset_id": "U",
            "status": "LEARNED",
            "lifecycle_event": "source_initial",
            "event_index": 0,
            "prediction_index": None,
        },
        {
            "experiment_id": experiment_id,
            "dataset_id": "T",
            "status": "NOT_LEARNED_NO_UPDATE",
            "lifecycle_event": "domain_end",
            "event_index": 1,
            "prediction_index": 9,
        },
        {
            "experiment_id": experiment_id,
            "dataset_id": "C",
            "status": "NOT_LEARNED_NO_UPDATE",
            "lifecycle_event": "domain_end",
            "event_index": 2,
            "prediction_index": 19,
        },
        {
            "experiment_id": experiment_id,
            "dataset_id": "B",
            "status": "NOT_LEARNED_NO_UPDATE",
            "lifecycle_event": "domain_end",
            "event_index": 3,
            "prediction_index": 29,
        },
    ]
    rows = [
        {
            "source_study": "S4",
            "experiment_id": experiment_id,
            "evaluated_domain": record["dataset_id"],
            "learned_state_status": (
                "LEARNED_REFERENCE" if record["status"] == "LEARNED" else "NOT_LEARNED_NO_UPDATE"
            ),
            "evaluation_slice_digest": str(index) * 64,
            "lifecycle_event": record["lifecycle_event"],
            "stage": index,
            "event_index": record["event_index"],
            "prediction_index": record["prediction_index"],
            "update_evidence_domain": "",
        }
        for index, record in enumerate(records)
    ]
    _validate_study4_learned_state_metadata(
        records,
        {experiment_id: ("U", "T", "C", "B")},
        rows,
    )
    corrupted = [
        *rows,
        {
            **rows[1],
            "learned_state_status": "LEARNED_REFERENCE",
            "evaluation_slice_digest": "f" * 64,
            "lifecycle_event": "post_accept",
            "event_index": 4,
            "prediction_index": 11,
            "update_evidence_domain": "T",
        },
    ]
    with pytest.raises(Study5ThreatAuditError, match="conflicting learned-state tags"):
        _validate_study4_learned_state_metadata(
            records,
            {experiment_id: ("U", "T", "C", "B")},
            corrupted,
        )
