from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import danids.evaluation.study5 as study5
from danids.attacks import FROZEN_STUDY5_DATASET_FINGERPRINTS
from danids.cli import main
from danids.evaluation.study5 import (
    ALL_OUTPUT_FILES,
    OUTPUT_SCHEMAS,
    Study5ThreatAuditError,
    evaluate_study5_threats,
    validate_study5_threat_audit,
)
from danids.evaluation.threat_estimands import physical_slice_digest, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "study5" / "attack_ontology_v1.yaml"


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _row(
    *,
    study: str,
    experiment: str,
    method: str,
    event: str,
    event_index: int,
    stage: int,
    domain: str,
    label: str,
    family: str,
    support: int,
    tp: int,
    learned: str = "",
) -> dict[str, object]:
    sequence = "U-T-C-B"
    start = {"U": 80, "T": 800}[domain]
    stop = {"U": 100, "T": 1_000}[domain]
    payload = {
        "dataset_fingerprint": FROZEN_STUDY5_DATASET_FINGERPRINTS[domain],
        "dataset_id": domain,
        "stratum": "PERMANENT_HOLDOUT",
        "row_start": start,
        "row_stop": stop,
    }
    recall = tp / support
    low, high = wilson_interval(tp, support)
    model_digest = _sha({"experiment": experiment, "event": event})
    analysis_role = {
        "S1": "STUDY1_STATIC_MATRIX",
        "S2": "STUDY2_STATIC_REFERENCE" if method == "static" else "STUDY2_ADAPTIVE",
        "S4": "STUDY4_CONFIRMATORY",
    }[study]
    evaluation = _sha(
        {
            **payload,
            "source_study": study,
            "analysis_role": analysis_role,
            "experiment_id": experiment,
            "method": method,
            "sequence": sequence.split("-"),
            "seed": 42,
            "lifecycle_event": event,
            "stage": stage,
            "event_index": event_index,
            "prediction_index": None,
            "window_id": None,
            "model_digest": model_digest,
        }
    )
    novelty = (
        "PREVIOUSLY_SEEN"
        if (domain, family)
        in {
            ("U", "Availability / Impact"),
        }
        else "PREVIOUSLY_UNSEEN"
    )
    return {
        "source_study": study,
        "experiment_id": experiment,
        "method": method,
        "sequence": sequence,
        "seed": 42,
        "stratum": "PERMANENT_HOLDOUT",
        "lifecycle_event": event,
        "stage": stage,
        "event_index": event_index,
        "prediction_index": "",
        "window_id": "",
        "evaluated_domain": domain,
        "native_attack_label": label,
        "mapping_status": "MAPPED",
        "semantic_family": family,
        "support": support,
        "tp": tp,
        "fn": support - tp,
        "recall": recall,
        "wilson95_low": low,
        "wilson95_high": high,
        "supported_n50": support >= 50,
        "novelty_status": novelty,
        "family_label_available_before_prediction": (
            True if family == "Availability / Impact" else ""
        ),
        "label_availability_status": (
            "AVAILABLE_SOURCE_LABELLED_EXPOSURE"
            if family == "Availability / Impact"
            else "UNAVAILABLE_INSUFFICIENT_ARTIFACT_PROVENANCE"
        ),
        "aggregate_support": support,
        "aggregate_tp": tp,
        "aggregate_fn": support - tp,
        "aggregate_recall": recall,
        "operating_envelope_state": "",
        "model_digest": model_digest,
        "physical_slice_start": start,
        "physical_slice_stop": stop,
        "physical_slice_identity": json.dumps(payload, sort_keys=True, separators=(",", ":")),
        "physical_slice_digest": physical_slice_digest(payload),
        "evaluation_slice_digest": evaluation,
        "learned_state_status": learned,
        "update_evidence_domain": domain if learned == "LEARNED_REFERENCE" else "",
    }


def _extraction() -> SimpleNamespace:
    rows: list[dict[str, object]] = []
    s1 = "E1_STATIC_MLP_U-T-C-B_s42"
    rows.extend(
        [
            _row(
                study="S1",
                experiment=s1,
                method="static",
                event="source_initial",
                event_index=1,
                stage=1,
                domain="U",
                label="DoS",
                family="Availability / Impact",
                support=100,
                tp=95,
                learned="LEARNED_REFERENCE",
            ),
            _row(
                study="S1",
                experiment=s1,
                method="static",
                event="final",
                event_index=4,
                stage=4,
                domain="U",
                label="DoS",
                family="Availability / Impact",
                support=100,
                tp=95,
            ),
            _row(
                study="S1",
                experiment=s1,
                method="static",
                event="domain_end",
                event_index=4,
                stage=2,
                domain="T",
                label="xss",
                family="Application / Web Injection",
                support=100,
                tp=70,
            ),
        ]
    )
    s2 = "E2_NAIVEFT_U-T-C-B_B100_D1_s42"
    for event, index, tp, learned in (
        ("source_initial", 1, 95, "LEARNED_REFERENCE"),
        ("final", 10, 90, ""),
    ):
        rows.append(
            _row(
                study="S2",
                experiment=s2,
                method="naive_ft",
                event=event,
                event_index=index,
                stage=1 if event == "source_initial" else 4,
                domain="U",
                label="DoS",
                family="Availability / Impact",
                support=100,
                tp=tp,
                learned=learned,
            )
        )
    for event, index, tp, learned in (
        ("post_adapt", 3, 80, "LEARNED_REFERENCE"),
        ("domain_end", 4, 75, ""),
        ("final", 10, 70, ""),
    ):
        rows.append(
            _row(
                study="S2",
                experiment=s2,
                method="naive_ft",
                event=event,
                event_index=index,
                stage=2 if event != "final" else 4,
                domain="T",
                label="xss",
                family="Application / Web Injection",
                support=100,
                tp=tp,
                learned=learned,
            )
        )
    s4 = "E4_STATIC_U-T-C-B_s42"
    for event, index, domain, label, family, tp, learned in (
        (
            "source_initial",
            1,
            "U",
            "DoS",
            "Availability / Impact",
            95,
            "LEARNED_REFERENCE",
        ),
        ("final", 10, "U", "DoS", "Availability / Impact", 95, ""),
        (
            "domain_end",
            3,
            "T",
            "xss",
            "Application / Web Injection",
            60,
            "NOT_LEARNED_NO_UPDATE",
        ),
        (
            "final",
            10,
            "T",
            "xss",
            "Application / Web Injection",
            55,
            "",
        ),
    ):
        rows.append(
            _row(
                study="S4",
                experiment=s4,
                method="STATIC",
                event=event,
                event_index=index,
                stage=(0 if event == "source_initial" else 1 if event == "domain_end" else 3),
                domain=domain,
                label=label,
                family=family,
                support=100,
                tp=tp,
                learned=learned,
            )
        )
    sources = {
        "version": study5.STUDY5_THREAT_AUDIT_VERSION,
        "runs": [
            {"source_study": "S1", "method": "static", "experiment_id": s1},
            {"source_study": "S2", "method": "naive_ft", "experiment_id": s2},
            {"source_study": "S4", "method": "STATIC", "experiment_id": s4},
        ],
        "aggregates": [],
        "study4_learned_state_availability": [
            {
                "experiment_id": s4,
                "dataset_id": "U",
                "status": "LEARNED",
                "lifecycle_event": "source_initial",
                "event_index": 1,
                "prediction_index": None,
            },
            {
                "experiment_id": s4,
                "dataset_id": "T",
                "status": "NOT_LEARNED_NO_UPDATE",
                "lifecycle_event": "domain_end",
                "event_index": 3,
                "prediction_index": None,
            },
        ],
    }
    return SimpleNamespace(native_rows=tuple(rows), source_artifacts=sources)


@pytest.fixture
def audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        study5,
        "load_study5_source_artifacts",
        lambda *_args, **_kwargs: _extraction(),
    )
    # The writer/round-trip fixture is intentionally a three-run miniature.  Exact
    # 75-run source-matrix enforcement is exercised by test_study5_sources.py and the
    # dedicated source-metadata corruption test below.
    monkeypatch.setattr(study5, "_validate_source_artifacts", lambda *_args: None)
    return evaluate_study5_threats(
        contract_path=CONTRACT,
        study1_run_dirs=[tmp_path / "s1"],
        study1_evaluation_dir=tmp_path / "s1-eval",
        study2_static_run_dirs=[tmp_path / "s2-static"],
        study2_run_dirs=[tmp_path / "s2"],
        study2_evaluation_dir=tmp_path / "s2-eval",
        study4_run_dirs=[tmp_path / "s4"],
        study4_evaluation_dir=tmp_path / "s4-eval",
        output_dir=tmp_path / "audit",
    )


def _reseal(root: Path) -> None:
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in sorted(ALL_OUTPUT_FILES.difference({study5.MANIFEST_FILENAME}))
    }
    (root / study5.MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "version": study5.STUDY5_THREAT_AUDIT_VERSION,
                "files": files,
                "bundle_digest": study5._digest(files),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_artifact_only_evaluation_writes_exact_contract_and_self_validates(audit: Path) -> None:
    assert {path.name for path in audit.iterdir()} == ALL_OUTPUT_FILES
    for name, columns in OUTPUT_SCHEMAS.items():
        assert (audit / name).read_text(encoding="utf-8").splitlines()[0] == ",".join(columns)
    summary = json.loads((audit / "study5_summary.json").read_text(encoding="utf-8"))
    assert summary["family_conditioned_binary_detection_not_attribution"] is True
    assert summary["raw_data_accessed"] is False
    assert summary["models_scored"] is False
    assert summary["study4_family_learned_state_availability"]["NOT_LEARNED_NO_UPDATE"] == 1
    validate_study5_threat_audit(audit)


def test_evaluation_is_write_once(audit: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        study5,
        "load_study5_source_artifacts",
        lambda *_args, **_kwargs: _extraction(),
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluate_study5_threats(
            contract_path=CONTRACT,
            study1_run_dirs=[],
            study1_evaluation_dir=audit,
            study2_static_run_dirs=[],
            study2_run_dirs=[],
            study2_evaluation_dir=audit,
            study4_run_dirs=[],
            study4_evaluation_dir=audit,
            output_dir=audit,
        )


def test_resealed_derived_metric_corruption_is_rejected(audit: Path) -> None:
    path = audit / "semantic_supported_summary.csv"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("0.95", "0.94", 1), encoding="utf-8")
    _reseal(audit)
    with pytest.raises(Study5ThreatAuditError, match="differs from recomputation"):
        validate_study5_threat_audit(audit)


def test_resealed_native_count_corruption_is_rejected(audit: Path) -> None:
    path = audit / "native_family_long.csv"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(",95,5,0.95,", ",94,6,0.95,", 1), encoding="utf-8")
    _reseal(audit)
    with pytest.raises(Study5ThreatAuditError, match="TP/FN"):
        validate_study5_threat_audit(audit)


def test_study5_threat_cli_help_is_available() -> None:
    with pytest.raises(SystemExit) as raised:
        main(["evaluate-study5-threats", "--help"])
    assert raised.value.code == 0


def test_source_metadata_contract_fails_closed_before_summary_derivation() -> None:
    with pytest.raises(Study5ThreatAuditError, match="metadata schema"):
        study5._validate_source_artifacts({}, [], study5.load_study5_contract(CONTRACT))


def test_study1_supported_transfer_uses_one_paired_semantic_family_set() -> None:
    experiment = "E1_STATIC_MLP_U-T-C-B_s42"
    native: list[dict[str, object]] = []
    for event, stage, domain, cells in (
        (
            "source_initial",
            1,
            "U",
            (
                ("DoS", "Availability / Impact", 90),
                ("Reconnaissance", "Reconnaissance / Discovery", 70),
            ),
        ),
        (
            "domain_end",
            2,
            "T",
            (
                ("ddos", "Availability / Impact", 80),
                ("scanning", "Reconnaissance / Discovery", 70),
            ),
        ),
    ):
        evaluation_rows = [
            _row(
                study="S1",
                experiment=experiment,
                method="static",
                event=event,
                event_index=stage,
                stage=stage,
                domain=domain,
                label=label,
                family=family,
                support=100,
                tp=tp,
            )
            for label, family, tp in cells
        ]
        aggregate_tp = sum(int(row["tp"]) for row in evaluation_rows)
        for row in evaluation_rows:
            row.update(
                {
                    "aggregate_support": 200,
                    "aggregate_tp": aggregate_tp,
                    "aggregate_fn": 200 - aggregate_tp,
                    "aggregate_recall": aggregate_tp / 200,
                }
            )
        native.extend(evaluation_rows)

    rows = study5._study1_supported_transfer(study5._semantic_rows(native))

    assert len(rows) == 3
    row = next(row for row in rows if row["target_domain"] == "T")
    assert row["paired_supported_family_count"] == 2
    assert json.loads(str(row["paired_supported_family_identities"])) == [
        "Availability / Impact",
        "Reconnaissance / Discovery",
    ]
    assert row["source_macro_supported_family_recall"] == pytest.approx(0.8)
    assert row["target_macro_supported_family_recall"] == pytest.approx(0.75)
    assert row["macro_supported_family_recall_loss"] == pytest.approx(0.05)
    assert json.loads(str(row["source_worst_family_identities"])) == ["Reconnaissance / Discovery"]


def test_semantic_summary_preserves_unmapped_only_physical_slice() -> None:
    base = _row(
        study="S1",
        experiment="E1_STATIC_MLP_U-T-C-B_s42",
        method="static",
        event="domain_end",
        event_index=2,
        stage=2,
        domain="T",
        label="injection",
        family="Application / Web Injection",
        support=100,
        tp=50,
    )
    base["mapping_status"] = "UNMAPPED"
    base["semantic_family"] = ""

    semantic = study5._semantic_rows([base])
    summary = study5._supported_summaries(
        semantic,
        level="SEMANTIC",
        base_rows=[base],
    )

    assert semantic == []
    assert len(summary) == 1
    assert summary[0]["supported_family_count"] == 0
    assert summary[0]["macro_supported_family_recall"] is None
    assert summary[0]["worst_supported_family_recall"] is None
    assert json.loads(str(summary[0]["worst_family_identities"])) == []


def test_supported_worst_family_ties_use_exact_count_ratios() -> None:
    first = _row(
        study="S1",
        experiment="E1_STATIC_MLP_U-T-C-B_s42",
        method="static",
        event="source_initial",
        event_index=1,
        stage=1,
        domain="U",
        label="DoS",
        family="Availability / Impact",
        support=1_000_000,
        tp=999_999,
    )
    second = {
        **first,
        "native_attack_label": "Reconnaissance",
        "support": 999_999,
        "tp": 999_998,
        "fn": 1,
        "recall": 999_998 / 999_999,
    }

    summary = study5._supported_summaries([first, second], level="NATIVE")[0]

    assert json.loads(str(summary["worst_family_identities"])) == ["U::Reconnaissance"]


def test_safe_envelope_secondary_hidden_failure_is_separately_named() -> None:
    experiment = "E4_STATIC_U-T-C-B_s42"
    native: list[dict[str, object]] = []
    for event, event_index, stage, cells, aggregate_tp, state, learned in (
        (
            "source_initial",
            1,
            0,
            (
                ("DoS", "Availability / Impact", 95),
                ("Reconnaissance", "Reconnaissance / Discovery", 95),
            ),
            190,
            "SAFE",
            "LEARNED_REFERENCE",
        ),
        (
            "final",
            10,
            3,
            (
                ("DoS", "Availability / Impact", 70),
                ("Reconnaissance", "Reconnaissance / Discovery", 98),
            ),
            168,
            "SAFE",
            "",
        ),
    ):
        members = [
            _row(
                study="S4",
                experiment=experiment,
                method="STATIC",
                event=event,
                event_index=event_index,
                stage=stage,
                domain="U",
                label=label,
                family=family,
                support=100,
                tp=tp,
                learned=learned,
            )
            for label, family, tp in cells
        ]
        for row in members:
            row.update(
                {
                    "aggregate_support": 200,
                    "aggregate_tp": aggregate_tp,
                    "aggregate_fn": 200 - aggregate_tp,
                    "aggregate_recall": aggregate_tp / 200,
                    "operating_envelope_state": state,
                }
            )
        native.extend(members)
    semantic = study5._semantic_rows(native)

    failures = study5._hidden_failures(native, semantic, [])

    availability = [
        row for row in failures if row["family_identity"] in {"U::DoS", "Availability / Impact"}
    ]
    assert availability
    assert {row["hidden_failure_variant"] for row in availability} == {
        "SECONDARY_SAFE_OPERATING_ENVELOPE"
    }


def test_study2_forgetting_uses_post_adapt_and_excludes_static_reference() -> None:
    native = list(_extraction().native_rows)
    rows = study5._family_forgetting(native, study5._semantic_rows(native))
    target = [
        row
        for row in rows
        if row["source_study"] == "S2"
        and row["family_level"] == "SEMANTIC"
        and row["evaluated_domain"] == "T"
    ]

    assert len(target) == 1
    assert target[0]["method"] == "naive_ft"
    assert target[0]["learned_lifecycle_event"] == "post_adapt"
    assert target[0]["forgetting"] == pytest.approx(0.10)
    assert all(row["method"] != "static" for row in rows if row["source_study"] == "S2")
