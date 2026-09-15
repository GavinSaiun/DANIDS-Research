# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from danids.cli import build_parser
from danids.evaluation import thesis_assets
from danids.evaluation.thesis_assets import (
    ASSET_VERSION,
    CAPTIONS,
    DIGEST_POLICY,
    DISPLAY_SPECS,
    FIGURE_STEMS,
    GENERATION_COMMAND,
    SCIENTIFIC_PROCESSING,
    SOURCE_PATHS,
    TABLE_STEMS,
    _captions_markdown,
    _figure_f1,
    _hash_mode,
    _output_record,
    _projection_checks,
    _table_t1,
    _table_t2,
    _table_t4,
    verify_thesis_assets,
)
from danids.evaluation.thesis_assets import _sha256 as _canonical_sha256
from danids.evaluation.thesis_style import DOMAIN_LABELS, export_figure, thesis_style


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_thesis_display_contract_is_exact_and_uses_frozen_sources() -> None:
    assert FIGURE_STEMS == {
        "F1": "F01_danids_pipeline",
        "F2": "F02_protocol_information_boundary",
        "F3": "F03_static_transfer",
        "F4": "F04_adaptation_tradeoff",
        "F5": "F05_model_health",
        "F6": "F06_core_state_machine",
        "F7": "F07_study4_safety_resource",
        "F8": "F08_family_failure",
        "F9": "F09_order_replay_effects",
        "F10": "F10_evidence_synthesis",
    }
    assert TABLE_STEMS == {
        "T1": "T01_rq_hypothesis_map",
        "T2": "T02_literature_positioning",
        "T3": "T03_study_design_crosswalk",
        "T4": "T04_policy_qualification",
    }
    assert set(DISPLAY_SPECS) == {*FIGURE_STEMS, *TABLE_STEMS}
    assert SOURCE_PATHS["study5b_verdict"].endswith("all_order_synthesis_verdict.json")
    assert SOURCE_PATHS["policy_qualification"] == (
        "study4/policy-qualification-v1-final/policy_qualification.json"
    )
    assert SOURCE_PATHS["literature_t02"] == "docs/literature_t02_evidence.md"
    assert all("runs/" not in path for path in SOURCE_PATHS.values())


def test_display_provenance_meets_the_frozen_minimum() -> None:
    assert DISPLAY_SPECS["T2"].sources == ("literature_t02", "freeze", "plan")
    assert DISPLAY_SPECS["F2"].sources == (
        "freeze",
        "decisions",
        "study3_contract",
        "study4_contract",
        "study5_contract",
        "study5b_contract",
    )
    assert DISPLAY_SPECS["F6"].sources == ("freeze", "decisions", "study4_contract")
    assert DISPLAY_SPECS["F10"].sources == (
        "freeze",
        "study1_summary",
        "study2_summary",
        "study3_summary",
        "study4_results",
        "study5_summary",
        "study5b_verdict",
    )
    assert DISPLAY_SPECS["T3"].sources == (
        "study1_summary",
        "study2_summary",
        "study3_summary",
        "study3_contract",
        "study4_summary",
        "study4_contract",
        "study5_summary",
        "study5_contract",
        "study5b_summary",
        "study5b_contract",
    )
    assert DISPLAY_SPECS["T4"].sources == ("policy_qualification",)


def test_domain_labels_use_the_exact_frozen_dataset_ids(tmp_path: Path) -> None:
    assert DOMAIN_LABELS == {
        "U": "NF-UNSW-NB15-v3",
        "T": "NF-ToN-IoT-v3",
        "C": "NF-CSE-CIC-IDS2018-v3",
        "B": "NF-BoT-IoT-v3",
    }
    sources = {key: tmp_path / path for key, path in SOURCE_PATHS.items()}
    caption_text = _captions_markdown(sources, tmp_path)
    assert "C = NF-CSE-CIC-IDS2018-v3" in caption_text
    assert "CICIDS2017" not in caption_text


def test_frozen_ledger_and_wording_guardrails_are_preserved() -> None:
    table = _table_t1()
    ledger = str(table.loc[table["Research question"] == "Frozen ledger", "Frozen wording"].iloc[0])
    assert ledger == (
        "H1 SUPPORTED; H2 SUPPORTED; H3 NOT_SUPPORTED; H4 NOT_SUPPORTED; "
        "H6 NOT_TESTABLE; H7 NOT_SUPPORTED; H8 NOT_TESTABLE; H9 PARTIAL; H10 PARTIAL"
    )
    caption_text = "\n".join(CAPTIONS.values()).lower()
    assert "same-window harm screening" in caption_text
    assert "core reduced accepted update frequency, not supervision or labels" in caption_text
    assert "family-conditioned binary detection" in caption_text
    assert "mixed prior/prospective evidence" in caption_text
    assert "safe adaptation is impossible" in caption_text


def test_wording_guardrail_allows_only_the_frozen_early_warning_negation() -> None:
    assert (
        thesis_assets._wording_guardrail_violations(
            "This is same-window harm screening, not anticipatory early warning."
        )
        == ()
    )
    assert thesis_assets._wording_guardrail_violations(
        "The method provides anticipatory early warning."
    ) == ("anticipatory early warning",)


def test_t01_exactly_maps_the_five_reader_facing_research_questions() -> None:
    table = _table_t1()
    rows = table.loc[table["Research question"].str.fullmatch(r"RQ[1-5]")].to_dict("records")
    assert rows == [
        {
            "Research question": "RQ1",
            "Frozen wording": "How severe and direction-dependent is intrusion-detection degradation across heterogeneous network domains, and what operational trade-offs arise when standard continual-learning methods adapt one evolving detector?",
            "Study evidence": "Studies 1, 2",
            "Hypotheses": "H1",
            "Home chapter": "Chapter 4",
            "Final answer": "Static cross-domain degradation was substantial and asymmetric across recall, false-positive burden, and ranking performance. Straightforward adaptation did not solve the deployment problem: target-domain recall recovery could coexist with extreme false-positive burden or reduced earlier-domain competence, and threshold-free retention did not guarantee acceptable operation at the frozen threshold. Study 1's sequence positions are reporting positions, not causal order effects.",
        },
        {
            "Research question": "RQ2",
            "Frozen wording": "Which observable distributional and model-state signals can distinguish harmful operating-envelope violations from harmless domain change, and do combined or sparsely supervised health models generalise reliably?",
            "Study evidence": "Study 3",
            "Hypotheses": "H2; H3; H9",
            "Home chapter": "Chapter 5",
            "Final answer": "Many harmful windows were recognisable from permitted label-free observables, and the reviewed combined-unlabelled gradient-boosting model was the strongest label-free comparator selected for the frozen Core monitor. Distribution shift was not equivalent to harm, however, and neither combined signals nor sparse delayed labels dominated every simpler comparator across models, metrics, and grouped generalisation protocols. This is strong same-window harm screening, not anticipatory early warning.",
        },
        {
            "Research question": "RQ3",
            "Frozen wording": "Under the frozen B100/D1 supervision regime and A0--A4 action space, can health-aware selective adaptation maintain or restore operating-envelope safety while reducing supervision, accepted updates, and operational forgetting relative to unconditional adaptation?",
            "Study evidence": "Study 4",
            "Hypotheses": "H4; H10",
            "Home chapter": "Chapter 6",
            "Final answer": "No. Core reduced accepted update frequency relative to Always-Adapt, but did not reduce requested labels—the frozen label-usage artifact records equal requested-label totals—and did not maintain comparable operational safety. The descriptively favourable operational-forgetting component and fewer accepted updates did not rescue the no-supervision-reduction and failed comparable-safety components of H10. The scheduled Always-Adapt treatment was slightly worse than Static despite using 3,600 labels across its 12 runs, so unconditional adaptation did not restore aggregate safety. The non-deployable one-step Offline Oracle's sparse selected updates supported the minimum-intervention motivation, yet its approximately 19.06% compliance also showed that most observed harm was not recoverable under the frozen B100/D1 and A0--A4 regime. This does not establish that safe adaptation is impossible in general. DANIDS-Policy failed closed before fitting and was never deployed.",
        },
        {
            "Research question": "RQ4",
            "Frozen wording": "To what extent do aggregate binary metrics conceal family-specific detection failures across deployment histories, and what is the evidential boundary between family-conditioned detection, attack attribution, and open-set recognition?",
            "Study evidence": "Study 5A",
            "Hypotheses": "H6; H8",
            "Home chapter": "Chapter 7",
            "Final answer": "Aggregate binary recall could remain within the frozen loss tolerance while a physically supported family experienced a larger loss. The estimand is family-conditioned binary detection recall. No attribution predictions, structured embedding comparison, explicit UNKNOWN decision, or open-set experiment was performed, so H6 and H8 are not testable. The 27 hidden-family rows are repeated lifecycle/representation output rows, not 27 independent failures, and history-relative unseen status is not a real-world zero-day claim.",
        },
        {
            "Research question": "RQ5",
            "Frozen wording": "Does replay-aware adaptation robustly reduce supported attack-family forgetting and improve final previous-domain competence across deployment orders relative to target-only full fine-tuning?",
            "Study evidence": "Studies 2, 5B",
            "Hypotheses": "H7",
            "Home chapter": "Chapter 7",
            "Final answer": "No order-robust replay advantage was found. Positive U-T-C-B replay competence effects had been observed before TASK-009; the three missing rotations were then frozen prospectively and failed to reproduce the benefit. B-U-T-C showed materially adverse replay effects. The conclusion is deployment-order heterogeneity and reversal, not merely a near-zero pooled effect. The final four-order synthesis mixes prior and prospective evidence. Its primary level is mapped semantic families with physical support n >= 50; only previous domains in positions 1--3 are eligible, while position 4 is excluded. Effects are aggregated unweighted from family to domain to rotation--seed, and rotation--seed units—not flows or family rows—are the experimental units.",
        },
    ]
    assert not table["Study evidence"].astype(str).str.fullmatch("Synthesis").any()


def test_t02_is_a_complete_deterministic_literature_contract() -> None:
    table = _table_t2()
    substantive_columns = [
        "Prediction target",
        "Information available",
        "Typical unit",
        "Operational criterion",
        "Unresolved DANIDS layer",
    ]
    assert table.columns.tolist() == [
        "Area",
        *substantive_columns,
        "Citation status",
    ]
    assert table["Area"].tolist() == [
        "Cross-domain NIDS",
        "Continual learning",
        "Drift monitoring",
        "Selective adaptation",
        "Family/open-set evaluation",
    ]

    substantive = table.loc[:, substantive_columns]
    assert substantive.size == 25
    assert not substantive.isna().to_numpy().any()
    assert all(str(value).strip() for value in substantive.to_numpy().ravel())
    complete_text = "\n".join(table.astype(str).to_numpy().ravel().tolist())
    assert "CITATION_REQUIRED" not in complete_text
    assert "Pending verified literature" not in complete_text

    assert table["Citation status"].tolist() == [
        "Apruzzese et al. (2022); Layeghy & Portmann (2023); Layeghy et al. (2023)",
        "Parisi et al. (2019); Kirkpatrick et al. (2017); Lopez-Paz & Ranzato (2017); Delgado et al. (2026)",
        "Lu et al. (2019); Gama et al. (2004); Gretton et al. (2012); Rabanser et al. (2019)",
        "Horchulhack et al. (2022); Niu et al. (2022, 2023); Yoo et al. (2024)",
        "Elmasry et al. (2019); Cruz et al. (2017); Baye et al. (2023); Yu et al. (2024)",
    ]
    assert table["Unresolved DANIDS layer"].tolist() == [
        "Separating detected shift, fixed-threshold operational harm and sequential recoverability",
        "Label-free harm recognition and audited minimum intervention under hidden boundaries and delayed labels",
        "Whether same-window change is operationally harmful and safely repairable",
        "Health-conditioned A0-A4 choice under B100/D1 with audit-gated promotion or rollback",
        "DANIDS provides family-conditioned binary detection recall only, with no attribution or open-set output",
    ]
    assert "without claiming that the cited areas ignore deployment shift" in CAPTIONS["T2"]
    t02_guardrail = thesis_assets.INTERPRETATION_NOTES["T2"]
    assert "not a novelty census" in t02_guardrail
    assert "same-window harm screening" in t02_guardrail
    assert "bounded to the frozen B100/D1 and A0-A4 regime" in t02_guardrail


def test_t03_records_prospective_freezes_and_mixed_evidence_timing() -> None:
    table = thesis_assets._table_t3().set_index("Study")
    assert table.loc["Study 5A", "Evidence phase"] == (
        "Ontology and estimands frozen after Studies 1--4 existed but before artifact-only "
        "threat-effect analysis over the prior Study 1, 2, and 4 source bundles"
    )
    assert table.loc["Study 5B", "Evidence phase"] == (
        "9 prior U-T-C-B runs + 27 prospective missing-rotation runs; "
        "mixed prior/prospective four-order synthesis"
    )


def test_policy_qualification_table_is_a_direct_frozen_projection(tmp_path: Path) -> None:
    source = tmp_path / "policy_qualification.json"
    payload = {
        "calibration_component_count": 103,
        "minimum_recommendation_requirement": 100,
        "minimum_model_changing_selection_requirement": 25,
        "required_wilson_lower_bound": 0.9,
        "minimum_successes_required_at_n_100": 96,
        "maximum_achievable_confirmed_success_count": 10,
        "maximum_empirical_success_rate_at_minimum_recommendations": 0.1,
        "maximum_wilson_lower_bound": 0.055229137060675094,
        "action_models_fitted": False,
        "status": "DISABLED_INSUFFICIENT_CALIBRATION_SUCCESS_SUPPORT",
        "fail_closed_reason": "frozen reason",
    }
    source.write_text(json.dumps(payload), encoding="utf-8")
    table = _table_t4(source).set_index("Qualification field")["Frozen value"].to_dict()
    assert table["Calibration physical components"] == 103
    assert table["Minimum successes at n=100"] == 96
    assert table["Action models fitted"] is False
    assert table["Policy status"] == "DISABLED_INSUFFICIENT_CALIBRATION_SUCCESS_SUPPORT"


def test_vector_and_300_dpi_exports_are_byte_deterministic(tmp_path: Path) -> None:
    roots = [tmp_path / "first", tmp_path / "second"]
    outputs: list[list[Path]] = []
    for root in roots:
        with thesis_style():
            figure = _figure_f1()
            outputs.append(export_figure(figure, root, FIGURE_STEMS["F1"]))
            plt.close(figure)
    assert [_raw_sha256(path) for path in outputs[0]] == [_raw_sha256(path) for path in outputs[1]]
    png = (roots[0] / "png" / "F01_danids_pipeline.png").read_bytes()
    width = int.from_bytes(png[16:20], "big")
    height = int.from_bytes(png[20:24], "big")
    assert width >= 2_400
    assert height >= 900
    assert (
        (roots[0] / "svg" / "F01_danids_pipeline.svg")
        .read_text(encoding="utf-8")
        .lstrip()
        .startswith("<?xml")
    )
    assert (roots[0] / "pdf" / "F01_danids_pipeline.pdf").read_bytes().startswith(b"%PDF")


def _minimal_projection_sources(tmp_path: Path) -> dict[str, Path]:
    def csv(name: str, rows: list[dict[str, object]]) -> Path:
        path = tmp_path / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    sources = {
        "study1_seed": csv(
            "s1.csv",
            [
                {"metric": "tpr", "source_domain": "U", "target_domain": "T", "mean": 0.5},
                {"metric": "fpr", "source_domain": "U", "target_domain": "T", "mean": 0.2},
            ],
        ),
        "study2_holdouts": csv("hold.csv", [{"seed": 42, "method": "er"}]),
        "study2_forgetting": csv("forget.csv", [{"metric": "pr_auc"}]),
        "study3_prevalence": csv("prev.csv", [{"window_count": 5}]),
        "study3_models": csv(
            "models.csv",
            [{"split_kind": "leave_current_domain_out", "metric": "harmful_auprc"}],
        ),
        "study3_correlations": csv("corr.csv", [{"signal": "mmd"}]),
        "study4_safety": csv(
            "safety.csv",
            [
                {
                    "method": method,
                    "operating_envelope_compliance_rate": 0.2,
                    "unsafe_exposure_windows": 10,
                }
                for method in ("STATIC", "ALWAYS_ADAPT", "DANIDS_CORE", "OFFLINE_ORACLE")
            ],
        ),
        "study4_labels": csv(
            "labels.csv",
            [
                {"method": method, "labels_requested": 100}
                for method in ("STATIC", "ALWAYS_ADAPT", "DANIDS_CORE", "OFFLINE_ORACLE")
            ],
        ),
        "study5_hidden": csv("hidden.csv", [{"family": "x"}, {"family": "y"}]),
        "study5_forgetting": csv("family_forgetting.csv", [{"family": "x"}]),
        "study5b_orders": csv(
            "orders.csv",
            [
                {
                    "analysis_layer": "ALL_ORDER_SYNTHESIS",
                    "dimension": "FAMILY_FORGETTING_REDUCTION",
                }
            ],
        ),
    }
    study4_results = tmp_path / "study4.json"
    study4_results.write_text(
        json.dumps(
            {
                "resource_totals": {
                    method: {"accepted_updates": index}
                    for index, method in enumerate(
                        ("STATIC", "ALWAYS_ADAPT", "DANIDS_CORE", "OFFLINE_ORACLE")
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    verdict = tmp_path / "verdict.json"
    verdict.write_text(json.dumps({"overall_verdict": "NOT_SUPPORTED"}), encoding="utf-8")
    sources["study4_results"] = study4_results
    sources["study5b_verdict"] = verdict
    return sources


def test_projection_checks_record_exact_frozen_counts_and_values(tmp_path: Path) -> None:
    checks = _projection_checks(_minimal_projection_sources(tmp_path))
    assert checks["F3"]["cell_rows"] == 2
    assert checks["F4"]["pr_auc_forgetting_rows"] == 1
    assert checks["F5"] == {
        "state_window_count": 5,
        "harmful_auprc_grouped_rows": 1,
        "correlation_rows": 1,
    }
    assert checks["F7"]["DANIDS_CORE"]["accepted_updates"] == 2
    assert checks["F8"]["hidden_artifact_rows"] == 2
    assert checks["F9"] == {"plotted_order_rows": 1, "all_order_verdict": "NOT_SUPPORTED"}


def _write_minimal_valid_manifest(
    tmp_path: Path, *, create_source: bool
) -> tuple[Path, Path, dict[str, Any]]:
    output_root = tmp_path / "thesis" / "assets"
    with thesis_style():
        figure = _figure_f1()
        template_outputs = export_figure(figure, output_root, FIGURE_STEMS["F1"])
        plt.close(figure)
    captions = output_root / "captions.md"
    captions.write_text("safe wording\n", encoding="utf-8", newline="\n")
    records = []
    for display_id, stem in FIGURE_STEMS.items():
        if display_id == "F1":
            figure_outputs = template_outputs
        else:
            figure_outputs = []
            for template in template_outputs:
                path = output_root / template.parent.name / f"{stem}{template.suffix}"
                path.write_bytes(template.read_bytes())
                figure_outputs.append(path)
        source_records = []
        for key in DISPLAY_SPECS[display_id].sources:
            relative = SOURCE_PATHS[key]
            source = tmp_path / relative
            if create_source:
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("frozen\n", encoding="utf-8", newline="\n")
            source_records.append(
                {
                    "path": relative,
                    "sha256": _canonical_sha256(source) if create_source else "0" * 64,
                    "hash_mode": _hash_mode(source),
                }
            )
        records.append(
            {
                "display_id": display_id,
                "display_type": "figure",
                "chapter": DISPLAY_SPECS[display_id].chapter,
                "generator": f"danids.evaluation.thesis_assets:_figure_{display_id.lower()}",
                "source_artifacts": source_records,
                "output_files": [_output_record(path, output_root) for path in figure_outputs],
                "frozen_takeaway": DISPLAY_SPECS[display_id].takeaway,
            }
        )
    table_root = output_root / "tables"
    table_root.mkdir(parents=True)
    for display_id, stem in TABLE_STEMS.items():
        table_csv = table_root / f"{stem}.csv"
        table_md = table_root / f"{stem}.md"
        table_csv.write_text("a\n1\n", encoding="utf-8", newline="\n")
        table_md.write_text("| a |\n| --- |\n| 1 |\n", encoding="utf-8", newline="\n")
        source_records = []
        for key in DISPLAY_SPECS[display_id].sources:
            relative = SOURCE_PATHS[key]
            source = tmp_path / relative
            if create_source:
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("frozen\n", encoding="utf-8", newline="\n")
            source_records.append(
                {
                    "path": relative,
                    "sha256": _canonical_sha256(source) if create_source else "0" * 64,
                    "hash_mode": _hash_mode(source),
                }
            )
        records.append(
            {
                "display_id": display_id,
                "display_type": "table",
                "chapter": DISPLAY_SPECS[display_id].chapter,
                "generator": f"danids.evaluation.thesis_assets:_table_{display_id.lower()}",
                "source_artifacts": source_records,
                "output_files": [
                    _output_record(table_csv, output_root),
                    _output_record(table_md, output_root),
                ],
                "frozen_takeaway": DISPLAY_SPECS[display_id].takeaway,
            }
        )
    manifest = {
        "artifact_version": ASSET_VERSION,
        "digest_policy": DIGEST_POLICY,
        "generation_command": GENERATION_COMMAND,
        "scientific_processing": SCIENTIFIC_PROCESSING,
        "display_count": 14,
        "figure_count": 10,
        "table_count": 4,
        "projection_checks": {},
        "displays": records,
        "supplementary_files": [_output_record(captions, output_root)],
    }
    (output_root / "visual_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return output_root, template_outputs[0], manifest


def test_manifest_verification_rejects_output_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root, first_svg, _manifest = _write_minimal_valid_manifest(tmp_path, create_source=True)
    monkeypatch.setattr(thesis_assets, "_projection_checks", lambda _sources: {})
    verify_thesis_assets(tmp_path, output_root)
    first_svg.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="generated output digest differs"):
        verify_thesis_assets(tmp_path, output_root)


def test_outputs_only_verification_is_strict_but_does_not_require_frozen_sources(
    tmp_path: Path,
) -> None:
    output_root, _first_svg, manifest = _write_minimal_valid_manifest(tmp_path, create_source=False)
    verify_thesis_assets(tmp_path, output_root, verify_sources=False)
    with pytest.raises(FileNotFoundError, match="missing frozen thesis sources"):
        verify_thesis_assets(tmp_path, output_root)

    captions = output_root / "captions.md"
    captions.write_bytes(captions.read_bytes().replace(b"\n", b"\r\n"))
    verify_thesis_assets(tmp_path, output_root, verify_sources=False)

    manifest["displays"][0]["chapter"] = "Wrong chapter"
    (output_root / "visual_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="display metadata differs"):
        verify_thesis_assets(tmp_path, output_root, verify_sources=False)
    manifest["displays"][0]["chapter"] = DISPLAY_SPECS["F1"].chapter

    manifest["displays"][0]["source_artifacts"][0]["path"] = "wrong/source.json"
    (output_root / "visual_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source provenance differs"):
        verify_thesis_assets(tmp_path, output_root, verify_sources=False)
    first_source_key = DISPLAY_SPECS["F1"].sources[0]
    manifest["displays"][0]["source_artifacts"][0]["path"] = SOURCE_PATHS[first_source_key]

    manifest["generation_command"] = "wrong command"
    (output_root / "visual_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="generation command differs"):
        verify_thesis_assets(tmp_path, output_root, verify_sources=False)
    manifest["generation_command"] = GENERATION_COMMAND

    actual_bytes = manifest["supplementary_files"][0]["bytes"]
    manifest["supplementary_files"][0]["bytes"] = -1
    (output_root / "visual_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="byte count is invalid"):
        verify_thesis_assets(tmp_path, output_root, verify_sources=False)
    manifest["supplementary_files"][0]["bytes"] = actual_bytes + 1
    (output_root / "visual_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="supplementary output digest differs"):
        verify_thesis_assets(tmp_path, output_root, verify_sources=False)


def test_text_sha256_is_stable_across_lf_and_crlf_without_weakening_binary_hashes(
    tmp_path: Path,
) -> None:
    lf = tmp_path / "lf.md"
    crlf = tmp_path / "crlf.md"
    lf.write_bytes(b"first\nsecond\n")
    crlf.write_bytes(b"first\r\nsecond\r\n")
    assert _raw_sha256(lf) != _raw_sha256(crlf)
    assert _canonical_sha256(lf) == _canonical_sha256(crlf)
    assert _hash_mode(lf) == "canonical-lf-utf8"

    raw_lf = tmp_path / "lf.bin"
    raw_crlf = tmp_path / "crlf.bin"
    raw_lf.write_bytes(lf.read_bytes())
    raw_crlf.write_bytes(crlf.read_bytes())
    assert _canonical_sha256(raw_lf) != _canonical_sha256(raw_crlf)
    assert _hash_mode(raw_lf) == "raw-bytes"


def test_manifest_verification_rejects_digest_policy_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "thesis" / "assets"
    output_root.mkdir(parents=True)
    manifest = {
        "artifact_version": ASSET_VERSION,
        "digest_policy": {**DIGEST_POLICY, "algorithm": "sha1"},
        "generation_command": GENERATION_COMMAND,
        "scientific_processing": SCIENTIFIC_PROCESSING,
    }
    (output_root / "visual_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(thesis_assets, "_resolve_sources", lambda _root: {})
    with pytest.raises(ValueError, match="digest policy differs"):
        verify_thesis_assets(tmp_path, output_root)


def test_cli_exposes_single_complete_regeneration_command() -> None:
    args = build_parser().parse_args(["generate-thesis-assets"])
    assert args.command == "generate-thesis-assets"
    assert args.repo_root is None
    assert args.output_dir is None
