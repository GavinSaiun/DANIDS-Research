from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from danids.cli import build_parser
from danids.evaluation import thesis_assets
from danids.evaluation.thesis_assets import (
    ASSET_VERSION,
    CAPTIONS,
    DISPLAY_SPECS,
    FIGURE_STEMS,
    SOURCE_PATHS,
    TABLE_STEMS,
    _figure_f1,
    _output_record,
    _projection_checks,
    _table_t1,
    _table_t4,
    verify_thesis_assets,
)
from danids.evaluation.thesis_style import export_figure, thesis_style


def _sha256(path: Path) -> str:
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
    assert all("runs/" not in path for path in SOURCE_PATHS.values())


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
    assert [_sha256(path) for path in outputs[0]] == [_sha256(path) for path in outputs[1]]
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


def test_manifest_verification_rejects_output_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "thesis" / "assets"
    source = tmp_path / "source.txt"
    source.write_text("frozen", encoding="utf-8")
    with thesis_style():
        figure = _figure_f1()
        figure_outputs = export_figure(figure, output_root, "shared")
        plt.close(figure)
    table_root = output_root / "tables"
    table_root.mkdir(parents=True)
    table_csv = table_root / "shared.csv"
    table_md = table_root / "shared.md"
    table_csv.write_text("a\n1\n", encoding="utf-8")
    table_md.write_text("| a |\n| --- |\n| 1 |\n", encoding="utf-8")
    captions = output_root / "captions.md"
    captions.write_text("safe wording", encoding="utf-8")
    source_record = {"path": "source.txt", "sha256": _sha256(source)}
    records = []
    for display_id in FIGURE_STEMS:
        records.append(
            {
                "display_id": display_id,
                "source_artifacts": [source_record],
                "output_files": [_output_record(path, output_root) for path in figure_outputs],
            }
        )
    for display_id in TABLE_STEMS:
        records.append(
            {
                "display_id": display_id,
                "source_artifacts": [source_record],
                "output_files": [
                    _output_record(table_csv, output_root),
                    _output_record(table_md, output_root),
                ],
            }
        )
    manifest = {
        "artifact_version": ASSET_VERSION,
        "display_count": 14,
        "figure_count": 10,
        "table_count": 4,
        "projection_checks": {},
        "displays": records,
        "supplementary_files": [_output_record(captions, output_root)],
    }
    (output_root / "visual_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(thesis_assets, "_resolve_sources", lambda _root: {})
    monkeypatch.setattr(thesis_assets, "_projection_checks", lambda _sources: {})
    verify_thesis_assets(tmp_path, output_root)
    figure_outputs[0].write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="generated output digest differs"):
        verify_thesis_assets(tmp_path, output_root)


def test_cli_exposes_single_complete_regeneration_command() -> None:
    args = build_parser().parse_args(["generate-thesis-assets"])
    assert args.command == "generate-thesis-assets"
    assert args.repo_root is None
    assert args.output_dir is None
