# ruff: noqa: E501, RUF001
"""Generate the frozen DANIDS 2.0 thesis visualisation pack.

This module is intentionally artifact-only: it reads validated Study 1--5 result
bundles and documentation, never raw data, checkpoints, or experiment runners.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mpl_colors
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from danids.evaluation.thesis_style import (
    DOMAIN_LABELS,
    DOMAIN_ORDER,
    METHOD_LABELS,
    PALETTE,
    export_figure,
    thesis_style,
)

plt.switch_backend("Agg")

ASSET_VERSION = "danids-thesis-assets-v1"
GENERATION_COMMAND = "python -m danids generate-thesis-assets"
SCIENTIFIC_PROCESSING = (
    "frozen-artifact-only; no experiments, training, rescoring, or hypothesis recomputation"
)
_WORDING_GUARDRAIL_BANNED = (
    "anticipatory early warning",
    "core reduced labels",
    "study 5a performs attribution",
    "study 5a performs open-set",
    "four-order h7 synthesis is wholly prospective",
)
_WORDING_GUARDRAIL_ALLOWED_NEGATIONS = ("not anticipatory early warning",)
DIGEST_POLICY = {
    "algorithm": "sha256",
    "encoding": "lowercase hexadecimal",
    "binary_input": "raw bytes",
    "text_input": "UTF-8 bytes after CRLF and CR are canonicalised to LF",
    "bytes_field": "length of the hash input after the selected hash mode",
}
TEXT_HASH_SUFFIXES = frozenset({".csv", ".json", ".md", ".svg", ".txt", ".yaml", ".yml"})
FIGURE_STEMS = {
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
TABLE_STEMS = {
    "T1": "T01_rq_hypothesis_map",
    "T2": "T02_literature_positioning",
    "T3": "T03_study_design_crosswalk",
    "T4": "T04_policy_qualification",
}

SOURCE_PATHS = {
    "freeze": "docs/thesis_evidence_freeze.md",
    "plan": "docs/thesis_master_plan.md",
    "decisions": "docs/decisions.md",
    "literature_t02": "docs/literature_t02_evidence.md",
    "study1_seed": "study1/static-s42-s44/study1_seed_summary.csv",
    "study1_transfer": "study1/static-s42-s44/study1_transfer_long.csv",
    "study1_summary": "study1/static-s42-s44/study1_summary.json",
    "study2_methods": "study2/u-t-c-b-s42-s44/study2_method_summary.csv",
    "study2_forgetting": "study2/u-t-c-b-s42-s44/study2_forgetting.csv",
    "study2_holdouts": "study2/u-t-c-b-s42-s44/study2_final_holdouts.csv",
    "study2_summary": "study2/u-t-c-b-s42-s44/study2_summary.json",
    "study3_prevalence": "study3/health-s42-s44/study3_state_prevalence.csv",
    "study3_models": "study3/health-s42-s44/study3_model_summary.csv",
    "study3_correlations": "study3/health-s42-s44/study3_shift_harm_correlations.csv",
    "study3_summary": "study3/health-s42-s44/study3_summary.json",
    "study3_contract": "study3/health-s42-s44/evaluation_contract.json",
    "study4_safety": "study4/e4-confirmatory-final/study4_safety_compliance.csv",
    "study4_labels": "study4/e4-confirmatory-final/study4_label_query_usage.csv",
    "study4_actions": "study4/e4-confirmatory-final/study4_action_distribution.csv",
    "study4_paired": "study4/e4-confirmatory-final/study4_core_vs_always_paired.csv",
    "study4_summary": "study4/e4-confirmatory-final/study4_summary.json",
    "study4_contract": "study4/e4-confirmatory-final/evaluation_contract.json",
    "study4_results": "study4/e4-confirmatory-analysis/study4_confirmatory_results.json",
    "policy_qualification": "study4/policy-qualification-v1-final/policy_qualification.json",
    "study5_hidden": "study5/threat-audit-v1/hidden_family_failures.csv",
    "study5_forgetting": "study5/threat-audit-v1/family_forgetting.csv",
    "study5_transfer": "study5/threat-audit-v1/study1_supported_transfer.csv",
    "study5_contract": "study5/threat-audit-v1/study5_contract.json",
    "study5_summary": "study5/threat-audit-v1/study5_summary.json",
    "study5b_orders": "study5/task009-study5b-all-order-replay-v1/order_summary.csv",
    "study5b_methods": "study5/task009-study5b-all-order-replay-v1/method_dimension_summary.csv",
    "study5b_verdict": "study5/task009-study5b-all-order-replay-v1/all_order_synthesis_verdict.json",
    "study5b_summary": "study5/task009-study5b-all-order-replay-v1/study5b_summary.json",
    "study5b_contract": "study5/task009-study5b-all-order-replay-v1/task009_contract.json",
}


@dataclass(frozen=True)
class DisplaySpec:
    chapter: str
    sources: tuple[str, ...]
    takeaway: str


DISPLAY_SPECS = {
    "F1": DisplaySpec(
        "Chapter 1",
        ("freeze", "decisions"),
        "Deployment shift, observable health, intervention and retained competence are related but non-equivalent problems.",
    ),
    "T1": DisplaySpec(
        "Chapter 1",
        ("freeze", "plan"),
        "The research questions, studies and frozen hypothesis ledger are traceable without reinterpretation.",
    ),
    "T2": DisplaySpec(
        "Chapter 2",
        ("literature_t02", "freeze", "plan"),
        "Adjacent literature addresses parts of the pipeline, while the interfaces among shift, operational harm and constrained recoverability remain distinct.",
    ),
    "F2": DisplaySpec(
        "Chapter 3",
        (
            "freeze",
            "decisions",
            "study3_contract",
            "study4_contract",
            "study5_contract",
            "study5b_contract",
        ),
        "The prequential protocol separates policy-visible information from evaluator-only truth and permanent holdouts.",
    ),
    "T3": DisplaySpec(
        "Chapter 3",
        (
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
        ),
        "Each study has an explicit unit, evidence phase, supervision contract and hypothesis role.",
    ),
    "F3": DisplaySpec(
        "Chapter 4",
        ("study1_seed", "study1_transfer"),
        "Static transfer is directional and target-dependent, with operational false-alarm burden distinct from recall.",
    ),
    "F4": DisplaySpec(
        "Chapter 4",
        ("study2_methods", "study2_forgetting", "study2_holdouts"),
        "Representational retention and frozen-threshold operation are distinct; scheduled adaptation did not reliably restore safety.",
    ),
    "F5": DisplaySpec(
        "Chapter 5",
        ("study3_prevalence", "study3_models", "study3_correlations", "study3_summary"),
        "Same-window harm screening can recognise many violations, but health states are imbalanced and signal relationships vary by domain.",
    ),
    "F6": DisplaySpec(
        "Chapter 6",
        ("freeze", "decisions", "study4_contract"),
        "DANIDS-Core acts only on permitted health and released evidence, with audit-gated acceptance and exact rollback.",
    ),
    "F7": DisplaySpec(
        "Chapter 6",
        ("study4_safety", "study4_labels", "study4_actions", "study4_paired", "study4_results"),
        "Core reduced accepted update frequency, not labels, and did not achieve comparable safety; sequence mattered.",
    ),
    "T4": DisplaySpec(
        "Chapter 6",
        ("policy_qualification",),
        "DANIDS-Policy failed closed before model fitting because calibration success support could not meet the frozen rule.",
    ),
    "F8": DisplaySpec(
        "Chapter 7",
        ("study5_hidden", "study5_forgetting", "study5_transfer", "study5_contract"),
        "Aggregate recall can conceal family-conditioned binary detection failures, while learned-reference availability is sparse.",
    ),
    "F9": DisplaySpec(
        "Chapter 7",
        ("study5b_orders", "study5b_methods", "study5b_verdict"),
        "Positive prior U-T-C-B replay competence effects did not generalise prospectively across deployment orders.",
    ),
    "F10": DisplaySpec(
        "Chapter 8",
        (
            "freeze",
            "study1_summary",
            "study2_summary",
            "study3_summary",
            "study4_results",
            "study5_summary",
            "study5b_verdict",
        ),
        "Operational harm was easier to recognise than repair under the frozen B100/D1 and A0-A4 regime.",
    ),
}


def _hash_mode(path: Path) -> str:
    return "canonical-lf-utf8" if path.suffix.lower() in TEXT_HASH_SUFFIXES else "raw-bytes"


def _canonical_text_bytes(path: Path) -> bytes:
    text = path.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return text.encode("utf-8")


def _digest_byte_count(path: Path) -> int:
    if _hash_mode(path) == "canonical-lf-utf8":
        return len(_canonical_text_bytes(path))
    return path.stat().st_size


def _wording_guardrail_violations(text: str) -> tuple[str, ...]:
    """Return unsupported claims while preserving explicit frozen negations."""

    screened = text.lower()
    for allowed in _WORDING_GUARDRAIL_ALLOWED_NEGATIONS:
        screened = screened.replace(allowed, "")
    return tuple(phrase for phrase in _WORDING_GUARDRAIL_BANNED if phrase in screened)


def _sha256(path: Path) -> str:
    """Hash text canonically across platforms and binary artifacts byte-for-byte."""

    digest = hashlib.sha256()
    if _hash_mode(path) == "canonical-lf-utf8":
        digest.update(_canonical_text_bytes(path))
        return digest.hexdigest()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve_sources(repo_root: Path) -> dict[str, Path]:
    sources = {key: repo_root / relative for key, relative in SOURCE_PATHS.items()}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen thesis sources:\n" + "\n".join(missing))
    return sources


def _box(
    axis: Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    face: str = "white",
    edge: str = PALETTE["blue"],
    linestyle: str = "-",
    fontsize: float = 8.0,
) -> None:
    axis.add_patch(
        patches.FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.2,
            linestyle=linestyle,
        )
    )
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )


def _arrow(
    axis: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = PALETTE["black"],
    linestyle: str = "-",
) -> None:
    axis.add_patch(
        patches.FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=color,
            linestyle=linestyle,
        )
    )


def _table_markdown(frame: pd.DataFrame) -> str:
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    header = "| " + " | ".join(clean(column) for column in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows]) + "\n"


def _write_table(frame: pd.DataFrame, output_root: Path, stem: str) -> list[Path]:
    table_root = output_root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    csv_path = table_root / f"{stem}.csv"
    md_path = table_root / f"{stem}.md"
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    md_path.write_text(_table_markdown(frame), encoding="utf-8", newline="\n")
    return [csv_path, md_path]


def _figure_f1() -> Figure:
    figure, axis = plt.subplots(figsize=(8.4, 3.6))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    axis.add_patch(patches.Rectangle((0.02, 0.46), 0.96, 0.40, color="#F4F8FB", zorder=-2))
    axis.text(0.03, 0.89, "Policy-visible operational lane", color=PALETTE["blue"], weight="bold")
    labels = [
        "Cross-domain\nchange",
        "Observable\nhealth",
        "Intervention\nchoice",
        "Accept / rollback\nmodel state",
        "Current, prior-domain\nand family outcomes",
    ]
    xs = [0.02, 0.205, 0.39, 0.585, 0.78]
    widths = [0.15, 0.15, 0.16, 0.16, 0.20]
    for x, width, label in zip(xs, widths, labels, strict=True):
        _box(axis, (x, 0.58), width, 0.17, label, face="white", fontsize=7.2)
    for index in range(4):
        _arrow(axis, (xs[index] + widths[index], 0.665), (xs[index + 1], 0.665))
    axis.add_patch(patches.Rectangle((0.02, 0.08), 0.96, 0.27, color="#F3F3F3", zorder=-2))
    axis.text(0.03, 0.38, "Evaluator-only lane", color=PALETTE["grey"], weight="bold")
    for x, text in [
        (0.10, "Truth / holdout\nscoring"),
        (0.42, "Counterfactual and\nverdict evidence"),
        (0.74, "Frozen hypothesis\nassessment"),
    ]:
        _box(axis, (x, 0.14), 0.18, 0.12, text, face="white", edge=PALETTE["grey"], linestyle="--")
    for x, text in [
        (0.23, "false-alarm burden"),
        (0.50, "threat-family composition"),
        (0.76, "deployment order"),
    ]:
        axis.text(x, 0.50, text, ha="center", va="center", fontsize=6.5, color=PALETTE["red"])
    figure.suptitle("DANIDS evidence pipeline: change, recognition, action and outcome")
    return figure


def _figure_f2() -> Figure:
    figure, axis = plt.subplots(figsize=(9.2, 4.4))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    axis.text(
        0.02, 0.94, "Dataset-native chronological stream (50,000-flow windows)", weight="bold"
    )
    for index, domain in enumerate(DOMAIN_ORDER):
        x = 0.03 + index * 0.235
        _box(
            axis, (x, 0.80), 0.18, 0.09, f"Domain {domain}\n{DOMAIN_LABELS[domain]}", face="#F4F8FB"
        )
        if index < 3:
            _arrow(axis, (x + 0.18, 0.845), (x + 0.235, 0.845))
    events = [
        "Predict + label-free\nhealth",
        "Record prediction\nfor evaluation",
        "Release previous\nwindow labels",
        "Query",
        "Candidate\nupdate",
        "Audit",
        "Accept or\nrollback",
    ]
    for index, event in enumerate(events):
        x = 0.025 + index * 0.139
        _box(axis, (x, 0.53), 0.115, 0.12, event, fontsize=6.8)
        if index < len(events) - 1:
            _arrow(axis, (x + 0.115, 0.59), (x + 0.139, 0.59))
    axis.text(
        0.03,
        0.69,
        "Predict-first chronology; one-window delayed release",
        color=PALETTE["blue"],
        weight="bold",
    )
    _box(
        axis,
        (0.03, 0.27),
        0.25,
        0.13,
        "Replay memory\ntraining eligible after release",
        face="#E8F5F0",
        edge=PALETTE["green"],
    )
    _box(
        axis,
        (0.375, 0.27),
        0.25,
        0.13,
        "Audit memory\nnever in optimizer batches",
        face="#FFF4DB",
        edge=PALETTE["orange"],
    )
    _box(
        axis,
        (0.72, 0.27),
        0.25,
        0.13,
        "Permanent holdout\nevaluator-only",
        face="#FBE9E3",
        edge=PALETTE["red"],
        linestyle="--",
    )
    axis.add_patch(
        patches.Rectangle(
            (0.02, 0.04), 0.96, 0.13, facecolor="#F3F3F3", edgecolor=PALETTE["grey"], linestyle="--"
        )
    )
    axis.text(
        0.50,
        0.105,
        "Prohibited from Core, querying, training, calibration, replay and audit:\n"
        "domain/source/task/boundary identities • future/unreleased labels • evaluator truth/verdict • permanent holdouts",
        ha="center",
        va="center",
        fontsize=7.5,
    )
    figure.suptitle("Frozen prequential protocol and information boundary")
    return figure


def _figure_f6() -> Figure:
    figure, axis = plt.subplots(figsize=(8.5, 4.7))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    _box(axis, (0.03, 0.73), 0.18, 0.12, "Monitored health\nstate", face="#F4F8FB")
    _box(axis, (0.28, 0.73), 0.20, 0.12, "Query eligible?\nB100 / query25 / D1", face="#F4F8FB")
    _box(axis, (0.55, 0.73), 0.18, 0.12, "Released\nevidence", face="#F4F8FB")
    _box(axis, (0.80, 0.73), 0.16, 0.12, "Action\nselection", face="#F4F8FB")
    for start, end in [
        ((0.21, 0.79), (0.28, 0.79)),
        ((0.48, 0.79), (0.55, 0.79)),
        ((0.73, 0.79), (0.80, 0.79)),
    ]:
        _arrow(axis, start, end)
    actions = [
        ("A0", "No-op"),
        ("A1", "Recalibrate\n(strictly feasible)"),
        ("A2", "Head update"),
        ("A4", "Replay update"),
    ]
    for index, (code, label) in enumerate(actions):
        x = 0.05 + index * 0.235
        _box(axis, (x, 0.47), 0.18, 0.13, f"{code}\n{label}", face="#E8F5F0", edge=PALETTE["green"])
        _arrow(axis, (0.88, 0.73), (x + 0.09, 0.60), color=PALETTE["blue"])
    _box(
        axis,
        (0.27, 0.24),
        0.20,
        0.12,
        "Candidate state\n(deployed state unchanged)",
        face="#FFF4DB",
        edge=PALETTE["orange"],
    )
    _box(axis, (0.54, 0.24), 0.15, 0.12, "Audit gate", face="#FFF4DB", edge=PALETTE["orange"])
    _box(
        axis,
        (0.76, 0.24),
        0.19,
        0.12,
        "Accept or exact\nrollback / fallback",
        face="#FFF4DB",
        edge=PALETTE["orange"],
    )
    _arrow(axis, (0.23, 0.47), (0.33, 0.36))
    _arrow(axis, (0.46, 0.47), (0.40, 0.36))
    _arrow(axis, (0.70, 0.47), (0.43, 0.36))
    _arrow(axis, (0.93, 0.47), (0.46, 0.36))
    _arrow(axis, (0.47, 0.30), (0.54, 0.30))
    _arrow(axis, (0.69, 0.30), (0.76, 0.30))
    axis.text(
        0.50,
        0.135,
        "A3 full fine-tuning: unavailable to normal Core",
        ha="center",
        color=PALETTE["red"],
        weight="bold",
    )
    axis.text(
        0.50,
        0.075,
        "Offline Oracle: evaluator-visible, non-deployable upper-bound comparator",
        ha="center",
        color=PALETTE["purple"],
    )
    axis.text(
        0.50,
        0.02,
        "Core has no evaluator-truth or permanent-holdout access",
        ha="center",
        color=PALETTE["grey"],
    )
    figure.suptitle("DANIDS-Core: delayed evidence, bounded actions and audit-gated state change")
    return figure


def _figure_f10() -> Figure:
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    layers = [
        ("1", "Cross-domain change", "H1 SUPPORTED", PALETTE["green"]),
        (
            "2",
            "Same-window harm screening",
            "H2 SUPPORTED • H3 NOT SUPPORTED • H9 PARTIAL",
            PALETTE["blue"],
        ),
        ("3", "Intervention and recoverability", "H4 NOT SUPPORTED • H10 PARTIAL", PALETTE["red"]),
        (
            "4",
            "Threat-family and order robustness",
            "H7 NOT SUPPORTED • H6/H8 NOT TESTABLE",
            PALETTE["orange"],
        ),
        ("5", "Thesis synthesis", "Harm recognition was easier than repair", PALETTE["purple"]),
    ]
    for index, (number, title, status, color) in enumerate(layers):
        y = 0.81 - index * 0.17
        _box(
            axis,
            (0.09, y),
            0.82,
            0.115,
            f"Layer {number}  |  {title}\n{status}",
            face="#FAFAFA",
            edge=color,
            fontsize=8.2,
        )
        if index < len(layers) - 1:
            _arrow(axis, (0.50, y), (0.50, y - 0.055), color=PALETTE["grey"])
    axis.text(
        0.50,
        0.045,
        "Evidence boundary: four related NetFlow-v3 domains • compact MLP • three seeds • B100/D1 • A0-A4",
        ha="center",
        fontsize=7.2,
    )
    axis.text(
        0.50,
        0.01,
        "No causal proof or general impossibility claim",
        ha="center",
        fontsize=7.2,
        color=PALETTE["red"],
    )
    figure.suptitle("DANIDS 2.0 frozen evidence synthesis")
    return figure


def _annotated_heatmap(
    axis: Axes,
    values: np.ndarray,
    *,
    title: str,
    color_label: str,
    norm: mpl_colors.Normalize,
    display_values: np.ndarray | None = None,
) -> None:
    image = axis.imshow(values, cmap="viridis", norm=norm, aspect="equal")
    axis.set_xticks(
        range(4), [DOMAIN_LABELS[domain] for domain in DOMAIN_ORDER], rotation=35, ha="right"
    )
    axis.set_yticks(range(4), [DOMAIN_LABELS[domain] for domain in DOMAIN_ORDER])
    axis.set_xlabel("Target domain")
    axis.set_ylabel("Source domain")
    axis.set_title(title, loc="left", weight="bold")
    shown = values if display_values is None else display_values
    for row in range(4):
        for column in range(4):
            value = shown[row, column]
            text = "—" if np.isnan(value) else f"{value:.3g}"
            mapped = (
                cast(float, image.norm(values[row, column]))
                if not np.isnan(values[row, column])
                else 0
            )
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if mapped > 0.55 else PALETTE["black"],
            )
        axis.add_patch(
            patches.Rectangle(
                (row - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1.6
            )
        )
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(color_label)


def _figure_f3(sources: dict[str, Path]) -> Figure:
    seed_summary = pd.read_csv(sources["study1_seed"])

    def matrix(metric: str) -> np.ndarray:
        selected = seed_summary.loc[seed_summary["metric"].astype(str).str.lower() == metric]
        pivot = selected.pivot(index="source_domain", columns="target_domain", values="mean")
        return pivot.reindex(index=DOMAIN_ORDER, columns=DOMAIN_ORDER).to_numpy(dtype=float)

    tpr = matrix("tpr")
    fpr = matrix("fpr")
    budget_ratio = fpr / 0.001
    positive = budget_ratio[np.isfinite(budget_ratio) & (budget_ratio > 0)]
    lower = max(float(positive.min()) if positive.size else 0.01, 0.01)
    upper = max(float(positive.max()) if positive.size else 1.0, 1.0)
    plotted_ratio = np.maximum(budget_ratio, lower)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), constrained_layout=True)
    _annotated_heatmap(
        axes[0],
        tpr,
        title="A  Mean attack recall",
        color_label="Mean TPR",
        norm=mpl_colors.Normalize(vmin=0, vmax=1),
    )
    _annotated_heatmap(
        axes[1],
        plotted_ratio,
        display_values=budget_ratio,
        title="B  Mean false-positive budget ratio",
        color_label="FPR / 0.001 (log scale)",
        norm=mpl_colors.LogNorm(vmin=lower, vmax=upper),
    )
    figure.suptitle("Static cross-domain transfer is directional and operationally asymmetric")
    figure.text(
        0.50,
        -0.01,
        "Diagonal cells are within-domain references; off-diagonal cells show directional transfer.",
        ha="center",
        fontsize=7.3,
    )
    return figure


def _figure_f4(sources: dict[str, Path]) -> Figure:
    holdouts = pd.read_csv(sources["study2_holdouts"])
    forgetting = pd.read_csv(sources["study2_forgetting"])
    method_order = ["static", "naive_ft", "ewc", "er", "ft_mem"]
    method_colors = dict(
        zip(
            method_order,
            [
                PALETTE["grey"],
                PALETTE["blue"],
                PALETTE["orange"],
                PALETTE["green"],
                PALETTE["purple"],
            ],
            strict=True,
        )
    )
    figure = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(1, 1.25))
    for panel, domain in enumerate(("T", "C", "B")):
        axis = figure.add_subplot(grid[0, panel])
        rows = holdouts.loc[holdouts["holdout_dataset_id"].astype(str) == domain].copy()
        for index, method in enumerate(method_order):
            values = rows.loc[rows["method"].astype(str).str.lower() == method, "tpr"].astype(float)
            if values.empty:
                continue
            offsets = np.linspace(-0.08, 0.08, len(values))
            axis.scatter(
                index + offsets,
                values,
                s=12,
                alpha=0.45,
                color=method_colors[method],
                edgecolor="none",
            )
            axis.scatter(
                index,
                values.mean(),
                s=42,
                color=method_colors[method],
                marker="D",
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        axis.set_xticks(
            range(len(method_order)),
            [METHOD_LABELS[name.upper()] for name in method_order],
            rotation=45,
            ha="right",
        )
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(
            f"{chr(65 + panel)}  Final {domain} holdout\n{DOMAIN_LABELS[domain]}",
            fontsize=8.5,
        )
        if panel == 0:
            axis.set_ylabel("Attack recall (TPR)")
        axis.grid(axis="y", color=PALETTE["light_grey"], linewidth=0.6)
    axis = figure.add_subplot(grid[1, :])
    pr_rows = forgetting.loc[forgetting["metric"].astype(str).str.lower() == "pr_auc"].copy()
    pr_rows = pr_rows.loc[pr_rows["domain_id"].astype(str) != "B"]
    final_rows = holdouts.copy()
    final_rows["fpr_budget_ratio_plot"] = pd.to_numeric(
        final_rows["fpr_budget_ratio"], errors="coerce"
    )
    missing_ratio = final_rows["fpr_budget_ratio_plot"].isna()
    final_rows.loc[missing_ratio, "fpr_budget_ratio_plot"] = (
        pd.to_numeric(final_rows.loc[missing_ratio, "fpr"], errors="coerce") / 0.001
    )
    joined = pr_rows.merge(
        final_rows[["seed", "method", "holdout_dataset_id", "fpr_budget_ratio_plot"]],
        left_on=["seed", "method", "domain_id"],
        right_on=["seed", "method", "holdout_dataset_id"],
        how="inner",
        validate="one_to_one",
    )
    markers = {"T": "o", "C": "s", "U": "^"}
    for method in method_order[1:]:
        rows = joined.loc[joined["method"].astype(str).str.lower() == method]
        for domain, marker in markers.items():
            domain_rows = rows.loc[rows["domain_id"].astype(str) == domain]
            axis.scatter(
                domain_rows["forgetting"],
                domain_rows["fpr_budget_ratio_plot"],
                s=32,
                marker=marker,
                color=method_colors[method],
                alpha=0.7,
                label=f"{METHOD_LABELS[method.upper()]} · {domain}"
                if not domain_rows.empty
                else None,
            )
    axis.axhline(
        1.0, color=PALETTE["black"], linestyle="--", linewidth=0.9, label="Frozen FPR budget"
    )
    axis.set_yscale("log")
    axis.set_xlabel("PR-AUC forgetting (maximum since learned − final)")
    axis.set_ylabel("Final false-positive budget ratio (FPR / 0.001, log scale)")
    axis.set_title("D  Representational forgetting versus frozen-threshold operation", loc="left")
    axis.grid(color=PALETTE["light_grey"], linewidth=0.5)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False
    )
    figure.suptitle(
        "Scheduled adaptation: retained discrimination does not guarantee safe operation"
    )
    return figure


def _pretty_feature_set(value: str) -> str:
    return {
        "all": "All signals",
        "combined_unlabelled": "Combined unlabelled",
        "conformal": "Conformal",
        "distribution": "Distribution shift",
        "predictive": "Predictive uncertainty",
    }.get(value, value.replace("_", " ").title())


def _pretty_predictor(value: str) -> str:
    return {
        "gradient_boosting": "Gradient boosting",
        "logistic_regression": "Logistic regression",
    }.get(value, value.replace("_", " ").title())


def _figure_f5(sources: dict[str, Path]) -> Figure:
    prevalence = pd.read_csv(sources["study3_prevalence"])
    models = pd.read_csv(sources["study3_models"])
    correlations = pd.read_csv(sources["study3_correlations"])
    figure = plt.figure(figsize=(7.2, 7.1), constrained_layout=True)
    grid = figure.add_gridspec(3, 1, height_ratios=(0.9, 1.4, 1.65))

    axis = figure.add_subplot(grid[0])
    state_order = ("SAFE", "UNCERTAIN", "HARMFUL")
    state_colors = {
        "SAFE": PALETTE["green"],
        "UNCERTAIN": PALETTE["orange"],
        "HARMFUL": PALETTE["red"],
    }
    grouped = prevalence.groupby(["current_domain", "health_state"], as_index=False)[
        "window_count"
    ].sum()
    overall = grouped.groupby("health_state", as_index=False)["window_count"].sum()
    overall["current_domain"] = "Overall"
    grouped = pd.concat([grouped, overall], ignore_index=True)
    categories = [*DOMAIN_ORDER, "Overall"]
    left = np.zeros(len(categories))
    for state in state_order:
        values = []
        for category in categories:
            subset = grouped.loc[
                (grouped["current_domain"].astype(str) == category)
                & (grouped["health_state"].astype(str) == state),
                "window_count",
            ]
            total = grouped.loc[
                grouped["current_domain"].astype(str) == category, "window_count"
            ].sum()
            values.append(float(subset.sum()) / float(total) if total else 0.0)
        axis.barh(
            categories,
            values,
            left=left,
            color=state_colors[state],
            label=state,
            edgecolor="white",
            linewidth=0.4,
        )
        left += np.asarray(values)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Share of evaluated windows")
    axis.set_title("A  Health-state prevalence by current domain", loc="left")
    axis.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.48))

    axis = figure.add_subplot(grid[1])
    selected = models.loc[
        (models["split_kind"].astype(str) == "leave_current_domain_out")
        & (models["metric"].astype(str) == "harmful_auprc")
    ].copy()
    selected["label"] = (
        selected["feature_set"].astype(str).map(_pretty_feature_set)
        + " · "
        + selected["predictor"].astype(str).map(_pretty_predictor)
    )
    selected = selected.sort_values("macro_mean")
    y = np.arange(len(selected))
    x = selected["macro_mean"].astype(float).to_numpy()
    lower = x - selected["bootstrap_95_low"].astype(float).to_numpy()
    upper = selected["bootstrap_95_high"].astype(float).to_numpy() - x
    axis.errorbar(
        x,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor=PALETTE["grey"],
        capsize=2,
        linewidth=0.8,
    )
    frozen = (selected["feature_set"].astype(str) == "combined_unlabelled") & (
        selected["predictor"].astype(str) == "gradient_boosting"
    )
    axis.scatter(
        x[~frozen.to_numpy()], y[~frozen.to_numpy()], color=PALETTE["blue"], s=36, zorder=3
    )
    axis.scatter(
        x[frozen.to_numpy()],
        y[frozen.to_numpy()],
        facecolor="white",
        edgecolor=PALETTE["red"],
        linewidth=1.6,
        s=48,
        zorder=4,
    )
    axis.set_yticks(y, selected["label"])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Leave-current-domain-out harmful AUPRC (bootstrap 95% interval)")
    axis.set_title("B  Same-window harmful-state discrimination", loc="left")
    axis.text(
        0.99,
        0.03,
        "Red outline: frozen combined-unlabelled gradient boosting monitor",
        transform=axis.transAxes,
        ha="right",
        fontsize=7,
        color=PALETTE["red"],
    )

    subgrid = grid[2].subgridspec(1, 2)
    signal_order = [
        "dist_mmd2_linear_rbf",
        "dist_domain_classifier_auroc",
        "dist_wasserstein_mean",
        "dist_covariance_relative_frobenius",
        "dist_training_standardized_composite",
    ]
    signal_labels = [
        "MMD²",
        "Domain classifier AUROC",
        "Wasserstein",
        "Covariance shift",
        "Composite shift",
    ]
    for index, target in enumerate(("eval_fpr_budget_ratio", "eval_tpr_loss_from_reference")):
        axis = figure.add_subplot(subgrid[0, index])
        rows = correlations.loc[correlations["evaluator_target"].astype(str) == target]
        pivot = rows.pivot(
            index="signal", columns="held_out_current_domain", values="spearman_correlation"
        )
        matrix_values = pivot.reindex(index=signal_order, columns=DOMAIN_ORDER).to_numpy(
            dtype=float
        )
        image = axis.imshow(matrix_values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        axis.set_xticks(range(4), DOMAIN_ORDER)
        axis.set_yticks(range(len(signal_order)), signal_labels if index == 0 else [])
        axis.set_xlabel("Held-out current domain")
        axis.set_title(
            f"C{index + 1}  {'FPR budget ratio' if target == 'eval_fpr_budget_ratio' else 'TPR loss'}",
            loc="left",
        )
        for row in range(matrix_values.shape[0]):
            for column in range(matrix_values.shape[1]):
                if np.isfinite(matrix_values[row, column]):
                    axis.text(
                        column,
                        row,
                        f"{matrix_values[row, column]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                    )
        if index == 1:
            colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            colorbar.set_label("Spearman correlation")
    figure.suptitle("Study 3: health-state imbalance and domain-variable same-window signals")
    return figure


def _figure_f7(sources: dict[str, Path]) -> Figure:
    safety = pd.read_csv(sources["study4_safety"])
    labels = pd.read_csv(sources["study4_labels"])
    paired = pd.read_csv(sources["study4_paired"])
    results = _load_json(sources["study4_results"])
    method_order = ["STATIC", "ALWAYS_ADAPT", "DANIDS_CORE", "OFFLINE_ORACLE"]
    method_colors = [PALETTE["grey"], PALETTE["orange"], PALETTE["blue"], PALETTE["purple"]]
    run_count = int(results["experimental_completion"]["run_count"]) // len(method_order)
    resources = results["resource_totals"]
    safety = safety.set_index("method").reindex(method_order)
    labels = labels.set_index("method").reindex(method_order)

    figure = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    axis = figure.add_subplot(grid[0, 0])
    compliance = safety["operating_envelope_compliance_rate"].astype(float).to_numpy() * 100
    bars = axis.bar(range(4), compliance, color=method_colors, hatch=["", "//", "..", "xx"])
    unsafe_per_run = safety["unsafe_exposure_windows"].astype(float).to_numpy() / run_count
    for bar, unsafe in zip(bars, unsafe_per_run, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.35,
            f"{unsafe:.1f}\nunsafe/run",
            ha="center",
            va="bottom",
            fontsize=6.8,
        )
    axis.set_xticks(
        range(4), [METHOD_LABELS[method] for method in method_order], rotation=30, ha="right"
    )
    axis.set_ylabel("Operating-envelope compliance (%)")
    axis.set_ylim(0, float(max(compliance)) + 5)
    axis.set_title("A  Safety compliance", loc="left")

    axis = figure.add_subplot(grid[0, 1])
    updates = np.asarray(
        [float(resources[method]["accepted_updates"]) / run_count for method in method_order]
    )
    labels_per_run = labels["labels_requested"].astype(float).to_numpy() / run_count
    label_offsets = {
        "STATIC": (6, 6),
        "ALWAYS_ADAPT": (6, 6),
        "DANIDS_CORE": (8, -30),
        "OFFLINE_ORACLE": (8, 6),
    }
    for index, method in enumerate(method_order):
        axis.scatter(
            updates[index],
            unsafe_per_run[index],
            s=48 + labels_per_run[index] * 0.45,
            color=method_colors[index],
            alpha=0.78,
            edgecolor=PALETTE["black"],
            linewidth=0.5,
        )
        axis.annotate(
            f"{METHOD_LABELS[method]}\n{labels_per_run[index]:.0f} labels/run",
            (updates[index], unsafe_per_run[index]),
            xytext=label_offsets[method],
            textcoords="offset points",
            fontsize=6.8,
        )
    axis.set_xlabel("Accepted updates per run")
    axis.set_ylabel("Unsafe exposure windows per run")
    axis.set_title("B  Intervention resource–safety plane", loc="left")
    axis.text(
        0.98,
        0.03,
        "Offline Oracle is non-deployable",
        transform=axis.transAxes,
        ha="right",
        fontsize=7,
        color=PALETTE["purple"],
    )

    axis = figure.add_subplot(grid[1, :])
    sequence_order = ["B-U-T-C", "C-B-U-T", "T-C-B-U", "U-T-C-B"]
    for index, sequence in enumerate(sequence_order):
        values = paired.loc[
            paired["sequence"].astype(str) == sequence, "core_minus_always_unsafe_exposure"
        ].astype(float)
        axis.scatter(
            values,
            np.full(len(values), index) + np.linspace(-0.08, 0.08, len(values)),
            color=PALETTE["blue"],
            alpha=0.55,
            s=25,
        )
        axis.scatter(values.mean(), index, color=PALETTE["red"], marker="D", s=45, zorder=3)
    axis.axvline(0, color=PALETTE["black"], linestyle="--", linewidth=0.9)
    axis.set_yticks(range(len(sequence_order)), sequence_order)
    axis.set_xlabel("Core − Always-Adapt unsafe exposure windows (negative favours Core)")
    axis.set_title("C  Paired sequence dependence (descriptive; seed points and mean)", loc="left")
    axis.grid(axis="x", color=PALETTE["light_grey"], linewidth=0.6)
    figure.suptitle("Study 4: sparse accepted updates did not deliver comparable safety")
    return figure


def _figure_f8(sources: dict[str, Path]) -> Figure:
    hidden = pd.read_csv(sources["study5_hidden"], keep_default_na=False)
    forgetting = pd.read_csv(sources["study5_forgetting"], keep_default_na=False)
    figure = plt.figure(figsize=(7.2, 6.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.15, 1.0))

    axis = figure.add_subplot(grid[0, :])
    representation_colors = {"NATIVE": PALETTE["blue"], "SEMANTIC": PALETTE["orange"]}
    for family_level, rows in hidden.groupby(hidden["family_level"].astype(str).str.upper()):
        axis.scatter(
            pd.to_numeric(rows["aggregate_recall_loss"]),
            pd.to_numeric(rows["family_recall_loss"]),
            s=34,
            alpha=0.55,
            color=representation_colors.get(family_level, PALETTE["grey"]),
            label=f"{family_level.title()} family artifact row",
        )
    axis.axvline(0.10, color=PALETTE["grey"], linestyle="--", linewidth=0.9)
    axis.axhline(0.10, color=PALETTE["grey"], linestyle="--", linewidth=0.9)
    xmax = max(0.12, float(pd.to_numeric(hidden["aggregate_recall_loss"]).max()) * 1.08)
    ymax = max(0.12, float(pd.to_numeric(hidden["family_recall_loss"]).max()) * 1.08)
    xmin = min(-0.01, float(pd.to_numeric(hidden["aggregate_recall_loss"]).min()) * 1.05)
    axis.add_patch(
        patches.Rectangle(
            (xmin, 0.10),
            0.10 - xmin,
            max(0, ymax - 0.10),
            facecolor=PALETTE["red"],
            alpha=0.10,
            edgecolor="none",
        )
    )
    axis.text(
        xmin + 0.006,
        ymax * 0.96,
        "Aggregate-stable /\nfamily-harm region",
        fontsize=7,
        color=PALETTE["red"],
        va="top",
    )
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(min(-0.01, float(pd.to_numeric(hidden["family_recall_loss"]).min()) * 1.05), ymax)
    axis.set_xlabel("Aggregate binary recall loss")
    axis.set_ylabel("Family-conditioned binary recall loss")
    axis.set_title("A  Aggregate metrics can conceal supported family-slice failures", loc="left")
    axis.legend(frameon=False, loc="upper right")
    axis.text(
        0.99,
        0.02,
        f"{len(hidden)} artifact rows; repeated views are not independent failures",
        transform=axis.transAxes,
        ha="right",
        fontsize=7,
    )

    sequence_parts = forgetting["sequence"].astype(str).str.split("-")
    forgetting = forgetting.assign(
        domain_position=[
            parts.index(domain) + 1
            for parts, domain in zip(
                sequence_parts, forgetting["evaluated_domain"].astype(str), strict=True
            )
        ],
        family_level_normalized=forgetting["family_level"].astype(str).str.upper(),
        available=forgetting["learned_state_status"].astype(str) == "LEARNED_REFERENCE",
    )
    supported = forgetting.loc[
        (pd.to_numeric(forgetting["learned_support"], errors="coerce") >= 50)
        | (pd.to_numeric(forgetting["final_support"], errors="coerce") >= 50)
    ]
    for panel, family_level in enumerate(("NATIVE", "SEMANTIC")):
        axis = figure.add_subplot(grid[1, panel])
        rows = supported.loc[supported["family_level_normalized"] == family_level]
        strata = sorted(rows["source_study"].astype(str).unique())
        categories: list[str] = []
        available_shares: list[float] = []
        unavailable_shares: list[float] = []
        for stratum in strata:
            for position in (1, 2, 3, 4):
                subset = rows.loc[
                    (rows["source_study"].astype(str) == stratum)
                    & (rows["domain_position"] == position)
                ]
                if subset.empty:
                    continue
                share = float(subset["available"].mean())
                categories.append(f"{stratum} · P{position}")
                available_shares.append(share)
                unavailable_shares.append(1 - share)
        positions = np.arange(len(categories))
        axis.barh(
            positions, available_shares, color=PALETTE["green"], label="Learned reference available"
        )
        axis.barh(
            positions,
            unavailable_shares,
            left=available_shares,
            color=PALETTE["light_grey"],
            hatch="//",
            label="Unavailable",
        )
        axis.set_yticks(positions, categories)
        axis.invert_yaxis()
        axis.set_xlim(0, 1)
        axis.set_xlabel("Share of physically supported family-slice rows")
        axis.set_title(f"B{panel + 1}  {family_level.title()} representation", loc="left")
        if panel == 1:
            axis.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    figure.suptitle(
        "Study 5A: family-conditioned binary detection and learned-reference availability"
    )
    return figure


def _figure_f9(sources: dict[str, Path]) -> Figure:
    orders = pd.read_csv(sources["study5b_orders"])
    rows = orders.loc[
        (orders["analysis_layer"].astype(str) == "ALL_ORDER_SYNTHESIS")
        & orders["dimension"]
        .astype(str)
        .isin(("FAMILY_FORGETTING_REDUCTION", "FINAL_PREVIOUS_DOMAIN_COMPETENCE_GAIN"))
    ].copy()
    sequence_order = ["U-T-C-B", "T-C-B-U", "C-B-U-T", "B-U-T-C"]
    method_order = ["er", "ft_mem"]
    method_colors = {"er": PALETTE["green"], "ft_mem": PALETTE["purple"]}
    method_markers = {"er": "o", "ft_mem": "s"}
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), sharey=True, constrained_layout=True)
    dimensions = [
        ("FAMILY_FORGETTING_REDUCTION", "A  Family-forgetting reduction"),
        ("FINAL_PREVIOUS_DOMAIN_COMPETENCE_GAIN", "B  Final previous-domain competence gain"),
    ]
    for axis, (dimension, title) in zip(axes, dimensions, strict=True):
        dimension_rows = rows.loc[rows["dimension"].astype(str) == dimension]
        for sequence_index, sequence in enumerate(sequence_order):
            if sequence == "U-T-C-B":
                axis.axhspan(
                    sequence_index - 0.43, sequence_index + 0.43, color=PALETTE["sky"], alpha=0.10
                )
            else:
                axis.axhspan(
                    sequence_index - 0.43,
                    sequence_index + 0.43,
                    color=PALETTE["orange"],
                    alpha=0.035,
                )
            for method_index, method in enumerate(method_order):
                match = dimension_rows.loc[
                    (dimension_rows["sequence"].astype(str) == sequence)
                    & (dimension_rows["replay_method"].astype(str) == method)
                ]
                if match.empty:
                    continue
                row = match.iloc[0]
                seed_values = json.loads(str(row["seed_unit_values"]))
                y = sequence_index + (-0.12 if method_index == 0 else 0.12)
                axis.scatter(
                    [float(item["value"]) for item in seed_values],
                    np.full(len(seed_values), y),
                    s=19,
                    color=method_colors[method],
                    alpha=0.35,
                    marker=method_markers[method],
                )
                axis.scatter(
                    float(row["median_effect"]),
                    y,
                    s=52,
                    color=method_colors[method],
                    edgecolor="white",
                    linewidth=0.5,
                    marker=method_markers[method],
                    label=METHOD_LABELS[method.upper()] if sequence_index == 0 else None,
                    zorder=3,
                )
        axis.axvline(0, color=PALETTE["black"], linestyle="--", linewidth=0.9)
        axis.set_yticks(range(4), sequence_order)
        axis.invert_yaxis()
        axis.set_xlabel("Paired replay effect (positive favours replay)")
        axis.set_title(title, loc="left")
        axis.grid(axis="x", color=PALETTE["light_grey"], linewidth=0.5)
        axis.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=2)
    figure.text(0.01, 0.01, "Blue band: prior U-T-C-B evidence", fontsize=7, color=PALETTE["blue"])
    figure.text(
        0.99,
        0.01,
        "Other three rotations: prospectively frozen extension",
        ha="right",
        fontsize=7,
        color=PALETTE["orange"],
    )
    figure.suptitle("Study 5B: replay effects are not robust to deployment order")
    return figure


def _table_t1() -> pd.DataFrame:
    rows = [
        (
            "RQ1",
            "How severe and direction-dependent is intrusion-detection degradation across heterogeneous network domains, and what operational trade-offs arise when standard continual-learning methods adapt one evolving detector?",
            "Studies 1, 2",
            "H1",
            "Chapter 4",
            "Static cross-domain degradation was substantial and asymmetric across recall, false-positive burden, and ranking performance. Straightforward adaptation did not solve the deployment problem: target-domain recall recovery could coexist with extreme false-positive burden or reduced earlier-domain competence, and threshold-free retention did not guarantee acceptable operation at the frozen threshold. Study 1's sequence positions are reporting positions, not causal order effects.",
        ),
        (
            "RQ2",
            "Which observable distributional and model-state signals can distinguish harmful operating-envelope violations from harmless domain change, and do combined or sparsely supervised health models generalise reliably?",
            "Study 3",
            "H2; H3; H9",
            "Chapter 5",
            "Many harmful windows were recognisable from permitted label-free observables, and the reviewed combined-unlabelled gradient-boosting model was the strongest label-free comparator selected for the frozen Core monitor. Distribution shift was not equivalent to harm, however, and neither combined signals nor sparse delayed labels dominated every simpler comparator across models, metrics, and grouped generalisation protocols. This is strong same-window harm screening, not anticipatory early warning.",
        ),
        (
            "RQ3",
            "Under the frozen B100/D1 supervision regime and A0--A4 action space, can health-aware selective adaptation maintain or restore operating-envelope safety while reducing supervision, accepted updates, and operational forgetting relative to unconditional adaptation?",
            "Study 4",
            "H4; H10",
            "Chapter 6",
            "No. Core reduced accepted update frequency relative to Always-Adapt, but did not reduce requested labels—the frozen label-usage artifact records equal requested-label totals—and did not maintain comparable operational safety. The descriptively favourable operational-forgetting component and fewer accepted updates did not rescue the no-supervision-reduction and failed comparable-safety components of H10. The scheduled Always-Adapt treatment was slightly worse than Static despite using 3,600 labels across its 12 runs, so unconditional adaptation did not restore aggregate safety. The non-deployable one-step Offline Oracle's sparse selected updates supported the minimum-intervention motivation, yet its approximately 19.06% compliance also showed that most observed harm was not recoverable under the frozen B100/D1 and A0--A4 regime. This does not establish that safe adaptation is impossible in general. DANIDS-Policy failed closed before fitting and was never deployed.",
        ),
        (
            "RQ4",
            "To what extent do aggregate binary metrics conceal family-specific detection failures across deployment histories, and what is the evidential boundary between family-conditioned detection, attack attribution, and open-set recognition?",
            "Study 5A",
            "H6; H8",
            "Chapter 7",
            "Aggregate binary recall could remain within the frozen loss tolerance while a physically supported family experienced a larger loss. The estimand is family-conditioned binary detection recall. No attribution predictions, structured embedding comparison, explicit UNKNOWN decision, or open-set experiment was performed, so H6 and H8 are not testable. The 27 hidden-family rows are repeated lifecycle/representation output rows, not 27 independent failures, and history-relative unseen status is not a real-world zero-day claim.",
        ),
        (
            "RQ5",
            "Does replay-aware adaptation robustly reduce supported attack-family forgetting and improve final previous-domain competence across deployment orders relative to target-only full fine-tuning?",
            "Studies 2, 5B",
            "H7",
            "Chapter 7",
            "No order-robust replay advantage was found. Positive U-T-C-B replay competence effects had been observed before TASK-009; the three missing rotations were then frozen prospectively and failed to reproduce the benefit. B-U-T-C showed materially adverse replay effects. The conclusion is deployment-order heterogeneity and reversal, not merely a near-zero pooled effect. The final four-order synthesis mixes prior and prospective evidence. Its primary level is mapped semantic families with physical support n >= 50; only previous domains in positions 1--3 are eligible, while position 4 is excluded. Effects are aggregated unweighted from family to domain to rotation--seed, and rotation--seed units—not flows or family rows—are the experimental units.",
        ),
    ]
    ledger = "H1 SUPPORTED; H2 SUPPORTED; H3 NOT_SUPPORTED; H4 NOT_SUPPORTED; H6 NOT_TESTABLE; H7 NOT_SUPPORTED; H8 NOT_TESTABLE; H9 PARTIAL; H10 PARTIAL"
    frame = pd.DataFrame(
        rows,
        columns=[
            "Research question",
            "Frozen wording",
            "Study evidence",
            "Hypotheses",
            "Home chapter",
            "Final answer",
        ],
    )
    frame.loc[len(frame)] = [
        "Frozen ledger",
        ledger,
        "—",
        "—",
        "docs/thesis_evidence_freeze.md",
        "Verbatim status record",
    ]
    return frame


def _table_t2() -> pd.DataFrame:
    rows = [
        (
            "Cross-domain NIDS",
            "Per-flow binary attack detection or attack-specific classification after a network or dataset change",
            "Labeled source data; target or cross-network use ranges from held-out testing to unlabeled adaptation or labeled augmentation",
            "Directional source-to-target corpus pair under a common flow-feature schema",
            "Target F1 or related classification performance and degradation from a same-domain reference",
            "Separating detected shift, fixed-threshold operational harm and sequential recoverability",
            "Apruzzese et al. (2022); Layeghy & Portmann (2023); Layeghy et al. (2023)",
        ),
        (
            "Continual learning",
            "Current-task prediction while retaining earlier-task or domain competence in one evolving model",
            "Sequential labels; optional stored exemplars, task identity or parameter-importance state",
            "Task or domain stage by retained test set, summarized over a sequence",
            "High current and final performance with low forgetting or favourable backward transfer",
            "Label-free harm recognition and audited minimum intervention under hidden boundaries and delayed labels",
            "Parisi et al. (2019); Kirkpatrick et al. (2017); Lopez-Paz & Ranzato (2017); Delgado et al. (2026)",
        ),
        (
            "Drift monitoring",
            "Distribution mismatch or change point, or rising labeled prediction error",
            "Reference data or model plus current unlabeled features or scores; labels for direct error monitoring",
            "Current sample or window versus a reference window, or a sequential error stream",
            "Controlled shift or error alarms with low false alarms and detection delay",
            "Whether same-window change is operationally harmful and safely repairable",
            "Lu et al. (2019); Gama et al. (2004); Gretton et al. (2012); Rabanser et al. (2019)",
        ),
        (
            "Selective adaptation",
            "Whether, when and on which samples or batches a deployed model should adapt",
            "Current unlabeled inputs, predictions, entropy or shift proxies; sometimes delayed performance feedback",
            "Test sample or minibatch, stream step or detected-drift episode",
            "Shifted-domain accuracy plus update cost, forgetting, stability or collapse avoidance",
            "Health-conditioned A0-A4 choice under B100/D1 with audit-gated promotion or rollback",
            "Horchulhack et al. (2022); Niu et al. (2022, 2023); Yoo et al. (2024)",
        ),
        (
            "Family/open-set evaluation",
            "Known-class attribution or explicit UNKNOWN rejection, distinct from binary detection sliced by true family",
            "Flow features; held-out classes for open-set tests; evaluator-only true families for conditional recall",
            "Flow or connection summarized per class or family within a physical evaluation slice",
            "Per-class or macro metrics and known-versus-unknown rejection trade-offs, reported separately",
            "DANIDS provides family-conditioned binary detection recall only, with no attribution or open-set output",
            "Elmasry et al. (2019); Cruz et al. (2017); Baye et al. (2023); Yu et al. (2024)",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "Area",
            "Prediction target",
            "Information available",
            "Typical unit",
            "Operational criterion",
            "Unresolved DANIDS layer",
            "Citation status",
        ],
    )


def _table_t3() -> pd.DataFrame:
    rows = [
        (
            "Study 1",
            "Static cross-domain transfer",
            "Static",
            "4 sources × 4 targets",
            "42–44",
            "source × target × seed",
            "None",
            "Evaluator-only outcomes",
            "Prior benchmark",
            "H1",
        ),
        (
            "Study 2",
            "Sequential adaptation and retention",
            "Static; NaiveFT; EWC; ER; FT-Mem",
            "U-T-C-B",
            "42–44",
            "sequence × seed",
            "B100/D1",
            "Released labels only",
            "Prior sequential evidence",
            "Supports RQ1 context",
        ),
        (
            "Study 3",
            "Same-window model-health estimation",
            "Logistic; gradient boosting",
            "4 rotations",
            "42–44",
            "held-out current domain",
            "Frozen delayed-label signals",
            "Policy-visible features separated from evaluator truth",
            "Prospective confirmatory extraction",
            "H2; H3; H9",
        ),
        (
            "Study 4",
            "Selective intervention and recoverability",
            "Static; Always-Adapt; Core; Offline Oracle",
            "4 rotations",
            "42–44",
            "rotation × seed",
            "B100/D1",
            "Core sees health/released evidence; Oracle is evaluator-visible",
            "Frozen confirmatory",
            "H4; H10",
        ),
        (
            "Study 5A",
            "Family-conditioned binary detection audit",
            "Artifact-only",
            "Studies 1, 2 and 4",
            "Inherited",
            "physically supported family slice",
            "No new supervision",
            "Evaluator-only audit",
            "Ontology and estimands frozen after Studies 1--4 existed but before artifact-only threat-effect analysis over the prior Study 1, 2, and 4 source bundles",
            "H6; H8",
        ),
        (
            "Study 5B",
            "All-order replay-retention robustness",
            "NaiveFT; ER; FT-Mem",
            "4 rotations",
            "42–44",
            "rotation × seed",
            "B100/D1",
            "No policy controller",
            "9 prior U-T-C-B runs + 27 prospective missing-rotation runs; mixed prior/prospective four-order synthesis",
            "H7",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "Study",
            "Question / estimand",
            "Methods",
            "Domains / orders",
            "Seeds",
            "Experimental unit",
            "Supervision",
            "Information boundary",
            "Evidence phase",
            "Hypothesis",
        ],
    )


def _table_t4(source: Path) -> pd.DataFrame:
    value = _load_json(source)
    rows = [
        ("Calibration physical components", value["calibration_component_count"]),
        ("Minimum recommendations", value["minimum_recommendation_requirement"]),
        (
            "Minimum model-changing selections",
            value["minimum_model_changing_selection_requirement"],
        ),
        ("Required Wilson 95% lower bound", value["required_wilson_lower_bound"]),
        ("Minimum successes at n=100", value["minimum_successes_required_at_n_100"]),
        ("Maximum confirmed successes", value["maximum_achievable_confirmed_success_count"]),
        (
            "Maximum empirical success rate at n=100",
            value["maximum_empirical_success_rate_at_minimum_recommendations"],
        ),
        ("Maximum Wilson 95% lower bound", value["maximum_wilson_lower_bound"]),
        ("Action models fitted", value["action_models_fitted"]),
        ("Policy status", value["status"]),
        ("Fail-closed reason", value["fail_closed_reason"]),
    ]
    return pd.DataFrame(rows, columns=["Qualification field", "Frozen value"])


CAPTIONS = {
    "F1": (
        "DANIDS evidence pipeline. The solid lane contains operationally available quantities; the shaded dashed lane is evaluator-only. "
        "The display separates deployment shift, observable model health, intervention choice, state acceptance or rollback, and retained outcomes."
    ),
    "T1": "Research-question and hypothesis map. Hypothesis statuses reproduce the frozen ledger verbatim and are not re-estimated here.",
    "T2": (
        "Verified literature-positioning matrix. Representative adjacent work is compared by prediction target, information regime, evaluation unit and success criterion. "
        "The final column states the remaining interface question for DANIDS without claiming that the cited areas ignore deployment shift."
    ),
    "F2": (
        "Frozen prequential chronology and information boundary. Prediction and label-free health extraction precede truth observation; labels release one window later. "
        "Permanent holdouts and evaluator truth cannot enter Core, querying, training, calibration, replay, or audit."
    ),
    "T3": (
        "Study-design crosswalk. Study 5A froze its ontology and estimands after Studies 1--4 existed but before artifact-only threat-effect analysis over the prior Study 1, 2, and 4 source bundles. "
        "Study 5B combines nine prior U-T-C-B runs with 27 prospective missing-rotation runs in a mixed prior/prospective four-order synthesis."
    ),
    "F3": (
        "Static cross-domain transfer. Cells show seed-mean attack recall and the operational false-positive budget ratio on a logarithmic colour scale. "
        "Directional transfer asymmetry is descriptive; no significance claim is implied."
    ),
    "F4": (
        "Scheduled adaptation trade-off. Seed points and summaries separate final target-domain recall from the relationship between PR-AUC forgetting and frozen-threshold false-positive burden. "
        "Representational retention is not treated as evidence of safe operation."
    ),
    "F5": (
        "Study 3 model-health evidence. Panels show health-state prevalence, leave-current-domain-out harmful-state discrimination, and selected signal correlations. "
        "These are same-window harm screening results, not anticipatory forecasts."
    ),
    "F6": (
        "DANIDS-Core action lifecycle. Candidate changes are audited before promotion and rejected candidates restore exact deployed state. "
        "A3 is unavailable to normal Core; Offline Oracle is a separate non-deployable comparator with evaluator visibility."
    ),
    "F7": (
        "Study 4 safety and resource outcomes. Core reduced accepted update frequency, not supervision or labels, and comparable safety was not supported. "
        "Sequence-level differences are descriptive; Offline Oracle is non-deployable."
    ),
    "T4": (
        "DANIDS-Policy qualification. The model-independent upper bound could not meet the frozen Wilson criterion, so Policy was disabled before action-model fitting."
    ),
    "F8": (
        "Study 5A family-conditioned binary detection audit. The upper panel contrasts aggregate and family-slice recall loss; the lower panels show learned-reference availability. "
        "Artifact rows and native/semantic views are not independent failures, and this is not multiclass attribution or open-set recognition."
    ),
    "F9": (
        "Study 5B paired replay effects by deployment order. Positive values favour replay. U-T-C-B is prior evidence; T-C-B-U, C-B-U-T, and B-U-T-C form the prospectively frozen extension. "
        "The four-order synthesis is therefore mixed prior/prospective evidence."
    ),
    "F10": (
        "Frozen evidence synthesis. Across the four related NetFlow-v3 domains, operational harm was easier to recognise than repair under the frozen B100/D1 and A0-A4 regime. "
        "This bounded result is neither causal proof nor a claim that safe adaptation is impossible in general."
    ),
}


INTERPRETATION_NOTES = {
    "F1": "Do not collapse recognition, intervention and outcome into a single capability claim.",
    "T1": "H3 is NOT_SUPPORTED; H9 and H10 use the compact final status PARTIAL.",
    "T2": (
        "Read the five areas as adjacent capabilities, not a novelty census. DANIDS uses same-window harm screening, "
        "and its repair claims remain bounded to the frozen B100/D1 and A0-A4 regime."
    ),
    "F2": "Evaluator-only truth is used for scoring, never deployed control.",
    "T3": "Units are study-level experimental units, not flows, windows or family rows treated as replicates.",
    "F3": "Do not infer symmetric transfer from a source-target pair.",
    "F4": "Discuss failed repair under the frozen B100/D1 and A0-A4 regime.",
    "F5": "Use the phrase same-window harm screening; avoid language suggesting future prediction.",
    "F6": "Do not imply that DANIDS-Policy was deployed; it failed qualification before fitting.",
    "F7": "State accepted-update sparsity and explicitly report that Core and Always-Adapt used equal label totals.",
    "T4": "The failure is a frozen support qualification result, not post-hoc threshold tuning.",
    "F8": "Use family-conditioned binary detection recall; do not claim attribution or open-set recognition.",
    "F9": "Do not call the four-order H7 synthesis wholly prospective.",
    "F10": "Do not generalise beyond the frozen evidence boundary or claim safe adaptation is impossible.",
}


def _captions_markdown(sources: dict[str, Path], repo_root: Path) -> str:
    lines = [
        "# DANIDS 2.0 thesis display captions",
        "",
        "Generated deterministically from frozen Study 1–5 artifacts. No experiment, rescoring, or new scientific analysis is performed.",
        "",
        "Domain labels: U = NF-UNSW-NB15-v3; T = NF-ToN-IoT-v3; C = NF-CSE-CIC-IDS2018-v3; B = NF-BoT-IoT-v3.",
        "",
    ]
    for display_id in (*FIGURE_STEMS, *TABLE_STEMS):
        spec = DISPLAY_SPECS[display_id]
        lines.extend(
            [
                f"## {display_id} — {FIGURE_STEMS.get(display_id, TABLE_STEMS.get(display_id, ''))}",
                "",
                CAPTIONS[display_id],
                "",
                "Sources: "
                + ", ".join(
                    f"`{sources[key].relative_to(repo_root).as_posix()}`" for key in spec.sources
                )
                + ".",
                "",
                "Interpretation guardrail: " + INTERPRETATION_NOTES[display_id],
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _output_record(path: Path, output_root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(output_root).as_posix(),
        "sha256": _sha256(path),
        "hash_mode": _hash_mode(path),
        "bytes": _digest_byte_count(path),
    }


def _projection_checks(sources: dict[str, Path]) -> dict[str, Any]:
    study1 = pd.read_csv(sources["study1_seed"])
    holdouts = pd.read_csv(sources["study2_holdouts"])
    forgetting = pd.read_csv(sources["study2_forgetting"])
    prevalence = pd.read_csv(sources["study3_prevalence"])
    models = pd.read_csv(sources["study3_models"])
    correlations = pd.read_csv(sources["study3_correlations"])
    safety = pd.read_csv(sources["study4_safety"])
    labels = pd.read_csv(sources["study4_labels"])
    study4_results = _load_json(sources["study4_results"])
    hidden = pd.read_csv(sources["study5_hidden"])
    family_forgetting = pd.read_csv(sources["study5_forgetting"])
    orders = pd.read_csv(sources["study5b_orders"])
    verdict = _load_json(sources["study5b_verdict"])
    study1_selected = study1.loc[
        study1["metric"].astype(str).str.lower().isin(("tpr", "fpr"))
    ].sort_values(["metric", "source_domain", "target_domain"])
    study4_methods: dict[str, Any] = {}
    for method in ("STATIC", "ALWAYS_ADAPT", "DANIDS_CORE", "OFFLINE_ORACLE"):
        safety_row = safety.loc[safety["method"].astype(str) == method].iloc[0]
        label_row = labels.loc[labels["method"].astype(str) == method].iloc[0]
        study4_methods[method] = {
            "compliance_rate": float(safety_row["operating_envelope_compliance_rate"]),
            "unsafe_exposure_windows": int(safety_row["unsafe_exposure_windows"]),
            "labels_requested": int(label_row["labels_requested"]),
            "accepted_updates": int(study4_results["resource_totals"][method]["accepted_updates"]),
        }
    order_rows = orders.loc[
        (orders["analysis_layer"].astype(str) == "ALL_ORDER_SYNTHESIS")
        & orders["dimension"]
        .astype(str)
        .isin(("FAMILY_FORGETTING_REDUCTION", "FINAL_PREVIOUS_DOMAIN_COMPETENCE_GAIN"))
    ]
    return {
        "F3": {
            "cell_rows": len(study1_selected),
            "canonical_values_sha256": hashlib.sha256(
                study1_selected.to_csv(index=False, lineterminator="\n").encode("utf-8")
            ).hexdigest(),
        },
        "F4": {
            "final_holdout_rows": len(holdouts),
            "pr_auc_forgetting_rows": int(
                (forgetting["metric"].astype(str).str.lower() == "pr_auc").sum()
            ),
        },
        "F5": {
            "state_window_count": int(prevalence["window_count"].sum()),
            "harmful_auprc_grouped_rows": int(
                (
                    (models["split_kind"].astype(str) == "leave_current_domain_out")
                    & (models["metric"].astype(str) == "harmful_auprc")
                ).sum()
            ),
            "correlation_rows": len(correlations),
        },
        "F7": study4_methods,
        "F8": {
            "hidden_artifact_rows": len(hidden),
            "family_forgetting_rows": len(family_forgetting),
        },
        "F9": {
            "plotted_order_rows": len(order_rows),
            "all_order_verdict": verdict["overall_verdict"],
        },
    }


def generate_thesis_assets(
    repo_root: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Generate all frozen thesis displays and return their auditable manifest."""

    repo_root = repo_root.resolve()
    output_root = (output_root or repo_root / "thesis" / "assets").resolve()
    sources = _resolve_sources(repo_root)
    source_hashes_before = {key: _sha256(path) for key, path in sources.items()}
    for directory in (
        output_root,
        output_root / "png",
        output_root / "pdf",
        output_root / "svg",
        output_root / "tables",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    figure_builders: dict[str, Any] = {
        "F1": _figure_f1,
        "F2": _figure_f2,
        "F3": lambda: _figure_f3(sources),
        "F4": lambda: _figure_f4(sources),
        "F5": lambda: _figure_f5(sources),
        "F6": _figure_f6,
        "F7": lambda: _figure_f7(sources),
        "F8": lambda: _figure_f8(sources),
        "F9": lambda: _figure_f9(sources),
        "F10": _figure_f10,
    }
    table_builders: dict[str, Any] = {
        "T1": _table_t1,
        "T2": _table_t2,
        "T3": _table_t3,
        "T4": lambda: _table_t4(sources["policy_qualification"]),
    }
    display_records: list[dict[str, Any]] = []
    with thesis_style():
        for display_id, stem in FIGURE_STEMS.items():
            figure = figure_builders[display_id]()
            output_paths = export_figure(figure, output_root, stem)
            plt.close(figure)
            spec = DISPLAY_SPECS[display_id]
            display_records.append(
                {
                    "display_id": display_id,
                    "display_type": "figure",
                    "chapter": spec.chapter,
                    "generator": f"danids.evaluation.thesis_assets:_figure_{display_id.lower()}",
                    "source_artifacts": [
                        {
                            "path": sources[key].relative_to(repo_root).as_posix(),
                            "sha256": source_hashes_before[key],
                            "hash_mode": _hash_mode(sources[key]),
                        }
                        for key in spec.sources
                    ],
                    "output_files": [_output_record(path, output_root) for path in output_paths],
                    "frozen_takeaway": spec.takeaway,
                }
            )
    for display_id, stem in TABLE_STEMS.items():
        frame = table_builders[display_id]()
        output_paths = _write_table(frame, output_root, stem)
        spec = DISPLAY_SPECS[display_id]
        display_records.append(
            {
                "display_id": display_id,
                "display_type": "table",
                "chapter": spec.chapter,
                "generator": f"danids.evaluation.thesis_assets:_table_{display_id.lower()}",
                "source_artifacts": [
                    {
                        "path": sources[key].relative_to(repo_root).as_posix(),
                        "sha256": source_hashes_before[key],
                        "hash_mode": _hash_mode(sources[key]),
                    }
                    for key in spec.sources
                ],
                "output_files": [_output_record(path, output_root) for path in output_paths],
                "frozen_takeaway": spec.takeaway,
            }
        )

    captions_path = output_root / "captions.md"
    captions_path.write_text(_captions_markdown(sources, repo_root), encoding="utf-8", newline="\n")
    source_hashes_after = {key: _sha256(path) for key, path in sources.items()}
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("a frozen source artifact changed during thesis asset generation")
    manifest: dict[str, Any] = {
        "artifact_version": ASSET_VERSION,
        "digest_policy": DIGEST_POLICY,
        "generation_command": GENERATION_COMMAND,
        "scientific_processing": SCIENTIFIC_PROCESSING,
        "display_count": len(display_records),
        "figure_count": len(FIGURE_STEMS),
        "table_count": len(TABLE_STEMS),
        "projection_checks": _projection_checks(sources),
        "displays": display_records,
        "supplementary_files": [_output_record(captions_path, output_root)],
    }
    manifest_path = output_root / "visual_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    verify_thesis_assets(repo_root, output_root)
    return manifest


def _validate_rendered_file(path: Path) -> None:
    if path.suffix == ".png":
        with path.open("rb") as handle:
            header = handle.read(24)
        if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"invalid PNG output: {path}")
        width, height = struct.unpack(">II", header[16:24])
        if min(width, height) < 900:
            raise ValueError(f"PNG resolution is below thesis requirement: {path}")
    elif path.suffix == ".pdf":
        if not path.read_bytes().startswith(b"%PDF"):
            raise ValueError(f"invalid PDF output: {path}")
    elif path.suffix == ".svg":
        prefix = path.read_text(encoding="utf-8")[:1000]
        if "<svg" not in prefix:
            raise ValueError(f"invalid SVG output: {path}")


def _expected_output_paths(display_id: str) -> list[str]:
    if display_id in FIGURE_STEMS:
        stem = FIGURE_STEMS[display_id]
        return [f"svg/{stem}.svg", f"pdf/{stem}.pdf", f"png/{stem}.png"]
    stem = TABLE_STEMS[display_id]
    return [f"tables/{stem}.csv", f"tables/{stem}.md"]


def _validate_digest_metadata(record: dict[str, Any], *, include_bytes: bool) -> None:
    relative = record.get("path")
    digest = record.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError("thesis asset manifest digest path is invalid")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"thesis asset manifest SHA-256 is invalid: {relative}")
    if record.get("hash_mode") != _hash_mode(Path(relative)):
        raise ValueError(f"thesis asset manifest hash mode differs: {relative}")
    if include_bytes and (
        not isinstance(record.get("bytes"), int)
        or isinstance(record.get("bytes"), bool)
        or record["bytes"] < 0
    ):
        raise ValueError(f"thesis asset manifest byte count is invalid: {relative}")


def verify_thesis_assets(
    repo_root: Path,
    output_root: Path | None = None,
    *,
    verify_sources: bool = True,
) -> None:
    """Verify source/output digests, display roster, formats, and wording guardrails."""

    repo_root = repo_root.resolve()
    output_root = (output_root or repo_root / "thesis" / "assets").resolve()
    manifest_path = output_root / "visual_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("artifact_version") != ASSET_VERSION:
        raise ValueError("thesis asset manifest version differs")
    if manifest.get("digest_policy") != DIGEST_POLICY:
        raise ValueError("thesis asset manifest digest policy differs")
    if manifest.get("generation_command") != GENERATION_COMMAND:
        raise ValueError("thesis asset manifest generation command differs")
    if manifest.get("scientific_processing") != SCIENTIFIC_PROCESSING:
        raise ValueError("thesis asset manifest scientific-processing declaration differs")
    if (
        manifest.get("display_count") != 14
        or manifest.get("figure_count") != 10
        or manifest.get("table_count") != 4
    ):
        raise ValueError("thesis display roster differs")
    actual_ids = [record["display_id"] for record in manifest["displays"]]
    expected_ids = [*FIGURE_STEMS, *TABLE_STEMS]
    if actual_ids != expected_ids:
        raise ValueError("thesis display ordering or identities differ")
    sources: dict[str, Path] | None = None
    if verify_sources:
        sources = _resolve_sources(repo_root)
        if manifest.get("projection_checks") != _projection_checks(sources):
            raise ValueError("thesis display projection checks differ")
    for record in manifest["displays"]:
        display_id = record["display_id"]
        spec = DISPLAY_SPECS[display_id]
        is_figure = display_id in FIGURE_STEMS
        expected_metadata = {
            "display_type": "figure" if is_figure else "table",
            "chapter": spec.chapter,
            "generator": (
                f"danids.evaluation.thesis_assets:_figure_{display_id.lower()}"
                if is_figure
                else f"danids.evaluation.thesis_assets:_table_{display_id.lower()}"
            ),
            "frozen_takeaway": spec.takeaway,
        }
        if any(record.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError(f"thesis display metadata differs: {display_id}")
        expected_sources = [SOURCE_PATHS[key] for key in spec.sources]
        actual_sources = [source.get("path") for source in record["source_artifacts"]]
        if actual_sources != expected_sources:
            raise ValueError(f"thesis display source provenance differs: {display_id}")
        expected_outputs = _expected_output_paths(display_id)
        actual_outputs = [output.get("path") for output in record["output_files"]]
        if actual_outputs != expected_outputs:
            raise ValueError(f"thesis display output roster differs: {record['display_id']}")
        for source in record["source_artifacts"]:
            _validate_digest_metadata(source, include_bytes=False)
            if verify_sources:
                path = repo_root / source["path"]
                if not path.is_file() or _sha256(path) != source["sha256"]:
                    raise ValueError(f"frozen source digest differs: {path}")
        for output in record["output_files"]:
            _validate_digest_metadata(output, include_bytes=True)
            path = output_root / output["path"]
            if (
                not path.is_file()
                or _sha256(path) != output["sha256"]
                or _digest_byte_count(path) != output["bytes"]
            ):
                raise ValueError(f"generated output digest differs: {path}")
            _validate_rendered_file(path)
    supplementary = manifest["supplementary_files"]
    if [output.get("path") for output in supplementary] != ["captions.md"]:
        raise ValueError("thesis supplementary output roster differs")
    for output in supplementary:
        _validate_digest_metadata(output, include_bytes=True)
        path = output_root / output["path"]
        if (
            not path.is_file()
            or _sha256(path) != output["sha256"]
            or _digest_byte_count(path) != output["bytes"]
        ):
            raise ValueError(f"supplementary output digest differs: {path}")
    text_paths = [output_root / "captions.md", *(output_root / "tables").glob("*.md")]
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in text_paths)
    found = _wording_guardrail_violations(text)
    if found:
        raise ValueError("wording guardrail violated: " + ", ".join(found))
