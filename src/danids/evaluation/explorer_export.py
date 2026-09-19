"""Deterministic public-evidence presentation only; no scientific evaluation imports."""

# ruff: noqa: RUF001
# Mathematical minus signs and en dashes are intentional public display typography.

from __future__ import annotations

import ast
import hashlib
import json
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

VERSION = "danids-explorer-v1"
# Reviewed public-source identities, not identities of ignored experiment bundles.
SOURCE_HASHES = {
    "RDX_RESULTS.md": "bd07c9287e5f8a0804552eb7bc7db33d03c3b57b096c72b29b2143a05f0ebee5",
    "docs/thesis_evidence_freeze.md": (
        "6d6c996f5a70ad3a63104f87f715afb4c921f9025c51ea0e927c5c33228a7c11"
    ),
    "thesis/assets/visual_manifest.json": (
        "0d03c5894c2e30f357132f6d0f293e05cab9c159c660731dff90af93236aecbb"
    ),
    "thesis/assets/svg/F03_static_transfer.svg": (
        "77bab8e4d4fbad0493cdca3e31a31789714d6b4b3c3d8f0329917380bab02482"
    ),
    "src/danids/config/core.py": "4564691bd72e8687b8ba5075fe21f0c789547bfafa777c37ad4ca2ff47ab592f",
    "src/danids/health/states.py": (
        "6481fece908227bc2ada14af933495d4b14ef3b145ee11097dd10f5e94db8e61"
    ),
    "docs/decisions.md": "031198fd5be7f1afc930672488493d4467da6d59f1180cea6c43291f4d30bdd6",
    "pyproject.toml": "9f8ef69b50b9b908b6c84b74c2676e0acf80895bb078c4913f770d19ea1f7304",
}


def canonical_bytes(text: str) -> bytes:
    """Use the existing thesis text-hash convention on every platform."""
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_explorer_projection(repo_root: Path) -> dict[str, bytes]:
    """Read only reviewed public sources. Missing/changed evidence fails closed."""
    root = repo_root.resolve()
    texts: dict[str, str] = {}
    for name, expected in SOURCE_HASHES.items():
        source = root / name
        if source.resolve() != source or not source.is_file():
            raise ValueError(f"Explorer source missing or redirected: {name}")
        text = source.read_text(encoding="utf-8")
        if _sha(canonical_bytes(text)) != expected:
            raise ValueError(f"Explorer source identity differs: {name}")
        texts[name] = text

    def envelope(data: dict[str, Any], *names: str) -> dict[str, Any]:
        return {
            "schema_version": VERSION,
            "sources": [
                {"path": name, "sha256": SOURCE_HASHES[name], "hash_mode": "canonical-lf-utf8"}
                for name in names
            ],
            "data": data,
        }

    public = "RDX_RESULTS.md"
    manifest_path = "thesis/assets/visual_manifest.json"
    visual = json.loads(texts[manifest_path])
    svg_path = "thesis/assets/svg/F03_static_transfer.svg"
    f3 = next(item for item in visual["displays"] if item["display_id"] == "F3")
    svg_record = next(item for item in f3["output_files"] if item["path"].endswith(".svg"))
    if svg_record["sha256"] != SOURCE_HASHES[svg_path]:
        raise ValueError("F03 manifest/source disagreement")
    svg = ET.fromstring(texts[svg_path])
    ns = {"s": "http://www.w3.org/2000/svg"}

    def svg_text(index: int) -> str:
        node = svg.find(f".//s:g[@id='text_{index}']/s:text", ns)
        if node is None:
            raise ValueError("Missing F03 annotation")
        return "".join(node.itertext())

    if svg_text(27) != "A  Mean attack recall" or svg_text(54) != (
        "B  Mean false-positive budget ratio"
    ):
        raise ValueError("F03 panel semantics differ")
    domains = ["U", "T", "C", "B"]
    cells = [
        {
            "source": source,
            "target": target,
            "recall": svg_text(11 + i * 4 + j),
            "fpr_budget_ratio": svg_text(38 + i * 4 + j),
        }
        for i, source in enumerate(domains)
        for j, target in enumerate(domains)
    ]
    if any(
        not 0 <= float(cell["recall"]) <= 1 or not 0 <= float(cell["fpr_budget_ratio"]) <= 1000
        for cell in cells
    ):
        raise ValueError("Invalid F03 annotation range")
    core_path = "src/danids/config/core.py"
    tree = ast.parse(texts[core_path])
    features = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "CORE_HEALTH_FEATURES"
            for target in node.targets
        )
    )
    if len(features) != 28 or len(set(features)) != 28:
        raise ValueError("Core feature contract differs")
    freeze = "docs/thesis_evidence_freeze.md"
    ledger = dict(
        re.findall(r"\b(H\d+)\s+(SUPPORTED|NOT_SUPPORTED|NOT_TESTABLE|PARTIAL)\b", texts[freeze])
    )
    if len(ledger) != 9:
        raise ValueError("Frozen hypothesis ledger unavailable")
    # Curated transcription of the hash-bound public document, not fresh calculation.
    pathways = [
        {
            "id": "recognition",
            "title": "Harm recognised",
            "numerator": 22539,
            "denominator": 32133,
            "percent": "70.14%",
            "unit": "evaluator-HARMFUL window",
            "outcome": "PREDICTED_HARMFUL in the same window",
            "population": "Frozen Study-4 matrix",
        },
        {
            "id": "capability",
            "title": "One-step capability",
            "numerator": 9,
            "denominator": 7776,
            "percent": "≈0.12%",
            "unit": "eligible harmful Oracle decision",
            "outcome": "Any confirmed A0–A4 one-step success",
            "population": "Offline Oracle only",
        },
        {
            "id": "immediate",
            "title": "Immediate deployed recovery",
            "numerator": 4,
            "denominator": 130,
            "percent": "3.08%",
            "unit": "accepted deployed intervention",
            "outcome": "Immediate SAFE successor",
            "population": "Accepted model-changing interventions",
        },
        {
            "id": "sustained",
            "title": "Sustained deployed recovery",
            "numerator": 2,
            "denominator": 65,
            "percent": "3.08%",
            "unit": "sustained-recovery-assessable intervention",
            "outcome": "Two consecutive SAFE windows",
            "population": "Assessable two-window horizons only",
        },
    ]
    budgets = [
        {
            "budget": "B100",
            "timing": "historical",
            "optimizer_rows": 80,
            "eligible": 7776,
            "mean_p1": 0.212,
            "median_p1": 0,
            "mean_p2": 0.188,
            "median_p2": 0,
        },
        {
            "budget": "B400",
            "timing": "prospective",
            "optimizer_rows": 380,
            "eligible": 7480,
            "mean_p1": 0.416,
            "median_p1": 0,
            "mean_p2": 0.416,
            "median_p2": 0,
        },
        {
            "budget": "B1600",
            "timing": "prospective",
            "optimizer_rows": 1580,
            "eligible": 7428,
            "mean_p1": 0.568,
            "median_p1": 0,
            "mean_p2": 0.550,
            "median_p2": 0,
        },
    ]
    contrasts = [
        {"contrast": name, "mean_pp": mean, "median_pp": 0, "positive_rotations": "1/4"}
        for name, mean in [("B400 − B100", 0.228), ("B1600 − B100", 0.362), ("B1600 − B400", 0.134)]
    ]
    docs = {
        "overview.json": envelope(
            {
                "chain": [
                    "Cross-domain shift",
                    "Operational degradation",
                    "Harmful operation",
                    "Recognition",
                    "Intervention",
                    "Recovery",
                    "Training-evidence sensitivity",
                ],
                "ledger": ledger,
                "limitations": [
                    "Study 5A measures family-conditioned binary detection recall, "
                    "not attribution or open-set recognition.",
                    "H7 combines prior U-T-C-B evidence with three prospectively frozen missing "
                    "rotations; replay benefits did not generalise.",
                    "Four related NetFlow-v3 domains, one compact MLP and three seeds "
                    "bound the evidence.",
                    "RDX is post-freeze evidence, not Study 6 or part of the original Study-4 "
                    "confirmatory design.",
                ],
            },
            freeze,
            public,
        ),
        "cross_domain.json": envelope(
            {
                "domains": domains,
                "cells": cells,
                "precision": "Published rounded F03 annotations",
                "aggregation": "Mean across seeds 42–44; not seed-level evidence",
                "unavailable": ["AUROC", "PR-AUC", "Seed-level values", "Unrounded FPR"],
            },
            manifest_path,
            svg_path,
        ),
        "model_health.json": envelope(
            {
                "target_fpr": 0.001,
                "recall_loss_tolerance": 0.10,
                "confidence": "Wilson 95%",
                "feature_groups": [
                    "Distribution shift",
                    "Score / output behaviour",
                    "Uncertainty / confidence",
                    "Representation",
                    "Conformal set behaviour",
                ],
                "features": list(features),
                "states": [
                    {
                        "name": "SAFE",
                        "rule": "FPR upper bound ≤ target AND recall lower bound ≥ "
                        "reference recall − tolerance.",
                    },
                    {
                        "name": "UNCERTAIN",
                        "rule": "Neither sufficient evidence of violation nor both safety "
                        "conditions; includes missing support.",
                    },
                    {
                        "name": "HARMFUL",
                        "rule": "FPR lower bound > target OR recall upper bound < "
                        "reference recall − tolerance.",
                    },
                ],
            },
            core_path,
            freeze,
            "src/danids/health/states.py",
            "docs/decisions.md",
        ),
        "study4.json": envelope({"methods": visual["projection_checks"]["F7"]}, manifest_path),
        "rdx.json": envelope(
            {
                "pathways": pathways,
                "core_recognised": 5715,
                "core_harmful": 8290,
                "budgets": budgets,
                "contrasts": contrasts,
                "paired_units": 12,
                "gate_median_pp": 2,
                "gate_positive_rotations": "3/4",
                "decision": "TRAINING_EVIDENCE_INCREASE_NOT_MATERIALLY_RECOVERABILITY_EXPANDING",
                "interpretation": "B100 training scarcity alone was insufficient to explain the "
                "observed recovery ceiling under the frozen compact model and A0–A4 "
                "intervention family.",
                "actions": [
                    {
                        "id": "A0",
                        "name": "NO_OP",
                        "detail": "Keep the deployed model and threshold unchanged.",
                    },
                    {
                        "id": "A1",
                        "name": "RECALIBRATE",
                        "detail": "Threshold-only candidate; strict benign support of at least "
                        "3,838 at FPR 0.001 with two-sided Wilson 95%, "
                        "normally infeasible under B100.",
                    },
                    {
                        "id": "A2",
                        "name": "HEAD_UPDATE",
                        "detail": "Train the head using legitimately released training evidence.",
                    },
                    {
                        "id": "A3",
                        "name": "FULL_FINE_TUNE",
                        "detail": "Full-model fine-tuning; Oracle-selectable, not Core-selectable. "
                        "Also available to prescribed adaptation baselines.",
                    },
                    {
                        "id": "A4",
                        "name": "REPLAY_UPDATE",
                        "detail": "Update with legitimate current training evidence and historical "
                        "replay; audit rows never train.",
                    },
                ],
            },
            public,
            "docs/decisions.md",
        ),
        "provenance.json": envelope(
            {
                "freeze_tag": "v2.0-thesis-freeze",
                "rdx_tag": "rdx-training-evidence-final",
                "software_version": "v2.1.0",
                "bundle_digest": "c28ef8a5b73db863ec18be0c3f21110defde34967623214268e169312362fbc7",
                "contract_sha256": (
                    "8a9ee12ad626014b3903c9c34f8117fb01ebabcf60808b0de507bd5fadfac8a8"
                ),
                "scope": "Public visualization projections; not a revalidation of ignored "
                "evidence bundles.",
            },
            public,
            manifest_path,
            "pyproject.toml",
        ),
    }
    outputs = {name: _json_bytes(value) for name, value in docs.items()}
    outputs["manifest.json"] = _json_bytes(
        {
            "schema_version": VERSION,
            "hash_mode": "canonical-lf-utf8",
            "files": [
                {"path": name, "sha256": _sha(content), "bytes": len(content)}
                for name, content in sorted(outputs.items())
            ],
        }
    )
    return outputs


def export_explorer_data(repo_root: Path, output_dir: Path, *, check: bool = False) -> Path:
    """Write only a new presentation directory, or verify an identical existing one."""
    outputs = build_explorer_projection(repo_root)
    output = output_dir.absolute()
    if output.resolve() != output:
        raise ValueError("Explorer output must not be redirected")
    if output.exists():
        if not output.is_dir() or {p.name for p in output.iterdir()} != set(outputs):
            raise ValueError(
                "Explorer output directory is not an exact projection; refusing overwrite"
            )
        for name, content in outputs.items():
            path = output / name
            if (
                path.is_symlink()
                or not path.is_file()
                or canonical_bytes(path.read_text(encoding="utf-8")) != content
            ):
                raise ValueError(f"Explorer projection differs: {name}; refusing overwrite")
        return output
    if check:
        raise ValueError("Explorer output missing")
    # Do not create presentation subdirectories inside any protected evidence tree.
    protected = [
        repo_root / name
        for name in (
            "data",
            "datasets",
            "runs",
            "results",
            "rdx",
            "thesis/assets",
            "dist",
            "study1",
            "study2",
            "study3",
            "study4",
            "study5",
            "manifests",
            "checkpoints",
        )
    ]
    if any(output.is_relative_to(path.resolve()) for path in protected):
        raise ValueError("Explorer output cannot be inside scientific evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".explorer-", dir=output.parent) as tmp:
        staging = Path(tmp) / "evidence"
        staging.mkdir()
        for name, content in outputs.items():
            (staging / name).write_bytes(content)
        staging.rename(output)
    return output
