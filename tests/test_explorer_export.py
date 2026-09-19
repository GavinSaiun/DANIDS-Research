"""Explorer uses a tiny public allowlist; no experiment execution is involved."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from danids.evaluation.explorer_export import (
    SOURCE_HASHES,
    VERSION,
    build_explorer_projection,
    canonical_bytes,
    export_explorer_data,
)

ROOT = Path(__file__).resolve().parents[1]


def public_clone(tmp_path: Path) -> Path:
    root = tmp_path / "clone"
    for name in SOURCE_HASHES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    return root


def test_clean_clone_determinism_and_source_immutability(tmp_path: Path) -> None:
    root = public_clone(tmp_path)
    before = {p: (root / p).read_bytes() for p in SOURCE_HASHES}
    expected = build_explorer_projection(ROOT)
    assert build_explorer_projection(root) == expected
    assert not (root / "runs").exists()
    output = tmp_path / "export"
    export_explorer_data(root, output)
    export_explorer_data(root, output, check=True)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == expected
    assert before == {p: (root / p).read_bytes() for p in SOURCE_HASHES}


def test_exact_schema_values_and_manifest() -> None:
    outputs = build_explorer_projection(ROOT)
    assert set(outputs) == {
        "overview.json",
        "cross_domain.json",
        "model_health.json",
        "study4.json",
        "rdx.json",
        "provenance.json",
        "manifest.json",
    }
    manifest = json.loads(outputs["manifest.json"])
    assert set(manifest) == {"schema_version", "hash_mode", "files"}
    for entry in manifest["files"]:
        content = outputs[entry["path"]]
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()
        assert entry["bytes"] == len(content)
        doc = json.loads(content)
        assert set(doc) == {"schema_version", "sources", "data"}
        assert doc["schema_version"] == VERSION
        for source in doc["sources"]:
            assert set(source) == {"path", "sha256", "hash_mode"}
            assert source["sha256"] == SOURCE_HASHES[source["path"]]
    rdx = json.loads(outputs["rdx.json"])["data"]
    assert [(p["numerator"], p["denominator"], p["percent"]) for p in rdx["pathways"]] == [
        (22539, 32133, "70.14%"),
        (9, 7776, "≈0.12%"),
        (4, 130, "3.08%"),
        (2, 65, "3.08%"),
    ]
    assert (rdx["core_recognised"], rdx["core_harmful"]) == (5715, 8290)
    assert [(b["optimizer_rows"], b["mean_p1"], b["mean_p2"]) for b in rdx["budgets"]] == [
        (80, 0.212, 0.188),
        (380, 0.416, 0.416),
        (1580, 0.568, 0.550),
    ]
    assert all(b["median_p1"] == b["median_p2"] == 0 for b in rdx["budgets"])
    assert [b["timing"] for b in rdx["budgets"]] == ["historical", "prospective", "prospective"]
    assert rdx["paired_units"] == 12
    assert rdx["gate_median_pp"] == 2
    assert rdx["gate_positive_rotations"] == "3/4"
    assert [c["mean_pp"] for c in rdx["contrasts"]] == [0.228, 0.362, 0.134]
    transfer = json.loads(outputs["cross_domain.json"])["data"]
    assert len(transfer["cells"]) == 16
    assert transfer["cells"][1] == {
        "source": "U",
        "target": "T",
        "recall": "0.582",
        "fpr_budget_ratio": "895",
    }
    assert "rounded" in transfer["precision"]


def test_wording_and_frozen_ledger() -> None:
    docs = {k: json.loads(v) for k, v in build_explorer_projection(ROOT).items()}
    ledger = docs["overview.json"]["data"]["ledger"]
    assert ledger == {
        "H1": "SUPPORTED",
        "H2": "SUPPORTED",
        "H3": "NOT_SUPPORTED",
        "H4": "NOT_SUPPORTED",
        "H6": "NOT_TESTABLE",
        "H7": "NOT_SUPPORTED",
        "H8": "NOT_TESTABLE",
        "H9": "PARTIAL",
        "H10": "PARTIAL",
    }
    assert "not Study 6" in str(docs["overview.json"])
    assert "not attribution" in str(docs["overview.json"])
    assert "not Core-selectable" in str(docs["rdx.json"])
    assert "frozen compact model" in docs["rdx.json"]["data"]["interpretation"]
    features = docs["model_health.json"]["data"]["features"]
    assert len(features) == len(set(features)) == 28
    assert all(f.startswith(("dist_", "model_")) for f in features)


@pytest.mark.parametrize("missing", [True, False])
def test_changed_or_missing_source_fails_before_write(tmp_path: Path, missing: bool) -> None:
    root = public_clone(tmp_path)
    source = root / "RDX_RESULTS.md"
    if missing:
        source.unlink()
    else:
        source.write_text("Changed evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        export_explorer_data(root, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_no_raw_or_ignored_file_access(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_text
    accessed: list[str] = []

    def guarded(path: Path, *args: object, **kwargs: object) -> str:
        relative = path.relative_to(ROOT).as_posix()
        assert relative in SOURCE_HASHES
        accessed.append(relative)
        return original(path, encoding="utf-8")

    monkeypatch.setattr(Path, "read_text", guarded)
    build_explorer_projection(ROOT)
    assert set(accessed) == set(SOURCE_HASHES)


def test_refuses_overwrite_and_protected_output(tmp_path: Path) -> None:
    root = public_clone(tmp_path)
    output = tmp_path / "export"
    export_explorer_data(root, output)
    (output / "rdx.json").write_bytes(b"different")
    with pytest.raises(ValueError, match="refusing overwrite"):
        export_explorer_data(root, output)
    assert (output / "rdx.json").read_bytes() == b"different"
    with pytest.raises(ValueError, match="scientific evidence"):
        export_explorer_data(root, root / "study4" / "new-output")
    with pytest.raises(ValueError, match="missing"):
        export_explorer_data(root, tmp_path / "absent", check=True)
    assert not (tmp_path / "absent").exists()


def test_line_endings_and_committed_projection(tmp_path: Path) -> None:
    root = public_clone(tmp_path)
    for name in SOURCE_HASHES:
        path = root / name
        path.write_bytes(canonical_bytes(path.read_text(encoding="utf-8")).replace(b"\n", b"\r\n"))
    assert build_explorer_projection(root) == build_explorer_projection(ROOT)
    export_explorer_data(ROOT, ROOT / "explorer/public/evidence", check=True)
