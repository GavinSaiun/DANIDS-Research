from __future__ import annotations

import tomllib
from importlib.metadata import metadata, version
from pathlib import Path

import yaml

import danids

REPO_ROOT = Path(__file__).parents[1]

FROZEN_HYPOTHESIS_LEDGER = """H1  SUPPORTED
H2  SUPPORTED
H3  NOT_SUPPORTED
H4  NOT_SUPPORTED
H6  NOT_TESTABLE
H7  NOT_SUPPORTED
H8  NOT_TESTABLE
H9  PARTIAL
H10 PARTIAL"""


def test_public_package_version_is_current_software_release() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    assert danids.__version__ == "2.1.0"
    assert project["version"] == danids.__version__
    assert pyproject["build-system"]["requires"] == ["setuptools>=77"]
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert project["authors"] == [{"name": "Gavin Sun"}]
    assert version("danids-research") == danids.__version__
    installed = metadata("danids-research")
    assert installed["Author"] == "Gavin Sun"
    assert installed["License-Expression"] == "MIT"
    assert installed.get_all("License-File") == ["LICENSE"]


def test_citation_metadata_is_minimal_verified_and_unarchived() -> None:
    citation = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation == {
        "cff-version": "1.2.0",
        "message": (
            "If you use this software or its frozen research artifacts, please cite it "
            "using this metadata."
        ),
        "title": (
            "DANIDS: Deployment-Aware Network Intrusion Detection under Sequential "
            "Cross-Domain Shift"
        ),
        "type": "software",
        "version": "2.1.0",
        "license": "MIT",
        "authors": [{"family-names": "Sun", "given-names": "Gavin"}],
        "repository-code": "https://github.com/GavinSaiun/DANIDS-Research",
    }


def test_mit_license_is_consistent_across_release_metadata() -> None:
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    licensing = (REPO_ROOT / "docs" / "LICENSING.md").read_text(encoding="utf-8")

    assert license_text.startswith("MIT License\n\nCopyright (c) 2026 Gavin Sun\n")
    assert "Permission is hereby granted, free of charge" in license_text
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in license_text
    assert "[MIT License](LICENSE)" in readme
    assert "[`docs/LICENSING.md`](docs/LICENSING.md)" in readme
    assert "root [`LICENSE`](../LICENSE)" in licensing
    obsolete_doc = "_".join(("LICENSE", "DECISION", "REQUIRED")) + ".md"
    assert not (REPO_ROOT / "docs" / obsolete_doc).exists()

    release_text = "\n".join((readme, licensing))
    for stale_phrase in (
        "rights reserved",
        "no public-use",
        "no standalone public-use",
        "unresolved public-release",
    ):
        assert stale_phrase.casefold() not in release_text.casefold()


def test_public_docs_separate_frozen_studies_from_post_freeze_rdx() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    rdx_results = (REPO_ROOT / "RDX_RESULTS.md").read_text(encoding="utf-8")
    thesis_results = (REPO_ROOT / "THESIS_RESULTS.md").read_text(encoding="utf-8")
    normalized_rdx = " ".join(rdx_results.split())
    normalized_thesis = " ".join(thesis_results.split())

    assert "DANIDS 2.0 Studies 1--5 were frozen on 15 September 2026" in readme
    assert "RDX does not alter the original hypothesis ledger" in readme
    assert "[RDX results](RDX_RESULTS.md)" in readme
    assert "separately" in normalized_rdx
    assert "versioned evidence completed and frozen after" in normalized_rdx
    assert "RDX is not Study 6" in normalized_rdx
    assert "B100 is historical evidence" in normalized_rdx
    assert "B400 and B1600 are prospective treatments" in normalized_rdx
    assert "## Post-freeze Recoverability Diagnostic Extension (RDX)" in thesis_results
    assert "No budget sensitivity was part of the original" in normalized_thesis
    assert "RDX later performed a separately versioned" in normalized_thesis


def test_original_frozen_hypothesis_ledger_is_unchanged() -> None:
    paths = (
        REPO_ROOT / "README.md",
        REPO_ROOT / "THESIS_RESULTS.md",
        REPO_ROOT / "docs" / "thesis_evidence_freeze.md",
    )
    for path in paths:
        assert FROZEN_HYPOTHESIS_LEDGER in path.read_text(encoding="utf-8")


def test_rdx_release_metadata_is_dataset_free_and_integrity_bound() -> None:
    rdx_results = (REPO_ROOT / "RDX_RESULTS.md").read_text(encoding="utf-8")
    reproducibility = (REPO_ROOT / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    decision = "TRAINING_EVIDENCE_INCREASE_NOT_MATERIALLY_RECOVERABILITY_EXPANDING"
    bundle = "c28ef8a5b73db863ec18be0c3f21110defde34967623214268e169312362fbc7"
    contract = "8a9ee12ad626014b3903c9c34f8117fb01ebabcf60808b0de507bd5fadfac8a8"
    assert decision in rdx_results
    assert bundle in rdx_results and bundle in reproducibility
    assert contract in rdx_results and contract in reproducibility
    assert "not present in a clean Git clone" in rdx_results
    assert "### 2.1.0" in changelog
    assert "### 2.0.0" in changelog
