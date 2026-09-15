from __future__ import annotations

import tomllib
from importlib.metadata import metadata, version
from pathlib import Path

import yaml

import danids

REPO_ROOT = Path(__file__).parents[1]


def test_public_package_version_is_thesis_release() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    assert danids.__version__ == "2.0.0"
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
        "version": "2.0.0",
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
