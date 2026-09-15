from __future__ import annotations

import tomllib
from importlib.metadata import version
from pathlib import Path

import yaml

import danids


def test_public_package_version_is_thesis_release() -> None:
    pyproject = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert danids.__version__ == "2.0.0"
    assert pyproject["project"]["version"] == danids.__version__
    assert version("danids-research") == danids.__version__


def test_citation_metadata_is_minimal_verified_and_unreleased() -> None:
    citation = yaml.safe_load(
        (Path(__file__).parents[1] / "CITATION.cff").read_text(encoding="utf-8")
    )
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
        "authors": [{"family-names": "Sun", "given-names": "Gavin"}],
        "repository-code": "https://github.com/GavinSaiun/DANIDS-Research",
    }
