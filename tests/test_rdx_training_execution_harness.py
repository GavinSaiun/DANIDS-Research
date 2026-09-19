"""Focused regressions for RDX-006 execution-artifact representation."""

from __future__ import annotations

import torch
import yaml

from danids.experiments.rdx_training_execution import _canonical_digest, _json_compatible


def test_contract_identity_ignores_yaml_tuple_list_representation_only() -> None:
    tuple_contract = {"supervision": {"query_states": ("UNCERTAIN", "HARMFUL")}}
    yaml_contract = {"supervision": {"query_states": ["UNCERTAIN", "HARMFUL"]}}

    assert _canonical_digest(tuple_contract) == _canonical_digest(yaml_contract)


def test_resolved_artifact_normalizes_scalar_subclasses_and_tuples() -> None:
    normalized = _json_compatible(
        {
            "code": {"torch": torch.__version__},
            "query_states": ("UNCERTAIN", "HARMFUL"),
        }
    )

    assert type(normalized["code"]["torch"]) is str
    assert normalized["query_states"] == ["UNCERTAIN", "HARMFUL"]
    assert yaml.safe_load(yaml.safe_dump(normalized)) == normalized
