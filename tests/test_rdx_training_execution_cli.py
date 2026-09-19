"""Focused CLI tests for the explicit RDX-006 execution gate."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import danids.cli as cli_module
from danids.cli import build_parser, main
from danids.config.rdx_training_evidence import RDX004Budget


def _arguments(*, execute: bool = True, smoke: bool = True) -> list[str]:
    arguments = [
        "run-rdx004-training-evidence",
        "--execution-config",
        "configs/experiments/rdx/rdx006_execution_v1.yaml",
        "--datasets-config",
        "configs/datasets.yaml",
        "--preflight-dir",
        "rdx/training-evidence-preflight-v1",
        "--budget",
        "B1600",
        "--rotation",
        "U",
        "T",
        "C",
        "B",
        "--seed",
        "42",
        "--manifest-dir",
        "manifests/study1-s42",
        "--output-root",
        ".",
        "--device",
        "cpu",
    ]
    if smoke:
        arguments.append("--smoke")
    if execute:
        arguments.append("--execute")
    return arguments


def test_parser_exposes_only_prospective_budget_seed_and_explicit_execute() -> None:
    args = build_parser().parse_args(_arguments())

    assert args.budget is RDX004Budget.B1600
    assert args.rotation == ["U", "T", "C", "B"]
    assert args.seed == 42
    assert args.smoke is True
    assert args.execute is True
    assert args.output_root == Path(".")

    with pytest.raises(SystemExit):
        build_parser().parse_args([value if value != "B1600" else "B100" for value in _arguments()])
    with pytest.raises(SystemExit):
        build_parser().parse_args([value if value != "42" else "45" for value in _arguments()])
    invalid_domain = _arguments()
    invalid_domain[invalid_domain.index("U", invalid_domain.index("--rotation"))] = "X"
    with pytest.raises(SystemExit):
        build_parser().parse_args(invalid_domain)


def test_handler_rejects_missing_execute_before_loading_inputs(capsys: Any) -> None:
    assert main(_arguments(execute=False)) == 2
    captured = capsys.readouterr()
    assert "requires explicit --execute" in captured.err


def test_handler_rejects_nonfrozen_rotation_before_loading_inputs(capsys: Any) -> None:
    arguments = _arguments()
    start = arguments.index("--rotation") + 1
    arguments[start : start + 4] = ["U", "T", "B", "C"]

    assert main(arguments) == 2
    captured = capsys.readouterr()
    assert "one exact frozen rotation" in captured.err


def test_handler_forwards_typed_authorized_request(
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    registry = object()
    execution_config = object()
    calls: list[tuple[object, object, dict[str, object]]] = []

    monkeypatch.setattr(
        cli_module.DatasetRegistry,
        "from_yaml",
        lambda path: registry,
    )
    monkeypatch.setattr(
        cli_module,
        "load_rdx006_execution_config",
        lambda path: execution_config,
    )
    module = types.ModuleType("danids.experiments.rdx_training_execution")

    def fake_runner(
        actual_registry: object,
        actual_config: object,
        **kwargs: object,
    ) -> Path:
        calls.append((actual_registry, actual_config, kwargs))
        return Path("runs/rdx004-training-evidence-smoke-v1/test")

    module.run_rdx004_training_evidence = fake_runner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    assert main(_arguments()) == 0
    assert calls == [
        (
            registry,
            execution_config,
            {
                "budget": RDX004Budget.B1600,
                "rotation": ("U", "T", "C", "B"),
                "seed": 42,
                "preflight_dir": Path("rdx/training-evidence-preflight-v1"),
                "manifest_dir": Path("manifests/study1-s42"),
                "output_root": Path("."),
                "device_name": "cpu",
                "smoke": True,
                "execute": True,
            },
        )
    ]
    reported = json.loads(capsys.readouterr().out)["run_directory"]
    assert Path(reported).name == "test"
