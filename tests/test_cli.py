from __future__ import annotations

import json
from pathlib import Path

from danids.cli import build_parser, main


def test_validate_generate_and_dry_run_end_to_end(
    tmp_path: Path,
    datasets_config: Path,
    experiment_config: Path,
    capsys: object,
) -> None:
    assert main(["validate", "--datasets-config", str(datasets_config)]) == 0
    # capsys is supplied by pytest; avoid binding this test to its concrete private type.
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert len(json.loads(captured.out)["datasets"]) == 4

    output = tmp_path / "manifests"
    assert (
        main(
            [
                "generate-manifests",
                "--datasets-config",
                str(datasets_config),
                "--experiment-config",
                str(experiment_config),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()  # type: ignore[attr-defined]
    assert sorted(path.name for path in output.glob("*.json")) == [
        "stage-01-B.json",
        "stage-02-U.json",
        "stage-03-C.json",
        "stage-04-T.json",
    ]

    assert (
        main(
            [
                "dry-run",
                "--datasets-config",
                str(datasets_config),
                "--experiment-config",
                str(experiment_config),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert summary["sequence"] == ["B", "U", "C", "T"]
    assert [stage["windows"] for stage in summary["stages"]] == [0, 4, 4, 4]
    assert [stage["holdout_rows"] for stage in summary["stages"]] == [5, 5, 5, 5]


def test_task009_launcher_and_evaluator_cli_contracts() -> None:
    parser = build_parser()
    launch = parser.parse_args(
        [
            "launch-study5b-replay-retention",
            "--contract",
            "configs/study5/task009_h7_all_order_v1.yaml",
            "--preflight-only",
        ]
    )
    assert launch.preflight_only is True
    assert launch.device == "cpu"
    assert launch.workers == 1

    evaluate = parser.parse_args(
        [
            "evaluate-study5b-replay-robustness",
            "--contract",
            "configs/study5/task009_h7_all_order_v1.yaml",
            "--task008-dir",
            "study5/threat-audit-v1",
            "--study1-run",
            "runs/E1_STATIC_MLP_T-C-B-U_s42",
            "--study2-run",
            "runs/E2_ER_T-C-B-U_B100_D1_s42",
            "--output-dir",
            "study5/task009-study5b-all-order-replay-v1",
        ]
    )
    assert evaluate.study1_run_dirs == [Path("runs/E1_STATIC_MLP_T-C-B-U_s42")]
    assert evaluate.study2_run_dirs == [Path("runs/E2_ER_T-C-B-U_B100_D1_s42")]
