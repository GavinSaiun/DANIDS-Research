from __future__ import annotations

import json
from pathlib import Path

from danids.cli import main


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
