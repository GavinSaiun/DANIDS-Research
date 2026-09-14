"""Create the artifact-only thesis analysis for frozen confirmatory Study 4."""

from __future__ import annotations

import argparse
from pathlib import Path

from danids.evaluation.study4_analysis import analyze_study4_confirmatory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a complete Study-4 evaluation and create its deterministic "
            "artifact-only analysis package."
        )
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        required=True,
        help="Validated complete Study-4 evaluation directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New analysis output directory; existing paths are never overwritten.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = analyze_study4_confirmatory(args.evaluation_dir, args.output_dir)
    print(f"Study-4 confirmatory analysis written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
