"""Shared, deterministic visual style for the frozen DANIDS thesis displays."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure

PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#222222",
    "grey": "#7A7A7A",
    "light_grey": "#E6E6E6",
    "white": "#FFFFFF",
}

DOMAIN_ORDER = ("U", "T", "C", "B")
DOMAIN_LABELS = {
    "U": "NF-UNSW-NB15-v3",
    "T": "NF-ToN-IoT-v3",
    "C": "NF-CSE-CIC-IDS2018-v3",
    "B": "NF-BoT-IoT-v3",
}
METHOD_ORDER = ("STATIC", "NAIVE_FT", "EWC", "ER", "FT_MEM")
METHOD_LABELS = {
    "STATIC": "Static",
    "NAIVE_FT": "NaiveFT",
    "EWC": "EWC",
    "ER": "ER",
    "FT_MEM": "FT-Mem",
    "ALWAYS_ADAPT": "Always-Adapt",
    "DANIDS_CORE": "DANIDS-Core",
    "OFFLINE_ORACLE": "Offline Oracle",
}


@contextmanager
def thesis_style() -> Iterator[None]:
    """Apply the repository-wide thesis style without leaking global rcParams."""

    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "axes.edgecolor": PALETTE["black"],
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.titlesize": 11,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "danids-thesis-assets-v1",
        }
    ):
        yield


def export_figure(figure: Figure, output_root: Path, stem: str) -> list[Path]:
    """Export a figure in deterministic thesis formats."""

    paths = [
        output_root / "svg" / f"{stem}.svg",
        output_root / "pdf" / f"{stem}.pdf",
        output_root / "png" / f"{stem}.png",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        paths[0],
        format="svg",
        bbox_inches="tight",
        metadata={"Creator": "DANIDS thesis asset generator", "Date": None},
    )
    svg_text = paths[0].read_text(encoding="utf-8")
    paths[0].write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    figure.savefig(
        paths[1],
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "DANIDS thesis asset generator",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(
        paths[2],
        format="png",
        dpi=360,
        bbox_inches="tight",
        metadata={"Software": "DANIDS thesis asset generator"},
    )
    return paths
