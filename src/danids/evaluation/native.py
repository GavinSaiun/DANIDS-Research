"""Exact dataset-native attack-label recall metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def native_attack_recall_rows(
    binary_labels: NDArray[np.int8],
    native_labels: NDArray[np.object_],
    scores: NDArray[np.float64],
    threshold: float,
) -> tuple[list[dict[str, Any]], float | None, float | None]:
    attack = np.asarray(binary_labels) == 1
    labels = np.asarray(native_labels, dtype=object)
    predicted = np.asarray(scores) >= threshold
    rows: list[dict[str, Any]] = []
    for label in sorted({str(value) for value in labels[attack]}):
        selected = attack & (labels.astype(str) == label)
        support = int(selected.sum())
        recall = float(np.sum(predicted & selected) / support)
        rows.append({"native_attack_label": label, "support": support, "recall": recall})
    recalls = [float(row["recall"]) for row in rows]
    return rows, (float(np.mean(recalls)) if recalls else None), (min(recalls) if recalls else None)
