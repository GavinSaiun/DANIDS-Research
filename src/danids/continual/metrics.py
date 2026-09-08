"""Deterministic continual-learning metrics over evaluator-only holdout records."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

CONTINUAL_METRICS = ("pr_auc", "roc_auc", "tpr")
ADAPTATION_METRICS = (*CONTINUAL_METRICS, "fpr")


def _value(row: Mapping[str, Any], metric: str) -> float | None:
    raw = row.get(metric)
    if raw is None or raw == "":
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def adaptation_gains(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    stages = sorted({int(row["stage"]) for row in rows if row["event"] == "post_adapt"})
    for stage in stages:
        post_rows = [
            row for row in rows if int(row["stage"]) == stage and row["event"] == "post_adapt"
        ]
        current_domain = next(
            str(row["adaptation_domain"]) for row in post_rows if row.get("adaptation_domain")
        )
        pre = next(
            row
            for row in rows
            if int(row["stage"]) == stage
            and row["event"] == "pre_adapt"
            and row["holdout_dataset_id"] == current_domain
        )
        post = next(row for row in post_rows if row["holdout_dataset_id"] == current_domain)
        for metric in ADAPTATION_METRICS:
            before = _value(pre, metric)
            after = _value(post, metric)
            result.append(
                {
                    "stage": stage,
                    "domain_id": current_domain,
                    "metric": metric,
                    "pre_adapt": before,
                    "post_adapt": after,
                    "adaptation_gain": None if before is None or after is None else after - before,
                    "direction": "lower_is_better" if metric == "fpr" else "higher_is_better",
                }
            )
    return result


def final_forgetting(
    rows: Sequence[Mapping[str, Any]], sequence: Sequence[str]
) -> list[dict[str, Any]]:
    final_rows = {str(row["holdout_dataset_id"]): row for row in rows if row["event"] == "final"}
    result: list[dict[str, Any]] = []
    for learned_stage, domain in enumerate(sequence, start=1):
        learned_event = "source_initial" if learned_stage == 1 else "post_adapt"
        learned_row_index, learned = next(
            (index, row)
            for index, row in enumerate(rows)
            if str(row["holdout_dataset_id"]) == domain
            and int(row["stage"]) == learned_stage
            and row["event"] == learned_event
        )
        learned_event_index = learned.get("event_index")
        if learned_event_index is None or learned_event_index == "":
            eligible = [
                row for row in rows[learned_row_index:] if str(row["holdout_dataset_id"]) == domain
            ]
        else:
            learned_order = int(learned_event_index)
            eligible = [
                row
                for row in rows
                if str(row["holdout_dataset_id"]) == domain
                and row.get("event_index") not in {None, ""}
                and int(row["event_index"]) >= learned_order
            ]
        final = final_rows[domain]
        for metric in CONTINUAL_METRICS:
            values = [value for row in eligible if (value := _value(row, metric)) is not None]
            final_value = _value(final, metric)
            maximum = max(values) if values else None
            result.append(
                {
                    "domain_id": domain,
                    "learned_stage": learned_stage,
                    "metric": metric,
                    "maximum_since_learned": maximum,
                    "final": final_value,
                    "forgetting": None
                    if maximum is None or final_value is None
                    else maximum - final_value,
                }
            )
    return result


def backward_transfer(
    rows: Sequence[Mapping[str, Any]], sequence: Sequence[str]
) -> list[dict[str, Any]]:
    final_rows = {str(row["holdout_dataset_id"]): row for row in rows if row["event"] == "final"}
    result: list[dict[str, Any]] = []
    for learned_stage, domain in enumerate(sequence[:-1], start=1):
        learned_event = "source_initial" if learned_stage == 1 else "post_adapt"
        learned = next(
            row
            for row in rows
            if row["event"] == learned_event
            and int(row["stage"]) == learned_stage
            and row["holdout_dataset_id"] == domain
        )
        final = final_rows[domain]
        for metric in CONTINUAL_METRICS:
            learned_value = _value(learned, metric)
            final_value = _value(final, metric)
            result.append(
                {
                    "domain_id": domain,
                    "learned_stage": learned_stage,
                    "metric": metric,
                    "learned_state": learned_value,
                    "final": final_value,
                    "bwt": None
                    if learned_value is None or final_value is None
                    else final_value - learned_value,
                }
            )
    for metric in CONTINUAL_METRICS:
        values = [
            float(row["bwt"])
            for row in result
            if row["metric"] == metric and row["bwt"] is not None
        ]
        result.append(
            {
                "domain_id": "MEAN",
                "learned_stage": None,
                "metric": metric,
                "learned_state": None,
                "final": None,
                "bwt": statistics.fmean(values) if values else None,
            }
        )
    return result


def stage_seen_domain_metrics(
    rows: Sequence[Mapping[str, Any]], sequence: Sequence[str]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for stage in range(2, len(sequence) + 1):
        stage_rows = [
            row for row in rows if int(row["stage"]) == stage and row["event"] == "post_adapt"
        ]
        by_domain = {str(row["holdout_dataset_id"]): row for row in stage_rows}
        seen = sequence[:stage]
        previous = sequence[: stage - 1]
        for metric in CONTINUAL_METRICS:
            seen_values = [
                value for domain in seen if (value := _value(by_domain[domain], metric)) is not None
            ]
            previous_values = [
                value
                for domain in previous
                if (value := _value(by_domain[domain], metric)) is not None
            ]
            result.append(
                {
                    "stage": stage,
                    "metric": metric,
                    "average_seen_domain": statistics.fmean(seen_values) if seen_values else None,
                    "worst_previous_domain": min(previous_values) if previous_values else None,
                }
            )
    return result


__all__ = [
    "ADAPTATION_METRICS",
    "CONTINUAL_METRICS",
    "adaptation_gains",
    "backward_transfer",
    "final_forgetting",
    "stage_seen_domain_metrics",
]
