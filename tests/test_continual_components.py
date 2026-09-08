"""Focused scientific-invariant tests for TASK-004 components."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from danids.config.continual import (
    AdaptationConfig,
    ContinualMemoryConfig,
    SupervisionConfig,
    load_continual_experiment_config,
)
from danids.continual.adaptation import adapt_er, adapt_ft_mem, adapt_naive_ft
from danids.continual.ewc import EWCState, estimate_diagonal_fisher, ewc_penalty
from danids.continual.memory import (
    ExemplarMemory,
    embedding_herding_exemplars,
    replay_epoch_batches,
    uniform_exemplars,
)
from danids.continual.metrics import (
    adaptation_gains,
    backward_transfer,
    final_forgetting,
    stage_seen_domain_metrics,
)
from danids.continual.supervision import (
    DelayedLabelQueue,
    SupervisionSchedule,
    generate_supervision_schedule,
    load_or_create_supervision_schedule,
)
from danids.data.manifests import IndexRange, SourceFingerprint, SplitManifest
from danids.data.preprocessing import NumericPreprocessor
from danids.data.types import LearningBatch, PartitionKind, PredictionView
from danids.models.mlp import StaticMLP, model_state_digest
from danids.streaming.prequential import PrequentialWindow

FEATURES = tuple(f"F{index:02d}" for index in range(47))
SEQUENCE = ("U", "T", "C", "B")


def _batch(
    size: int,
    *,
    kind: PartitionKind = PartitionKind.INITIAL_TRAIN,
    offset: int = 0,
    seed: int = 1,
) -> LearningBatch:
    rng = np.random.default_rng(seed)
    labels = (np.arange(size) % 2).astype(np.int8)
    return LearningBatch(
        rng.normal(size=(size, 47)).astype(np.float32),
        labels,
        np.where(labels == 1, "Attack", "Benign").astype(object),
        pd.DataFrame({"Timestamp": np.arange(offset, offset + size)}),
        np.arange(offset, offset + size, dtype=np.int64),
        FEATURES,
        kind,
    )


def _manifest(domain: str, role: str, *, fingerprint: str | None = None) -> SplitManifest:
    row_count = 125_000
    initial = role == "initial"
    return SplitManifest(
        dataset_id=domain,
        dataset_name=domain,
        domain_role="initial" if initial else "later",
        split_version="chronological-v1",
        generator_version="task001-manifest-v1",
        generation_seed=42,
        source=SourceFingerprint("fake.csv", 1, 1, fingerprint or domain * 64),
        row_count=row_count,
        timestamp_column="Timestamp",
        chronological_start="0",
        chronological_end=str(row_count - 1),
        source_was_chronologically_sorted=True,
        feature_contract_version="core-common-primary-v2-no-ports",
        feature_columns=FEATURES,
        metadata_columns=("Timestamp",),
        excluded_columns=("Label", "Attack"),
        binary_label_column="Label",
        native_attack_column="Attack",
        initial_train=IndexRange(0, 75_000) if initial else None,
        validation=IndexRange(75_000, 100_000) if initial else None,
        online_stream=None if initial else IndexRange(0, 100_000),
        permanent_holdout=IndexRange(100_000, 125_000),
    )


def _manifests() -> tuple[SplitManifest, ...]:
    return tuple(
        _manifest(domain, "initial" if index == 0 else "later")
        for index, domain in enumerate(SEQUENCE)
    )


def _window(batch: LearningBatch, window_id: int) -> PrequentialWindow:
    return PrequentialWindow(
        window_id=window_id,
        prediction_view=PredictionView(
            batch.features, batch.metadata, batch.row_positions, batch.feature_columns
        ),
        binary_labels=batch.binary_labels,
        native_attack_labels=batch.native_attack_labels,
        final_partial=False,
    )


def _model_and_preprocessor() -> tuple[StaticMLP, NumericPreprocessor]:
    source = _batch(20)
    preprocessor = NumericPreprocessor().fit(source)
    torch.manual_seed(4)
    return StaticMLP(47, dropout=0.0), preprocessor


def test_all_task004_configs_load_with_frozen_defaults() -> None:
    for method in ("naive_ft", "ewc", "er", "ft_mem"):
        config = load_continual_experiment_config(
            Path("configs/experiments") / f"task004_{method}_u-t-c-b.yaml"
        )
        assert config.experiment.study == "E2"
        assert config.experiment.sequence == SEQUENCE
        assert config.experiment.window_size == 50_000
        assert config.experiment.boundary_mode == "boundary_aware_control"
        assert config.supervision == SupervisionConfig()
        assert config.memory == ContinualMemoryConfig()
        assert config.adaptation.method == method
        assert config.adaptation.learning_rate == 1e-4
        assert config.adaptation.epochs == 20


def test_runtime_seed_override_updates_method_id() -> None:
    config = load_continual_experiment_config(
        "configs/experiments/task004_ewc_u-t-c-b.yaml"
    ).with_runtime_overrides(seed=44, adaptation_epochs=2)
    assert config.experiment.experiment_id == "E2_EWC_U-T-C-B_B100_D1_s44"
    assert config.experiment.seed == 44
    assert config.adaptation.epochs == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [("label_budget_per_later_domain", 99), ("label_delay_windows", 0), ("schedule", "random")],
)
def test_supervision_config_rejects_nonfrozen_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="TASK-004"):
        replace(SupervisionConfig(), **{field: value}).validate()


def test_adaptation_config_rejects_changed_optimizer_contract() -> None:
    with pytest.raises(ValueError, match="AdamW"):
        replace(AdaptationConfig("naive_ft"), learning_rate=1e-3).validate()


def test_schedule_is_deterministic_distinct_and_label_blind() -> None:
    manifests = _manifests()
    first = generate_supervision_schedule(manifests, seed=42)
    second = generate_supervision_schedule(manifests, seed=42)
    assert first == second
    assert len(first.entries) == 3
    for entry in first.entries:
        assert len(entry.chronological_positions) == 100
        assert len(set(entry.chronological_positions)) == 100
        assert entry.chronological_positions == tuple(sorted(entry.chronological_positions))
        assert min(entry.chronological_positions) >= 0
        assert max(entry.chronological_positions) < 50_000
        assert entry.label_return_window == 1


def test_schedule_changes_with_seed() -> None:
    assert generate_supervision_schedule(_manifests(), seed=42) != generate_supervision_schedule(
        _manifests(), seed=43
    )


def test_schedule_rejects_manifest_fingerprint_change() -> None:
    schedule = generate_supervision_schedule(_manifests(), seed=42)
    changed = list(_manifests())
    changed[1] = _manifest("T", "later", fingerprint="x" * 64)
    with pytest.raises(ValueError, match="fingerprint"):
        schedule.validate(tuple(changed))


def test_schedule_round_trip_is_byte_stable(tmp_path: Path) -> None:
    path = tmp_path / "schedule.json"
    first = load_or_create_supervision_schedule(path, _manifests(), seed=42)
    payload = path.read_bytes()
    second = load_or_create_supervision_schedule(path, _manifests(), seed=42)
    assert first == second == SupervisionSchedule.from_json(path)
    assert path.read_bytes() == payload


def test_schedule_refuses_overwrite_for_different_seed(tmp_path: Path) -> None:
    path = tmp_path / "schedule.json"
    load_or_create_supervision_schedule(path, _manifests(), seed=42)
    with pytest.raises(ValueError, match="differs"):
        load_or_create_supervision_schedule(path, _manifests(), seed=43)


def test_delayed_labels_cannot_be_released_before_return_prediction() -> None:
    schedule = generate_supervision_schedule(_manifests(), seed=42)
    entry = schedule.entries[0]
    first = _batch(50_000, kind=PartitionKind.ONLINE_STREAM)
    first_window = _window(first, 0)
    first_window.mark_predicted([0.5] * len(first))
    observed = first_window.observe()
    queue = DelayedLabelQueue(entry)
    queue.request(observed)
    second_window = _window(_batch(2, kind=PartitionKind.ONLINE_STREAM, offset=50_000), 1)
    with pytest.raises(RuntimeError, match="only after"):
        queue.release_after_prediction(second_window)
    second_window.mark_predicted([0.5, 0.5])
    second_window.observe()
    released = queue.release_after_prediction(second_window)
    assert released is not None
    assert len(released) == 100
    assert released.partition_kind is PartitionKind.ONLINE_STREAM


def test_delayed_labels_release_exactly_once() -> None:
    entry = generate_supervision_schedule(_manifests(), seed=42).entries[0]
    first = _window(_batch(50_000, kind=PartitionKind.ONLINE_STREAM), 0)
    first.mark_predicted()
    queue = DelayedLabelQueue(entry)
    queue.request(first.observe())
    second = _window(_batch(1, kind=PartitionKind.ONLINE_STREAM, offset=50_000), 1)
    second.mark_predicted()
    second.observe()
    assert queue.release_after_prediction(second) is not None
    with pytest.raises(RuntimeError, match="already released"):
        queue.release_after_prediction(second)


def test_uniform_memory_is_deterministic_and_capped() -> None:
    batch = _batch(20)
    first = uniform_exemplars(batch, domain_id="U", capacity=7, seed=4)
    second = uniform_exemplars(batch, domain_id="U", capacity=7, seed=4)
    assert first.size == 7
    assert first.batch.row_positions.tolist() == second.batch.row_positions.tolist()
    memory = ExemplarMemory(7)
    memory.add(first)
    assert memory.total_size == 7
    assert memory.manifest()["audit_memory_size"] == 0


def test_memory_rejects_duplicate_domain_and_over_capacity() -> None:
    batch = _batch(10)
    memory = ExemplarMemory(5)
    with pytest.raises(ValueError, match="exceeds"):
        memory.add(uniform_exemplars(batch, domain_id="U", capacity=6, seed=1))
    memory.add(uniform_exemplars(batch, domain_id="U", capacity=5, seed=1))
    with pytest.raises(ValueError, match="already contains"):
        memory.add(uniform_exemplars(batch, domain_id="U", capacity=5, seed=2))


def test_embedding_herding_is_deterministic_and_stratified() -> None:
    model, preprocessor = _model_and_preprocessor()
    batch = _batch(20)
    first = embedding_herding_exemplars(
        batch, model, preprocessor, domain_id="T", capacity=6, device=torch.device("cpu")
    )
    second = embedding_herding_exemplars(
        batch, model, preprocessor, domain_id="T", capacity=6, device=torch.device("cpu")
    )
    assert first.batch.row_positions.tolist() == second.batch.row_positions.tolist()
    assert np.bincount(first.batch.binary_labels, minlength=2).tolist() == [3, 3]
    assert first.selection == "embedding_mean_herding"


def test_replay_batches_mix_target_and_historical_examples() -> None:
    target = _batch(10, kind=PartitionKind.ONLINE_STREAM, offset=100)
    replay = _batch(20, offset=0)
    batches = replay_epoch_batches(target, replay, batch_size=8, seed=9)
    assert all(np.any(batch.row_positions >= 100) for batch in batches)
    assert all(np.any(batch.row_positions < 100) for batch in batches)
    assert all(len(batch) <= 8 for batch in batches)


def test_fisher_is_per_example_nonnegative_and_records_positions() -> None:
    model, preprocessor = _model_and_preprocessor()
    batch = _batch(4)
    consolidation = estimate_diagonal_fisher(
        model, preprocessor, batch, domain_id="U", device=torch.device("cpu")
    )
    assert consolidation.row_positions == (0, 1, 2, 3)
    assert all(torch.all(value >= 0) for value in consolidation.fisher.values())
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_ewc_penalty_is_zero_at_optimum_and_positive_after_change() -> None:
    model, preprocessor = _model_and_preprocessor()
    state = EWCState()
    state.add(
        estimate_diagonal_fisher(
            model, preprocessor, _batch(4), domain_id="U", device=torch.device("cpu")
        )
    )
    assert float(ewc_penalty(model, state, coefficient=100.0)) == pytest.approx(0.0)
    with torch.no_grad():
        next(model.parameters()).add_(0.1)
    assert float(ewc_penalty(model, state, coefficient=100.0)) > 0.0


@pytest.mark.parametrize("method", ["naive_ft", "er", "ft_mem"])
def test_adaptation_methods_update_only_model(method: str) -> None:
    model, preprocessor = _model_and_preprocessor()
    target = _batch(10, kind=PartitionKind.ONLINE_STREAM, offset=100)
    replay = _batch(8)
    preprocessor_digest = preprocessor.state_digest()
    before = model_state_digest(model)
    config = AdaptationConfig(method=method, epochs=1)  # type: ignore[arg-type]
    if method == "naive_ft":
        result = adapt_naive_ft(
            model, preprocessor, target, config, device=torch.device("cpu"), seed=5
        )
    elif method == "er":
        result = adapt_er(
            model, preprocessor, target, replay, config, device=torch.device("cpu"), seed=5
        )
    else:
        result = adapt_ft_mem(
            model, preprocessor, target, replay, config, device=torch.device("cpu"), seed=5
        )
    assert result.model_digest_before == before
    assert result.model_digest_after != before
    assert preprocessor.state_digest() == preprocessor_digest
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert result.replay_rows_available == (0 if method == "naive_ft" else 8)


def _metric_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    values = {
        "source_initial": (1, {"U": 0.80}),
        "pre_adapt": (2, {"U": 0.78, "T": 0.40}),
        "post_adapt": (2, {"U": 0.75, "T": 0.60}),
        "pre_adapt_3": (3, {"U": 0.73, "T": 0.58, "C": 0.30}),
        "post_adapt_3": (3, {"U": 0.70, "T": 0.55, "C": 0.65}),
        "pre_adapt_4": (4, {"U": 0.69, "T": 0.54, "C": 0.62, "B": 0.20}),
        "post_adapt_4": (4, {"U": 0.68, "T": 0.52, "C": 0.60, "B": 0.70}),
        "final": (4, {"U": 0.68, "T": 0.52, "C": 0.60, "B": 0.70}),
    }
    for key, (stage, domains) in values.items():
        event = key.split("_3")[0].split("_4")[0]
        adaptation_domain = SEQUENCE[stage - 1] if event in {"pre_adapt", "post_adapt"} else ""
        for domain, value in domains.items():
            rows.append(
                {
                    "stage": stage,
                    "event": event,
                    "adaptation_domain": adaptation_domain,
                    "holdout_dataset_id": domain,
                    "pr_auc": value,
                    "roc_auc": value,
                    "tpr": value,
                }
            )
    return rows


def test_adaptation_gain_is_post_minus_pre() -> None:
    gains = adaptation_gains(_metric_rows())
    row = next(item for item in gains if item["stage"] == 2 and item["metric"] == "pr_auc")
    assert row["adaptation_gain"] == pytest.approx(0.20)


def test_adaptation_fpr_delta_is_marked_lower_is_better() -> None:
    rows = _metric_rows()
    for row in rows:
        row["fpr"] = 0.02 if row["event"] == "pre_adapt" else 0.01
    gains = adaptation_gains(rows)
    fpr = next(item for item in gains if item["stage"] == 2 and item["metric"] == "fpr")
    assert fpr["adaptation_gain"] == pytest.approx(-0.01)
    assert fpr["direction"] == "lower_is_better"


def test_final_forgetting_is_peak_since_learning_minus_final() -> None:
    rows = final_forgetting(_metric_rows(), SEQUENCE)
    source = next(item for item in rows if item["domain_id"] == "U" and item["metric"] == "pr_auc")
    assert source["forgetting"] == pytest.approx(0.12)


def test_backward_transfer_uses_learned_state_and_reports_mean() -> None:
    rows = backward_transfer(_metric_rows(), SEQUENCE)
    source = next(item for item in rows if item["domain_id"] == "U" and item["metric"] == "pr_auc")
    mean = next(item for item in rows if item["domain_id"] == "MEAN" and item["metric"] == "pr_auc")
    assert source["bwt"] == pytest.approx(-0.12)
    assert mean["bwt"] == pytest.approx((-0.12 - 0.08 - 0.05) / 3)


def test_stage_metrics_use_seen_and_previous_domains_only() -> None:
    rows = stage_seen_domain_metrics(_metric_rows(), SEQUENCE)
    stage2 = next(item for item in rows if item["stage"] == 2 and item["metric"] == "pr_auc")
    assert stage2["average_seen_domain"] == pytest.approx(0.675)
    assert stage2["worst_previous_domain"] == pytest.approx(0.75)
