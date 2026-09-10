from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import danids.policy.query as query_module
from danids.adaptation.supervision import IncrementalDelayedSupervision
from danids.data.types import ObservedStreamEvaluation, PartitionKind, PredictionView
from danids.policy.query import (
    CORE_QUERY_BATCH_SIZE,
    CoreDelayedSupervision,
    HistoricalScopeClosure,
    QuerySelectionStatus,
    select_core_query,
    validate_core_query_selection,
    write_core_query_log,
)
from danids.streaming.prequential import (
    PrequentialWindow,
    WindowState,
    derive_supervision_scope_token,
)


def _view(count: int = 100, *, offset: int = 1_000) -> PredictionView:
    return PredictionView(
        np.arange(count * 2, dtype=np.float32).reshape(count, 2),
        pd.DataFrame({"timestamp": np.arange(count)}),
        np.arange(offset, offset + count, dtype=np.int64),
        ("f0", "f1"),
    )


def _window(window_id: int, *, offset: int, scope_token: str = "a" * 64) -> PrequentialWindow:
    view = _view(offset=offset)
    return PrequentialWindow(
        window_id=window_id,
        prediction_view=view,
        binary_labels=np.asarray([index % 2 for index in range(len(view))], dtype=np.int8),
        native_attack_labels=np.asarray(["attack"] * len(view), dtype=object),
        final_partial=False,
        partition_kind=PartitionKind.ONLINE_STREAM,
        supervision_scope_token=scope_token,
    )


def test_sha_query_is_exact_label_blind_and_deterministic() -> None:
    first = select_core_query(
        _view(), seed=42, opaque_scope_token="scope-opaque-01", window_id=7, query_ordinal=0
    )
    second = select_core_query(
        _view(), seed=42, opaque_scope_token="scope-opaque-01", window_id=7, query_ordinal=0
    )

    assert first == second
    assert first.status is QuerySelectionStatus.SELECTED
    assert len(first.selected_positions) == CORE_QUERY_BATCH_SIZE
    assert first.selected_positions == tuple(sorted(first.selected_positions))
    assert set(first.selected_positions).issubset(set(_view().row_positions))
    validate_core_query_selection(first, _view())


def test_query_recomputation_uses_bounded_exact_label_free_memo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_module._select_core_query_cached.cache_clear()
    calls = 0
    original = query_module._rank_digest

    def counted(**kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(query_module, "_rank_digest", counted)
    view = _view()
    selection = select_core_query(
        view, seed=42, opaque_scope_token="scope", window_id=0, query_ordinal=0
    )
    assert calls == len(view)
    validate_core_query_selection(selection, _view())
    assert calls == len(view)
    assert query_module._select_core_query_cached.cache_info().maxsize == 16


def test_query_identity_changes_ranking_without_labels() -> None:
    base = select_core_query(
        _view(), seed=42, opaque_scope_token="opaque-a", window_id=0, query_ordinal=0
    )
    later = select_core_query(
        _view(), seed=42, opaque_scope_token="opaque-a", window_id=1, query_ordinal=1
    )
    other_scope = select_core_query(
        _view(), seed=42, opaque_scope_token="opaque-b", window_id=0, query_ordinal=0
    )

    assert base.selected_positions != later.selected_positions
    assert base.selected_positions != other_scope.selected_positions


def test_query_requires_exact_25_or_fails_without_partial_selection() -> None:
    result = select_core_query(
        _view(24), seed=42, opaque_scope_token="scope", window_id=0, query_ordinal=0
    )

    assert result.status is QuerySelectionStatus.INFEASIBLE
    assert result.selected_positions == ()
    assert result.selected_positions_digest is None
    assert result.reason == "fewer_than_25_current_window_rows"


def test_query_rejects_label_bearing_evaluator_capability() -> None:
    view = _view()
    labelled = ObservedStreamEvaluation(
        view.features,
        np.zeros(len(view), dtype=np.int8),
        np.asarray(["benign"] * len(view), dtype=object),
        view.metadata,
        view.row_positions,
        view.feature_columns,
    )

    with pytest.raises(TypeError, match="label-free"):
        select_core_query(
            labelled,  # type: ignore[arg-type]
            seed=42,
            opaque_scope_token="scope",
            window_id=0,
            query_ordinal=0,
        )


def test_query_validation_rejects_changed_persisted_positions() -> None:
    result = select_core_query(
        _view(), seed=42, opaque_scope_token="scope", window_id=0, query_ordinal=0
    )
    corrupt = replace(result, seed=43)

    with pytest.raises(ValueError, match="recomputation"):
        validate_core_query_selection(corrupt, _view())


def test_query_log_is_write_once(tmp_path: Path) -> None:
    result = select_core_query(
        _view(), seed=42, opaque_scope_token="scope", window_id=0, query_ordinal=0
    )
    output = tmp_path / "query_log.json"

    write_core_query_log(output, [result])
    with pytest.raises(FileExistsError, match="overwrite"):
        write_core_query_log(output, [result])


def test_core_delayed_supervision_releases_after_next_prediction_before_evaluator() -> None:
    supervision = CoreDelayedSupervision(seed=42, opaque_scope_token="a" * 64)
    first = _window(0, offset=0)
    first.mark_predicted(np.zeros(100))
    assert supervision.release_after_prediction(first) is None
    selection = supervision.select(first.prediction_view, window_id=0)
    supervision.register_after_prediction(first, selection)

    assert first.state is WindowState.PREDICTED
    assert supervision.pending_count == 1
    second = _window(1, offset=100)
    second.mark_predicted(np.zeros(100))
    released = supervision.release_after_prediction(second)

    assert second.state is WindowState.PREDICTED
    assert released is not None
    assert tuple(int(value) for value in released.row_positions) == selection.selected_positions
    assert supervision.pending_count == 0
    assert supervision.available_count == 25


def test_fresh_counterfactual_gates_share_only_immutable_payload() -> None:
    template = _window(3, offset=300)
    first = template.fresh_gate()
    second = template.fresh_gate()

    assert first is not second
    assert first.prediction_view is second.prediction_view
    first.mark_predicted(np.zeros(len(first.prediction_view)))
    first_observed = first.observe()
    assert second.state is WindowState.AWAITING_PREDICTION
    assert template.state is WindowState.AWAITING_PREDICTION
    with pytest.raises(ValueError):
        first_observed.binary_labels[0] = 1


def test_final_query_releases_across_scope_boundary_then_issues_closure() -> None:
    old_scope = "a" * 64
    supervision = CoreDelayedSupervision(seed=42, opaque_scope_token=old_scope)
    final = _window(0, offset=0, scope_token=old_scope)
    final.mark_predicted(np.zeros(100))
    supervision.release_after_prediction(final, prediction_index=11)
    selection = supervision.select(final.prediction_view, window_id=0)
    provenance = supervision.register_after_prediction(
        final,
        selection,
        prediction_index=11,
    )
    assert provenance is not None
    assert provenance.query_window == 11
    assert provenance.release_window == 12

    boundary = _window(0, offset=100, scope_token="b" * 64)
    boundary.mark_predicted(np.zeros(100))
    with pytest.raises(RuntimeError, match="pending cross-boundary"):
        supervision.close_at_administrative_boundary(
            boundary,
            activation_boundary_index=12,
        )

    released = supervision.release_after_prediction(boundary, prediction_index=12)
    assert released is not None
    assert tuple(int(value) for value in released.row_positions) == selection.selected_positions
    closure = supervision.close_at_administrative_boundary(
        boundary,
        activation_boundary_index=12,
    )
    assert closure.opaque_scope_token == old_scope
    assert closure.final_scope_window_id == 0
    assert closure.activation_boundary_index == 12
    assert closure.successful_query_count == 1
    assert closure.released_label_count == 25
    assert closure.released_row_positions_digest == selection.selected_positions_digest
    assert supervision.is_closed
    with pytest.raises(RuntimeError, match="already closed"):
        supervision.select(final.prediction_view, window_id=1)


def test_historical_scope_closure_cannot_be_constructed_by_policy_callers() -> None:
    with pytest.raises(TypeError, match="issued only"):
        HistoricalScopeClosure(
            opaque_scope_token="a" * 64,
            final_scope_window_id=0,
            activation_boundary_index=1,
            successful_query_count=0,
            released_label_count=0,
            released_row_positions_digest=None,
            supervision_state_digest="b" * 64,
            _capability_token=object(),
        )


def test_core_delayed_supervision_enforces_one_pending_and_four_queries() -> None:
    supervision = CoreDelayedSupervision(seed=7, opaque_scope_token="a" * 64)
    for window_id in range(4):
        window = _window(window_id, offset=window_id * 100)
        window.mark_predicted(np.zeros(100))
        supervision.release_after_prediction(window)
        selection = supervision.select(window.prediction_view, window_id=window_id)
        supervision.register_after_prediction(window, selection)
        if window_id == 0:
            extra = select_core_query(
                window.prediction_view,
                seed=7,
                opaque_scope_token="a" * 64,
                window_id=0,
                query_ordinal=1,
            )
            with pytest.raises(RuntimeError, match="pending"):
                supervision.register_after_prediction(window, extra)

    release_window = _window(4, offset=400)
    release_window.mark_predicted(np.zeros(100))
    supervision.release_after_prediction(release_window)
    assert supervision.query_count == 4
    assert supervision.used_budget == 100
    assert supervision.remaining_budget == 0
    with pytest.raises(ValueError, match="ordinal"):
        supervision.select(release_window.prediction_view, window_id=4)
    closure = supervision.close_at_administrative_boundary(
        release_window, activation_boundary_index=4
    )
    assert closure.released_label_count == 100
    release_window.observe()


def test_supervision_scope_token_is_deterministic_and_binds_windows() -> None:
    token = derive_supervision_scope_token(
        source_sha256="1" * 64,
        split_version="task001-v1",
        row_start=0,
        row_stop=80,
    )
    assert token == derive_supervision_scope_token(
        source_sha256="1" * 64,
        split_version="task001-v1",
        row_start=0,
        row_stop=80,
    )
    supervision = CoreDelayedSupervision(seed=42, opaque_scope_token=token)
    wrong_stream = _window(0, offset=0, scope_token="b" * 64)
    wrong_stream.mark_predicted(np.zeros(100))

    with pytest.raises(ValueError, match="different opaque"):
        supervision.release_after_prediction(wrong_stream)


def test_core_release_rejects_already_observed_window() -> None:
    supervision = CoreDelayedSupervision(seed=42, opaque_scope_token="a" * 64)
    window = _window(0, offset=0)
    window.mark_predicted(np.zeros(100))
    window.observe()

    with pytest.raises(RuntimeError, match="before evaluator"):
        supervision.release_after_prediction(window)


def test_holdout_provenance_cannot_be_laundered_into_delayed_learning() -> None:
    view = _view()
    holdout_window = PrequentialWindow(
        window_id=0,
        prediction_view=view,
        binary_labels=np.zeros(len(view), dtype=np.int8),
        native_attack_labels=np.asarray(["benign"] * len(view), dtype=object),
        final_partial=False,
        partition_kind=PartitionKind.PERMANENT_HOLDOUT,
        supervision_scope_token=None,
    )
    holdout_window.mark_predicted(np.zeros(len(view)))
    queue = IncrementalDelayedSupervision("opaque")

    with pytest.raises(TypeError, match="proven online-stream"):
        queue.request_from_predicted_window(holdout_window, list(range(1_000, 1_025)))


def test_prequential_window_rejects_misaligned_hidden_labels() -> None:
    view = _view()
    with pytest.raises(ValueError, match="misaligned"):
        PrequentialWindow(
            window_id=0,
            prediction_view=view,
            binary_labels=np.zeros(len(view) - 1, dtype=np.int8),
            native_attack_labels=np.asarray(["benign"] * len(view), dtype=object),
            final_partial=False,
            partition_kind=PartitionKind.ONLINE_STREAM,
            supervision_scope_token="a" * 64,
        )
