from __future__ import annotations

import numpy as np
import pytest

from danids.config.experiment import ExperimentConfig
from danids.data.loading import InitialDomainPartitions, LaterDomainPartitions, load_partitions
from danids.data.manifests import generate_split_manifest
from danids.data.preprocessing import NumericPreprocessor
from danids.data.registry import DatasetRegistry
from danids.data.schema import FeatureContract
from danids.streaming.prequential import ProtocolOrderError, WindowState


def _loaded(
    registry: DatasetRegistry,
    contract: FeatureContract,
    experiment: ExperimentConfig,
    dataset_id: str,
    role: str,
) -> InitialDomainPartitions | LaterDomainPartitions:
    manifest = generate_split_manifest(
        registry[dataset_id],
        contract,
        role=role,  # type: ignore[arg-type]
        split_version=experiment.split_version,
        seed=experiment.seed,
        splits=experiment.splits,
    )
    return load_partitions(registry[dataset_id], contract, manifest, window_size=6)


def test_preprocessor_fits_only_initial_history_and_not_future(
    registry: DatasetRegistry, contract: FeatureContract, experiment: ExperimentConfig
) -> None:
    partitions = _loaded(registry, contract, experiment, "B", "initial")
    assert isinstance(partitions, InitialDomainPartitions)
    processor = NumericPreprocessor().fit(partitions.train)
    # F1 is chronological 0..14 in train: median 7. Full-data median would be 12.
    assert processor.medians[0] == pytest.approx(7.0)
    before = processor.medians
    assert np.isfinite(processor.transform(partitions.train)).all()
    transformed = processor.transform(partitions.holdout)
    assert transformed.dtype == np.float32
    np.testing.assert_array_equal(processor.medians, before)
    with pytest.raises(TypeError, match="LearningBatch"):
        processor.fit(partitions.holdout)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="LearningBatch"):
        processor.fit(partitions.validation)  # type: ignore[arg-type]


def test_observe_is_impossible_before_prediction(
    registry: DatasetRegistry, contract: FeatureContract, experiment: ExperimentConfig
) -> None:
    partitions = _loaded(registry, contract, experiment, "U", "later")
    assert isinstance(partitions, LaterDomainPartitions)
    window = next(iter(partitions.stream))
    assert not hasattr(window.prediction_view, "binary_labels")
    with pytest.raises(ProtocolOrderError, match="before"):
        window.observe()
    window.mark_predicted(np.zeros(len(window.prediction_view)))
    observed = window.observe()
    assert window.state is WindowState.OBSERVED
    assert len(observed.binary_labels) == 6
    with pytest.raises(ProtocolOrderError, match="already observed"):
        window.observe()


def test_cursor_cannot_advance_past_unpredicted_window(
    registry: DatasetRegistry, contract: FeatureContract, experiment: ExperimentConfig
) -> None:
    partitions = _loaded(registry, contract, experiment, "U", "later")
    assert isinstance(partitions, LaterDomainPartitions)
    cursor = iter(partitions.stream)
    next(cursor)
    with pytest.raises(ProtocolOrderError, match="previous window"):
        next(cursor)


def test_windows_are_deterministic_chronological_and_keep_partial_final(
    registry: DatasetRegistry, contract: FeatureContract, experiment: ExperimentConfig
) -> None:
    partitions = _loaded(registry, contract, experiment, "C", "later")
    assert isinstance(partitions, LaterDomainPartitions)

    def consume_positions() -> tuple[list[list[int]], list[int], list[bool]]:
        positions: list[list[int]] = []
        sizes: list[int] = []
        partial: list[bool] = []
        for window in partitions.stream:
            positions.append(window.prediction_view.row_positions.tolist())
            sizes.append(len(window.prediction_view))
            partial.append(window.is_final_partial)
            window.mark_predicted()
            window.observe()
        return positions, sizes, partial

    first = consume_positions()
    second = consume_positions()
    assert first == second
    assert first[1] == [6, 6, 6, 2]
    assert first[2] == [False, False, False, True]
    assert [value for group in first[0] for value in group] == list(range(20))
