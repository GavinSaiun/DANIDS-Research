"""Deterministic, label-blind Study-2 supervision schedules and delayed release."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from danids.data.manifests import SplitManifest
from danids.data.types import LearningBatch, ObservedStreamEvaluation, PartitionKind
from danids.streaming.prequential import PrequentialWindow, WindowState

SUPERVISION_SCHEDULE_VERSION = "task004-first-window-uniform-v1"


@dataclass(frozen=True, slots=True)
class SupervisionEntry:
    stage: int
    dataset_id: str
    manifest_source_sha256: str
    stream_window_index: int
    chronological_positions: tuple[int, ...]
    query_count: int
    query_window: int
    label_return_window: int


@dataclass(frozen=True, slots=True)
class SupervisionSchedule:
    version: str
    seed: int
    sequence: tuple[str, ...]
    entries: tuple[SupervisionEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "sequence": list(self.sequence),
            "entries": [
                {**asdict(entry), "chronological_positions": list(entry.chronological_positions)}
                for entry in self.entries
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def write(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json()
        if output.exists():
            if output.read_text(encoding="utf-8") != payload:
                raise FileExistsError(f"refusing to overwrite a different schedule: {output}")
            return
        output.write_text(payload, encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> SupervisionSchedule:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list):
            raise ValueError("invalid supervision schedule")
        entries = tuple(
            SupervisionEntry(
                stage=int(item["stage"]),
                dataset_id=str(item["dataset_id"]),
                manifest_source_sha256=str(item["manifest_source_sha256"]),
                stream_window_index=int(item["stream_window_index"]),
                chronological_positions=tuple(
                    int(value) for value in item["chronological_positions"]
                ),
                query_count=int(item["query_count"]),
                query_window=int(item["query_window"]),
                label_return_window=int(item["label_return_window"]),
            )
            for item in raw["entries"]
        )
        schedule = cls(
            version=str(raw["version"]),
            seed=int(raw["seed"]),
            sequence=tuple(str(value) for value in raw["sequence"]),
            entries=entries,
        )
        schedule.validate()
        return schedule

    def validate(self, manifests: tuple[SplitManifest, ...] | None = None) -> None:
        if self.version != SUPERVISION_SCHEDULE_VERSION:
            raise ValueError(f"unsupported supervision schedule version: {self.version}")
        if len(self.sequence) != 4 or len(self.entries) != 3:
            raise ValueError("Study-2 schedule requires four domains and three later entries")
        for expected_stage, entry in enumerate(self.entries, start=2):
            if (
                entry.stage != expected_stage
                or entry.dataset_id != self.sequence[expected_stage - 1]
            ):
                raise ValueError("supervision entry stage/dataset differs from sequence")
            if entry.stream_window_index != 0 or entry.query_window != 0:
                raise ValueError("Study-2 queries must come from the first window")
            if entry.label_return_window != 1:
                raise ValueError("Study-2 labels must return after the second-window prediction")
            if entry.query_count != 100 or len(entry.chronological_positions) != 100:
                raise ValueError("Study-2 schedule must select exactly 100 positions")
            if len(set(entry.chronological_positions)) != 100:
                raise ValueError("supervision positions must be distinct")
            if tuple(sorted(entry.chronological_positions)) != entry.chronological_positions:
                raise ValueError("supervision positions must be chronologically ordered")
        if manifests is not None:
            if tuple(item.dataset_id for item in manifests) != self.sequence:
                raise ValueError("schedule sequence differs from manifests")
            for entry, manifest in zip(self.entries, manifests[1:], strict=True):
                if entry.manifest_source_sha256 != manifest.source.sha256:
                    raise ValueError("schedule dataset fingerprint differs from manifest")
                if manifest.online_stream is None:
                    raise ValueError("later schedule manifest lacks online stream")
                first_stop = min(manifest.online_stream.start + 50_000, manifest.online_stream.stop)
                if any(
                    position < manifest.online_stream.start or position >= first_stop
                    for position in entry.chronological_positions
                ):
                    raise ValueError("schedule contains positions outside the first stream window")
                if any(
                    manifest.permanent_holdout.start <= position < manifest.permanent_holdout.stop
                    for position in entry.chronological_positions
                ):
                    raise ValueError("schedule contains permanent-holdout positions")


def generate_supervision_schedule(
    manifests: tuple[SplitManifest, ...], *, seed: int, query_count: int = 100
) -> SupervisionSchedule:
    """Select label-blind first-window positions using only manifest identity and seed."""

    if len(manifests) != 4 or query_count != 100:
        raise ValueError("TASK-004 requires four manifests and exactly 100 labels/domain")
    entries: list[SupervisionEntry] = []
    sequence = tuple(item.dataset_id for item in manifests)
    for stage, manifest in enumerate(manifests[1:], start=2):
        if manifest.domain_role != "later" or manifest.online_stream is None:
            raise ValueError("supervision requires later-domain online-stream manifests")
        first_start = manifest.online_stream.start
        first_stop = min(first_start + 50_000, manifest.online_stream.stop)
        if first_stop - first_start < query_count:
            raise ValueError("later domain first window contains fewer than 100 flows")
        identity = json.dumps(
            {
                "version": SUPERVISION_SCHEDULE_VERSION,
                "seed": seed,
                "stage": stage,
                "dataset_id": manifest.dataset_id,
                "source_sha256": manifest.source.sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        derived_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
        rng = np.random.default_rng(derived_seed)
        positions = tuple(
            sorted(
                int(value)
                for value in rng.choice(first_stop - first_start, query_count, replace=False)
                + first_start
            )
        )
        entries.append(
            SupervisionEntry(
                stage=stage,
                dataset_id=manifest.dataset_id,
                manifest_source_sha256=manifest.source.sha256,
                stream_window_index=0,
                chronological_positions=positions,
                query_count=query_count,
                query_window=0,
                label_return_window=1,
            )
        )
    schedule = SupervisionSchedule(
        version=SUPERVISION_SCHEDULE_VERSION,
        seed=seed,
        sequence=sequence,
        entries=tuple(entries),
    )
    schedule.validate(manifests)
    return schedule


def load_or_create_supervision_schedule(
    path: str | Path, manifests: tuple[SplitManifest, ...], *, seed: int
) -> SupervisionSchedule:
    output = Path(path)
    expected = generate_supervision_schedule(manifests, seed=seed)
    if output.exists():
        loaded = SupervisionSchedule.from_json(output)
        loaded.validate(manifests)
        if loaded != expected:
            raise ValueError("existing supervision schedule differs from resolved inputs")
        return loaded
    expected.write(output)
    return expected


class DelayedLabelQueue:
    """Keep queried labels inaccessible until the scheduled post-prediction release."""

    def __init__(self, entry: SupervisionEntry) -> None:
        self.entry = entry
        self._pending: LearningBatch | None = None
        self._released = False

    def request(self, observed: ObservedStreamEvaluation) -> None:
        if self._pending is not None:
            raise RuntimeError("labels were already requested")
        position_to_index = {
            int(position): index for index, position in enumerate(observed.row_positions)
        }
        try:
            indices = np.asarray(
                [position_to_index[position] for position in self.entry.chronological_positions],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError("scheduled query position is absent from the first window") from exc
        self._pending = LearningBatch(
            observed.features[indices],
            observed.binary_labels[indices],
            observed.native_attack_labels[indices],
            observed.metadata.iloc[indices],
            observed.row_positions[indices],
            observed.feature_columns,
            PartitionKind.ONLINE_STREAM,
        )

    def release_after_prediction(self, window: PrequentialWindow) -> LearningBatch | None:
        if window.state is not WindowState.OBSERVED:
            raise RuntimeError("labels may be released only after the return window was predicted")
        if window.window_id < self.entry.label_return_window:
            return None
        if self._pending is None:
            raise RuntimeError("no queried labels are pending")
        if self._released:
            raise RuntimeError("queried labels were already released")
        self._released = True
        return self._pending


__all__ = [
    "SUPERVISION_SCHEDULE_VERSION",
    "DelayedLabelQueue",
    "SupervisionEntry",
    "SupervisionSchedule",
    "generate_supervision_schedule",
    "load_or_create_supervision_schedule",
]
