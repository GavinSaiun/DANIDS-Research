"""Content-addressed write-once cache for model-independent distribution signals."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

SIGNAL_CACHE_VERSION = "task005-distribution-signals-v1"
REFERENCE_CACHE_VERSION = "task005-source-reference-v1"


class DistributionSignalCache:
    """Reuse only values with an identical explicit semantic identity."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @staticmethod
    def _key(identity: Mapping[str, Any]) -> str:
        payload = {
            "version": SIGNAL_CACHE_VERSION,
            "identity": dict(identity),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def load(self, identity: Mapping[str, Any]) -> dict[str, Any] | None:
        key = self._key(identity)
        path = self.root / f"{key}.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("distribution cache payload must be an object")
        if value.get("version") != SIGNAL_CACHE_VERSION or value.get("identity") != dict(identity):
            raise ValueError("distribution cache identity differs from requested signals")
        result = value.get("result")
        if not isinstance(result, dict):
            raise ValueError("distribution cache result must be an object")
        return result

    def store(self, identity: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        key = self._key(identity)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{key}.json"
        payload = (
            json.dumps(
                {
                    "version": SIGNAL_CACHE_VERSION,
                    "identity": dict(identity),
                    "result": dict(result),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise FileExistsError("refusing to overwrite different cached health signals")
            return
        path.write_text(payload, encoding="utf-8")


class SourceReferenceCache:
    """Content-addressed cache for bounded source arrays and validation predictions."""

    _ARRAYS = (
        "training_positions",
        "transformed_features",
        "embeddings",
        "validation_labels",
        "validation_scores",
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @staticmethod
    def _identity_json(identity: Mapping[str, Any]) -> str:
        return json.dumps(dict(identity), sort_keys=True, separators=(",", ":"))

    @classmethod
    def _key(cls, identity: Mapping[str, Any]) -> str:
        payload = f"{REFERENCE_CACHE_VERSION}|{cls._identity_json(identity)}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def load(self, identity: Mapping[str, Any]) -> dict[str, NDArray[Any]] | None:
        path = self.root / f"{self._key(identity)}.npz"
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["version"].item()) != REFERENCE_CACHE_VERSION:
                raise ValueError("source-reference cache version differs")
            if str(cached["identity_json"].item()) != self._identity_json(identity):
                raise ValueError("source-reference cache identity differs")
            if set(cached.files) != {"version", "identity_json", *self._ARRAYS}:
                raise ValueError("source-reference cache array schema differs")
            result = {name: np.asarray(cached[name]).copy() for name in self._ARRAYS}
        positions = result["training_positions"]
        transformed = result["transformed_features"]
        embeddings = result["embeddings"]
        labels = result["validation_labels"]
        scores = result["validation_scores"]
        if positions.ndim != 1 or transformed.ndim != 2 or embeddings.ndim != 2:
            raise ValueError("invalid source-reference cache array dimensions")
        if transformed.shape[0] != len(positions) or embeddings.shape != (len(positions), 64):
            raise ValueError("invalid source-reference cache sample alignment")
        if labels.ndim != 1 or scores.shape != labels.shape or len(labels) == 0:
            raise ValueError("invalid source-reference cache validation alignment")
        if not np.isfinite(transformed).all() or not np.isfinite(embeddings).all():
            raise ValueError("source-reference cache contains non-finite features")
        if not np.isfinite(scores).all() or not np.isin(labels, (0, 1)).all():
            raise ValueError("source-reference cache contains invalid validation values")
        return result

    def store(
        self,
        identity: Mapping[str, Any],
        arrays: Mapping[str, NDArray[Any]],
    ) -> None:
        if set(arrays) != set(self._ARRAYS):
            raise ValueError("source-reference cache payload has the wrong arrays")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{self._key(identity)}.npz"
        if path.exists():
            existing = self.load(identity)
            assert existing is not None
            if any(not np.array_equal(existing[name], arrays[name]) for name in self._ARRAYS):
                raise FileExistsError("refusing to overwrite a different source-reference cache")
            return
        with path.open("xb") as handle:
            np.savez_compressed(
                handle,
                version=np.asarray(REFERENCE_CACHE_VERSION),
                identity_json=np.asarray(self._identity_json(identity)),
                training_positions=arrays["training_positions"],
                transformed_features=arrays["transformed_features"],
                embeddings=arrays["embeddings"],
                validation_labels=arrays["validation_labels"],
                validation_scores=arrays["validation_scores"],
            )


__all__ = [
    "REFERENCE_CACHE_VERSION",
    "SIGNAL_CACHE_VERSION",
    "DistributionSignalCache",
    "SourceReferenceCache",
]
