"""Versioned baseline snapshots of anchor-set geometry.

A baseline is what you keep from the *old* model: not the raw vectors, but
the compact geometry summary — anchor ids, per-anchor norms, and the
condensed pairwise cosine matrix, all rounded to six decimals. For 256
anchors that is ~33 KB of JSON, small enough to commit next to your code,
and it means you can delete the old vectors (or the old model) entirely and
still get a drift verdict months later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

from .errors import BaselineError, InputError
from .geometry import Geometry, condensed_length
from .linalg import norm
from .vectors import VectorSet, load_vectors

__all__ = ["Baseline", "snapshot", "load_baseline", "load_reference", "FORMAT_VERSION"]

FORMAT_NAME = "vecdrift-baseline"
FORMAT_VERSION = 1
_ROUND = 6  # decimals kept for norms and similarities


@dataclass
class Baseline:
    """Geometry summary of one anchor set under one embedding model."""

    ids: List[str]
    dim: int
    norms: List[float]
    sims: List[float]  # condensed upper triangle, same order as Geometry
    label: str = ""
    source: str = field(default="", compare=False)

    def __len__(self) -> int:
        return len(self.ids)

    def geometry(self) -> Geometry:
        return Geometry(self.ids, self.sims)

    def to_dict(self) -> dict:
        return {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "label": self.label,
            "count": len(self.ids),
            "dim": self.dim,
            "ids": self.ids,
            "norms": self.norms,
            "pair_sims": self.sims,
        }

    def save(self, path: Union[str, Path]) -> None:
        payload = json.dumps(self.to_dict(), sort_keys=True, indent=2)
        Path(path).write_text(payload + "\n", encoding="utf-8")


def snapshot(vector_set: VectorSet, label: str = "") -> Baseline:
    """Build a baseline from raw vectors: norms + condensed cosine matrix."""
    geometry = Geometry.from_vectors(vector_set)
    return Baseline(
        ids=list(vector_set.ids),
        dim=vector_set.dim,
        norms=[round(norm(v), _ROUND) for v in vector_set.vectors],
        sims=[round(s, _ROUND) for s in geometry.sims],
        label=label,
        source=vector_set.source,
    )


def _from_dict(data: dict, source: str) -> Baseline:
    if data.get("format") != FORMAT_NAME:
        raise BaselineError(f"{source}: not a {FORMAT_NAME} file")
    version = data.get("version")
    if version != FORMAT_VERSION:
        raise BaselineError(
            f"{source}: baseline format version {version!r} is not supported "
            f"(this build reads version {FORMAT_VERSION})"
        )
    for key in ("ids", "norms", "pair_sims", "dim"):
        if key not in data:
            raise BaselineError(f"{source}: missing field {key!r}")
    ids = data["ids"]
    norms = data["norms"]
    sims = data["pair_sims"]
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise BaselineError(f"{source}: \"ids\" must be a list of strings")
    if len(set(ids)) != len(ids):
        raise BaselineError(f"{source}: duplicate ids in baseline")
    if not isinstance(norms, list) or len(norms) != len(ids):
        raise BaselineError(f"{source}: \"norms\" must have one entry per id")
    if not isinstance(sims, list) or len(sims) != condensed_length(len(ids)):
        raise BaselineError(
            f"{source}: \"pair_sims\" must hold {condensed_length(len(ids))} "
            f"entries for {len(ids)} ids, got {len(sims) if isinstance(sims, list) else 'non-list'}"
        )
    return Baseline(
        ids=list(ids),
        dim=int(data["dim"]),
        norms=[float(x) for x in norms],
        sims=[float(x) for x in sims],
        label=str(data.get("label", "")),
        source=source,
    )


def load_baseline(path: Union[str, Path]) -> Baseline:
    """Load and validate a saved baseline file."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BaselineError(f"{path}: {exc.strerror or exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BaselineError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise BaselineError(f"{path}: not a {FORMAT_NAME} file")
    return _from_dict(data, str(path))


def load_reference(path: Union[str, Path]) -> Baseline:
    """Load either side of a comparison from disk.

    Accepts a saved baseline *or* a raw vector export; a raw export is
    snapshotted in memory. Detection: a ``.json`` file whose top level says
    ``"format": "vecdrift-baseline"`` is a baseline, everything else goes
    through the vector loaders.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise InputError(f"{path}: {exc.strerror or exc}") from exc
        except json.JSONDecodeError as exc:
            raise InputError(f"{path}: invalid JSON: {exc.msg}") from exc
        if isinstance(data, dict) and data.get("format") == FORMAT_NAME:
            return _from_dict(data, str(path))
    return snapshot(load_vectors(path), label=path.stem)
