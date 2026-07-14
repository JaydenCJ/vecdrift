"""Loading and validating exported embedding vectors.

vecdrift never talks to a vector database or an embedding API: it consumes
files you export yourself. Three formats are accepted, chosen by extension:

* ``.jsonl`` — one JSON object per line: ``{"id": "...", "vector": [...]}``.
  Extra keys (``text``, ``tags``, …) are ignored, so a raw datastore dump
  usually works unmodified.
* ``.json``  — either a list of the same objects, an object with a
  ``"vectors"`` list, or a plain ``{"id": [floats], ...}`` mapping.
* ``.csv``   — header ``id,v0,v1,...`` (any names after ``id``), one row
  per anchor.

Validation is strict on purpose: duplicate ids, ragged dimensions, NaN/Inf
components, and zero vectors are all hard errors, because every one of them
silently corrupts a geometry comparison.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

from .errors import InputError

__all__ = ["VectorSet", "load_vectors", "write_jsonl"]


@dataclass
class VectorSet:
    """An ordered set of (id, vector) anchor pairs from a single export."""

    ids: List[str]
    vectors: List[List[float]]
    source: str = ""
    _index: Dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self._index:
            self._index = {anchor_id: i for i, anchor_id in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        """Dimensionality of the vectors (0 for an empty set)."""
        return len(self.vectors[0]) if self.vectors else 0

    def __contains__(self, anchor_id: str) -> bool:
        return anchor_id in self._index

    def vector(self, anchor_id: str) -> List[float]:
        return self.vectors[self._index[anchor_id]]

    def subset(self, keep_ids: Sequence[str]) -> "VectorSet":
        """A new VectorSet with only ``keep_ids``, in the order given."""
        ids: List[str] = []
        vectors: List[List[float]] = []
        for anchor_id in keep_ids:
            if anchor_id not in self._index:
                raise KeyError(anchor_id)
            ids.append(anchor_id)
            vectors.append(self.vectors[self._index[anchor_id]])
        return VectorSet(ids=ids, vectors=vectors, source=self.source)


def _validate_pairs(
    pairs: Iterable[Tuple[str, List[float]]], source: str
) -> VectorSet:
    ids: List[str] = []
    vectors: List[List[float]] = []
    seen: Dict[str, int] = {}
    dim = -1
    for position, (anchor_id, vector) in enumerate(pairs):
        where = f"{source}: entry {position} (id {anchor_id!r})"
        if anchor_id in seen:
            raise InputError(f"{where}: duplicate id (first seen at entry {seen[anchor_id]})")
        if not vector:
            raise InputError(f"{where}: empty vector")
        if dim == -1:
            dim = len(vector)
        elif len(vector) != dim:
            raise InputError(
                f"{where}: dimension {len(vector)} differs from first vector's {dim}"
            )
        cleaned: List[float] = []
        for j, component in enumerate(vector):
            if isinstance(component, bool) or not isinstance(component, (int, float)):
                raise InputError(f"{where}: component {j} is not a number")
            value = float(component)
            if not math.isfinite(value):
                raise InputError(f"{where}: component {j} is {component!r} (non-finite)")
            cleaned.append(value)
        if all(value == 0.0 for value in cleaned):
            raise InputError(f"{where}: zero vector (cosine geometry is undefined)")
        seen[anchor_id] = position
        ids.append(anchor_id)
        vectors.append(cleaned)
    if not ids:
        raise InputError(f"{source}: no vectors found")
    return VectorSet(ids=ids, vectors=vectors, source=source)


def _coerce_id(raw: object, where: str) -> str:
    if isinstance(raw, str):
        if raw == "":
            raise InputError(f"{where}: empty id")
        return raw
    if isinstance(raw, int) and not isinstance(raw, bool):
        return str(raw)
    raise InputError(f"{where}: id must be a string or integer, got {type(raw).__name__}")


def _pairs_from_obj(obj: object, where: str) -> Tuple[str, List[float]]:
    if not isinstance(obj, dict):
        raise InputError(f"{where}: expected a JSON object, got {type(obj).__name__}")
    if "id" not in obj:
        raise InputError(f"{where}: missing \"id\" field")
    if "vector" not in obj:
        raise InputError(f"{where}: missing \"vector\" field")
    vector = obj["vector"]
    if not isinstance(vector, list):
        raise InputError(f"{where}: \"vector\" must be a list")
    return _coerce_id(obj["id"], where), vector


def _parse_jsonl(text: str, source: str) -> VectorSet:
    pairs: List[Tuple[str, List[float]]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{source}:{lineno}"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"{where}: invalid JSON: {exc.msg}") from exc
        pairs.append(_pairs_from_obj(obj, where))
    return _validate_pairs(pairs, source)


def _parse_json(text: str, source: str) -> VectorSet:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputError(f"{source}: invalid JSON: {exc.msg}") from exc
    if isinstance(data, dict) and "vectors" in data:
        data = data["vectors"]
    pairs: List[Tuple[str, List[float]]] = []
    if isinstance(data, list):
        for i, obj in enumerate(data):
            pairs.append(_pairs_from_obj(obj, f"{source}: item {i}"))
    elif isinstance(data, dict):
        for key, vector in data.items():
            if not isinstance(vector, list):
                raise InputError(f"{source}: value for id {key!r} must be a list")
            pairs.append((_coerce_id(key, source), vector))
    else:
        raise InputError(f"{source}: expected a JSON list or object at top level")
    return _validate_pairs(pairs, source)


def _parse_csv(text: str, source: str) -> VectorSet:
    reader = csv.reader(text.splitlines())
    rows = [row for row in reader if row]
    if not rows:
        raise InputError(f"{source}: empty CSV file")
    header = rows[0]
    if not header or header[0].strip().lower() != "id":
        raise InputError(f"{source}: first CSV column must be named \"id\"")
    if len(header) < 2:
        raise InputError(f"{source}: CSV needs at least one vector column after \"id\"")
    pairs: List[Tuple[str, List[float]]] = []
    for lineno, row in enumerate(rows[1:], start=2):
        where = f"{source}:{lineno}"
        if len(row) != len(header):
            raise InputError(
                f"{where}: expected {len(header)} columns per the header, got {len(row)}"
            )
        vector: List[float] = []
        for j, cell in enumerate(row[1:]):
            try:
                vector.append(float(cell))
            except ValueError:
                raise InputError(f"{where}: column {j + 1} is not a number: {cell!r}") from None
        pairs.append((_coerce_id(row[0].strip(), where), vector))
    return _validate_pairs(pairs, source)


_PARSERS = {
    ".jsonl": _parse_jsonl,
    ".ndjson": _parse_jsonl,
    ".json": _parse_json,
    ".csv": _parse_csv,
}


def load_vectors(path: Union[str, Path]) -> VectorSet:
    """Load a vector export, dispatching on file extension."""
    path = Path(path)
    parser = _PARSERS.get(path.suffix.lower())
    if parser is None:
        supported = ", ".join(sorted(_PARSERS))
        raise InputError(f"{path}: unsupported extension (supported: {supported})")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"{path}: {exc.strerror or exc}") from exc
    return parser(text, str(path))


def write_jsonl(vector_set: VectorSet, path: Union[str, Path]) -> None:
    """Write a VectorSet back out as JSONL (used by ``vecdrift pick``)."""
    path = Path(path)
    lines = [
        json.dumps({"id": anchor_id, "vector": vector}, separators=(",", ":"))
        for anchor_id, vector in zip(vector_set.ids, vector_set.vectors)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
