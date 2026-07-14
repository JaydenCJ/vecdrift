# Baseline file format (`vecdrift-baseline`, version 1)

A baseline is a compact, committable summary of one anchor set's geometry
under one embedding model. It deliberately stores **no vectors**: once the
baseline is written you can delete the old export (or retire the old model)
and still get a drift verdict later.

## Layout

A single JSON object, written with sorted keys, two-space indentation, and a
trailing newline (so re-saving is byte-identical and diffs stay clean):

```json
{
  "format": "vecdrift-baseline",
  "version": 1,
  "label": "model-v1",
  "count": 48,
  "dim": 8,
  "ids": ["doc-00", "doc-01", "..."],
  "norms": [1.234567, 0.987654, "..."],
  "pair_sims": [0.912345, -0.043210, "..."]
}
```

| Field | Type | Meaning |
|---|---|---|
| `format` | string | Always `"vecdrift-baseline"`; how `compare` tells a baseline from a raw `.json` vector export |
| `version` | int | Format version. This build reads exactly `1` and refuses anything else |
| `label` | string | Free-form human label (`--label`); shown in report headers |
| `count` | int | Number of anchors; redundant with `len(ids)`, kept for human inspection |
| `dim` | int | Vector dimensionality of the snapshotted export |
| `ids` | string[] | Anchor ids, in export order. Must be unique |
| `norms` | number[] | L2 norm per anchor, aligned with `ids`, rounded to 6 decimals |
| `pair_sims` | number[] | Condensed upper triangle of the pairwise cosine matrix, rounded to 6 decimals |

## The condensed matrix

`pair_sims` stores cosine similarities for every unordered pair `(i, j)`
with `i < j`, iterated row by row: `(0,1), (0,2), …, (0,n-1), (1,2), …`.
For `n` anchors that is `n*(n-1)/2` entries; the flat index of `(i, j)` is
`i*n - i*(i+1)//2 + (j - i - 1)`. For 256 anchors this is 32,640 numbers,
roughly 33 KB of JSON — small enough to commit next to your retrieval code.

## Validation on load

`load_baseline` rejects, with a clear error message: a wrong `format`
marker, an unsupported `version`, missing fields, duplicate `ids`, a
`norms` array whose length differs from `ids`, and a `pair_sims` array
whose length is not exactly `n*(n-1)/2`.

## Rounding and precision

Norms and similarities are rounded to six decimal places on snapshot. Drift
worth acting on moves cosine similarities by 1e-2 or more; 1e-6 rounding
keeps files small and diffable while staying four orders of magnitude below
the signal. Comparisons therefore treat differences at the 1e-6 level as
noise, and the default `ok` thresholds are far above it.

## Compatibility promise

Any change to the meaning of an existing field bumps `version`, and the
loader keeps refusing versions it does not understand rather than guessing.
