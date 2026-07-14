# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-13

### Added

- Anchor-set geometry engine: condensed pairwise cosine matrix over exported
  vectors, with deterministic neighbor orderings (ties broken by anchor id)
  and id-based slicing so partially overlapping exports still compare.
- Vector export loaders for `.jsonl`/`.ndjson`, `.json` (list, `"vectors"`
  key, or id-to-vector mapping), and `.csv`, with strict validation:
  duplicate ids, ragged dimensions, NaN/Infinity components, booleans, zero
  vectors, and empty ids are all hard errors.
- Versioned baseline format (`vecdrift-baseline` v1): ids, dim, per-anchor
  norms, and the rounded condensed similarity matrix — commit the baseline,
  delete the old vectors, and compare months later (`docs/baseline-format.md`).
- Comparison engine: pairwise-similarity Pearson/Spearman correlation, mean
  and max |delta similarity| with the offending pair named, overlap@k and
  mean rank shift per anchor, worst-anchor ranking, and norm statistics when
  dimensions match.
- Three-tier verdict policy (`OK` / `WARN` / `RE-EMBED`) with inclusive,
  overridable thresholds and human-readable reasons for every failed gate;
  undefined (zero-variance) correlation never fails a gate by itself.
- `vecdrift` CLI: `snapshot`, `compare` (text or `--json`, `--fail-on
  never|warn|re-embed`, threshold flags), `inspect` (counts, norms,
  near-duplicate pairs), and `pick` (deterministic farthest-point anchor
  selection). Exit codes: 0 pass, 1 drift, 2 usage/input error.
- Deterministic example generator (`examples/generate_exports.py`): one
  corpus, a rotation-only "clean upgrade", and a drifting upgrade whose
  damaged documents vecdrift names exactly.
- 91 offline pytest tests and `scripts/smoke.sh` (prints `SMOKE OK`).

### Notes

- The repository ships no CI workflow; verification is local — `pip install -e '.[dev]' && pytest && bash scripts/smoke.sh`.

[0.1.0]: https://github.com/JaydenCJ/vecdrift/releases/tag/v0.1.0
