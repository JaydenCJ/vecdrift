# Drift metrics and the verdict policy

vecdrift never compares raw coordinates. Two models place the same text at
unrelated coordinates (different dims, arbitrary rotation), so the only
portable signal is **relative geometry**: how the anchors sit with respect
to each other. All metrics below are computed on the anchors *shared by
both sides* (matched by id), and all are deterministic — neighbor ties are
broken by ascending anchor id.

## Pairwise-structure metrics

For matched anchors, both sides yield a condensed vector of anchor-pair
cosine similarities (`n*(n-1)/2` numbers, same pair order).

| Metric | Definition | What it catches |
|---|---|---|
| Pearson correlation | linear correlation of the two similarity vectors | global geometry distortion; the classic representational-similarity signal |
| Spearman correlation | Pearson over tie-averaged ranks | monotonic-but-nonlinear warping that Pearson under-reports |
| mean \|Δ sim\| | average absolute change per anchor pair | the overall magnitude of movement |
| max \|Δ sim\| | worst single pair, reported with both ids | a localized break that averages would hide |

Correlation is reported as `n/a` when either side has zero variance (for
example an exactly orthonormal anchor set, where every pair similarity is
0). An `n/a` correlation never fails a verdict gate on its own — overlap
and delta still guard those geometries.

## Neighborhood metrics (the recall proxies)

Vector search consumes *rankings*, so these map most directly to what your
users experience. For each anchor, both sides rank all other matched
anchors by similarity; `k` defaults to 10 and is clamped to `matched - 1`.

| Metric | Definition | What it catches |
|---|---|---|
| overlap@k | fraction of the baseline top-k neighbors still in the candidate top-k, averaged over anchors | lost recall: results that used to surface and no longer do |
| min overlap@k | the single worst anchor, reported with its id | one region of the corpus breaking while the average looks fine |
| mean rank shift | mean displacement of each baseline top-k neighbor in the candidate ranking | re-ranking churn even when the top-k set survives |

The **worst anchors** list ranks matched anchors by (lowest overlap, then
largest mean |Δ sim|, then id) — these are the concrete documents to
spot-check first.

## Norm statistics

Mean/std/min/max of vector norms are reported for both sides, but only
labelled comparable when the dimensionalities match: norms across different
embedding spaces are meaningless. Within one model family, a shifted norm
distribution is an early hint of preprocessing or pooling changes.

## The verdict

Two inclusive gate tiers turn numbers into a decision:

| Gate | `OK` requires | `WARN` requires |
|---|---|---|
| mean overlap@k | ≥ 0.95 | ≥ 0.80 |
| Pearson correlation | ≥ 0.995 | ≥ 0.97 |
| mean \|Δ sim\| | ≤ 0.02 | ≤ 0.05 |

Every `OK` gate passing ⇒ **OK** (ship it). Otherwise every `WARN` gate
passing ⇒ **WARN** (spot-check top queries first). Otherwise ⇒
**RE-EMBED** (recall will visibly change; re-embed before switching).
Each non-OK verdict lists exactly which gates failed and by how much.

The defaults are tuned for retrieval corpora of ~50–500 anchors; override
any gate per corpus with the `--ok-*` / `--warn-*` flags. `--fail-on`
chooses which verdict level makes the exit code 1 in CI (default:
`re-embed`).
