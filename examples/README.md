# vecdrift examples

`generate_exports.py` builds three deterministic, synthetic embedding
exports of the same 48-document corpus — no model, no network, seeded RNG:

| File | Simulates | Expected verdict |
|---|---|---|
| `model_v1.jsonl` | the current embedding model (dim 8, 4 topic clusters) | — (the baseline) |
| `model_v2.jsonl` | a well-behaved upgrade: exact rotation + uniform scale + tiny noise | `OK` |
| `model_v3.jsonl` | a drifting upgrade to dim 12: 10 documents lose their cluster identity | `RE-EMBED` |

Run the whole story:

```bash
python3 examples/generate_exports.py /tmp/vecdrift-demo
cd /tmp/vecdrift-demo

vecdrift inspect model_v1.jsonl
vecdrift snapshot model_v1.jsonl -o baseline.json --label model-v1

vecdrift compare baseline.json model_v2.jsonl   # verdict: OK, exit 0
vecdrift compare baseline.json model_v3.jsonl   # verdict: RE-EMBED, exit 1
```

The drifted documents in `model_v3.jsonl` are exactly
`doc-03, doc-07, doc-11, doc-15, doc-19, doc-23, doc-27, doc-31, doc-35,
doc-39` — check the "worst anchors" section of the report and you will see
vecdrift naming documents from precisely that set.

Why v2 passes even though every raw coordinate changed: cosine geometry is
invariant under orthogonal maps and uniform scaling, and vecdrift compares
*relative* anchor-pair structure, never raw coordinates. That is also why
the dim-8 baseline can be compared against the dim-12 `model_v3.jsonl` at
all — and why the norm statistics section only appears when both sides share
a dimensionality.

`scripts/smoke.sh` runs this exact scenario end-to-end and asserts on the
verdicts and exit codes; `tests/test_examples.py` asserts that the generator
is bit-for-bit reproducible and that the worst anchors come from the
drifted set.
