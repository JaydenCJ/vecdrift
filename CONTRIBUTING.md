# Contributing to vecdrift

Thanks for your interest in contributing. Issues, discussions, and pull
requests are all welcome.

## Development setup

```bash
git clone https://github.com/JaydenCJ/vecdrift
cd vecdrift
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the checks

```bash
pytest                 # 91 unit + CLI + example tests, fully offline
bash scripts/smoke.sh  # end-to-end: snapshot -> compare OK -> compare RE-EMBED
```

Both must pass before a pull request is reviewed; `scripts/smoke.sh` must
print `SMOKE OK`. The whole suite runs offline in a few seconds and needs no
API keys, no vector database, and no model downloads.

## Ground rules

- **No new runtime dependencies.** The package is standard-library only;
  that is a feature. Test-only dependencies belong in the `dev` extra.
- **Determinism is a contract.** Neighbor orderings, anchor picks, and every
  reported number must be reproducible bit-for-bit across runs and machines:
  no RNG without a fixed seed, ties always broken by anchor id.
- **Baseline format changes need a version bump and docs.** Anything that
  changes the meaning of an existing field must bump `FORMAT_VERSION` and
  update `docs/baseline-format.md` in the same pull request.
- **Every public API needs an English docstring and a test.** Keep logic in
  pure, unit-testable modules; the CLI layer stays thin.
- **Keep the three READMEs aligned.** `README.md`, `README.zh.md`, and
  `README.ja.md` share the same structure; update all three when you change
  one (English is the authoritative version).

## Reporting bugs

Please include the `vecdrift --version` output, the exact command line, the
full report text (it contains no vector data, only ids and aggregate
numbers), and — if you can share them — the two exports or the baseline file
that reproduce the problem.

## Security

Please do not report security issues in public GitHub issues. Use GitHub's
private vulnerability reporting on this repository instead.
