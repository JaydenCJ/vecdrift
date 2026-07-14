#!/usr/bin/env bash
# Smoke test for vecdrift: generate the example exports, snapshot a baseline,
# and verify that a clean model upgrade passes while a drifting one fails.
# Self-contained: pure stdlib, no network, idempotent (works from a clean tree).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
if [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
fi

# The package has zero runtime dependencies, so running from src/ needs no install.
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/vecdrift-smoke.XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT

fail() { echo "SMOKE FAIL: $1" >&2; exit 1; }

echo "[smoke] python: $("$PYTHON" --version 2>&1)"

# 1. Generate the deterministic example exports (v1 base, v2 clean, v3 drifted).
"$PYTHON" "$ROOT/examples/generate_exports.py" "$WORKDIR" >/dev/null \
  || fail "generate_exports.py exited non-zero"
for f in model_v1.jsonl model_v2.jsonl model_v3.jsonl; do
  [ -f "$WORKDIR/$f" ] || fail "missing generated export $f"
done

# 2. inspect: sanity stats on the base export.
inspect_out="$("$PYTHON" -m vecdrift inspect "$WORKDIR/model_v1.jsonl")"
echo "$inspect_out" | sed 's/^/[inspect] /'
echo "$inspect_out" | grep -q "anchors : 48" || fail "inspect did not count 48 anchors"
echo "$inspect_out" | grep -q "dim     : 8" || fail "inspect did not report dim 8"

# 3. snapshot: freeze the v1 geometry into a baseline file.
snap_out="$("$PYTHON" -m vecdrift snapshot "$WORKDIR/model_v1.jsonl" \
  -o "$WORKDIR/baseline.json" --label model-v1)"
echo "$snap_out" | sed 's/^/[snapshot] /'
echo "$snap_out" | grep -q "48 anchors" || fail "snapshot did not report 48 anchors"
[ -f "$WORKDIR/baseline.json" ] || fail "baseline file missing"

# 4. compare vs the clean upgrade: verdict OK, exit code 0.
ok_out="$("$PYTHON" -m vecdrift compare "$WORKDIR/baseline.json" "$WORKDIR/model_v2.jsonl")" \
  || fail "compare against the clean upgrade should exit 0"
echo "$ok_out" | sed 's/^/[compare-ok] /'
echo "$ok_out" | grep -q "verdict: OK" || fail "clean upgrade did not grade OK"

# 5. compare vs the drifting upgrade: verdict RE-EMBED, exit code 1.
set +e
bad_out="$("$PYTHON" -m vecdrift compare "$WORKDIR/baseline.json" "$WORKDIR/model_v3.jsonl")"
bad_rc=$?
set -e
echo "$bad_out" | tail -6 | sed 's/^/[compare-drift] /'
[ "$bad_rc" -eq 1 ] || fail "drifting upgrade should exit 1, got $bad_rc"
echo "$bad_out" | grep -q "verdict: RE-EMBED" || fail "drifting upgrade did not grade RE-EMBED"
echo "$bad_out" | grep -q "worst anchors" || fail "drift report did not name worst anchors"

# 6. --json output must be machine-readable and carry the same verdict.
"$PYTHON" -m vecdrift compare "$WORKDIR/baseline.json" "$WORKDIR/model_v3.jsonl" \
  --json --fail-on never > "$WORKDIR/report.json" \
  || fail "--fail-on never should exit 0"
"$PYTHON" -c '
import json, sys
data = json.load(open(sys.argv[1]))
assert data["verdict"] == "RE-EMBED", data["verdict"]
assert data["alignment"]["matched"] == 48
' "$WORKDIR/report.json" || fail "JSON report malformed"

# 7. pick: deterministic anchor selection round-trips through compare.
"$PYTHON" -m vecdrift pick "$WORKDIR/model_v1.jsonl" -n 16 -o "$WORKDIR/anchors.jsonl" >/dev/null \
  || fail "pick exited non-zero"
[ "$(wc -l < "$WORKDIR/anchors.jsonl")" -eq 16 ] || fail "pick did not write 16 anchors"
"$PYTHON" -m vecdrift compare "$WORKDIR/anchors.jsonl" "$WORKDIR/model_v2.jsonl" >/dev/null \
  || fail "picked anchor subset should still compare OK against v2"

# 8. --version agrees with the package version.
version_out="$("$PYTHON" -m vecdrift --version)"
pkg_version="$("$PYTHON" -c 'import vecdrift; print(vecdrift.__version__)')"
[ "$version_out" = "vecdrift $pkg_version" ] \
  || fail "--version mismatch: '$version_out' vs package '$pkg_version'"

echo "SMOKE OK"
