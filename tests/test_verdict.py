"""Tests for the threshold policy and verdict grading."""

import pytest

from vecdrift.verdict import Thresholds, Verdict, evaluate


def test_perfect_metrics_and_exact_boundaries_grade_ok():
    verdict, reasons = evaluate(mean_overlap=1.0, correlation=1.0, mean_delta=0.0)
    assert verdict is Verdict.OK and reasons == []
    # Gates are inclusive: exactly meeting a threshold is a pass.
    verdict, _ = evaluate(mean_overlap=0.95, correlation=0.995, mean_delta=0.02)
    assert verdict is Verdict.OK


def test_any_single_metric_beyond_its_ok_gate_gives_warn_with_reason():
    verdict, reasons = evaluate(mean_overlap=0.94, correlation=1.0, mean_delta=0.0)
    assert verdict is Verdict.WARN
    assert any("overlap" in reason for reason in reasons)
    verdict, reasons = evaluate(mean_overlap=1.0, correlation=0.99, mean_delta=0.0)
    assert verdict is Verdict.WARN
    assert any("correlation" in reason for reason in reasons)
    verdict, reasons = evaluate(mean_overlap=1.0, correlation=1.0, mean_delta=0.03)
    assert verdict is Verdict.WARN
    assert any("delta" in reason for reason in reasons)


def test_any_warn_gate_failure_gives_reembed_citing_warn_tier():
    verdict, reasons = evaluate(mean_overlap=0.5, correlation=1.0, mean_delta=0.0)
    assert verdict is Verdict.REEMBED
    # For RE-EMBED the actionable information is which *warn* gate broke.
    _, reasons = evaluate(mean_overlap=0.5, correlation=0.5, mean_delta=0.5)
    assert len(reasons) == 3
    assert all("warn threshold" in reason for reason in reasons)


def test_none_correlation_never_fails_a_gate_by_itself():
    # Zero-variance geometry (e.g. an orthonormal anchor set) yields no
    # correlation signal; overlap and delta still decide.
    verdict, _ = evaluate(mean_overlap=1.0, correlation=None, mean_delta=0.0)
    assert verdict is Verdict.OK
    verdict, reasons = evaluate(mean_overlap=0.9, correlation=None, mean_delta=0.0)
    assert verdict is Verdict.WARN
    assert all("correlation" not in reason for reason in reasons)


def test_custom_thresholds_relax_the_gate():
    assert evaluate(0.9, 1.0, 0.0)[0] is Verdict.WARN
    relaxed = Thresholds(ok_overlap=0.85, warn_overlap=0.5)
    assert evaluate(0.9, 1.0, 0.0, relaxed)[0] is Verdict.OK


def test_thresholds_reject_inconsistent_or_out_of_range_values():
    with pytest.raises(ValueError, match="warn_overlap"):
        Thresholds(ok_overlap=0.9, warn_overlap=0.95)
    with pytest.raises(ValueError, match="warn_delta"):
        Thresholds(ok_delta=0.1, warn_delta=0.05)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Thresholds(ok_overlap=1.5)
    with pytest.raises(ValueError, match=">= 0"):
        Thresholds(ok_delta=-0.1)


def test_verdict_severity_ordering():
    assert Verdict.OK.severity < Verdict.WARN.severity < Verdict.REEMBED.severity
