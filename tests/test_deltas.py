from __future__ import annotations

from quantthink.metrics.deltas import compute_delta, compute_delta_rel, compute_eta


def test_compute_delta_basic():
    assert round(compute_delta(0.8, 0.6), 6) == 0.2


def test_compute_delta_rel():
    delta_rel = compute_delta_rel(0.8, 0.6)
    assert delta_rel is not None
    assert round(delta_rel, 6) == 0.25


def test_compute_delta_rel_zero_baseline():
    assert compute_delta_rel(0.0, 0.5) is None


def test_compute_eta():
    assert compute_eta(0.8, 4.0) == 0.2


def test_compute_eta_none_vram():
    assert compute_eta(0.8, None) is None


def test_compute_eta_zero_vram():
    assert compute_eta(0.8, 0.0) is None
