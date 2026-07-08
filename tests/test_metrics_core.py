from __future__ import annotations

import pytest

from quantthink.metrics.core import compute_metrics, evaluate_instance


def test_compute_metrics_basic_acc():
    results = [
        evaluate_instance("p1", 0, True, thinking_tokens=10, total_tokens=20),
        evaluate_instance("p2", 0, False, thinking_tokens=30, total_tokens=40),
    ]
    m = compute_metrics(results)
    assert m.n == 2
    assert m.acc == 0.5
    assert m.tl_mean == 20.0
    assert m.total_tokens_mean == 30.0


def test_compute_metrics_tl_split_solved_unsolved():
    results = [
        evaluate_instance("p1", 0, True, thinking_tokens=10, total_tokens=20),
        evaluate_instance("p2", 0, True, thinking_tokens=20, total_tokens=30),
        evaluate_instance("p3", 0, False, thinking_tokens=100, total_tokens=110),
    ]
    m = compute_metrics(results)
    assert m.tl_solved == 15.0
    assert m.tl_unsolved == 100.0


def test_compute_metrics_cts_is_tokens_per_correct_answer():
    results = [
        evaluate_instance("p1", 0, True, thinking_tokens=10, total_tokens=100),
        evaluate_instance("p2", 0, False, thinking_tokens=10, total_tokens=100),
    ]
    m = compute_metrics(results)
    # acc=0.5, total_tokens_mean=100 -> cts = 100/0.5 = 200
    assert m.cts == pytest.approx(200.0)


def test_compute_metrics_cts_none_when_all_wrong():
    results = [evaluate_instance("p1", 0, False, thinking_tokens=10, total_tokens=100)]
    m = compute_metrics(results)
    assert m.acc == 0.0
    assert m.cts is None


def test_compute_metrics_empty_raises():
    with pytest.raises(ValueError):
        compute_metrics([])


def test_compute_metrics_all_correct_no_unsolved_tl():
    results = [evaluate_instance("p1", 0, True, thinking_tokens=5, total_tokens=10)]
    m = compute_metrics(results)
    assert m.tl_unsolved is None
    assert m.tl_solved == 5.0
