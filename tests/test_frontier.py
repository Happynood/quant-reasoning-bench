from __future__ import annotations

from quantthink.budget.frontier import compute_frontier, select_for_budget


def _row(model, quant, acc, cts, vram_gb):
    return {
        "model": model,
        "quant": quant,
        "kv_quant": "fp16",
        "thinking_cap": None,
        "acc_mean": acc,
        "cts_mean": cts,
        "vram_gb": vram_gb,
    }


ROWS = [
    _row("R1-1.5B", "fp16", 0.8, 200.0, 3.5),
    _row("R1-1.5B", "Q8_0", 0.75, 180.0, 2.2),
    _row("R1-1.5B", "Q5_K_M", 0.7, 190.0, 1.7),
    _row("R1-1.5B", "Q4_K_M", 0.6, 220.0, 1.5),
]


def test_select_for_budget_picks_best_accuracy_within_budget():
    sel = select_for_budget(ROWS, budget_gb=2.0, objective="accuracy")
    assert sel.winner is not None
    assert sel.winner["quant"] == "Q5_K_M"  # highest acc among quants fitting <= 2.0GB


def test_select_for_budget_picks_best_cts_within_budget():
    sel = select_for_budget(ROWS, budget_gb=4.0, objective="cts")
    assert sel.winner is not None
    assert sel.winner["quant"] == "Q8_0"  # lowest cts among all quants


def test_select_for_budget_no_candidates_returns_none_winner():
    sel = select_for_budget(ROWS, budget_gb=0.5, objective="accuracy")
    assert sel.winner is None
    assert sel.candidates == []


def test_select_for_budget_widest_budget_matches_global_best():
    sel = select_for_budget(ROWS, budget_gb=10.0, objective="accuracy")
    assert sel.winner is not None
    assert sel.winner["quant"] == "fp16"


def test_compute_frontier_sweeps_grid():
    frontier = compute_frontier(ROWS, budget_grid=[1.5, 2.0, 4.0], objective="accuracy")
    assert len(frontier) == 3
    winners = [s.winner["quant"] if s.winner else None for s in frontier]
    assert winners == ["Q4_K_M", "Q5_K_M", "fp16"]
