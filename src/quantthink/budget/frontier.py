"""Memory-Budget Frontier (MBF) — the prescriptive core of this project.

For a target peak-VRAM budget B, computes c*(B) = the accuracy- (or CTS-)
optimal (q_w, q_kv, thinking_cap) config among all configs that fit in B.

Reuses this project's own Pareto-front logic (report/pareto.py, vendored
verbatim from quant-toolcall-bench) for the Pareto-optimality check, and
follows the same constrain-then-select shape as llm-inference-benchmark's
`recommend.py` (Constraints -> apply_constraints -> pick the best candidate)
rather than reimplementing selection from scratch. Operates on this project's
own leaderboard-row dicts (report/published.py's aggregate_leaderboard()
output — vram_gb/acc_mean/cts_mean) instead of llm-inference-benchmark's
tool-calling-specific RunRow dataclass, since that dataclass's mandatory
fields (p95_latency_ms, sanity_pass_rate, ...) don't apply here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from quantthink.report.pareto import pareto_front

Objective = Literal["accuracy", "cts"]


@dataclass(frozen=True)
class BudgetSelection:
    budget_gb: float
    objective: Objective
    winner: dict[str, Any] | None
    is_pareto_optimal: bool
    candidates: list[dict[str, Any]]


def _objective_key(objective: Objective) -> tuple[str, bool]:
    """Return (metric_key, maximize) for the given objective."""
    if objective == "accuracy":
        return "acc_mean", True
    return "cts_mean", False


def _feasible(
    rows: list[dict[str, Any]], budget_gb: float, metric_key: str
) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if r.get("vram_gb") is not None
        and r["vram_gb"] <= budget_gb
        and r.get(metric_key) is not None
    ]


def select_for_budget(
    rows: list[dict[str, Any]], budget_gb: float, objective: Objective = "accuracy"
) -> BudgetSelection:
    """Select c*(B): the objective-optimal leaderboard row that fits in budget_gb."""
    metric_key, maximize = _objective_key(objective)
    candidates = _feasible(rows, budget_gb, metric_key)

    if not candidates:
        return BudgetSelection(
            budget_gb=budget_gb,
            objective=objective,
            winner=None,
            is_pareto_optimal=False,
            candidates=[],
        )

    winner = (max if maximize else min)(candidates, key=lambda r: r[metric_key])

    front = pareto_front(
        candidates, x_key="vram_gb", y_key=metric_key, minimize_x=True, maximize_y=maximize
    )
    is_pareto_optimal = any(c is winner for c in front)

    return BudgetSelection(
        budget_gb=budget_gb,
        objective=objective,
        winner=winner,
        is_pareto_optimal=is_pareto_optimal,
        candidates=candidates,
    )


def compute_frontier(
    rows: list[dict[str, Any]], budget_grid: list[float], objective: Objective = "accuracy"
) -> list[BudgetSelection]:
    """Sweep select_for_budget() over a grid of VRAM budgets -> the frontier B -> c*(B)."""
    return [select_for_budget(rows, b, objective) for b in budget_grid]


def render_selection(selection: BudgetSelection) -> str:
    if selection.winner is None:
        return f"Budget {selection.budget_gb:.2f} GB: no config fits (0 candidates)."

    w = selection.winner
    metric_key, _ = _objective_key(selection.objective)
    metric_label = "Acc" if selection.objective == "accuracy" else "CTS"
    pareto_note = "; Pareto-optimal" if selection.is_pareto_optimal else ""
    return (
        f"Budget {selection.budget_gb:.2f} GB -> {w['model']} {w['quant']} "
        f"(kv={w['kv_quant']}, cap={w['thinking_cap']}): "
        f"{metric_label}={w[metric_key]:.3f}, VRAM={w['vram_gb']:.2f} GB "
        f"({len(selection.candidates)} candidate(s){pareto_note})"
    )
