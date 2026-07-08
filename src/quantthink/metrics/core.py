"""Acc, Thinking-Length (TL), Cost-to-Solve (CTS).

The genuinely new metric code for QuantThink (no sibling measures thinking
length or cost-to-solve; QuantCall/QuantMCP's `metrics/core.py` computes
SVR/TSA/AC/FCR instead and is not reused here).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstanceResult:
    problem_id: str
    seed: int
    correct: bool
    thinking_tokens: int
    total_tokens: int
    thinking_truncated: bool = False


@dataclass(frozen=True)
class MetricsResult:
    n: int
    acc: float
    tl_mean: float
    tl_solved: float | None
    tl_unsolved: float | None
    cts: float | None
    total_tokens_mean: float


def evaluate_instance(
    problem_id: str,
    seed: int,
    correct: bool,
    thinking_tokens: int,
    total_tokens: int,
    thinking_truncated: bool = False,
) -> InstanceResult:
    return InstanceResult(
        problem_id=problem_id,
        seed=seed,
        correct=correct,
        thinking_tokens=thinking_tokens,
        total_tokens=total_tokens,
        thinking_truncated=thinking_truncated,
    )


def compute_metrics(results: list[InstanceResult]) -> MetricsResult:
    """Compute Acc (4.1), TL (4.2), CTS (4.3) over N problems x K seeds.

    CTS = mean(total_tokens) / Acc; None when Acc is 0 (division undefined —
    an all-wrong config has no "cost per correct answer").
    """
    if not results:
        raise ValueError("results must not be empty")

    n = len(results)
    acc = sum(1 for r in results if r.correct) / n
    tl_mean = sum(r.thinking_tokens for r in results) / n
    total_tokens_mean = sum(r.total_tokens for r in results) / n

    solved = [r.thinking_tokens for r in results if r.correct]
    unsolved = [r.thinking_tokens for r in results if not r.correct]
    tl_solved = sum(solved) / len(solved) if solved else None
    tl_unsolved = sum(unsolved) / len(unsolved) if unsolved else None

    cts = total_tokens_mean / acc if acc > 0 else None

    return MetricsResult(
        n=n,
        acc=acc,
        tl_mean=tl_mean,
        tl_solved=tl_solved,
        tl_unsolved=tl_unsolved,
        cts=cts,
        total_tokens_mean=total_tokens_mean,
    )
