"""Memory-Budget Frontier (MBF) — the prescriptive core of this project.

For a target peak-VRAM budget B, computes c*(B) = the accuracy- (or CTS-)
optimal (q_w, q_kv, L) config among all configs that fit in B. Wraps the
constraint-based recommender from llm-inference-benchmark's `recommend.py`
(Constraints / apply_constraints / pareto_classify) rather than reimplementing
selection logic.

Ships in Phase 3, once real sweep results across the 2-4GB grid exist to
compute a frontier from. Phase 0 only reserves the module path.
"""

from __future__ import annotations


def compute_frontier(*args: object, **kwargs: object) -> None:
    raise NotImplementedError(
        "The Memory-Budget Frontier ships in Phase 3, once Phase 1-2 sweep "
        "results exist to compute c*(B) from across the 2-4GB budget grid."
    )
