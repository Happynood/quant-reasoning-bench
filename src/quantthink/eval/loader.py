"""Loads frozen benchmark subsets (fixed indices) for each eval tier.

Phase 0 ships only E0 (in-repo toy arithmetic, no download, CI-safe). E1
(GSM8K) and E2 (MATH-500 subset) are loaded from the frozen HF suite dataset
in Phase 1 — see `load_gsm8k`/`load_math500` below.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_TOY_PATH = Path(__file__).parent.parent.parent.parent / "data" / "smoke" / "e0_toy.jsonl"


@dataclass(frozen=True)
class Problem:
    id: str
    tier: str
    prompt: str
    ground_truth: str


def load_toy(path: str | Path | None = None) -> list[Problem]:
    """Load the in-repo E0 toy dataset (2 hand-crafted arithmetic problems)."""
    p = Path(path) if path is not None else _TOY_PATH
    problems: list[Problem] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            problems.append(
                Problem(
                    id=raw["id"],
                    tier=raw.get("tier", "E0"),
                    prompt=raw["prompt"],
                    ground_truth=raw["ground_truth"],
                )
            )
    return problems


def load_gsm8k(sample_size: int = 200, seed: int = 42) -> list[Problem]:
    """E1: GSM8K frozen subset. Implemented in Phase 1."""
    raise NotImplementedError(
        "E1 (GSM8K) loader ships in Phase 1, from the frozen quantthink-suite HF dataset."
    )


def load_math500(sample_size: int = 200, seed: int = 42) -> list[Problem]:
    """E2: MATH-500 frozen subset. Implemented in Phase 1."""
    raise NotImplementedError(
        "E2 (MATH-500) loader ships in Phase 1, from the frozen quantthink-suite HF dataset."
    )
