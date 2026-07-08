"""Loads frozen benchmark subsets (fixed indices) for each eval tier.

E0 (in-repo toy arithmetic) needs no download. E1 (GSM8K) and E2 (MATH-500)
are frozen once via `freeze_gsm8k`/`freeze_math500` into local JSONL files
under `data/suite/` (same shape as the toy set) so a benchmark run never
depends on the upstream dataset being reachable or unchanged — the frozen
files are what ships as the `quantthink-suite` HF dataset in Phase 4.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

_DATA_ROOT = Path(__file__).parent.parent.parent.parent / "data"
_TOY_PATH = _DATA_ROOT / "smoke" / "e0_toy.jsonl"
_GSM8K_PATH = _DATA_ROOT / "suite" / "gsm8k_e1.jsonl"
_MATH500_PATH = _DATA_ROOT / "suite" / "math500_e2.jsonl"

_THINK_INSTRUCTION = "\n\nThink step by step, then give your final answer as \\boxed{}."


@dataclass(frozen=True)
class Problem:
    id: str
    tier: str
    prompt: str
    ground_truth: str


def _load_jsonl(path: Path) -> list[Problem]:
    problems: list[Problem] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            problems.append(
                Problem(
                    id=raw["id"],
                    tier=raw["tier"],
                    prompt=raw["prompt"],
                    ground_truth=raw["ground_truth"],
                )
            )
    return problems


def load_toy(path: str | Path | None = None) -> list[Problem]:
    """Load the in-repo E0 toy dataset (2 hand-crafted arithmetic problems)."""
    return _load_jsonl(Path(path) if path is not None else _TOY_PATH)


def load_gsm8k(path: str | Path | None = None) -> list[Problem]:
    """E1: the frozen GSM8K subset written by `freeze_gsm8k` (see below)."""
    p = Path(path) if path is not None else _GSM8K_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist yet — run `python -m quantthink.eval.loader freeze-gsm8k` "
            "(or freeze_gsm8k()) once to materialize the frozen E1 subset."
        )
    return _load_jsonl(p)


def load_math500(path: str | Path | None = None) -> list[Problem]:
    """E2: the frozen MATH-500 subset written by `freeze_math500` (see below)."""
    p = Path(path) if path is not None else _MATH500_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist yet — run `python -m quantthink.eval.loader freeze-math500` "
            "(or freeze_math500()) once to materialize the frozen E2 subset."
        )
    return _load_jsonl(p)


def _write_jsonl(problems: list[Problem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in problems:
            f.write(
                json.dumps(
                    {"id": p.id, "tier": p.tier, "prompt": p.prompt, "ground_truth": p.ground_truth}
                )
                + "\n"
            )


def freeze_gsm8k(
    sample_size: int = 200, seed: int = 42, output_path: str | Path | None = None
) -> list[Problem]:
    """Sample a fixed subset of GSM8K's test split and write it as frozen JSONL.

    One-time (per sample_size/seed) generation step — not called at eval time.
    Ground truth is normalized from GSM8K's "...#### N" suffix to a bare "N"
    so every tier's checker sees a plain final-answer string.
    """
    from datasets import load_dataset  # type: ignore[import]

    from quantthink.eval.checkers import extract_gsm8k_ground_truth

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(ds)), min(sample_size, len(ds))))

    problems: list[Problem] = []
    for i in indices:
        row = ds[i]
        ground_truth = extract_gsm8k_ground_truth(row["answer"])
        if ground_truth is None:
            continue
        problems.append(
            Problem(
                id=f"gsm8k-{i}",
                tier="E1",
                prompt=row["question"] + _THINK_INSTRUCTION,
                ground_truth=ground_truth,
            )
        )

    _write_jsonl(problems, Path(output_path) if output_path is not None else _GSM8K_PATH)
    return problems


def freeze_math500(
    sample_size: int = 200, seed: int = 42, output_path: str | Path | None = None
) -> list[Problem]:
    """Sample a fixed subset of HuggingFaceH4/MATH-500's test split and freeze it.

    MATH-500's `answer` field is already a bare final-answer expression (no
    \\boxed{} wrapper), unlike its `solution` field — used directly as ground
    truth.
    """
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(ds)), min(sample_size, len(ds))))

    problems: list[Problem] = []
    for i in indices:
        row = ds[i]
        problems.append(
            Problem(
                id=f"math500-{row['unique_id']}",
                tier="E2",
                prompt=row["problem"] + _THINK_INSTRUCTION,
                ground_truth=row["answer"],
            )
        )

    _write_jsonl(problems, Path(output_path) if output_path is not None else _MATH500_PATH)
    return problems


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2 or sys.argv[1] not in ("freeze-gsm8k", "freeze-math500"):
        print("usage: python -m quantthink.eval.loader {freeze-gsm8k|freeze-math500}")
        sys.exit(1)
    if sys.argv[1] == "freeze-gsm8k":
        frozen = freeze_gsm8k()
        print(f"Wrote {len(frozen)} GSM8K problems to {_GSM8K_PATH}")
    else:
        frozen = freeze_math500()
        print(f"Wrote {len(frozen)} MATH-500 problems to {_MATH500_PATH}")
