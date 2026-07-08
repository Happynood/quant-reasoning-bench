"""Per-model recommended sampling defaults + fixed seed set.

New to QuantThink: the sibling projects default to temperature=0 (greedy);
reasoning models are known to degrade under greedy decoding, so the primary
run here uses each model's creator-recommended sampling with a FIXED set of
K seeds for reproducibility — a deliberate, disclosed departure documented in
docs/METHODOLOGY.md. A greedy control run is kept for one model to show the gap.
"""

from __future__ import annotations

from dataclasses import dataclass

# Fixed seed set used for every sampled (non-greedy) run, across all models and
# configs, so every result is reproducible. Recorded in every manifest.
DEFAULT_SEED_SET: tuple[int, ...] = (0, 1, 2, 3, 4)

GREEDY_SEED: int = 0


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    seeds: tuple[int, ...]


# Model-creator-recommended sampling params, verified against each model's card
# at build time (record any change in docs/RUN_REAL.md).
_MODEL_SAMPLING: dict[str, SamplingConfig] = {
    "deepseek-r1-distill-qwen-1.5b": SamplingConfig(
        temperature=0.6, top_p=0.95, seeds=DEFAULT_SEED_SET
    ),
    "qwen3-1.7b": SamplingConfig(temperature=0.6, top_p=0.95, seeds=DEFAULT_SEED_SET),
    "qwen3-0.6b": SamplingConfig(temperature=0.6, top_p=0.95, seeds=DEFAULT_SEED_SET),
    "mock": SamplingConfig(temperature=0.0, top_p=1.0, seeds=(0,)),
}


def get_sampling_config(model_key: str) -> SamplingConfig:
    key = model_key.strip().lower()
    if key not in _MODEL_SAMPLING:
        raise ValueError(
            f"No recommended sampling config for {model_key!r}; "
            f"add it to _MODEL_SAMPLING after checking the model card."
        )
    return _MODEL_SAMPLING[key]


def greedy_config(model_key: str) -> SamplingConfig:
    """A temperature=0 control config, for the disclosed greedy-vs-sampled comparison."""
    return SamplingConfig(temperature=0.0, top_p=1.0, seeds=(GREEDY_SEED,))


def enforce_thinking_cap(thinking_tokens: int, cap: int | None) -> bool:
    """Return True if the thinking segment exceeds the cap (i.e. was truncated).

    The cap itself is enforced by the backend's max_tokens during generation
    (decoding/sampling.py only classifies the outcome after the fact); this
    keeps the truncation decision in one place for the H5 thinking-cap study.
    """
    if cap is None:
        return False
    return thinking_tokens >= cap
