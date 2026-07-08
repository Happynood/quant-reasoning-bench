# Adapted from Happynood/quant-toolcall-bench @6b6e29e5c83a (src/quantcall/backends/mock.py).
# Diff: emits a deterministic fake <think>...</think> + final-answer completion
# instead of a fake tool-call JSON blob, so the E0 toy eval and CI smoke test can
# exercise the extractor/checker pipeline without a real model.
from __future__ import annotations

import time

from quantthink.backends.base import Backend, GenerationResult


class MockBackend(Backend):
    """Deterministic mock backend for CI and plumbing validation."""

    def __init__(
        self,
        model: str = "mock",
        latency_ms: float = 5.0,
        answer: str = "4",
        thinking_tokens: int = 8,
    ) -> None:
        self._model = model
        self._latency_s = latency_ms / 1000.0
        self._answer = answer
        self._thinking_tokens = thinking_tokens

    @property
    def name(self) -> str:
        return "mock"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
    ) -> GenerationResult:
        start = time.perf_counter()
        if self._latency_s > 0:
            time.sleep(self._latency_s)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        thinking = " ".join(["thinking"] * self._thinking_tokens)
        raw_output = f"<think>{thinking}</think>\nThe answer is \\boxed{{{self._answer}}}."

        input_tokens = sum(len(m.get("content", "").split()) for m in messages)
        output_tokens = len(raw_output.split())
        return GenerationResult(
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            tokens_per_second=output_tokens / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0,
        )
