# Adapted from Happynood/quant-toolcall-bench @6b6e29e5c83a (src/quantcall/backends/base.py).
# Diff: reasoning backends drive plain long-form generation, not tool-calling, so
# generate_toolcall()/ToolCallResult/tools_to_openai_spec() (all tool-schema-shaped)
# are replaced by generate()/GenerationResult with a single free-text prompt in and
# a plain completion out. The numeric fields (latency, VRAM, tok/s) and the ABC
# shape are otherwise unchanged.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    raw_output: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    ttft_ms: float | None = None
    peak_vram_mb: float | None = None
    tokens_per_second: float | None = None


class Backend(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
    ) -> GenerationResult: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
