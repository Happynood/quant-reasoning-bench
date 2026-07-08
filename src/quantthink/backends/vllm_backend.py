"""vLLM offline inference backend (vllm.LLM.chat()).

Adapted from Happynood/quant-toolcall-bench @6b6e29e5c83a
(src/quantcall/backends/vllm_backend.py). Diff: generate_toolcall(messages,
tools) -> generate(messages, max_tokens, temperature, top_p, seed); adds top_p
and per-call seed to SamplingParams.

Optional dependency — install with: uv sync --extra vllm
Requires a CUDA-capable GPU with enough VRAM for vLLM's paged-attention KV
cache allocator; vLLM has no CPU inference path. Same 4GB scope-limit
disclosure as the sibling projects — not verified on hardware this tight.
"""

from __future__ import annotations

import time

from quantthink.backends.base import Backend, GenerationResult


class VLLMBackend(Backend):
    """Inference backend using vLLM's offline `LLM.chat()` API."""

    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
    ) -> None:
        from vllm import LLM, SamplingParams

        self._sampling_params_cls = SamplingParams

        self._llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
        )

    @property
    def name(self) -> str:
        return "vllm"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
    ) -> GenerationResult:
        sampling_params = self._sampling_params_cls(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )

        t_start = time.perf_counter()
        outputs = self._llm.chat(messages, sampling_params=sampling_params)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        output = outputs[0]
        completion = output.outputs[0]
        raw_output: str = completion.text
        input_tokens: int = len(output.prompt_token_ids)
        output_tokens: int = len(completion.token_ids)

        return GenerationResult(
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            tokens_per_second=(
                output_tokens / (latency_ms / 1000.0)
                if latency_ms > 0 and output_tokens > 0
                else None
            ),
        )
