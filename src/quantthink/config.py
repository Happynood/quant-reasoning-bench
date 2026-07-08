# Adapted from Happynood/quant-toolcall-bench @6b6e29e5c83a (src/quantcall/config.py).
# Diff: BFCL-shaped fields (tiers T0-T6, chat_variant, use_gated_xlam, bfcl_data_dir,
# fcr_weights) replaced with reasoning-shaped fields (benchmark tiers E0-E5,
# kv_quant, thinking_cap, seeds list for sampled decoding). The
# Pydantic-v2 base pattern and per-backend config blocks are otherwise unchanged.
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class MockBackendConfig(BaseModel):
    latency_ms: float = Field(default=5.0, ge=0.0)


class LlamaCppBackendConfig(BaseModel):
    n_ctx: int = Field(default=8192, ge=1)
    n_gpu_layers: int = -1
    chat_format: str | None = None
    verbose: bool = False


class HFBackendConfig(BaseModel):
    device: str = "cpu"
    torch_dtype: Literal["float32", "float16", "bfloat16", "auto"] = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False


class OpenAIEndpointConfig(BaseModel):
    base_url: str = "http://localhost:8080/v1"
    api_key_env: str | None = None
    timeout_s: float = Field(default=120.0, gt=0.0)


class VLLMBackendConfig(BaseModel):
    tensor_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.9, gt=0.0, le=1.0)
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"


class QuantThinkConfig(BaseModel):
    model: str = "mock"
    backend: Literal["mock", "llama-cpp", "transformers", "vllm", "openai"] = "mock"
    quant: str = "fp16"
    kv_quant: Literal["fp16", "Q8", "Q4"] = "fp16"
    thinking_cap: int | None = None  # max thinking-token budget L; None = uncapped
    max_tokens: int = Field(default=512, ge=1)  # total generation budget (thinking + answer)
    tiers: list[str] = Field(default_factory=lambda: ["E0"])
    sample_size: int = Field(default=50, ge=1)
    seeds: list[int] = Field(default_factory=lambda: [0])
    temperature: float = Field(default=0.6, ge=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    greedy: bool = False  # disclosed greedy-decoding control run, for comparison against sampling
    mock: MockBackendConfig = Field(default_factory=MockBackendConfig)
    llama_cpp: LlamaCppBackendConfig = Field(default_factory=LlamaCppBackendConfig)
    hf: HFBackendConfig = Field(default_factory=HFBackendConfig)
    openai: OpenAIEndpointConfig = Field(default_factory=OpenAIEndpointConfig)
    vllm: VLLMBackendConfig = Field(default_factory=VLLMBackendConfig)


def load_config(path: str | Path) -> QuantThinkConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return QuantThinkConfig.model_validate(data or {})
