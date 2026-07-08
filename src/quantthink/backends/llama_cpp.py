# Adapted from Happynood/quant-toolcall-bench @6b6e29e5c83a (src/quantcall/backends/llama_cpp.py).
# The CUDA-preload workaround (_preload_cuda_libs) and _get_vram_mb are vendored
# verbatim — do not rediscover this fix, see docs/RUN_REAL.md.
# Diff: generate_toolcall(messages, tools) -> generate(messages, max_tokens,
# temperature, top_p, seed) (reasoning has no tool schema); adds KV-cache dtype
# control (type_k/type_v, new to QuantThink, see kvcache/config.py) and an
# explicit per-call seed (reasoning evals use sampled decoding with a fixed seed
# set, not the siblings' temperature=0 default).
from __future__ import annotations

import ctypes
import site
import subprocess
import time
from pathlib import Path

from quantthink.backends.base import Backend, GenerationResult
from quantthink.kvcache.config import KvDtype, kv_dtype_to_ggml_type


def _preload_cuda_libs() -> None:
    """Load CUDA runtime via ctypes (RTLD_GLOBAL) before importing llama_cpp.

    When llama-cpp-python is installed from a pre-built CUDA wheel the bundled
    libllama.so links against libcudart.so.12.  If the CUDA toolkit is not in
    the system library path (common on laptops with only the driver installed),
    the import fails with "libcudart.so.12: cannot open shared object file".

    Workaround: pre-load the shared object from the nvidia-cuda-runtime-cu12
    pip package with RTLD_GLOBAL so it is visible to every subsequent dlopen.
    """
    for site_dir in site.getsitepackages():
        for subpath in (
            "nvidia/cuda_runtime/lib",
            "nvidia/cublas/lib",
            "nvidia/cuda_nvrtc/lib",
        ):
            lib_dir = Path(site_dir) / subpath
            if not lib_dir.exists():
                continue
            for lib_file in sorted(lib_dir.glob("*.so*")):
                if lib_file.is_symlink():
                    continue
                try:
                    ctypes.CDLL(str(lib_file), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _get_vram_mb() -> float | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            return float(r.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


class LlamaCppBackend(Backend):
    """llama.cpp inference backend using llama-cpp-python.

    The primary backend for real GPU sweeps: GGUF weight quants x KV-cache
    dtype, uncapped or thinking-token-capped generation.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        kv_dtype: KvDtype = "fp16",
        chat_format: str | None = None,
        verbose: bool = False,
    ) -> None:
        _preload_cuda_libs()
        from llama_cpp import Llama

        self._model_path = model_path
        self._kv_dtype = kv_dtype

        # Quantized KV cache (type_k/type_v other than F16) requires flash attention
        # in llama.cpp — without it, context creation fails with "Failed to create
        # llama_context" (confirmed on this hardware/llama-cpp-python 0.3.33).
        kv_kwargs: dict[str, int | bool] = {}
        if kv_dtype != "fp16":
            ggml_type = kv_dtype_to_ggml_type(kv_dtype)
            kv_kwargs = {"type_k": ggml_type, "type_v": ggml_type, "flash_attn": True}

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            chat_format=chat_format,
            **kv_kwargs,
        )

    @property
    def name(self) -> str:
        return "llama-cpp"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
    ) -> GenerationResult:
        vram_before = _get_vram_mb()
        t_start = time.perf_counter()

        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed if seed is not None else -1,
        )

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        vram_after = _get_vram_mb()

        choice = response["choices"][0]
        msg = choice.get("message", {})
        raw_output: str = msg.get("content") or ""

        usage = response.get("usage", {})
        input_tokens: int = usage.get("prompt_tokens", 0)
        output_tokens: int = usage.get("completion_tokens", 0)
        peak_vram = (
            max(v for v in (vram_before, vram_after) if v is not None)
            if (vram_before is not None or vram_after is not None)
            else None
        )

        return GenerationResult(
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            peak_vram_mb=float(peak_vram) if peak_vram is not None else None,
            tokens_per_second=(
                output_tokens / (latency_ms / 1000.0)
                if latency_ms > 0 and output_tokens > 0
                else None
            ),
        )
