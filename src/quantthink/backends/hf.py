# Adapted from Happynood/quant-toolcall-bench @6b6e29e5c83a (src/quantcall/backends/hf.py).
# Diff: generate_toolcall(messages, tools) -> generate(messages, max_tokens,
# temperature, top_p, seed); apply_chat_template no longer passes tools=.
from __future__ import annotations

import os
import subprocess
import time

from quantthink.backends.base import Backend, GenerationResult

_DTYPE_MAP = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "auto": "auto",
}


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


class HFBackend(Backend):
    """HuggingFace Transformers inference backend (AutoModelForCausalLM).

    Optional dependency — install with: uv sync --extra transformers
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ) -> None:
        # httpx (used by huggingface_hub) rejects the bare socks:// proxy scheme
        # some shells export; drop it so HF Hub falls through to http(s)_proxy.
        os.environ.pop("ALL_PROXY", None)
        os.environ.pop("all_proxy", None)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._model_id = model_id
        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        quantization_config = None
        if load_in_4bit or load_in_8bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
            )

        dtype_name = _DTYPE_MAP.get(torch_dtype, "auto")
        dtype = getattr(torch, dtype_name) if dtype_name != "auto" else "auto"

        if quantization_config is not None:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                quantization_config=quantization_config,
                device_map=device,
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(
                device
            )
        self._model.eval()

    @property
    def name(self) -> str:
        return "transformers"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
    ) -> GenerationResult:
        torch = self._torch
        if seed is not None:
            torch.manual_seed(seed)
        vram_before = _get_vram_mb()

        prompt = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        input_len: int = inputs["input_ids"].shape[1]

        t_start = time.perf_counter()
        with torch.no_grad():
            gen_kwargs: dict = {
                "max_new_tokens": max_tokens,
                "pad_token_id": self._tokenizer.pad_token_id,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        vram_after = _get_vram_mb()

        output_len: int = output_ids.shape[1] - input_len
        raw_output: str = self._tokenizer.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        )

        peak_vram = (
            max(v for v in (vram_before, vram_after) if v is not None)
            if (vram_before is not None or vram_after is not None)
            else None
        )

        return GenerationResult(
            raw_output=raw_output,
            input_tokens=input_len,
            output_tokens=output_len,
            latency_ms=latency_ms,
            peak_vram_mb=float(peak_vram) if peak_vram is not None else None,
            tokens_per_second=(
                output_len / (latency_ms / 1000.0) if latency_ms > 0 and output_len > 0 else None
            ),
        )
