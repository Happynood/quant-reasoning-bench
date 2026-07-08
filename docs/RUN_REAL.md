# Running Real GPU Evaluations

This guide covers running QuantThink against actual quantized reasoning models on a GPU, and documents exactly what has been run so far to produce the numbers in this repository.

## Status

No real GPU sweep has been run yet — Phase 0 (this commit) only establishes the skeleton and a mock-backend pipeline. The first real sweep (M1: DeepSeek-R1-Distill-Qwen-1.5B, weight quants fp16/Q8_0/Q5_K_M/Q4_K_M, fp16 KV, uncapped, on GSM8K + MATH-500) is next; this file will be updated with the exact commands, real numbers, and any scope limitations hit.

## Prerequisites

- Python 3.11+, `uv` installed (`pip install uv`)
- A CUDA-capable NVIDIA GPU. Only the driver is required — the CUDA *toolkit* does not need to be installed (see "GPU offload without the CUDA toolkit" below).
- A downloaded GGUF model file

## Install with the llama.cpp backend

```bash
git clone https://github.com/Happynood/quant-reasoning-bench
cd quant-reasoning-bench
uv sync --extra llama-cpp
```

`pyproject.toml`'s `llama-cpp` extra pulls in `nvidia-cuda-runtime-cu12` and `nvidia-cublas-cu12` pip wheels alongside `llama-cpp-python`. These provide `libcudart.so.12` / `libcublas.so.12` for machines that only have the NVIDIA driver installed (common on laptops), not the full CUDA toolkit.

### GPU offload without the CUDA toolkit

If `llama_cpp` fails to import with `libcudart.so.12: cannot open shared object file`, this is expected on driver-only systems. `LlamaCppBackend` (`src/quantthink/backends/llama_cpp.py`) works around this automatically by preloading the CUDA `.so` files from the pip-installed `nvidia-cuda-runtime-cu12` / `nvidia-cublas-cu12` packages with `ctypes.RTLD_GLOBAL` before importing `llama_cpp` — no manual steps needed once the extras are installed.

Importing `llama_cpp` successfully does **not** mean CUDA is available: a plain `uv sync --extra llama-cpp` may resolve to a source build (no prebuilt wheel available for your platform/Python/CUDA combination), and a source build without `CMAKE_ARGS="-DGGML_CUDA=on"` is silently CPU-only. Use `make install-llama-cpp-cuda` to force a CUDA source build, and verify GPU offload actually happened (check `nvidia-smi` during a run) before trusting a timing/VRAM number.

## Dependency / landscape decisions

- **M4 (third-family stretch model):** not yet chosen — the small-reasoning-model landscape moves monthly. Will be verified and recorded here before Phase 6.
- **AWQ/GPTQ tooling (Phase 4):** `pyproject.toml`'s `quantize` extra currently lists `autoawq` and `gptqmodel` as placeholders; the currently-maintained package names will be re-verified immediately before Phase 4 and this note updated with the actual versions used.

## Real runs completed

None yet.
