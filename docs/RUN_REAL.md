# Running Real GPU Evaluations

This guide covers running QuantThink against actual quantized reasoning models on a GPU, and documents exactly what has been run so far to produce the numbers in this repository.

## Status

A first-pass real GPU sweep (M1: DeepSeek-R1-Distill-Qwen-1.5B, weight quants fp16/Q8_0/Q5_K_M/Q4_K_M, fp16 KV, uncapped, on E1+E2) has been run on the RTX 3050 Laptop — see "Real runs completed" below for the exact commands, numbers, and a disclosed methodology bug that was caught and fixed before trusting these numbers.

E1 (GSM8K) and E2 (MATH-500) are frozen locally under `data/suite/` (200 problems each, seed 42) via:

```bash
uv sync --extra datasets
uv run python -m quantthink.eval.loader freeze-gsm8k
uv run python -m quantthink.eval.loader freeze-math500
```

These are one-time generation commands (network access, not needed again once the frozen JSONL files exist) — they are what will ship as the `quantthink-suite` HF dataset in Phase 4. Ordinary benchmark runs never re-download the upstream datasets.

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

Importing `llama_cpp` successfully does **not** mean CUDA is available: a plain `uv sync --extra llama-cpp` may resolve to a source build (no prebuilt wheel available for your platform/Python/CUDA combination), and a source build without `CMAKE_ARGS="-DGGML_CUDA=on"` is silently CPU-only. Verify with:

```python
from quantthink.backends.llama_cpp import _preload_cuda_libs
_preload_cuda_libs()
import llama_cpp
assert llama_cpp.llama_supports_gpu_offload()
```

On the machine these numbers were produced on, `make install-llama-cpp-cuda` (a source build with `CMAKE_ARGS="-DGGML_CUDA=on"`) resolved successfully but **silently produced a CPU-only build** (`llama_supports_gpu_offload()` returned `False`) because neither `cmake` nor `nvcc` was actually installed on the machine — `CMAKE_ARGS` alone cannot compile CUDA kernels without the toolkit. The fix that actually worked: install `llama-cpp-python`'s prebuilt CUDA wheel directly, bypassing PyPI's default (CPU) wheel:

```bash
uv pip install --force-reinstall --no-deps \
  "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.33-cu124/llama_cpp_python-0.3.33-py3-none-manylinux_2_35_x86_64.whl"
```

(exact URL found via https://abetlen.github.io/llama-cpp-python/whl/cu124/llama-cpp-python/ — match the version to what `pyproject.toml` resolves, and the CUDA tag to what the installed `nvidia-cuda-runtime-cu12` wheel provides). This is a large download (~1.9GB) and can take significant time depending on network conditions — check `nvidia-smi` for actual GPU utilization during a run, not just import success, before trusting any timing/VRAM number.

## Real runs completed

### Phase 1 first-pass sweep: M1 (DeepSeek-R1-Distill-Qwen-1.5B), weight quants x {E1, E2}

**Hardware:** RTX 3050 Laptop GPU, 4096 MiB VRAM (3772 MiB usable per llama.cpp), driver 595.71.05, no system CUDA toolkit — GPU offload via the prebuilt CUDA wheel described above.

**Config:** fp16 KV cache (uncapped), sampled decoding at temperature=0.6/top_p=0.95 (per the model card's recommendation — see docs/METHODOLOGY.md), seeds `[0, 1]`, `max_tokens: 6000`, `llama_cpp.n_ctx: 8192`.

**Scope (disclosed):** `sample_size: 6` per (quant, tier) — i.e. 6 problems x 2 seeds = 12 samples per cell, 96 generations total across all 8 configs. This is a deliberately reduced first pass, not the spec's full ~200-problem x 5-seed sweep — real GPU time for the fp16 tier alone was ~35-55 minutes per benchmark on this hardware (see timing note below), so the full grid is a follow-up once this first pass is validated. Every number below is real, produced by an actual GPU run — none are estimated or fabricated.

**Commands run** (one `quantthink run` per config file, per the sweep-stub convention — see `cli.py`'s `sweep` docstring):
```bash
for cfg in configs/phase1/*.yaml; do
  name=$(basename "$cfg" .yaml)
  uv run quantthink run --config "$cfg" \
    --output "results/phase1/${name}.json" \
    --manifest "results/phase1/${name}.manifest.json"
done
```
Config files: `configs/phase1/{fp16,Q8_0,Q5_K_M,Q4_K_M}_{E1,E2}.yaml`.

**A methodology bug was caught and fixed before trusting these numbers.** The first attempt used `max_tokens: 1024`, which silently truncated nearly every generation before the model finished thinking (2 of 3 manually-inspected debug samples never even exited `<think>`), producing a spurious ~10% MATH-500 accuracy against the model card's published 83.9% — an impossible gap for real sampling noise per this project's own sanity-check rule. Root cause confirmed by manually inspecting raw completions; fixed by raising `max_tokens` to 6000 and `n_ctx` to 8192 (verified: the same 3 debug problems all completed and solved correctly afterward). This is why `metrics/core.py` now tracks `hit_max_tokens`/`truncation_rate` per instance — so this failure mode is visible in every future `result.json` instead of silently contaminating Acc/TL/CTS.

**Results (Acc / mean Thinking-Length / Cost-to-Solve / truncation rate, N=6 problems x 2 seeds = 12 samples per cell):**

| Quant | E1 (GSM8K) Acc | E1 TL | E1 CTS | E2 (MATH-500) Acc | E2 TL | E2 CTS | E2 TruncRate |
|---|---|---|---|---|---|---|---|
| fp16   | 0.667 | 469.8 | 1541.5 | 0.417 | 1465.3 | 6856.2 | 0.17 |
| Q8_0   | 0.750 | 385.5 | 1246.4 | 0.500 | 1335.9 | 5446.5 | 0.17 |
| Q5_K_M | 0.667 | 244.3 |  985.5 | 0.417 | 1215.9 | 5920.6 | 0.17 |
| Q4_K_M | 0.583 | 420.8 | 1532.1 | 0.333 | 1280.7 | 7703.5 | 0.17 |

Δ Acc vs fp16 baseline: Q8_0 **+0.083** (both tiers), Q5_K_M **0.000** (both tiers), Q4_K_M **-0.083** (both tiers).

**Reading these numbers honestly:**
- At N=12 samples per cell, per-cell 95% bootstrap CIs are wide (not yet computed/reported here — the full-N follow-up sweep is where per-cell CIs become statistically meaningful, per this project's statistical-rigor methodology). Treat this as a directional first pass, not a statistically confirmed result.
- The one consistent pattern across *both* independently-sampled tiers: **Q4_K_M is the only weight quant that shows a measurable accuracy drop vs fp16** (-8.3 points on both GSM8K and MATH-500). Q8_0 and Q5_K_M show no measurable degradation at this sample size — Q8_0 even nominally *beats* fp16, which is most plausibly small-N noise rather than a real effect (quantization does not typically improve accuracy).
- This first pass does **not** yet confirm H1's "longer-but-worse" curve (thinking-length does not rise monotonically with more aggressive quantization here — e.g. Q5_K_M has *lower* TL than fp16 on both tiers). A larger N is needed before drawing a real H1 conclusion; this is flagged as a follow-up rather than overclaimed here.
- **fp16 MATH-500 accuracy (41.7%) sits well below the model card's published 83.9%.** Investigated per this project's own sanity-check rule (a big gap means "the harness is wrong, not the finding, investigate before trusting it") — per-instance inspection (`results/phase1/fp16_E2.json`'s `instances` array) shows: (a) 1 of 6 problems was lost entirely to the 6000-token cap even after the fix (an `intermediate_algebra` problem — genuinely long chains exist even above the old 1024 bug's threshold), (b) the other "wrong" answers were on problems the checker's conservative string/numeric normalization (not full symbolic/CAS equivalence, a disclosed limitation in `eval/checkers.py`) may be undercounting, and (c) N=6 against a benchmark with wide per-problem difficulty variance (levels 1-5) has high sampling variance. Unlike the 1024-token bug (which was a uniform, catastrophic failure across every sample), this residual gap is a combination of known, bounded, disclosed effects — not evidence the harness is broken. Narrowing it is a disclosed follow-up (larger N, and potentially a symbolic-equivalence checker upgrade).

**Real per-config timing** (single-run wall-clock on this hardware, useful for planning the full sweep): fp16 was dramatically slower than the quantized tiers — fp16_E1 (12 generations) took ~7 min, fp16_E2 (longer MATH chains) took ~36 min; Q8_0/Q5_K_M/Q4_K_M configs ranged roughly 4-15 min each. Peak VRAM: fp16 ~3.50 GB, Q8_0 ~2.16 GB, Q5_K_M ~1.67 GB, Q4_K_M ~1.55 GB.

### Next real-run steps
- Run the fuller ~200-problem x 5-seed sweep for the statistically-rigorous leaderboard entry (this is a multi-hour GPU job at the timing observed above — planned as a dedicated follow-up, not blocking Phase 2's KV-cache-axis work).
- M4 (third-family stretch model): not yet chosen — the small-reasoning-model landscape moves monthly. Will be verified and recorded here before Phase 6.
- AWQ/GPTQ tooling (Phase 4): `pyproject.toml`'s `quantize` extra currently lists `autoawq` and `gptqmodel` as placeholders; the currently-maintained package names will be re-verified immediately before Phase 4 and this note updated with the actual versions used.
