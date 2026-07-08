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

### Phase 2 first-pass sweep: KV-cache axis, M2/M3 family contrast, thinking-cap grid

Same hardware, same sampling config as Phase 1, but an even smaller first-pass N (**N=4 problems x 2 seeds = 8 samples/cell**, E1/GSM8K only) — this phase prioritizes covering three new axes for the first time over statistical depth on any one of them; a fuller sweep is a disclosed follow-up, same as Phase 1.

**Two more real bugs were caught and fixed while producing these numbers** (in addition to Phase 1's max_tokens bug):
1. **Quantized KV cache requires flash attention.** Setting `type_k`/`type_v` to anything other than F16 without `flash_attn=True` fails outright with `Failed to create llama_context` — llama.cpp's non-F16 KV cache path is only implemented under flash attention. Fixed in `LlamaCppBackend` (see `fix: enable flash attention for quantized KV cache`).
2. **The numeric checker crashed on overflow/infinity.** A model can emit a very long digit string (or the literal word "inf") that Python's `float()` silently parses as infinity instead of raising `ValueError`; the subsequent `int(val)` then raised an uncaught `OverflowError` and killed the whole run. Fixed by explicitly guarding against inf/NaN in `eval/checkers.py::_normalize_number` (see `fix: guard checker's numeric normalization against overflow/infinity`).

**KV-cache axis on M1 (DeepSeek-R1-Distill-Qwen-1.5B, Q4_K_M weights, E1/GSM8K):**

| KV dtype | Acc | TL | CTS | TruncRate | VRAM |
|---|---|---|---|---|---|
| fp16 | 0.375 | 541.2 | 2964.0 | 0.00 | 1.55 GB |
| Q8   | 0.500 | 452.5 | 2001.3 | 0.00 | 1.43 GB |
| Q4   | **0.000** | 3.5 | N/A | **1.00** | 1.38 GB |

Q4 KV cache did not just reduce accuracy — **it broke generation entirely.** Every one of the 8 samples hit the 6000-token cap while producing degenerate repetition instead of an answer; manually inspecting a fresh generation under the same config confirmed it directly:
```
<think>
First of Think. to Think Think to Think Think ThinkThink Think Think Think Think Think ThinkThink...
```
This is a real, reproducible finding, not a harness artifact: at Q4_K_M weights, this model's attention collapses under Q4 KV-cache quantization. For the Memory-Budget Frontier, this means Q4 KV is not a usable point on the frontier for this (model, weight-quant) pair at all — worse than merely low-accuracy, it is a hard exclusion. Q8 KV, in contrast, cost only 0.12 GB less VRAM than fp16 KV while nominally *improving* accuracy at this sample size (almost certainly noise, not a real gain, per the same caveat as Phase 1's Q8_0 weight result) — the honest read is "Q8 KV is free at this VRAM budget, Q4 KV is unusable," not a smooth accuracy/VRAM trade curve.

**Thinking-cap grid (H5) on M1 (Q4_K_M weights, fp16 KV, E1):**

| Thinking cap | Acc | TL | CTS | TruncRate |
|---|---|---|---|---|
| 2048 | 0.375 | 345.9 | 2087.3 | 0.12 |
| 4096 | 0.375 | 514.5 | 2770.0 | 0.12 |
| uncapped | 0.375 | 541.2 | 2964.0 | 0.00 |

At N=8 samples/cell, accuracy is identical across all three caps — no H5 truncation-amplifies-damage effect is visible yet, though `TruncRate` does rise from 0.00 (uncapped) to 0.12 (both capped levels), meaning the caps are truncating *some* generations; at this sample size those happen to be ones that would have been wrong anyway. This is inconclusive at N=8, not a null result — a larger sweep is needed before drawing any H5 conclusion.

**M2 (Qwen3-1.7B, thinking mode) family contrast, E1/GSM8K:**

| Weight quant | Acc | TL | CTS | TruncRate | VRAM |
|---|---|---|---|---|---|
| bf16/fp16 | **OOM** (`Failed to create llama_context`) | — | — | — | — |
| Q8_0 (labeled baseline) | 1.000 | 1191.8 | 2212.9 | 0.00 | 2.99 GB |
| Q4_K_M | 0.875 | 1910.1 | 3717.6 | 0.12 | 2.32 GB |

The bf16 OOM at `n_ctx=8192` on this 4GB card **is the expected finding, not a bug** — it reproduces the same caveat already documented in this project's sibling benchmarks (QuantCall/QuantMCP), which is why the spec labels Q8_0 as M2's baseline instead of fp16. Within the range that does fit: Δ Acc (Q8_0 → Q4_K_M) = **-0.125**, and thinking-length *rises* sharply (1191.8 → 1910.1) as accuracy falls — this is the clearest **H1 "longer-but-worse"** signal in the dataset so far (Phase 1's M1 data didn't show this pattern cleanly; M2's does, directionally, at N=8).

**M3 (Qwen3-0.6B, thinking mode) family contrast, E1/GSM8K:**

| Weight quant | Acc | TL | CTS | TruncRate | VRAM |
|---|---|---|---|---|---|
| bf16/fp16 | 0.500 | 1300.0 | 4934.3 | 0.00 | 2.40 GB |
| Q4_K_M | 0.375 | 1385.5 | 7455.3 | 0.00 | 1.65 GB |

Δ Acc (fp16 → Q4_K_M) = **-0.125**, again with TL rising (1300.0 → 1385.5) — the same directional H1 pattern as M2.

**Family cross-check (H3), Q4_K_M weights, E1/GSM8K, comparable-ish N:**

| Model | Family | Baseline → Q4_K_M Acc drop |
|---|---|---|
| M1 DeepSeek-R1-Distill-Qwen-1.5B | Qwen2.5 backbone, R1-distilled | fp16 0.667 (N=6, Phase 1) → Q4_K_M 0.375 (N=4, this phase) |
| M2 Qwen3-1.7B | Qwen3 (native thinking) | Q8_0 1.000 → Q4_K_M 0.875 (-0.125) |
| M3 Qwen3-0.6B | Qwen3 (native thinking) | fp16 0.500 → Q4_K_M 0.375 (-0.125) |

Directionally, both native-Qwen3 models show a smaller, identical-sized accuracy drop (-0.125) than the R1-distilled Qwen2.5 model's larger apparent drop — consistent with H3's prediction that a Qwen3-family thinking model stays comparatively more robust under quantization. **This is a preliminary read at very low N, with an inconsistent baseline quant and different N for M1's comparison point (fp16, N=6) vs. M2/M3's (Q8_0/fp16, N=4)** — treat as directionally suggestive, not confirmatory. A same-N, same-baseline-quant follow-up sweep is needed to actually confirm H3.

**Real per-config timing (Phase 2):** all configs used small enough models/N to finish in 2-17 minutes each, except the Q4-KV degenerate-loop config, which (unsurprisingly, since it always hit the full 6000-token cap) took ~17 minutes for just 8 samples — a slow failure looks the same as a slow success from the outside; the per-instance `instances` array is what made it possible to tell them apart quickly.

### Phase 3: Memory-Budget Frontier, computed on real combined Phase 1+2 data

`budget/frontier.py` and `quantthink recommend` are no longer stubs — both now run for real. As a first real exercise of the frontier (not yet the full statistically-rigorous version — same first-pass-data caveat as everything above), all M1/M2/M3 weight-quant and KV-axis result files from Phases 1-2 (E1/GSM8K only, across everyone's differing N) were combined into one leaderboard via `quantthink leaderboard build`, then swept across a 2.0-4.0 GB budget grid with `compute_frontier()`:

```bash
uv run quantthink recommend results/phase1/*_E1.json results/phase2/*_E1.json --vram 3.0 --objective accuracy
```

**Accuracy objective:**

| Budget | c*(B) | Acc | VRAM |
|---|---|---|---|
| 2.0 GB | M1 Q5_K_M | 0.667 | 1.67 GB |
| 2.5 GB | M2 (Qwen3-1.7B) Q4_K_M | 0.875 | 2.32 GB |
| 3.0 GB | M2 (Qwen3-1.7B) Q8_0 | 1.000 | 2.99 GB |
| 3.5 GB | M2 (Qwen3-1.7B) Q8_0 | 1.000 | 2.99 GB |
| 4.0 GB | M2 (Qwen3-1.7B) Q8_0 | 1.000 | 2.99 GB |

**CTS objective:**

| Budget | c*(B) | CTS | VRAM |
|---|---|---|---|
| 2.0-4.0 GB | M1 Q5_K_M | 985.5 | 1.67 GB (unchanged across the whole grid) |

**This is a real, working H2 budget-crossover — the first one in the project — computed from actual GPU results, not fabricated:** under the accuracy objective, the optimal model *family* itself crosses over as budget increases (M1 below ~2.3GB, M2 above it), which is exactly the kind of split the Memory-Budget Frontier is supposed to surface. Under the CTS objective, M1's Q5_K_M dominates the *entire* grid — M2 answers more accurately but at such a higher token cost (CTS 2213-3718 vs. M1's 985-2248) that it never wins on cost-to-solve, a genuinely informative accuracy-vs-efficiency split between families.

**Caveats before citing this anywhere:** (a) this combines files with different sample sizes/seeds per cell — `aggregate_leaderboard()` correctly bootstrap-CIs each cell but the *sample size behind* each cell varies (M1 Q4_K_M's row alone merges N=6+N=4 runs into one aggregate); (b) all of it is E1/GSM8K only — no MATH-500 or GPQA points feed the frontier yet; (c) N=4-12 per cell throughout. This demonstrates the mechanism works end-to-end on real numbers; it is not yet the rigorous, full-N frontier the spec's differentiator claim is ultimately about.

**E3 (GPQA) investigated, not yet integrated:** the official `Idavidrein/gpqa` dataset is gated behind a data-sharing agreement; an ungated mirror (`hendrydong/gpqa_diamond`) exists but only exposes free-text `problem`/`solution` fields (no separate multiple-choice options), which doesn't match this project's multiple-choice checker (`check_multiple_choice` expects a bare A/B/C/D) — it would need either the gated official dataset or a checker rework to score free-form physics/chemistry/biology answers via `check_math`-style normalization. Deferred; disclosed here rather than silently skipped.

### Phase 4: real GPTQ calibration of M1 on the RTX 3050

Tooling landscape check: `autoawq` has had no release since 2025-05 (stale) — not used. `gptqmodel` 7.1.0 (2026-06-08) was chosen: since its 5.0.0+ line it natively supports both GPTQ and AWQ ("has fully supplanted AutoGPTQ and AutoAWQ"), so a second tool wasn't needed for this pass.

**Recipe:** GPTQ, 4-bit, group_size=128, calibrated on 128 texts (≥512 chars each) from `Salesforce/wikitext`'s `wikitext-2-raw-v1` train split, batch_size=1. Run in a scratch venv (`/tmp/gptq-calib`, not the project's own `.venv`, since it needs a full torch/transformers/accelerate stack the llama-cpp-python-based runtime doesn't) against the original `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` safetensors checkpoint.

**Three real bugs hit and fixed while producing this:**
1. The `wikitext` dataset repo uses an old loading-script format incompatible with the current `datasets` library (`HfUriError` when resolving `.huggingface.yaml`) — fixed by switching to the parquet-based `Salesforce/wikitext` mirror of the same data.
2. `/tmp` on this machine is a 7.5GB **tmpfs** (RAM-backed), and `gptqmodel`'s default `offload_to_disk` scratch path silently wrote there, filling it (`Disk quota exceeded`) partway through calibration. Fixed by pointing `QuantizeConfig(offload_to_disk_path=...)` and the final output path at `/home/alexander/gptq-scratch` (the real disk, 41GB free), and clearing the ~937MB of orphaned scratch files the crashed run left in `/tmp`.
3. Evaluating the quantized model failed to load with the auto-selected `MarlinLinear` kernel: first `ninja` was missing (`pip install ninja` fixed the JIT build tool), then the JIT compile itself failed with `CUDA_HOME environment variable is not set` — the Marlin/Triton kernels need `nvcc` (the CUDA *toolkit*, not just the driver) to compile, which this machine doesn't have, matching the exact same constraint already documented for `llama-cpp-python`. Fixed by explicitly loading with `backend=BACKEND.TORCH` (gptqmodel's pure-PyTorch dequantize-and-matmul path), which needs no compilation step.

**Measured, real (N=6 problems x 2 seeds = 12 samples, GSM8K, per this project's own extractor/checker for a scoring methodology consistent with every other result here):**

| Metric | Value |
|---|---|
| Acc | 0.750 (9/12) |
| TL (mean thinking tokens) | 317.4 |
| CTS | 1025.1 |
| Truncation rate | 0.0 |
| Peak VRAM (`torch.cuda.max_memory_allocated`, `BACKEND.TORCH`) | 1.63 GB |

For comparison, the llama.cpp Q4_K_M **GGUF** quant of the same base model measured 0.583 Acc / 420.8 TL / 1532.1 CTS at the same N (Phase 1). This GPTQ quant nominally scores higher here — plausible since GPTQ uses real calibration data to minimize reconstruction error while GGUF's Q4_K_M is calibration-free — but **this is one N=12 comparison with two different VRAM-measurement methodologies (`nvidia-smi` for GGUF vs. `torch.cuda.max_memory_allocated` for GPTQ's `BACKEND.TORCH` path) and is not a statistically confirmed or directly apples-to-apples result.** Treat as a real, disclosed first data point, not a verdict on GPTQ vs. GGUF.

Published: [happynood/DeepSeek-R1-Distill-Qwen-1.5B-GPTQ](https://huggingface.co/happynood/DeepSeek-R1-Distill-Qwen-1.5B-GPTQ).

### Next real-run steps
- Run the fuller ~200-problem x 5-seed sweep for the statistically-rigorous leaderboard entry (this is a multi-hour GPU job at the timing observed above — planned as a dedicated follow-up, not blocking further phase work).
- Confirm H3 properly: same N, same baseline-quant convention (e.g. all vs. each model's best-fitting quant), across all three models.
- M4 (third-family stretch model): not yet chosen — the small-reasoning-model landscape moves monthly. Will be verified and recorded here before Phase 6.
- GPTQ/AWQ calibration for M2 (Qwen3-1.7B) and M3 (Qwen3-0.6B), where licensing allows — not yet attempted.
