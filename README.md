# QuantThink

**Does quantization break *reasoning*?** A reproducible benchmark measuring how weight- and KV-cache quantization degrade small reasoning (long chain-of-thought) models on a 4GB consumer GPU — and, given a fixed VRAM budget, which quantization split actually maximizes accuracy.

> QuantCall/QuantMCP measured whether quantization breaks tool-calling. This measures whether it breaks reasoning — and, unlike them, it doesn't just report the damage, it tells you the accuracy-optimal config for your VRAM budget.

[![CI](https://github.com/Happynood/quant-reasoning-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/Happynood/quant-reasoning-bench/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Results dataset](https://img.shields.io/badge/🤗%20Dataset-Results-blue)](https://huggingface.co/datasets/happynood/quantthink-results)
[![Suite dataset](https://img.shields.io/badge/🤗%20Dataset-Suite-blue)](https://huggingface.co/datasets/happynood/quantthink-suite)
[![GGUF model](https://img.shields.io/badge/🤗%20Model-GGUF-green)](https://huggingface.co/happynood/DeepSeek-R1-Distill-Qwen-1.5B-GGUF)
[![GPTQ model](https://img.shields.io/badge/🤗%20Model-GPTQ-green)](https://huggingface.co/happynood/DeepSeek-R1-Distill-Qwen-1.5B-GPTQ)

## Status

**A real Memory-Budget Frontier, computed from actual GPU runs on an RTX 3050 Laptop (4GB VRAM):** under the accuracy objective, the optimal model *family* itself crosses over as your VRAM budget grows — DeepSeek-R1-Distill-Qwen-1.5B wins below ~2.3GB, Qwen3-1.7B wins above it. Under the Cost-to-Solve objective, the smaller model wins the *entire* 2-4GB grid, because the larger model's accuracy gain doesn't pay for its much higher token cost. Separately, Q4 KV-cache quantization was found to cause **total generation collapse** (not smooth degradation) for the R1-distill model at Q4_K_M weights — a real, reproducible finding, confirmed by inspecting raw model output.

This is a first-pass, disclosed-small-N result (N=4-12 samples per cell) — real, not fabricated, but not yet the full statistically-rigorous sweep. See [docs/RUN_REAL.md](docs/RUN_REAL.md) for the complete write-up, every caveat, and three real methodology bugs found and fixed along the way.

## Leaderboard (GSM8K, real GPU results)

| Model | Quant | Acc | TL | CTS | VRAM |
|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | fp16 | 0.667 | 469.8 | 1541.5 | 3.50 GB |
| DeepSeek-R1-Distill-Qwen-1.5B | Q8_0 | 0.750 | 385.5 | 1246.4 | 2.16 GB |
| DeepSeek-R1-Distill-Qwen-1.5B | Q5_K_M | 0.667 | 244.3 | 985.5 | 1.67 GB |
| DeepSeek-R1-Distill-Qwen-1.5B | Q4_K_M | 0.583 | 420.8 | 1532.1 | 1.55 GB |
| DeepSeek-R1-Distill-Qwen-1.5B | **GPTQ 4-bit (self-calibrated)** | 0.750 | 317.4 | 1025.1 | 1.63 GB* |
| Qwen3-1.7B (thinking) | Q8_0 | 1.000 | 1191.8 | 2212.9 | 2.99 GB |
| Qwen3-1.7B (thinking) | Q4_K_M | 0.875 | 1910.1 | 3717.6 | 2.32 GB |
| Qwen3-0.6B (thinking) | fp16 | 0.500 | 1300.0 | 4934.3 | 2.40 GB |
| Qwen3-0.6B (thinking) | Q4_K_M | 0.375 | 1385.5 | 7455.3 | 1.65 GB |

\* GPTQ VRAM measured via `torch.cuda.max_memory_allocated`, not directly comparable to the `nvidia-smi`-based GGUF measurements. Full table (including MATH-500, KV-cache axis, thinking-cap grid) in [docs/RUN_REAL.md](docs/RUN_REAL.md).

## Published artifacts

- [happynood/quantthink-suite](https://huggingface.co/datasets/happynood/quantthink-suite) — frozen eval subsets
- [happynood/quantthink-results](https://huggingface.co/datasets/happynood/quantthink-results) — real result.json + leaderboard
- [happynood/DeepSeek-R1-Distill-Qwen-1.5B-GGUF](https://huggingface.co/happynood/DeepSeek-R1-Distill-Qwen-1.5B-GGUF) — benchmarked GGUF quants
- [happynood/DeepSeek-R1-Distill-Qwen-1.5B-GPTQ](https://huggingface.co/happynood/DeepSeek-R1-Distill-Qwen-1.5B-GPTQ) — real, self-calibrated 4-bit GPTQ

## Quickstart (mock backend, zero GPU)

```bash
git clone https://github.com/Happynood/quant-reasoning-bench
cd quant-reasoning-bench
pip install uv
uv sync --dev
uv run quantthink run --config configs/smoke.yaml
```

## What it measures

- **Acc** — pass@1 over frozen benchmark subsets (GSM8K, MATH-500, GPQA-Diamond, AIME, LiveCodeBench), sampled decoding with a fixed seed set (not greedy — reasoning models degrade under greedy decoding).
- **TL (Thinking-Length)** — mean `<think>...</think>` token count, split by solved/unsolved.
- **CTS (Cost-to-Solve)** — expected tokens spent per *correct* answer; the honest efficiency metric that captures both fronts of the quantization tax (lower accuracy, longer generation).
- **Memory-Budget Frontier** — for a target VRAM budget B, the accuracy- (or CTS-) optimal (weight-quant, KV-quant, thinking-cap) config, via `quantthink recommend`.

See `docs/METHODOLOGY.md` for the full metric definitions and hypotheses (H1-H5).

## `quantthink recommend`

```bash
quantthink recommend results/*.json --vram 4.0 --objective accuracy
quantthink recommend results/*.json --vram 3.0 --objective cts
```

## Methodology

- Judge-free, deterministic checkers only (GSM8K `####`, `\boxed{}`, multiple-choice letter match) — no LLM-as-judge, no external API calls.
- Every result ships a manifest: git SHA, config hash, eval-suite hash, seed set, model revision, KV-cache dtype, thinking-cap, and hardware fingerprint.
- Bootstrap 95% CI on every metric and every Δ.

## Contributing

Submit your own model/hardware results via PR — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Related projects

- [quant-toolcall-bench (QuantCall)](https://github.com/Happynood/quant-toolcall-bench) — does quantization break tool-calling?
- [quant-mcp-bench (QuantMCP)](https://github.com/Happynood/quant-mcp-bench) — does quantization survive real MCP tool schemas?
- [llm-inference-benchmark](https://github.com/Happynood/llm-inference-benchmark) — the inference backend/recommender this project builds on.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
