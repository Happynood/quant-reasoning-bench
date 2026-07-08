# QuantThink

**Does quantization break *reasoning*?** A reproducible benchmark measuring how weight- and KV-cache quantization degrade small reasoning (long chain-of-thought) models on a 4GB consumer GPU — and, given a fixed VRAM budget, which quantization split actually maximizes accuracy.

> QuantCall/QuantMCP measured whether quantization breaks tool-calling. This measures whether it breaks reasoning — and, unlike them, it doesn't just report the damage, it tells you the accuracy-optimal config for your VRAM budget.

[![CI](https://github.com/Happynood/quant-reasoning-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/Happynood/quant-reasoning-bench/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Status

Early build — skeleton and mock-backend pipeline are in place; the first real GPU sweep (H1, the "longer-but-worse" curve on DeepSeek-R1-Distill-Qwen-1.5B) is next. This section will be replaced with the real headline result once it exists.

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

## `quantthink recommend` (ships in Phase 3)

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
