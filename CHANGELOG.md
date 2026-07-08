# Changelog

## [Unreleased]

### Added
- Phase 0: Project skeleton — Pydantic v2 config, Mock/llama.cpp/transformers/OpenAI/vLLM backends, thinking-segment extractor, judge-free checkers, Acc/TL/CTS metrics engine, E0 toy smoke eval, Click CLI, Makefile verify gate, CI (Python 3.11-3.13)
- Phase 1: Frozen GSM8K/MATH-500 eval subsets; real weight-quant sweep (fp16/Q8_0/Q5_K_M/Q4_K_M) on DeepSeek-R1-Distill-Qwen-1.5B; per-instance truncation-rate tracking
- Phase 2: KV-cache quantization axis, thinking-token-cap grid, Qwen3-1.7B/0.6B family-contrast sweep; flash-attention requirement for quantized KV cache
- Phase 3: Memory-Budget Frontier (`budget/frontier.py`) and a working `quantthink recommend`, reusing the vendored Pareto-front logic
- Phase 4: Published `quantthink-suite`/`quantthink-results` HF datasets, a benchmarked GGUF model card, and a real self-calibrated GPTQ 4-bit quantization (via `gptqmodel`)
- Phase 5: Public GitHub repository, CI verified green on Python 3.11/3.12/3.13
