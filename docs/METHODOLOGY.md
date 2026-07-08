# Methodology

## Metrics

- **Acc (pass@1)** — fraction of (problem, seed) pairs where the extracted final answer matches ground truth, averaged over K seeds per problem.
- **TL (Thinking-Length)** — mean `<think>...</think>` token count, reported overall and split by solved/unsolved.
- **CTS (Cost-to-Solve)** — mean total generated tokens divided by Acc; the expected token cost per correct answer. Undefined (reported as `null`) when Acc is 0.
- **Δ / Δrel** — absolute and relative change vs. the fp16-weights / fp16-KV / uncapped baseline for a given (model, backend, tier) scope, or an explicitly labeled fallback baseline where fp16 does not fit in 4GB.
- **η (efficiency)** — Acc divided by peak VRAM (GB).

## Sampling: sampled, not greedy

Unlike this project's sibling benchmarks (which default to greedy, temperature=0 decoding), QuantThink runs reasoning models with each model's creator-recommended sampling (typically temperature ≈ 0.6, top-p ≈ 0.95 — verified per model card at build time) and a **fixed set of K seeds**, because reasoning models are known to degrade under greedy decoding. This is a deliberate, disclosed departure. A greedy control run (`greedy: true` in config) is kept for at least one model to show the size of the gap.

## Checkers

All correctness checks are deterministic and judge-free:
- GSM8K: extracts the `#### N` ground truth and a `\boxed{}` or trailing-number model answer, compares as normalized numbers.
- MATH-500: extracts `\boxed{}` on both sides, with light LaTeX/whitespace normalization (not full symbolic/CAS equivalence — a disclosed, conservative scope limit; see `src/quantthink/eval/checkers.py`).
- GPQA-style multiple choice: extracts a bare A/B/C/D letter.

No LLM-as-judge and no external API calls are used anywhere in scoring.

## Reproducibility

Every result ships a manifest: git commit, config hash, eval-suite (dataset) hash, seed set, KV-cache dtype, thinking-token cap, and a hardware fingerprint (GPU name, driver, CUDA version). Bootstrap 95% confidence intervals are computed on every aggregate metric and every Δ.

## Hypotheses under test

- **H1** — more aggressive weight quantization decreases accuracy while increasing mean thinking-token length.
- **H2** — at a fixed peak-VRAM budget, the accuracy-optimal split between weight-bits and KV-cache-bits shifts with effective model size.
- **H3** — model family predicts quantization sensitivity better than parameter count (cross-checked against this project's sibling benchmarks).
- **H4** — ranking configs by Cost-to-Solve instead of raw accuracy reorders the leaderboard.
- **H5** — a fixed thinking-token cap hurts quantized models more than the fp16 baseline.

Real numbers for each hypothesis are added here as sweeps complete — see `docs/RUN_REAL.md`.
