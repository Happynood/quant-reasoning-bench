# Contributing

## Submitting Benchmark Results

To add a model/hardware config to the leaderboard:

1. Run the evaluation on your own hardware:
   ```bash
   quantthink run \
     --config configs/your_config.yaml \
     --output results/your_model_quant.json \
     --manifest results/your_model_quant.manifest.json
   ```

2. Verify the result file includes a manifest (git SHA, config hash, seed set, KV-cache dtype, thinking-cap, hardware fingerprint).

3. Open a PR adding only `results/your_model_quant.json` and `results/your_model_quant.manifest.json`. Do not edit the leaderboard table manually — it is generated from result files.

CI validates new results structurally on the mock backend; a maintainer reviews the manifest before merge.

## Code Contributions

### Setup

```bash
git clone https://github.com/Happynood/quant-reasoning-bench
cd quant-reasoning-bench
pip install uv
uv sync --dev
```

### Workflow

1. Fork and branch from `main`.
2. Write tests first (TDD) for any new behavior.
3. Run `make verify` locally before opening a PR — it must be green.
4. Keep PRs focused: one logical change per PR.
5. Follow Conventional Commits for commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`, `perf:`, `ci:`).

### Adding a new benchmark tier

New eval tiers live in `src/quantthink/eval/loader.py` with a deterministic, judge-free checker in `src/quantthink/eval/checkers.py`. Frozen subsets (fixed indices) belong in the `quantthink-suite` HF dataset, not hardcoded in the repo.

### Adding a new backend

Implement the `Backend` ABC in `src/quantthink/backends/base.py`. See `mock.py` for the minimal reference implementation and `llama_cpp.py` for the full case (weight quant + KV-cache dtype + VRAM measurement).
