# Results Schema

Single source of truth for the columns in the published `quantthink-results` dataset. Checked against `src/quantthink/report/published.py`'s `RUNS_COLS`/`LEADERBOARD_COLS`.

## `runs.csv` — one row per real run

| Column | Meaning |
|---|---|
| `model` | Canonical model name (path/quant-suffix stripped) |
| `quant` | Weight quant level (`fp16`, `Q8_0`, `Q5_K_M`, `Q4_K_M`, `AWQ`, `GPTQ`) |
| `kv_quant` | KV-cache dtype (`fp16`, `Q8`, `Q4`) |
| `thinking_cap` | Max thinking-token budget, or empty for uncapped |
| `backend` | Inference backend (`llama-cpp`, `transformers`, `vllm`, `openai`, `mock`) |
| `tier` | Benchmark tier(s), `+`-joined (e.g. `E1`, `E1+E2`) |
| `seed` | RNG seed for this run |
| `sample_size` | Number of problems sampled |
| `acc` | Accuracy (pass@1) |
| `tl_mean` | Mean thinking-length (tokens) |
| `cts` | Cost-to-Solve, or empty if accuracy is 0 |
| `vram_gb` | Peak VRAM in GB |
| `git_commit`, `config_sha256`, `dataset_sha256`, `timestamp` | Reproducibility manifest fields |

## `leaderboard.csv` — aggregated over seeds

| Column | Meaning |
|---|---|
| `model`, `quant`, `kv_quant`, `thinking_cap`, `backend`, `tier` | Grouping key |
| `n_seeds` | Number of seeds aggregated |
| `acc_mean`, `acc_ci_low`, `acc_ci_high` | Accuracy + bootstrap 95% CI |
| `tl_mean` | Mean thinking-length across seeds |
| `cts_mean` | Mean Cost-to-Solve across seeds |
| `vram_gb` | Mean peak VRAM |
| `eta` | Acc / VRAM efficiency score |
| `delta_acc_rel`, `delta_cts_rel` | Relative Δ vs. the scope's baseline quant |
| `baseline_quant` | Which quant was used as the Δ baseline in this scope |
