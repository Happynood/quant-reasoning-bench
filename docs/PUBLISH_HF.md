# HuggingFace Publishing Status

## Live now

- **[happynood/quantthink-suite](https://huggingface.co/datasets/happynood/quantthink-suite)** — frozen GSM8K (200) + MATH-500 (200) eval subsets, seed 42.
- **[happynood/quantthink-results](https://huggingface.co/datasets/happynood/quantthink-results)** — real Phase 1-2 `result.json` files + aggregated leaderboard.
- **[happynood/DeepSeek-R1-Distill-Qwen-1.5B-GGUF](https://huggingface.co/happynood/DeepSeek-R1-Distill-Qwen-1.5B-GGUF)** — GGUF quants (fp16/Q8_0/Q5_K_M/Q4_K_M), re-hosted unmodified from [bartowski's original conversion](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF), with this project's own measured Acc/TL/CTS per quant in the model card.

## Deferred: `quantthink-leaderboard` Gradio Space

Creating a **new** Gradio/Docker Space on free `cpu-basic` hardware currently returns `HTTP 402 Payment Required` — HuggingFace now requires a PRO subscription for this, which is an account/billing decision, not something this build should spend real money on unprompted. The existing sibling Spaces (`quantcall-leaderboard`, `quantmcp-leaderboard`) were created under different account/billing conditions and are unaffected.

**Options for the human to unblock this later:**
1. Subscribe to HF PRO, then re-run the publish step (`hf repo create happynood/quantthink-leaderboard --repo-type space --space-sdk gradio`).
2. Duplicate an existing Space instead of creating fresh: `hf repo duplicate happynood/quantcall-leaderboard happynood/quantthink-leaderboard --repo-type space`, then replace `app.py`/`README.md`/`requirements.txt`.
3. Host the Gradio app elsewhere (any environment that can run `pip install gradio pandas plotly huggingface_hub && python app.py`) and link to it from the GitHub README instead of an HF Space.

No app code exists yet for this Space — writing it is a same-day task once one of the above is unblocked: reuse the sibling projects' own `app.py` pattern (Gradio Blocks, tabs, Plotly charts pulling CSV/JSON from `quantthink-results` via `hf_hub_download`), and add the three QuantThink-specific panels: the H1 "longer-but-worse" scatter (Acc vs TL), the H2 Memory-Budget Frontier (c*(B) as VRAM budget slides 2→4GB), and the H4 CTS-reordering comparison (leaderboard sorted by Acc vs. by CTS side by side).

## AWQ/GPTQ calibrated models

Tooling landscape check (this build, per the project's dependency-verification rule): `autoawq` has not had a release since 2025-05 and should be treated as stale/unmaintained. `gptqmodel` and `llmcompressor` (the vLLM project's unified quantization tool, also covers AWQ-style recipes) are the current, actively-maintained options — see this file's own updates below once calibration is actually attempted for the chosen tool, version, and why.

Status: not yet attempted as of this writing — see the rest of this file (updated in place once a real calibration run happens) or `docs/RUN_REAL.md` for the outcome.
