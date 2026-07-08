# HuggingFace Publishing Status

## Live now

- **[happynood/quantthink-suite](https://huggingface.co/datasets/happynood/quantthink-suite)** — frozen GSM8K (200) + MATH-500 (200) eval subsets, seed 42.
- **[happynood/quantthink-results](https://huggingface.co/datasets/happynood/quantthink-results)** — real Phase 1-2 `result.json` files + aggregated leaderboard.
- **[happynood/DeepSeek-R1-Distill-Qwen-1.5B-GGUF](https://huggingface.co/happynood/DeepSeek-R1-Distill-Qwen-1.5B-GGUF)** — GGUF quants (fp16/Q8_0/Q5_K_M/Q4_K_M), re-hosted unmodified from [bartowski's original conversion](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF), with this project's own measured Acc/TL/CTS per quant in the model card.
- **[happynood/DeepSeek-R1-Distill-Qwen-1.5B-GPTQ](https://huggingface.co/happynood/DeepSeek-R1-Distill-Qwen-1.5B-GPTQ)** — a real, self-calibrated 4-bit GPTQ quantization (not re-hosted from anyone else), calibrated on this project's own RTX 3050 via `gptqmodel` 7.1.0. See "AWQ/GPTQ calibrated models" below for the full recipe and measured numbers.

## Deferred: `quantthink-leaderboard` Gradio Space

Creating a **new** Gradio/Docker Space on free `cpu-basic` hardware currently returns `HTTP 402 Payment Required` — HuggingFace now requires a PRO subscription for this, which is an account/billing decision, not something this build should spend real money on unprompted. The existing sibling Spaces (`quantcall-leaderboard`, `quantmcp-leaderboard`) were created under different account/billing conditions and are unaffected.

**Options for the human to unblock this later:**
1. Subscribe to HF PRO, then re-run the publish step (`hf repo create happynood/quantthink-leaderboard --repo-type space --space-sdk gradio`).
2. Duplicate an existing Space instead of creating fresh: `hf repo duplicate happynood/quantcall-leaderboard happynood/quantthink-leaderboard --repo-type space`, then replace `app.py`/`README.md`/`requirements.txt`.
3. Host the Gradio app elsewhere (any environment that can run `pip install gradio pandas plotly huggingface_hub && python app.py`) and link to it from the GitHub README instead of an HF Space.

No app code exists yet for this Space — writing it is a same-day task once one of the above is unblocked: reuse the sibling projects' own `app.py` pattern (Gradio Blocks, tabs, Plotly charts pulling CSV/JSON from `quantthink-results` via `hf_hub_download`), and add the three QuantThink-specific panels: the H1 "longer-but-worse" scatter (Acc vs TL), the H2 Memory-Budget Frontier (c*(B) as VRAM budget slides 2→4GB), and the H4 CTS-reordering comparison (leaderboard sorted by Acc vs. by CTS side by side).

## AWQ/GPTQ calibrated models

Tooling landscape check (per the project's dependency-verification rule): `autoawq` has not had a release since 2025-05 and is treated as stale/unmaintained — not used. `gptqmodel` 7.1.0 (released 2026-06-08, actively maintained) was chosen: as of its 5.0.0+ releases it has "fully supplanted AutoGPTQ and AutoAWQ," supporting both GPTQ and AWQ natively in one library, so a second tool (`llmcompressor`) wasn't needed for this pass.

**Status: DONE for M1.** Real GPTQ 4-bit calibration completed on the RTX 3050 and pushed to [happynood/DeepSeek-R1-Distill-Qwen-1.5B-GPTQ](https://huggingface.co/happynood/DeepSeek-R1-Distill-Qwen-1.5B-GPTQ) — see `docs/RUN_REAL.md` for the full recipe, real bugs hit and fixed along the way (a stale `wikitext` dataset loader, a full `/tmp` tmpfs, a missing `ninja`/CUDA-toolkit dependency for the Marlin kernel), and the measured Acc/TL/CTS.

M2/M3 (Qwen3-1.7B/0.6B) AWQ/GPTQ calibration: not yet attempted — a reasonable next step, not a blocker (see STATE.md for scope/time tradeoffs already made this build).
