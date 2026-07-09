# HuggingFace Publishing Status

## Live now

- **[happynood/quantthink-suite](https://huggingface.co/datasets/happynood/quantthink-suite)** — frozen GSM8K (200) + MATH-500 (200) eval subsets, seed 42.
- **[happynood/quantthink-results](https://huggingface.co/datasets/happynood/quantthink-results)** — real Phase 1-2 `result.json` files + aggregated leaderboard.
- **[happynood/DeepSeek-R1-Distill-Qwen-1.5B-GGUF](https://huggingface.co/happynood/DeepSeek-R1-Distill-Qwen-1.5B-GGUF)** — GGUF quants (fp16/Q8_0/Q5_K_M/Q4_K_M), re-hosted unmodified from [bartowski's original conversion](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF), with this project's own measured Acc/TL/CTS per quant in the model card.
- **[happynood/DeepSeek-R1-Distill-Qwen-1.5B-GPTQ](https://huggingface.co/happynood/DeepSeek-R1-Distill-Qwen-1.5B-GPTQ)** — a real, self-calibrated 4-bit GPTQ quantization (not re-hosted from anyone else), calibrated on this project's own RTX 3050 via `gptqmodel` 7.1.0. See "AWQ/GPTQ calibrated models" below for the full recipe and measured numbers.
- **[happynood/quantthink-leaderboard](https://huggingface.co/spaces/happynood/quantthink-leaderboard)** — static-SDK Space showing the real GSM8K leaderboard table and the H2 Memory-Budget Frontier finding, linking out to the datasets/models above.

## `quantthink-leaderboard` Space: static, not Gradio

Both **creating** a new Gradio/Docker Space on `cpu-basic` and **duplicating**
an existing one (`hf repos duplicate`) return `HTTP 402 Payment Required` —
HuggingFace requires a PRO subscription for either path on this account. The
sibling Spaces (`quantcall-leaderboard`, `quantmcp-leaderboard`) were created
under different account/billing conditions and are unaffected, but neither
create nor duplicate is available here without PRO.

Static Spaces (`--space-sdk static`) are free regardless, so the Space was
built as a single static HTML page instead: `hf repos create
happynood/quantthink-leaderboard --type space --space-sdk static`, then
content was pushed via a direct `git clone`/`git push` against the Space's
git remote — `hf upload`'s CLI path calls `repos/create` unconditionally
(even for an already-existing repo) and hits the same 402, so it can't be
used to push files to a static Space on this account either.

The page (`index.html`) hand-renders the same GSM8K leaderboard table and H2
Memory-Budget Frontier callout that live in the main README, plus links to
the datasets/models/GitHub repo. It has no live data-fetching (no
`hf_hub_download`, no charts) — reasonable for the current small-N result
set, but if the sweep grows (see STATE.md priority 2) this could be revisited
as a Gradio app (once/if PRO is available) for the H1/H4 interactive panels
described in earlier drafts of this doc.

**To upgrade later if PRO becomes available:** `hf repos delete
happynood/quantthink-leaderboard --type space`, then recreate with
`--space-sdk gradio` and build the interactive version (Blocks, tabs, Plotly
charts pulling from `quantthink-results` via `hf_hub_download`); add the H1
"longer-but-worse" scatter (Acc vs TL), the H2 frontier as VRAM budget slides
2→4GB, and the H4 CTS-reordering comparison (leaderboard sorted by Acc vs. by
CTS side by side).

## AWQ/GPTQ calibrated models

Tooling landscape check (per the project's dependency-verification rule): `autoawq` has not had a release since 2025-05 and is treated as stale/unmaintained — not used. `gptqmodel` 7.1.0 (released 2026-06-08, actively maintained) was chosen: as of its 5.0.0+ releases it has "fully supplanted AutoGPTQ and AutoAWQ," supporting both GPTQ and AWQ natively in one library, so a second tool (`llmcompressor`) wasn't needed for this pass.

**Status: DONE for M1.** Real GPTQ 4-bit calibration completed on the RTX 3050 and pushed to [happynood/DeepSeek-R1-Distill-Qwen-1.5B-GPTQ](https://huggingface.co/happynood/DeepSeek-R1-Distill-Qwen-1.5B-GPTQ) — see `docs/RUN_REAL.md` for the full recipe, real bugs hit and fixed along the way (a stale `wikitext` dataset loader, a full `/tmp` tmpfs, a missing `ninja`/CUDA-toolkit dependency for the Marlin kernel), and the measured Acc/TL/CTS.

M2/M3 (Qwen3-1.7B/0.6B) AWQ/GPTQ calibration: not yet attempted — a reasonable next step, not a blocker (see STATE.md for scope/time tradeoffs already made this build).
