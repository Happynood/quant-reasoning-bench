# Adapted from Happynood/quant-toolcall-bench @6b6e29e5c83a (src/quantcall/cli.py),
# following the quant-mcp-bench convention of a documented `sweep` stub (real
# sweeps are one `run` invocation per config in a shell loop — see
# docs/RUN_REAL.md — since GGUF quant filename conventions aren't uniform
# enough for a single --model template). `recommend` ships in Phase 3 with
# budget/frontier.py.
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from quantthink import __version__
from quantthink.config import QuantThinkConfig, load_config


@click.group()
@click.version_option(version=__version__, prog_name="quantthink")
def main() -> None:
    """QuantThink — does quantization break reasoning?"""


@main.command("run")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", default=None, type=click.Path())
@click.option("--manifest", "manifest_path", default=None, type=click.Path())
@click.option("--tier", "extra_tiers", multiple=True, help="Override tiers (repeatable)")
def run_cmd(
    config_path: str,
    output_path: str | None,
    manifest_path: str | None,
    extra_tiers: tuple[str, ...],
) -> None:
    """Run the reasoning benchmark against a backend."""
    from quantthink.manifest import write_manifest
    from quantthink.runner import run_eval, write_result

    cfg = load_config(config_path)
    if extra_tiers:
        cfg = cfg.model_copy(update={"tiers": list(extra_tiers)})

    backend = _build_backend(cfg)
    problems = _load_problems(cfg)

    if not problems:
        click.echo("No problems loaded — check your tiers config.", err=True)
        sys.exit(1)

    click.echo(
        f"Running {len(problems)} problem(s) x {len(cfg.seeds)} seed(s) | "
        f"backend={cfg.backend} model={cfg.model} quant={cfg.quant} kv_quant={cfg.kv_quant}"
    )
    result = run_eval(cfg, problems, backend, config_path=config_path)

    out = output_path or "result.json"
    write_result(result, out)
    click.echo(f"Result written to {out}")
    cts_str = f"{result.metrics.cts:.1f}" if result.metrics.cts is not None else "N/A"
    click.echo(f"  Acc={result.metrics.acc:.3f}  TL={result.metrics.tl_mean:.1f}  CTS={cts_str}")

    if manifest_path:
        write_manifest(result.manifest, manifest_path)
        click.echo(f"Manifest written to {manifest_path}")


@main.command("sweep")
@click.option("--model", required=True)
@click.option("--weight-quants", required=True, help="Comma-separated weight quant levels")
@click.option("--kv-quants", default="fp16", help="Comma-separated KV-cache quant levels")
@click.option("--think-caps", default="uncapped", help="Comma-separated thinking-token caps")
@click.option("--benchmarks", default="gsm8k", help="Comma-separated benchmark tiers")
def sweep_cmd(
    model: str,
    weight_quants: str,
    kv_quants: str,
    think_caps: str,
    benchmarks: str,
) -> None:
    """Sweep model x weight-quant x kv-quant x thinking-cap combinations (not yet implemented).

    Every real sweep in this project (see docs/RUN_REAL.md) is produced with
    one `quantthink run` invocation per (weight-quant, kv-quant, thinking-cap)
    config file in a shell loop, since GGUF quant filename conventions aren't
    uniform across models. This command is a placeholder for a future
    convenience wrapper around that same pattern, not a different code path.
    """
    click.echo(
        f"[sweep not yet implemented] model={model} weight_quants={weight_quants} "
        f"kv_quants={kv_quants} think_caps={think_caps} benchmarks={benchmarks}"
    )
    click.echo("Run one 'quantthink run --config <combo>.yaml' per combination instead.")


@main.command("compare")
@click.argument("result_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json", "csv"]))
@click.option("--output", "output_path", default=None, type=click.Path())
def compare_cmd(
    result_files: tuple[str, ...],
    fmt: str,
    output_path: str | None,
) -> None:
    """Compare multiple result.json files and show an Acc/TL/CTS table."""
    results: list[dict[str, Any]] = []
    for p in result_files:
        with open(p) as f:
            results.append(json.load(f))

    if fmt == "json":
        text = json.dumps(results, indent=2)
    else:
        lines = [f"{'Config':<40}  {'Acc':>6}  {'TL':>8}  {'CTS':>8}"]
        lines.append("-" * 70)
        for r in results:
            cfg = r.get("config", {})
            label = f"{cfg.get('model', '?')} {cfg.get('quant', '?')} kv={cfg.get('kv_quant', '?')}"
            cts = r.get("cts")
            cts_str = f"{cts:.1f}" if cts is not None else "N/A"
            acc = r.get("acc", 0)
            tl = r.get("tl_mean", 0)
            lines.append(f"{label:<40}  {acc:.3f}  {tl:>8.1f}  {cts_str:>8}")
        text = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(text + "\n")
        click.echo(f"Written to {output_path}")
    else:
        click.echo(text)


@main.command("recommend")
@click.argument("result_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--vram", required=True, type=float, help="Peak VRAM budget in GB")
@click.option("--objective", default="accuracy", type=click.Choice(["accuracy", "cts"]))
def recommend_cmd(result_files: tuple[str, ...], vram: float, objective: str) -> None:
    """Recommend the accuracy- (or CTS-) optimal config for a VRAM budget (Phase 3)."""
    raise click.ClickException(
        "quantthink recommend ships in Phase 3 (budget/frontier.py), once real "
        "sweep results across the 2-4GB grid exist to select from."
    )


@main.group("leaderboard")
def leaderboard_group() -> None:
    """Build the published leaderboard from result files."""


@leaderboard_group.command("build")
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--output-dir", default="leaderboard", show_default=True, type=click.Path())
def leaderboard_build_cmd(results_dir: str, output_dir: str) -> None:
    """Build runs.csv + leaderboard.{json,csv,md} from a directory of result files."""
    from quantthink.report.leaderboard import build_leaderboard

    board = build_leaderboard(results_dir, output_dir=output_dir)
    click.echo(f"Leaderboard built: {len(board['rows'])} aggregated row(s) from {results_dir!r}")
    click.echo(f"  {output_dir}/runs.csv")
    click.echo(f"  {output_dir}/leaderboard.json")
    click.echo(f"  {output_dir}/leaderboard.csv")
    click.echo(f"  {output_dir}/leaderboard.md")


@main.command("validate-config")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def validate_config_cmd(config_path: str) -> None:
    """Validate a QuantThink YAML config file."""
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Config: {config_path}")
    click.echo(f"  backend      : {cfg.backend}")
    click.echo(f"  model        : {cfg.model}")
    click.echo(f"  quant        : {cfg.quant}")
    click.echo(f"  kv_quant     : {cfg.kv_quant}")
    click.echo(f"  thinking_cap : {cfg.thinking_cap}")
    click.echo(f"  tiers        : {cfg.tiers}")
    click.echo(f"  sample_size  : {cfg.sample_size}")
    click.echo(f"  seeds        : {cfg.seeds}")
    click.echo("OK")


def _build_backend(cfg: QuantThinkConfig) -> Any:
    from quantthink.backends.mock import MockBackend

    if cfg.backend == "mock":
        return MockBackend(model=cfg.model, latency_ms=cfg.mock.latency_ms)
    if cfg.backend == "llama-cpp":
        from quantthink.backends.llama_cpp import LlamaCppBackend  # type: ignore[import]

        return LlamaCppBackend(
            model_path=cfg.model,
            n_ctx=cfg.llama_cpp.n_ctx,
            n_gpu_layers=cfg.llama_cpp.n_gpu_layers,
            kv_dtype=cfg.kv_quant,
            chat_format=cfg.llama_cpp.chat_format,
            verbose=cfg.llama_cpp.verbose,
        )
    if cfg.backend == "transformers":
        from quantthink.backends.hf import HFBackend  # type: ignore[import]

        return HFBackend(
            model_id=cfg.model,
            device=cfg.hf.device,
            torch_dtype=cfg.hf.torch_dtype,
            load_in_4bit=cfg.hf.load_in_4bit,
            load_in_8bit=cfg.hf.load_in_8bit,
        )
    if cfg.backend == "openai":
        from quantthink.backends.openai_endpoint import OpenAIEndpointBackend

        return OpenAIEndpointBackend(
            base_url=cfg.openai.base_url,
            model=cfg.model,
            timeout_s=cfg.openai.timeout_s,
            api_key_env=cfg.openai.api_key_env,
        )
    if cfg.backend == "vllm":
        from quantthink.backends.vllm_backend import VLLMBackend

        return VLLMBackend(
            model_id=cfg.model,
            tensor_parallel_size=cfg.vllm.tensor_parallel_size,
            gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
            dtype=cfg.vllm.dtype,
        )
    raise ValueError(f"Unknown backend: {cfg.backend!r}")


def _load_problems(cfg: QuantThinkConfig) -> list[Any]:
    from quantthink.eval.loader import load_gsm8k, load_math500, load_toy

    problems: list[Any] = []
    for tier in cfg.tiers:
        if tier == "E0":
            problems.extend(load_toy())
        elif tier == "E1":
            try:
                problems.extend(load_gsm8k(sample_size=cfg.sample_size, seed=cfg.seeds[0]))
            except NotImplementedError as exc:
                click.echo(str(exc), err=True)
        elif tier == "E2":
            try:
                problems.extend(load_math500(sample_size=cfg.sample_size, seed=cfg.seeds[0]))
            except NotImplementedError as exc:
                click.echo(str(exc), err=True)
        else:
            click.echo(f"Unknown tier {tier!r}; skipping.", err=True)

    if cfg.sample_size and len(problems) > cfg.sample_size:
        import random

        rng = random.Random(cfg.seeds[0] if cfg.seeds else 0)
        problems = rng.sample(problems, cfg.sample_size)

    return problems
