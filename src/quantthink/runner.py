# Adapted from Happynood/quant-toolcall-bench @6b6e29e5c83a (src/quantcall/runner.py).
# Diff: the single-run pipeline is a reasoning loop (build prompt -> generate ->
# extract thinking/answer -> deterministic checker -> Acc/TL/CTS) instead of
# tool-call parsing; iterates problems x K fixed seeds (sampled decoding, not
# greedy) and tracks thinking-cap truncation for the truncation-amplifies-
# quant-damage study.
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from quantthink.backends.base import Backend
from quantthink.config import QuantThinkConfig
from quantthink.eval.checkers import extract_final_answer, get_checker
from quantthink.eval.extractor import extract
from quantthink.eval.loader import Problem
from quantthink.manifest import RunManifest, collect_manifest, compute_dataset_sha256
from quantthink.metrics.core import (
    InstanceResult,
    MetricsResult,
    compute_metrics,
    evaluate_instance,
)


@dataclass
class RunResult:
    config: dict[str, Any]
    metrics: MetricsResult
    manifest: RunManifest
    total_latency_ms: float
    instance_results: list[InstanceResult] = field(default_factory=list)
    peak_vram_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.metrics.n,
            "acc": self.metrics.acc,
            "tl_mean": self.metrics.tl_mean,
            "tl_solved": self.metrics.tl_solved,
            "tl_unsolved": self.metrics.tl_unsolved,
            "cts": self.metrics.cts,
            "total_tokens_mean": self.metrics.total_tokens_mean,
            "truncation_rate": self.metrics.truncation_rate,
            "total_latency_ms": self.total_latency_ms,
            "vram_gb": (self.peak_vram_mb / 1024.0) if self.peak_vram_mb is not None else None,
            "config": self.config,
            "manifest": asdict(self.manifest),
            "instances": [asdict(r) for r in self.instance_results],
        }


def _benchmark_for_tier(tier: str) -> str:
    mapping = {"E0": "toy", "E1": "gsm8k", "E2": "math500", "E3": "gpqa"}
    if tier not in mapping:
        raise ValueError(f"Unknown tier {tier!r}; expected one of {list(mapping)}")
    return mapping[tier]


def _build_messages(problem: Problem) -> list[dict[str, str]]:
    return [{"role": "user", "content": problem.prompt}]


def run_eval(
    cfg: QuantThinkConfig,
    problems: list[Problem],
    backend: Backend,
    config_path: str | Path = "",
) -> RunResult:
    instance_results: list[InstanceResult] = []
    total_latency_ms = 0.0
    peak_vram_mb: float | None = None
    seeds = [0] if cfg.greedy else cfg.seeds
    temperature = 0.0 if cfg.greedy else cfg.temperature
    # thinking_cap bounds total generation length (thinking + answer), not just the
    # thinking segment: KV-cache growth (the actual VRAM-relevant quantity for the
    # Memory-Budget Frontier) scales with total generated tokens regardless of
    # phase, so this is a simpler and equally faithful proxy for "how many tokens
    # can I afford before I OOM" than a phase-specific stop would be.
    effective_max_tokens = (
        min(cfg.max_tokens, cfg.thinking_cap) if cfg.thinking_cap is not None else cfg.max_tokens
    )

    for problem in problems:
        checker = get_checker(_benchmark_for_tier(problem.tier))
        messages = _build_messages(problem)

        for seed in seeds:
            gen = backend.generate(
                messages,
                max_tokens=effective_max_tokens,
                temperature=temperature,
                top_p=cfg.top_p,
                seed=seed,
            )
            total_latency_ms += gen.latency_ms
            if gen.peak_vram_mb is not None:
                peak_vram_mb = max(peak_vram_mb or 0.0, gen.peak_vram_mb)

            extraction = extract(gen.raw_output)
            model_answer = extract_final_answer(extraction.answer)
            correct = checker(model_answer, problem.ground_truth)

            instance_results.append(
                evaluate_instance(
                    problem_id=problem.id,
                    seed=seed,
                    correct=correct,
                    thinking_tokens=len(extraction.thinking.split()),
                    total_tokens=gen.output_tokens,
                    thinking_truncated=extraction.thinking_truncated,
                    hit_max_tokens=gen.output_tokens >= effective_max_tokens,
                )
            )

    metrics = compute_metrics(instance_results)
    dataset_sha = compute_dataset_sha256(problems)
    manifest = collect_manifest(config_path, cfg, dataset_sha256=dataset_sha)

    config_dict: dict[str, Any] = {
        "model": cfg.model,
        "backend": cfg.backend,
        "quant": cfg.quant,
        "kv_quant": cfg.kv_quant,
        "thinking_cap": cfg.thinking_cap,
        "tiers": cfg.tiers,
        "sample_size": cfg.sample_size,
        "seeds": seeds,
        "temperature": temperature,
        "top_p": cfg.top_p,
        "greedy": cfg.greedy,
    }

    return RunResult(
        config=config_dict,
        metrics=metrics,
        manifest=manifest,
        total_latency_ms=total_latency_ms,
        instance_results=instance_results,
        peak_vram_mb=peak_vram_mb,
    )


def write_result(result: RunResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result.to_dict(), indent=2) + "\n")
