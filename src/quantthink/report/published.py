# Adapted from Happynood/quant-toolcall-bench @6b6e29e5c83a (src/quantcall/report/published.py).
# Diff: RUNS_COLS/LEADERBOARD_COLS carry acc/tl_mean/cts instead of svr/tsa/ac/fcr,
# and gain kv_quant + thinking_cap columns (new QuantThink axes); the
# grouping/baseline/aggregation logic is otherwise unchanged.
"""Build the published results dataset (runs.csv + leaderboard.csv).

Single source of truth for the schema shipped to the happynood/quantthink-results
HF dataset. `docs/RESULTS_SCHEMA.md` documents these same column lists.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from quantthink.metrics.deltas import compute_delta_rel, compute_eta
from quantthink.metrics.stats import bootstrap_ci

RUNS_COLS = [
    "model",
    "quant",
    "kv_quant",
    "thinking_cap",
    "backend",
    "tier",
    "seed",
    "sample_size",
    "acc",
    "tl_mean",
    "cts",
    "vram_gb",
    "git_commit",
    "config_sha256",
    "dataset_sha256",
    "timestamp",
]

LEADERBOARD_COLS = [
    "model",
    "quant",
    "kv_quant",
    "thinking_cap",
    "backend",
    "tier",
    "n_seeds",
    "acc_mean",
    "acc_ci_low",
    "acc_ci_high",
    "tl_mean",
    "cts_mean",
    "vram_gb",
    "eta",
    "delta_acc_rel",
    "delta_cts_rel",
    "baseline_quant",
]

# Higher rank = less lossy precision. Used to pick the Δ baseline quant per
# (model, backend, tier) scope: the highest-ranked quant actually present in
# that scope. fp16 wins when it fits; otherwise the best available quant is
# used and labeled explicitly via the baseline_quant column.
PRECISION_RANK: dict[str, int] = {
    "fp16": 4,
    "Q8_0": 3,
    "Q5_K_M": 2,
    "Q4_K_M": 1,
    "AWQ": 0,
    "GPTQ": 0,
}

GroupKey = tuple[str, str, str, str, str, str]


def _tier_str(tiers: list[str]) -> str:
    return "+".join(tiers)


_FP16_ALIASES = ("fp16", "bf16", "f16")


def sanitize_model_name(raw_model: str, quant: str) -> str:
    """Derive a canonical, path-free model name from a local GGUF path.

    Local GGUF filenames encode both the model and the quant, e.g.
    "/home/x/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf" for quant="Q4_K_M". If the
    raw path is published verbatim, every quant of the same model gets a
    different "model" string, which breaks (model,backend,tier) grouping in
    aggregate_leaderboard(). This strips the directory, extension, and the
    trailing quant suffix (including fp16/bf16/f16 filename aliases) so all
    quants of one model share the same canonical name.
    """
    name = raw_model.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if name.lower().endswith(".gguf"):
        name = name[: -len(".gguf")]

    suffixes = {quant} | (set(_FP16_ALIASES) if quant == "fp16" else set())
    for suf in suffixes:
        for candidate in (f"-{suf}", f"-{suf.upper()}", f"-{suf.lower()}"):
            if name.endswith(candidate):
                name = name[: -len(candidate)]
                break

    # bartowski-style GGUF repos duplicate the org in the filename, e.g.
    # "Qwen_Qwen3-0.6B" for org "Qwen", model "Qwen3-0.6B".
    if "_" in name:
        org, _, rest = name.partition("_")
        if rest.startswith(org):
            name = rest

    return name


def run_row(r: dict[str, Any]) -> dict[str, Any]:
    """Flatten one result.json into a runs.csv row (one row per real run)."""
    cfg = r.get("config", {})
    manifest = r.get("manifest", {})
    seeds = cfg.get("seeds", [])
    return {
        "model": sanitize_model_name(cfg.get("model", ""), cfg.get("quant", "")),
        "quant": cfg.get("quant", ""),
        "kv_quant": cfg.get("kv_quant", "fp16"),
        "thinking_cap": cfg.get("thinking_cap"),
        "backend": cfg.get("backend", ""),
        "tier": _tier_str(cfg.get("tiers", [])),
        "seed": seeds[0] if seeds else "",
        "sample_size": cfg.get("sample_size", ""),
        "acc": r.get("acc", 0.0),
        "tl_mean": r.get("tl_mean", 0.0),
        "cts": r.get("cts"),
        "vram_gb": r.get("vram_gb"),
        "git_commit": manifest.get("git_commit", ""),
        "config_sha256": manifest.get("config_sha256", ""),
        "dataset_sha256": manifest.get("dataset_sha256", ""),
        "timestamp": manifest.get("timestamp", ""),
    }


def _group_key(row: dict[str, Any]) -> GroupKey:
    return (
        row["model"],
        row["quant"],
        str(row["kv_quant"]),
        str(row["thinking_cap"]),
        row["backend"],
        row["tier"],
    )


def _scope_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row["model"], row["backend"], row["tier"])


def _pick_baseline_quants(run_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], str]:
    baseline_by_scope: dict[tuple[str, str, str], str] = {}
    for row in run_rows:
        scope = _scope_key(row)
        quant = row["quant"]
        rank = PRECISION_RANK.get(quant, -1)
        current = baseline_by_scope.get(scope)
        if current is None or rank > PRECISION_RANK.get(current, -1):
            baseline_by_scope[scope] = quant
    return baseline_by_scope


def aggregate_leaderboard(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-seed run rows into one row per (model,quant,kv_quant,cap,backend,tier)."""
    groups: dict[GroupKey, list[dict[str, Any]]] = {}
    for row in run_rows:
        groups.setdefault(_group_key(row), []).append(row)

    baseline_by_scope = _pick_baseline_quants(run_rows)

    agg_rows: dict[GroupKey, dict[str, Any]] = {}
    for key, rows in groups.items():
        first = rows[0]
        model, quant, kv_quant, thinking_cap = (
            first["model"],
            first["quant"],
            first["kv_quant"],
            first["thinking_cap"],
        )
        backend, tier = first["backend"], first["tier"]
        n = len(rows)
        acc_vals = [float(r["acc"]) for r in rows]
        tl_vals = [float(r["tl_mean"]) for r in rows]
        cts_vals = [float(r["cts"]) for r in rows if isinstance(r["cts"], (int, float))]
        vram_vals = [float(r["vram_gb"]) for r in rows if isinstance(r["vram_gb"], (int, float))]

        acc_mean = sum(acc_vals) / n
        acc_lo, acc_hi = bootstrap_ci(acc_vals, n_resamples=2000, seed=42)
        vram_gb = (sum(vram_vals) / len(vram_vals)) if vram_vals else None
        cts_mean = (sum(cts_vals) / len(cts_vals)) if cts_vals else None

        agg_rows[key] = {
            "model": model,
            "quant": quant,
            "kv_quant": kv_quant,
            "thinking_cap": thinking_cap,
            "backend": backend,
            "tier": tier,
            "n_seeds": n,
            "acc_mean": acc_mean,
            "acc_ci_low": acc_lo,
            "acc_ci_high": acc_hi,
            "tl_mean": sum(tl_vals) / n,
            "cts_mean": cts_mean,
            "vram_gb": vram_gb,
            "eta": compute_eta(acc_mean, vram_gb),
            "delta_acc_rel": None,
            "delta_cts_rel": None,
            "baseline_quant": baseline_by_scope[(model, backend, tier)],
        }

    for row in agg_rows.values():
        model, backend, tier = row["model"], row["backend"], row["tier"]
        scope = (model, backend, tier)
        baseline_quant = baseline_by_scope[scope]
        if row["quant"] == baseline_quant and row["kv_quant"] == "fp16" and not row["thinking_cap"]:
            continue
        baseline_row = agg_rows.get((model, baseline_quant, "fp16", "None", backend, tier))
        if baseline_row is None:
            continue
        row["delta_acc_rel"] = compute_delta_rel(baseline_row["acc_mean"], row["acc_mean"])
        if baseline_row["cts_mean"] is not None and row["cts_mean"] is not None:
            row["delta_cts_rel"] = compute_delta_rel(baseline_row["cts_mean"], row["cts_mean"])

    return sorted(
        agg_rows.values(),
        key=lambda r: (
            r["model"],
            r["backend"],
            r["tier"],
            -PRECISION_RANK.get(r["quant"], -1),
        ),
    )


def write_csv(path: Path, cols: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in cols})
