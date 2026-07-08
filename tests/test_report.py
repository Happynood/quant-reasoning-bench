from __future__ import annotations

from quantthink.report.pareto import pareto_front
from quantthink.report.published import aggregate_leaderboard, run_row, sanitize_model_name


def _fake_result(model: str, quant: str, kv_quant: str, acc: float, vram_gb: float) -> dict:
    return {
        "acc": acc,
        "tl_mean": 100.0,
        "cts": (100.0 / acc) if acc > 0 else None,
        "vram_gb": vram_gb,
        "config": {
            "model": model,
            "quant": quant,
            "kv_quant": kv_quant,
            "thinking_cap": None,
            "backend": "llama-cpp",
            "tiers": ["E1"],
            "sample_size": 200,
            "seeds": [0],
        },
        "manifest": {
            "git_commit": "abc123",
            "config_sha256": "x",
            "dataset_sha256": "y",
            "timestamp": "2026-07-08T00:00:00Z",
        },
    }


def test_sanitize_model_name_strips_gguf_and_quant_suffix():
    raw = "/home/x/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf"
    assert sanitize_model_name(raw, "Q4_K_M") == "Qwen3-0.6B"


def test_aggregate_leaderboard_computes_delta_vs_fp16_baseline():
    results = [
        _fake_result("R1-1.5B", "fp16", "fp16", acc=0.8, vram_gb=3.5),
        _fake_result("R1-1.5B", "Q4_K_M", "fp16", acc=0.6, vram_gb=1.2),
    ]
    rows = [run_row(r) for r in results]
    agg = aggregate_leaderboard(rows)

    by_quant = {r["quant"]: r for r in agg}
    assert by_quant["fp16"]["delta_acc_rel"] is None  # baseline itself has no delta
    assert by_quant["Q4_K_M"]["delta_acc_rel"] is not None
    assert round(by_quant["Q4_K_M"]["delta_acc_rel"], 3) == 0.25  # (0.8-0.6)/0.8


def test_pareto_front_keeps_non_dominated_points():
    points = [
        {"vram_gb": 1.0, "acc": 0.5},
        {"vram_gb": 2.0, "acc": 0.6},
        {"vram_gb": 3.0, "acc": 0.5},  # dominated by vram_gb=1.0 point (lower vram, same acc)
    ]
    front = pareto_front(points, x_key="vram_gb", y_key="acc")
    assert {"vram_gb": 3.0, "acc": 0.5} not in front
    assert len(front) == 2
