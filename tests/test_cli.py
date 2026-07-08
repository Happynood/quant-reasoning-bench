from __future__ import annotations

from click.testing import CliRunner

from quantthink.cli import main


def test_validate_config(smoke_config_path):
    runner = CliRunner()
    result = runner.invoke(main, ["validate-config", "--config", str(smoke_config_path)])
    assert result.exit_code == 0
    assert "OK" in result.output


def test_run_writes_result_and_manifest(smoke_config_path, tmp_path):
    out = tmp_path / "result.json"
    manifest = tmp_path / "manifest.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--config",
            str(smoke_config_path),
            "--output",
            str(out),
            "--manifest",
            str(manifest),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert manifest.exists()


def test_leaderboard_build_on_empty_dir(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    out_dir = tmp_path / "leaderboard"
    runner = CliRunner()
    args = ["leaderboard", "build", str(results_dir), "--output-dir", str(out_dir)]
    result = runner.invoke(main, args)
    assert result.exit_code == 0, result.output
    assert (out_dir / "leaderboard.md").exists()


def _fake_result_file(path, model, quant, acc, vram_gb) -> None:
    import json

    path.write_text(
        json.dumps(
            {
                "acc": acc,
                "tl_mean": 100.0,
                "cts": (100.0 / acc) if acc > 0 else None,
                "vram_gb": vram_gb,
                "config": {
                    "model": model,
                    "quant": quant,
                    "kv_quant": "fp16",
                    "thinking_cap": None,
                    "backend": "llama-cpp",
                    "tiers": ["E1"],
                    "sample_size": 10,
                    "seeds": [0],
                },
                "manifest": {
                    "git_commit": "x",
                    "config_sha256": "x",
                    "dataset_sha256": "x",
                    "timestamp": "2026-07-08T00:00:00Z",
                },
            }
        )
    )


def test_recommend_picks_best_accuracy_within_budget(tmp_path):
    fp16 = tmp_path / "fp16.json"
    q4 = tmp_path / "q4.json"
    _fake_result_file(fp16, "R1-1.5B", "fp16", acc=0.8, vram_gb=3.5)
    _fake_result_file(q4, "R1-1.5B", "Q4_K_M", acc=0.6, vram_gb=1.5)

    runner = CliRunner()
    result = runner.invoke(main, ["recommend", str(fp16), str(q4), "--vram", "2.0"])
    assert result.exit_code == 0, result.output
    assert "Q4_K_M" in result.output


def test_recommend_fails_when_nothing_fits_budget(tmp_path):
    fp16 = tmp_path / "fp16.json"
    _fake_result_file(fp16, "R1-1.5B", "fp16", acc=0.8, vram_gb=3.5)

    runner = CliRunner()
    result = runner.invoke(main, ["recommend", str(fp16), "--vram", "1.0"])
    assert result.exit_code != 0
