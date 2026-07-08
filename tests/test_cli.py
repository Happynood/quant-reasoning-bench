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


def test_recommend_not_yet_implemented(tmp_path):
    fake_result = tmp_path / "r.json"
    fake_result.write_text("{}")
    runner = CliRunner()
    result = runner.invoke(main, ["recommend", str(fake_result), "--vram", "4.0"])
    assert result.exit_code != 0
