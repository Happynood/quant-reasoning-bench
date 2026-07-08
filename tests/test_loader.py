from __future__ import annotations

import pytest

from quantthink.eval.loader import load_gsm8k, load_math500, load_toy


def test_load_toy_returns_two_problems(toy_jsonl_path):
    problems = load_toy(toy_jsonl_path)
    assert len(problems) == 2
    assert problems[0].tier == "E0"
    assert problems[0].ground_truth == "4"


def test_load_gsm8k_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gsm8k(tmp_path / "does_not_exist.jsonl")


def test_load_math500_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_math500(tmp_path / "does_not_exist.jsonl")


def test_load_gsm8k_reads_frozen_fixture(tmp_path):
    fixture = tmp_path / "gsm8k.jsonl"
    fixture.write_text('{"id": "gsm8k-0", "tier": "E1", "prompt": "q", "ground_truth": "42"}\n')
    problems = load_gsm8k(fixture)
    assert len(problems) == 1
    assert problems[0].ground_truth == "42"
    assert problems[0].tier == "E1"
