from __future__ import annotations

import pytest

from quantthink.eval.loader import load_gsm8k, load_math500, load_toy


def test_load_toy_returns_two_problems(toy_jsonl_path):
    problems = load_toy(toy_jsonl_path)
    assert len(problems) == 2
    assert problems[0].tier == "E0"
    assert problems[0].ground_truth == "4"


def test_load_gsm8k_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        load_gsm8k()


def test_load_math500_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        load_math500()
