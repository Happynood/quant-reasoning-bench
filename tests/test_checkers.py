from __future__ import annotations

import pytest

from quantthink.eval.checkers import (
    check_math,
    check_multiple_choice,
    check_numeric,
    extract_final_answer,
    extract_gsm8k_ground_truth,
    get_checker,
)


def test_extract_final_answer_prefers_boxed():
    assert extract_final_answer("blah \\boxed{42} blah") == "42"


def test_extract_final_answer_falls_back_to_last_number():
    assert extract_final_answer("so the total is 15 apples") == "15"


def test_extract_final_answer_none_when_nothing_found():
    assert extract_final_answer("no numbers here") is None


def test_extract_gsm8k_ground_truth():
    assert extract_gsm8k_ground_truth("some reasoning\n#### 42") == "42"


def test_check_numeric_matches_with_formatting_noise():
    assert check_numeric("$42", "42") is True
    assert check_numeric("1,000", "1000") is True
    assert check_numeric("42", "43") is False


def test_check_numeric_none_inputs():
    assert check_numeric(None, "42") is False
    assert check_numeric("42", None) is False


def test_check_numeric_does_not_crash_on_overflowing_number():
    # A model can output a huge digit string (e.g. a runaway generation) that
    # float() silently parses as inf rather than raising ValueError.
    huge = "9" * 400
    assert check_numeric(huge, "42") is False
    assert check_numeric("42", huge) is False


def test_check_numeric_does_not_crash_on_literal_infinity():
    assert check_numeric("inf", "42") is False
    assert check_numeric("-inf", "42") is False


def test_check_math_boxed_normalization():
    assert check_math("\\frac{1}{2}", "\\frac{1}{2}") is True
    assert check_math("42", "42") is True


def test_check_multiple_choice():
    assert check_multiple_choice("The answer is B", "B") is True
    assert check_multiple_choice("A", "B") is False


def test_get_checker_unknown_raises():
    with pytest.raises(ValueError):
        get_checker("not-a-benchmark")


def test_get_checker_returns_callable():
    checker = get_checker("gsm8k")
    assert checker("4", "4") is True
