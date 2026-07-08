"""Deterministic, judge-free answer checkers — one per benchmark family.

New to QuantThink. No LLM-as-judge anywhere: each checker extracts a
final answer from the model's post-thinking segment and compares it to the
ground truth with benchmark-appropriate normalization only (never semantic/
LLM judgment).
"""

from __future__ import annotations

import re
from collections.abc import Callable

_BOXED_RE = re.compile(r"\\boxed\{")
_GSM8K_GT_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"-?\$?[\d,]+(?:\.\d+)?")
_MC_LETTER_RE = re.compile(r"\b([A-D])\b")


def _find_last_boxed(text: str) -> str | None:
    """Return the contents of the last \\boxed{...} in text, handling nested braces."""
    last: str | None = None
    for m in _BOXED_RE.finditer(text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last = text[start : i - 1]
    return last


def _normalize_number(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if not s:
        return None
    try:
        val = float(s)
    except ValueError:
        return None
    if val in (float("inf"), float("-inf")) or val != val:  # inf or NaN
        return None
    try:
        return str(int(val)) if val == int(val) else str(val)
    except (OverflowError, ValueError):
        return None


def extract_final_answer(answer_segment: str) -> str | None:
    """Extract the model's final answer from the post-thinking text.

    Preference order: last \\boxed{...} (the standard reasoning-model
    convention), then the last bare number in the segment.
    """
    boxed = _find_last_boxed(answer_segment)
    if boxed is not None:
        return boxed.strip()
    numbers = _NUMBER_RE.findall(answer_segment)
    if numbers:
        return numbers[-1]
    return None


def extract_gsm8k_ground_truth(gt_field: str) -> str | None:
    """GSM8K ground truth is formatted as '...#### 42'."""
    m = _GSM8K_GT_RE.search(gt_field)
    if m:
        return m.group(1)
    return gt_field.strip() or None


def check_numeric(model_answer: str | None, ground_truth: str | None) -> bool:
    """GSM8K / numeric-MATH checker: normalize both sides as numbers and compare."""
    if model_answer is None or ground_truth is None:
        return False
    a = _normalize_number(model_answer)
    b = _normalize_number(ground_truth)
    if a is not None and b is not None:
        return a == b
    return model_answer.strip() == ground_truth.strip()


def _normalize_math_expr(s: str) -> str:
    """Light LaTeX/whitespace normalization for MATH-500 boxed answers.

    Documented scope limit: this is string/numeric normalization, not symbolic
    equivalence (no sympy/CAS) — e.g. "1/2" and "0.5" will NOT match. Disclosed
    in docs/RUN_REAL.md as a conservative-accuracy trade-off (undercounts some
    true positives rather than risking false positives from a judge).
    """
    s = s.strip()
    s = re.sub(r"\\[!,;: ]", "", s)  # \! \, \; \: spacing commands
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    s = s.replace("\\left", "").replace("\\right", "")
    return s


def check_math(model_answer: str | None, ground_truth: str | None) -> bool:
    if model_answer is None or ground_truth is None:
        return False
    a, b = _normalize_math_expr(model_answer), _normalize_math_expr(ground_truth)
    if a == b:
        return True
    return check_numeric(model_answer, ground_truth)


def check_multiple_choice(model_answer: str | None, ground_truth: str | None) -> bool:
    """GPQA-style checker: extract a bare A/B/C/D letter and compare."""
    if model_answer is None or ground_truth is None:
        return False
    a_letters = _MC_LETTER_RE.findall(model_answer.upper())
    b_letters = _MC_LETTER_RE.findall(ground_truth.upper())
    if a_letters and b_letters:
        return a_letters[-1] == b_letters[-1]
    return model_answer.strip().upper() == ground_truth.strip().upper()


Checker = Callable[[str | None, str | None], bool]

CHECKERS: dict[str, Checker] = {
    "gsm8k": check_numeric,
    "math500": check_math,
    "gpqa": check_multiple_choice,
    "toy": check_numeric,
}


def get_checker(benchmark: str) -> Checker:
    if benchmark not in CHECKERS:
        raise ValueError(f"Unknown benchmark {benchmark!r}; expected one of {list(CHECKERS)}")
    return CHECKERS[benchmark]
