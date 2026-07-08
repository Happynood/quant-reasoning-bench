from __future__ import annotations

from quantthink.eval.extractor import extract


def test_extract_standard_think_block():
    raw = "<think>let me work this out</think>\nThe answer is \\boxed{4}."
    ext = extract(raw)
    assert ext.thinking == "let me work this out"
    assert "\\boxed{4}" in ext.answer
    assert not ext.thinking_truncated


def test_extract_no_opening_tag_but_has_closing():
    raw = "some reasoning here</think>final answer"
    ext = extract(raw)
    assert ext.thinking == "some reasoning here"
    assert ext.answer == "final answer"


def test_extract_truncated_thinking_no_closing_tag():
    raw = "<think>still thinking and never finished"
    ext = extract(raw)
    assert ext.thinking_truncated
    assert ext.answer == ""
    assert "still thinking" in ext.thinking


def test_extract_plain_answer_no_think_tags():
    raw = "The answer is 4."
    ext = extract(raw)
    assert ext.thinking == ""
    assert ext.answer == "The answer is 4."
    assert not ext.thinking_truncated
