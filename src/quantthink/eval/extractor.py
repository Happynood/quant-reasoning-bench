"""Split a reasoning model's raw output into a thinking segment and a final answer.

New to QuantThink — no sibling has this (tool-calling models don't emit a
<think> block). DeepSeek-R1-distill and Qwen3 (thinking mode) both wrap their
chain-of-thought in <think>...</think>; the final answer is whatever follows.
If the closing </think> tag is missing (the model ran out of tokens mid-thought,
e.g. under an aggressive thinking-cap), the entire output is treated as the
thinking segment and the answer segment is empty — this is the expected,
disclosed failure mode H5 measures (truncation costs the answer, not just length).
"""

from __future__ import annotations

from dataclasses import dataclass

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


@dataclass(frozen=True)
class Extraction:
    thinking: str
    answer: str
    thinking_truncated: bool


def extract(raw_output: str) -> Extraction:
    """Split raw_output into (thinking, answer).

    Handles three shapes seen across model families:
    - "<think>...</think>answer"           (DeepSeek-R1-distill, Qwen3 default)
    - "...</think>answer" (no opening tag — some chat templates pre-seed it)
    - no </think> at all                   (truncated by a thinking-token cap)
    """
    close_idx = raw_output.find(_THINK_CLOSE)
    if close_idx == -1:
        # No closing tag: either the model never entered thinking mode (plain
        # answer) or thinking was cut off by a token cap. Heuristic: if an
        # opening tag is present, treat the whole thing as (truncated) thinking.
        if _THINK_OPEN in raw_output:
            thinking = raw_output.split(_THINK_OPEN, 1)[1]
            return Extraction(thinking=thinking.strip(), answer="", thinking_truncated=True)
        return Extraction(thinking="", answer=raw_output.strip(), thinking_truncated=False)

    before_close = raw_output[:close_idx]
    open_idx = before_close.find(_THINK_OPEN)
    thinking = before_close[open_idx + len(_THINK_OPEN) :] if open_idx != -1 else before_close
    answer = raw_output[close_idx + len(_THINK_CLOSE) :]
    return Extraction(thinking=thinking.strip(), answer=answer.strip(), thinking_truncated=False)


def count_tokens_whitespace(text: str) -> int:
    """Cheap, backend-agnostic token proxy: whitespace-split word count.

    Real runs use the backend's own reported token counts (GenerationResult.
    output_tokens) for TL/CTS; this is only used where no tokenizer count is
    available (e.g. re-deriving TL from a stored raw_output in a report step).
    """
    return len(text.split())
