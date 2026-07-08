from __future__ import annotations

from quantthink.backends.base import GenerationResult
from quantthink.backends.mock import MockBackend


def test_mock_backend_returns_generation_result():
    backend = MockBackend(model="mock", latency_ms=0)
    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    result = backend.generate(messages, max_tokens=64, temperature=0.0, top_p=1.0, seed=0)
    assert isinstance(result, GenerationResult)
    assert result.raw_output
    assert result.latency_ms >= 0
    assert result.input_tokens > 0
    assert result.output_tokens > 0


def test_mock_backend_emits_think_tags_and_boxed_answer():
    backend = MockBackend(answer="42", thinking_tokens=5, latency_ms=0)
    result = backend.generate(
        [{"role": "user", "content": "q"}], max_tokens=64, temperature=0.0, top_p=1.0
    )
    assert "<think>" in result.raw_output
    assert "</think>" in result.raw_output
    assert "\\boxed{42}" in result.raw_output


def test_mock_backend_name():
    backend = MockBackend()
    assert backend.name == "mock"
