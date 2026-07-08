from __future__ import annotations

from quantthink.config import QuantThinkConfig, load_config


def test_default_config():
    cfg = QuantThinkConfig()
    assert cfg.backend == "mock"
    assert cfg.kv_quant == "fp16"
    assert cfg.thinking_cap is None
    assert cfg.seeds == [0]


def test_load_smoke_config(smoke_config_path):
    cfg = load_config(smoke_config_path)
    assert cfg.model == "mock"
    assert cfg.backend == "mock"
    assert cfg.tiers == ["E0"]
    assert cfg.greedy is True


def test_config_rejects_unknown_backend():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QuantThinkConfig(backend="not-a-backend")  # type: ignore[arg-type]
