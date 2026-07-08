from __future__ import annotations

from quantthink.backends.mock import MockBackend
from quantthink.config import load_config
from quantthink.eval.loader import load_toy
from quantthink.runner import run_eval


def test_e2e_mock_backend_e0_toy(smoke_config_path):
    cfg = load_config(smoke_config_path)
    backend = MockBackend(model=cfg.model, latency_ms=0, answer="4")
    problems = load_toy()

    result = run_eval(cfg, problems, backend, config_path=smoke_config_path)

    assert result.metrics.n == len(problems)
    # MockBackend always answers "4"; only the first toy problem's ground truth is "4".
    assert 0.0 < result.metrics.acc < 1.0
    assert result.manifest.model == "mock"
    assert result.manifest.kv_dtype == "fp16"


def test_e2e_result_to_dict_has_required_fields(smoke_config_path):
    cfg = load_config(smoke_config_path)
    backend = MockBackend(model=cfg.model, latency_ms=0, answer="4")
    problems = load_toy()
    result = run_eval(cfg, problems, backend, config_path=smoke_config_path)

    d = result.to_dict()
    for key in ("acc", "tl_mean", "truncation_rate", "config", "manifest", "instances"):
        assert key in d


def test_e2e_hit_max_tokens_flagged_when_generation_uses_full_budget(smoke_config_path):
    cfg = load_config(smoke_config_path)
    cfg = cfg.model_copy(update={"max_tokens": 5})
    backend = MockBackend(model=cfg.model, latency_ms=0, answer="4", thinking_tokens=20)
    problems = load_toy()

    result = run_eval(cfg, problems, backend, config_path=smoke_config_path)

    # MockBackend ignores max_tokens and always emits a fixed-length output, so
    # every instance's output_tokens exceeds the tiny max_tokens=5 budget.
    assert all(r.hit_max_tokens for r in result.instance_results)
    assert result.metrics.truncation_rate == 1.0
