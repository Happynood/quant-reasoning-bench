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
    for key in ("acc", "tl_mean", "config", "manifest"):
        assert key in d
