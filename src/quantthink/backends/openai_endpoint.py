"""OpenAI-compatible HTTP endpoint backend (/v1/chat/completions).

Adapted from Happynood/quant-toolcall-bench @6b6e29e5c83a
(src/quantcall/backends/openai_endpoint.py). Diff: generate_toolcall(messages,
tools) -> generate(messages, max_tokens, temperature, top_p, seed); no
tools/tool_choice payload fields.

Works with any server that speaks the OpenAI chat-completions protocol —
llama.cpp's own server, llama_cpp.server, Ollama, vLLM's OpenAI server, etc.

API key handling: if api_key_env is set, the key is read from that
environment variable at call time. The key is never logged or stored in config.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.request import urlopen

from quantthink.backends.base import Backend, GenerationResult

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

# A proxy-free opener, used for loopback requests only. Python's stdlib
# no_proxy parsing does not understand CIDR notation (e.g. "127.0.0.0/8"),
# so a system proxy env var can otherwise silently intercept calls to a
# local inference server that should never leave the machine.
_no_proxy_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _is_loopback_host(hostname: str | None) -> bool:
    return hostname in _LOOPBACK_HOSTS or bool(hostname) and hostname.startswith("127.")


def _open_url(req: urllib.request.Request, timeout: float) -> Any:
    if _is_loopback_host(req.host.split(":")[0] if req.host else None):
        return _no_proxy_opener.open(req, timeout=timeout)
    return urlopen(req, timeout=timeout)


class OpenAIEndpointBackend(Backend):
    """Backend that calls a /v1/chat/completions HTTP endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 120.0,
        api_key_env: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s
        self._api_key_env = api_key_env

    @property
    def name(self) -> str:
        return "openai"

    def _make_request(self, payload: dict[str, Any]) -> urllib.request.Request:
        url = f"{self._base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        key = os.environ.get(self._api_key_env) if self._api_key_env else None
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        return req

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
    ) -> GenerationResult:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if seed is not None:
            payload["seed"] = seed
        req = self._make_request(payload)

        try:
            t_start = time.perf_counter()
            with _open_url(req, timeout=self._timeout_s) as resp:
                raw = resp.read()
            latency_ms = (time.perf_counter() - t_start) * 1000.0
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"OpenAI-compatible endpoint request failed: {exc}") from exc

        data: dict[str, Any] = json.loads(raw)
        choice = data["choices"][0]
        msg = choice.get("message", {})
        raw_output = msg.get("content") or ""

        usage = data.get("usage") or {}
        input_tokens: int = usage.get("prompt_tokens", 0)
        output_tokens: int = usage.get("completion_tokens", 0)

        return GenerationResult(
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            tokens_per_second=(
                output_tokens / (latency_ms / 1000.0)
                if latency_ms > 0 and output_tokens > 0
                else None
            ),
        )
