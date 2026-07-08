"""KV-cache dtype plumbing into the llama.cpp backend.

New to QuantThink (no sibling has a KV-cache quant axis): maps the config-level
KV-dtype string ("fp16" / "Q8" / "Q4") onto llama-cpp-python's `type_k`/`type_v`
constructor args (`--cache-type-k`/`--cache-type-v` in the llama.cpp CLI).
"""

from __future__ import annotations

from typing import Literal

KvDtype = Literal["fp16", "Q8", "Q4"]

_KV_DTYPES: tuple[str, ...] = ("fp16", "Q8", "Q4")


def kv_dtype_to_ggml_type(kv_dtype: KvDtype) -> int:
    """Map a KV-cache dtype string to a llama_cpp GGML type constant.

    Imports llama_cpp lazily so this module has no hard dependency on the
    (optional, GPU-only) llama-cpp-python package.
    """
    import llama_cpp

    mapping = {
        "fp16": llama_cpp.GGML_TYPE_F16,
        "Q8": llama_cpp.GGML_TYPE_Q8_0,
        "Q4": llama_cpp.GGML_TYPE_Q4_0,
    }
    if kv_dtype not in mapping:
        raise ValueError(f"Unknown KV dtype {kv_dtype!r}; expected one of {_KV_DTYPES}")
    return mapping[kv_dtype]


def validate_kv_dtype(kv_dtype: str) -> KvDtype:
    if kv_dtype not in _KV_DTYPES:
        raise ValueError(f"Unknown KV dtype {kv_dtype!r}; expected one of {_KV_DTYPES}")
    return kv_dtype  # type: ignore[return-value]
