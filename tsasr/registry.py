"""Tiny plugin registry.

Every swappable component registers itself under a (kind, name) key:

    @register("stt", "my_engine")
    class MySTT(STT): ...

Kinds: vad | speaker | stt | denoiser
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

KINDS = ("vad", "speaker", "stt", "denoiser")
_REGISTRY: Dict[str, Dict[str, Type]] = {k: {} for k in KINDS}


def register(kind: str, name: str):
    if kind not in _REGISTRY:
        raise ValueError(f"unknown component kind {kind!r}; expected one of {KINDS}")

    def deco(cls):
        cls.registry_kind = kind
        cls.registry_name = name
        _REGISTRY[kind][name] = cls
        return cls

    return deco


def available(kind: str) -> List[str]:
    return sorted(_REGISTRY[kind])


def create(kind: str, name: str, params: Optional[Dict[str, Any]] = None):
    try:
        cls = _REGISTRY[kind][name]
    except KeyError:
        raise KeyError(f"No {kind} backend named {name!r}. Available: {available(kind)}") from None
    return cls(**(params or {}))
