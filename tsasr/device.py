"""Hardware auto-detection with graceful fallback.

Preference order: NVIDIA -> AMD -> Intel -> Apple -> CPU.

How that maps onto the libraries we use:

* PyTorch backends (speechbrain, funasr, hf_transformers, ...)
    "cuda"  NVIDIA (CUDA build) *and* AMD (ROCm build, same torch.cuda API)
    "xpu"   Intel GPUs (Arc / Iris Xe / Data Center) with the XPU build of torch
    "mps"   Apple Silicon (M1/M2/M3/...)
    "cpu"
  Which of these exist depends on which torch wheel you installed (see README / pyproject extras).

* CTranslate2 backend (faster-whisper): only "cuda" (NVIDIA) and "cpu".

`device: auto` in a preset walks the candidates in order. For each one the model is
loaded AND a 1-second dummy inference is run (the "probe"); if either fails - an
operator unsupported on MPS, missing cuDNN DLL, ... - it logs why and moves on to the
next candidate, finally CPU. An explicit device (e.g. `device: cuda:1`) is used
strictly, with no silent fallback.
"""
from __future__ import annotations

import functools
import gc
import logging
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np

from .base import SAMPLE_RATE

log = logging.getLogger("tsasr.device")
T = TypeVar("T")


def torch_candidates() -> List[str]:
    try:
        import torch
    except ImportError:
        return ["cpu"]
    out: List[str] = []
    try:
        if torch.cuda.is_available():  # NVIDIA CUDA or AMD ROCm
            out.append("cuda")
    except Exception:
        pass
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():  # Intel
            out.append("xpu")
    except Exception:
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():  # Apple
            out.append("mps")
    except Exception:
        pass
    out.append("cpu")
    return out


def ct2_candidates() -> List[str]:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return ["cuda", "cpu"]
    except Exception:
        pass
    return ["cpu"]


def candidates(requested: Optional[str], family: str = "torch") -> List[str]:
    r = (requested or "auto").strip().lower()
    if r != "auto":
        return [r]
    return ct2_candidates() if family == "ct2" else torch_candidates()


def describe_device(dev: str) -> str:
    """Human readable, e.g. 'cuda · NVIDIA CUDA · NVIDIA GeForce RTX 4070'."""
    base = (dev or "cpu").split(":")[0]
    try:
        import torch

        if base == "cuda":
            idx = int(dev.split(":")[1]) if ":" in dev else 0
            vendor = "AMD ROCm" if getattr(torch.version, "hip", None) else "NVIDIA CUDA"
            return f"cuda · {vendor} · {torch.cuda.get_device_name(idx)}"
        if base == "xpu":
            return f"xpu · Intel · {torch.xpu.get_device_name(0)}"
        if base == "mps":
            return "mps · Apple Silicon"
    except Exception:
        pass
    return dev


@functools.lru_cache(maxsize=1)
def hardware_summary() -> Dict[str, List[str]]:
    return {"torch": [describe_device(d) for d in torch_candidates()],
            "ctranslate2": [describe_device(d) for d in ct2_candidates()]}


def probe_noise(seconds: float = 1.0) -> np.ndarray:
    """Deterministic low-level noise used as a smoke-test input."""
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(seconds * SAMPLE_RATE)) * 0.05).astype(np.float32)


def build_with_fallback(build: Callable[[str], T], requested: Optional[str] = "auto",
                        family: str = "torch", probe: Optional[Callable[[T], None]] = None) -> Tuple[T, str]:
    """Try `build(device)` (+ optional `probe(obj)`) on each candidate device in order.
    Returns (object, device_used)."""
    errors: List[str] = []
    for dev in candidates(requested, family):
        try:
            obj = build(dev)
            if probe is not None and dev != "cpu":  # no point probing the last resort
                probe(obj)
            log.info("using device: %s", describe_device(dev))
            return obj, dev
        except Exception as e:  # noqa: BLE001 - we really do want to catch everything here
            errors.append(f"{dev}: {type(e).__name__}: {e}")
            log.warning("device %s failed (%s: %s) - trying next", dev, type(e).__name__, e)
            gc.collect()
    raise RuntimeError("could not initialise on any device:\n  " + "\n  ".join(errors))
