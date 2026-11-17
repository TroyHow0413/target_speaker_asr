from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import STT, Transcript
from ..device import build_with_fallback, probe_noise
from ..registry import register


@register("stt", "faster_whisper")
class FasterWhisperSTT(STT):
    """faster-whisper (CTranslate2). ~99 languages. `model_size_or_path` can be a size
    name (tiny/base/small/medium/large-v3) or a local CTranslate2 model folder.

    device: auto | cpu | cuda | cuda:1
      CTranslate2 only accelerates NVIDIA GPUs. On AMD / Intel / Apple machines "auto" ends up on
      CPU - use the `mlx_whisper` backend (Apple GPU) or `openai_http` + whisper.cpp (Vulkan: AMD/Intel/
      NVIDIA GPUs) for GPU speed there.
    compute_type: auto -> float16 on GPU, int8 on CPU.
    """

    def __init__(self, model_size_or_path: str = "small", device: str = "auto",
                 compute_type: str = "auto", beam_size: int = 5, cpu_threads: int = 0,
                 download_root: Optional[str] = None, initial_prompt: Optional[str] = None):
        from faster_whisper import WhisperModel

        try:  # On Windows, importing torch first puts its bundled CUDA/cuDNN DLLs on the
            import torch  # noqa: F401  # DLL search path, which CTranslate2 can then reuse.
        except ImportError:
            pass

        self.beam_size = beam_size
        self.initial_prompt = initial_prompt

        def build(dev: str):
            base, _, idx = dev.partition(":")
            ct = compute_type if compute_type != "auto" else ("float16" if base == "cuda" else "int8")
            extra = {"device_index": int(idx)} if idx else {}
            self.device = dev
            self.model = WhisperModel(model_size_or_path, device=base, compute_type=ct,
                                      cpu_threads=cpu_threads, download_root=download_root, **extra)
            return self

        _, self.device = build_with_fallback(build, device, "ct2", lambda _: self.transcribe(probe_noise(), None))

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> Transcript:
        segments, info = self.model.transcribe(
            np.asarray(audio, dtype=np.float32), language=language or None,
            beam_size=self.beam_size, initial_prompt=self.initial_prompt,
            condition_on_previous_text=False, vad_filter=False)
        text = "".join(s.text for s in segments).strip()
        return Transcript(text, info.language, {"language_prob": float(info.language_probability)})
