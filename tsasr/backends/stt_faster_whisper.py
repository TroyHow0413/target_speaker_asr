from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import STT, Transcript
from ..registry import register


@register("stt", "faster_whisper")
class FasterWhisperSTT(STT):
    """pip install faster-whisper. ~99 languages. `model_size_or_path` can be a
    size name (tiny/base/small/medium/large-v3) or a local CTranslate2 model folder
    (e.g. a language-specific fine-tune)."""

    def __init__(self, model_size_or_path: str = "small", device: str = "cpu",
                 compute_type: str = "int8", beam_size: int = 5, cpu_threads: int = 0,
                 download_root: Optional[str] = None, initial_prompt: Optional[str] = None):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_size_or_path, device=device, compute_type=compute_type,
                                  cpu_threads=cpu_threads, download_root=download_root)
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> Transcript:
        segments, info = self.model.transcribe(
            np.asarray(audio, dtype=np.float32), language=language or None,
            beam_size=self.beam_size, initial_prompt=self.initial_prompt,
            condition_on_previous_text=False, vad_filter=False)
        text = "".join(s.text for s in segments).strip()
        return Transcript(text, info.language, {"language_prob": float(info.language_probability)})
