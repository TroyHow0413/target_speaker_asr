from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import STT, Transcript
from ..device import probe_noise
from ..registry import register


@register("stt", "mlx_whisper")
class MLXWhisperSTT(STT):
    """Whisper on the Apple-Silicon GPU via MLX (M1/M2/M3/M4 Macs only).
    uv sync --extra mlx     (pip install mlx-whisper)

    `model` is a Hugging Face repo of MLX-converted weights, e.g.
      mlx-community/whisper-small-mlx
      mlx-community/whisper-large-v3-mlx
      mlx-community/whisper-large-v3-turbo
    """

    def __init__(self, model: str = "mlx-community/whisper-small-mlx"):
        import mlx_whisper

        self._mw = mlx_whisper
        self.model = model
        self.device = "mps · Apple GPU (MLX)"
        self.transcribe(probe_noise(), None)  # downloads + loads the model now, not on first utterance

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> Transcript:
        kw = {"language": language} if language else {}
        out = self._mw.transcribe(np.asarray(audio, dtype=np.float32), path_or_hf_repo=self.model,
                                  condition_on_previous_text=False, **kw)
        return Transcript(out.get("text", "").strip(), out.get("language") or language)
