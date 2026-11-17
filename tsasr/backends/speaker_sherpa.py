from __future__ import annotations

from pathlib import Path

import numpy as np

from ..base import SAMPLE_RATE, SpeakerEncoder
from ..config import resolve_path
from ..registry import register


@register("speaker", "sherpa_onnx")
class SherpaSpeakerEncoder(SpeakerEncoder):
    """pip install sherpa-onnx. Works with the 3D-Speaker, WeSpeaker and NeMo
    TitaNet speaker-embedding .onnx models from the sherpa-onnx releases page
    (good Chinese/English options, CPU friendly)."""

    min_seconds = 1.0

    def __init__(self, model: str, num_threads: int = 2, provider: str = "cpu"):
        import sherpa_onnx

        self._model_name = Path(model).stem
        self.device = provider  # default pip wheel is CPU-only; CUDA needs the separate "+cuda" sherpa-onnx wheel
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=resolve_path(model), num_threads=num_threads, debug=False, provider=provider)
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    @property
    def tag(self) -> str:
        return "sherpa_" + self._model_name

    def embed(self, audio: np.ndarray) -> np.ndarray:
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=np.asarray(audio, dtype=np.float32))
        stream.input_finished()
        return self._norm(self.extractor.compute(stream))
