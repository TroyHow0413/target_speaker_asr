from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import SAMPLE_RATE, STT, Transcript
from ..config import resolve_path
from ..registry import register

_PATH_KEYS = {"model", "tokens", "encoder", "decoder", "joiner", "paraformer", "hotwords_file"}


@register("stt", "sherpa_onnx")
class SherpaOnnxSTT(STT):
    """pip install sherpa-onnx  - lightweight CPU/ONNX recognisers; the same code runs
    on Android/iOS tablets.

    `kind` picks the OfflineRecognizer factory and the remaining params are passed
    straight to it:
      kind: sense_voice | paraformer | whisper | transducer | nemo_ctc | ...
    Example (SenseVoice): kind=sense_voice, model=..., tokens=..., num_threads=4
    Language is fixed by the chosen model, so the UI language hint is ignored here.
    """

    def __init__(self, kind: str = "sense_voice", **params):
        import sherpa_onnx

        factory = getattr(sherpa_onnx.OfflineRecognizer, f"from_{kind}", None)
        if factory is None:
            raise ValueError(f"sherpa_onnx.OfflineRecognizer has no factory from_{kind}")
        resolved = {k: (resolve_path(v) if k in _PATH_KEYS and isinstance(v, str) else v)
                    for k, v in params.items()}
        self.rec = factory(**resolved)

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> Transcript:
        s = self.rec.create_stream()
        s.accept_waveform(SAMPLE_RATE, np.asarray(audio, dtype=np.float32))
        self.rec.decode_stream(s)
        return Transcript(s.result.text.strip(), None)
