from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import STT, Transcript
from ..device import build_with_fallback, probe_noise
from ..registry import register


@register("stt", "funasr")
class FunASRSTT(STT):
    """funasr + modelscope + torch

    Strong for Chinese/Cantonese/Japanese/Korean/English:
      model="iic/SenseVoiceSmall"   multilingual (zh, yue, en, ja, ko)
      model="paraformer-zh"         Mandarin
      model="paraformer-en"         English
    Any other FunASR / ModelScope model id works too.

    device: auto | cpu | cuda | mps | xpu  (auto tries the GPU first and falls back to CPU if
    the model can't run there)
    """

    def __init__(self, model: str = "iic/SenseVoiceSmall", device: str = "auto",
                 model_kwargs: Optional[dict] = None, generate_kwargs: Optional[dict] = None):
        from funasr import AutoModel

        self.model_id = model
        self.generate_kwargs = dict(generate_kwargs or {})
        self.is_sensevoice = "sensevoice" in model.lower()

        def build(dev: str):
            fdev = "cuda:0" if dev == "cuda" else dev
            self.device = dev
            self.model = AutoModel(model=model, device=fdev, disable_update=True, **(model_kwargs or {}))
            return self

        _, self.device = build_with_fallback(build, device, "torch", lambda _: self.transcribe(probe_noise(), None))

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> Transcript:
        kw = dict(self.generate_kwargs)
        if self.is_sensevoice:
            kw.setdefault("language", language or "auto")
            kw.setdefault("use_itn", True)
        res = self.model.generate(input=np.asarray(audio, dtype=np.float32), cache={}, batch_size_s=60, **kw)
        text = res[0].get("text", "") if res else ""
        if "<|" in text:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess

            text = rich_transcription_postprocess(text)
        return Transcript(text.strip(), language)
