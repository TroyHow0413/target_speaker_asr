from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import STT, Transcript
from ..registry import register


@register("stt", "funasr")
class FunASRSTT(STT):
    """pip install funasr modelscope torch torchaudio

    Strong for Chinese/Cantonese/Japanese/Korean/English:
      model="iic/SenseVoiceSmall"   multilingual (zh, yue, en, ja, ko)
      model="paraformer-zh"         Mandarin
      model="paraformer-en"         English
    Any other FunASR / ModelScope model id works too.
    """

    def __init__(self, model: str = "iic/SenseVoiceSmall", device: str = "cpu",
                 model_kwargs: Optional[dict] = None, generate_kwargs: Optional[dict] = None):
        from funasr import AutoModel

        self.model_id = model
        self.model = AutoModel(model=model, device=device, disable_update=True, **(model_kwargs or {}))
        self.generate_kwargs = dict(generate_kwargs or {})
        self.is_sensevoice = "sensevoice" in model.lower()

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
