from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import SAMPLE_RATE, STT, Transcript
from ..device import build_with_fallback, probe_noise
from ..registry import register


@register("stt", "hf_transformers")
class HFTransformersSTT(STT):
    """transformers + torch - any Hugging Face ASR checkpoint, which is the escape hatch for
    languages Whisper handles poorly:

      facebook/mms-1b-all               1100+ languages (language = ISO-639-3, e.g. "eng", "zsm", "cmn")
      openai/whisper-*  or fine-tunes   language = ISO-639-1 ("ms", "ta", ...)
      facebook/wav2vec2-*, other CTC    language-specific checkpoints

    device: auto | cpu | cuda | xpu | mps. float16 is used on CUDA automatically.
    `generate_kwargs` is forwarded to the pipeline (whisper-style models only).
    """

    def __init__(self, model: str = "openai/whisper-small", device: str = "auto",
                 generate_kwargs: Optional[dict] = None, chunk_length_s: Optional[int] = None,
                 pass_language: Optional[bool] = None, **pipeline_kwargs):
        from transformers import pipeline

        self.generate_kwargs = dict(generate_kwargs or {})
        low = model.lower()
        self.is_mms = "mms" in low
        self.pass_language = ("whisper" in low) if pass_language is None else pass_language

        def build(dev: str):
            kw = dict(pipeline_kwargs)
            if chunk_length_s:
                kw["chunk_length_s"] = chunk_length_s
            if dev.startswith("cuda") and "torch_dtype" not in kw:
                import torch

                kw["torch_dtype"] = torch.float16
            self.device = dev
            self._mms_lang: Optional[str] = None
            self.pipe = pipeline("automatic-speech-recognition", model=model, device=dev, **kw)
            return self

        _, self.device = build_with_fallback(build, device, "torch", lambda _: self.transcribe(probe_noise(), None))

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> Transcript:
        gk = dict(self.generate_kwargs)
        if language:
            if self.is_mms:
                if language != self._mms_lang:
                    self.pipe.tokenizer.set_target_lang(language)
                    self.pipe.model.load_adapter(language)
                    self._mms_lang = language
            elif self.pass_language:
                gk["language"] = language
        inputs = {"raw": np.asarray(audio, dtype=np.float32), "sampling_rate": SAMPLE_RATE}
        out = self.pipe(inputs, generate_kwargs=gk) if gk else self.pipe(inputs)
        return Transcript(out["text"].strip(), language)
