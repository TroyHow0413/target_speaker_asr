from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..audio import wav_bytes
from ..base import STT, Transcript
from ..registry import register


@register("stt", "openai_http")
class OpenAICompatibleSTT(STT):
    """Talk to any *locally hosted* server that implements POST /v1/audio/transcriptions
    (whisper.cpp server, speaches, LocalAI, vLLM, ...). This lets you run an STT
    engine in its own process / environment / container.

    Keep `base_url` on localhost or your LAN - audio is sent there unmodified.
    pip install requests
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080", endpoint: str = "/v1/audio/transcriptions",
                 model: str = "whisper-1", api_key: Optional[str] = None, timeout: float = 60.0,
                 extra_fields: Optional[Dict[str, str]] = None):
        import requests

        self._requests = requests
        self.url = base_url.rstrip("/") + endpoint
        self.model, self.timeout = model, timeout
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.extra = dict(extra_fields or {})

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> Transcript:
        data = {"model": self.model, "response_format": "json", **self.extra}
        if language:
            data["language"] = language
        r = self._requests.post(self.url, headers=self.headers, data=data, timeout=self.timeout,
                                files={"file": ("audio.wav", wav_bytes(np.asarray(audio, dtype=np.float32)), "audio/wav")})
        r.raise_for_status()
        return Transcript(r.json().get("text", "").strip(), language)
