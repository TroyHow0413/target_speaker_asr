"""Importing this package registers every built-in backend.

Backend modules only import their heavy dependency (torch, funasr, ...) inside
__init__, so a missing optional package only breaks *that* backend.
"""
from . import (  # noqa: F401
    denoise,
    speaker_sherpa,
    speaker_speechbrain,
    stt_faster_whisper,
    stt_funasr,
    stt_hf,
    stt_mlx_whisper,
    stt_openai_http,
    stt_sherpa,
    vad_energy,
    vad_silero,
)
