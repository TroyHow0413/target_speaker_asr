"""Audio helpers (decode uploads, resample, PCM conversion)."""
from __future__ import annotations

import io
import shutil
import subprocess
from math import gcd

import numpy as np

from .base import SAMPLE_RATE


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    pcm = pcm[: len(pcm) // 2 * 2]
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def resample(audio: np.ndarray, sr_in: int, sr_out: int = SAMPLE_RATE) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    g = gcd(sr_in, sr_out)
    return resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)


def wav_bytes(audio: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _ffmpeg_decode(data: bytes, sr: int) -> np.ndarray:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("Could not decode this audio format. Upload WAV/FLAC/OGG, or install ffmpeg.")
    p = subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
        input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed to decode the upload: " + p.stderr.decode(errors="replace")[:300])
    return np.frombuffer(p.stdout, dtype="<f4").copy()


def load_audio_bytes(data: bytes, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode an uploaded file (wav/flac/ogg via soundfile, anything else via ffmpeg)
    into mono float32 at `target_sr`."""
    try:
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return resample(audio.mean(axis=1), sr, target_sr)
    except Exception:
        return _ffmpeg_decode(data, target_sr)
