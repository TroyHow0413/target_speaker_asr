from __future__ import annotations

import numpy as np

from ..base import SAMPLE_RATE, Denoiser
from ..registry import register


@register("denoiser", "none")
class NoDenoiser(Denoiser):
    """Pass-through (default; modern ASR is usually trained on noisy audio)."""

    def enhance(self, audio: np.ndarray) -> np.ndarray:
        return audio


def _patch_torchaudio_backend() -> None:
    """deepfilternet 0.5.x does `from torchaudio.backend.common import AudioMetaData`, a module
    that newer torchaudio removed. Provide a stand-in so the import works (only used for typing)."""
    import sys
    import types

    try:
        import torchaudio.backend.common  # noqa: F401
        return
    except ImportError:
        pass
    import torchaudio

    meta = getattr(torchaudio, "AudioMetaData", None)
    if meta is None:
        class meta:  # type: ignore[no-redef]
            pass
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = meta
    backend.common = common
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common
    torchaudio.backend = backend


@register("denoiser", "deepfilternet")
class DeepFilterNet(Denoiser):
    """pip install deepfilternet torch torchaudio  - real-time speech enhancement.
    Try it only if noise is heavy and always A/B test: denoising can hurt ASR."""

    def __init__(self, post_filter: bool = False):
        import torch
        import torchaudio
        _patch_torchaudio_backend()
        from df.enhance import enhance, init_df

        self._torch, self._ta, self._enhance = torch, torchaudio, enhance
        self.model, self.df_state, _ = init_df(post_filter=post_filter)
        self.sr = self.df_state.sr()

    def enhance(self, audio: np.ndarray) -> np.ndarray:
        torch, ta = self._torch, self._ta
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
            t = ta.functional.resample(t, SAMPLE_RATE, self.sr)
            out = self._enhance(self.model, self.df_state, t)
            out = ta.functional.resample(out, self.sr, SAMPLE_RATE)
        return out.squeeze(0).cpu().numpy().astype(np.float32)
