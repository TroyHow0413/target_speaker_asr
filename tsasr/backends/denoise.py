from __future__ import annotations

import numpy as np

from ..base import SAMPLE_RATE, Denoiser
from ..registry import register


@register("denoiser", "none")
class NoDenoiser(Denoiser):
    """Pass-through (default; modern ASR is usually trained on noisy audio)."""

    def enhance(self, audio: np.ndarray) -> np.ndarray:
        return audio


@register("denoiser", "deepfilternet")
class DeepFilterNet(Denoiser):
    """pip install deepfilternet torch torchaudio  - real-time speech enhancement.
    Try it only if noise is heavy and always A/B test: denoising can hurt ASR."""

    def __init__(self, post_filter: bool = False):
        import torch
        import torchaudio
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
