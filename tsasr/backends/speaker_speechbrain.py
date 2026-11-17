from __future__ import annotations

import re

import numpy as np

from ..base import SpeakerEncoder
from ..config import resolve_path
from ..device import build_with_fallback, probe_noise
from ..registry import register


@register("speaker", "speechbrain_ecapa")
class SpeechBrainECAPA(SpeakerEncoder):
    """pip/uv: speechbrain + torch (192-dim ECAPA-TDNN).
    device: auto | cpu | cuda | cuda:1 | xpu | mps"""

    min_seconds = 1.0

    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb",
                 savedir: str = "models/spkrec-ecapa-voxceleb", device: str = "auto"):
        import torch

        try:
            from speechbrain.inference.speaker import EncoderClassifier  # speechbrain >= 1.0
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier

        self._torch = torch
        self._source = source

        def build(dev: str):
            kwargs = dict(source=source, savedir=resolve_path(savedir), run_opts={"device": dev})
            try:  # avoid the symlink-privilege problem on Windows
                from speechbrain.utils.fetching import LocalStrategy

                kwargs["local_strategy"] = LocalStrategy.COPY
            except Exception:
                pass
            self.device = dev
            self.model = EncoderClassifier.from_hparams(**kwargs)
            return self

        _, self.device = build_with_fallback(build, device, "torch", lambda _: self.embed(probe_noise()))

    @property
    def tag(self) -> str:
        return "ecapa_" + re.sub(r"[^A-Za-z0-9]+", "_", self._source)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            wav = self._torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0).to(self.device)
            emb = self.model.encode_batch(wav)
        return self._norm(emb.squeeze().float().cpu().numpy())
