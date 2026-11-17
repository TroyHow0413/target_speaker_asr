from __future__ import annotations

import re

import numpy as np

from ..base import SpeakerEncoder
from ..config import resolve_path
from ..registry import register


@register("speaker", "speechbrain_ecapa")
class SpeechBrainECAPA(SpeakerEncoder):
    """pip install speechbrain torch torchaudio   (192-dim ECAPA-TDNN)"""

    min_seconds = 1.0

    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb",
                 savedir: str = "models/spkrec-ecapa-voxceleb", device: str = "cpu"):
        import torch

        try:
            from speechbrain.inference.speaker import EncoderClassifier  # speechbrain >= 1.0
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier

        kwargs = dict(source=source, savedir=resolve_path(savedir), run_opts={"device": device})
        try:  # avoid symlink privileges problem on Windows
            from speechbrain.utils.fetching import LocalStrategy

            kwargs["local_strategy"] = LocalStrategy.COPY
        except Exception:
            pass
        self._torch = torch
        self._device = device
        self._source = source
        self.model = EncoderClassifier.from_hparams(**kwargs)

    @property
    def tag(self) -> str:
        return "ecapa_" + re.sub(r"[^A-Za-z0-9]+", "_", self._source)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            wav = self._torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0).to(self._device)
            emb = self.model.encode_batch(wav)
        return self._norm(emb.squeeze().cpu().numpy())
