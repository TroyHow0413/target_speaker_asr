from __future__ import annotations

import numpy as np

from ..base import SAMPLE_RATE, VAD
from ..registry import register


@register("vad", "silero")
class SileroVAD(VAD):
    """pip install silero-vad   (torch, or onnxruntime with onnx=true)"""

    frame_samples = 512  # required by Silero at 16 kHz

    def __init__(self, onnx: bool = False):
        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        self.model = load_silero_vad(onnx=onnx)

    def reset(self) -> None:
        self.model.reset_states()

    def speech_prob(self, frame: np.ndarray) -> float:
        with self._torch.no_grad():
            return float(self.model(self._torch.from_numpy(frame), SAMPLE_RATE).item())
