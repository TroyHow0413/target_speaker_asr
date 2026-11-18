from __future__ import annotations

import logging

import numpy as np

from ..base import SAMPLE_RATE, VAD
from ..registry import register

log = logging.getLogger("tsasr.vad")


@register("vad", "silero")
class SileroVAD(VAD):
    """Silero VAD. Always runs on CPU.

    onnx=True (default) uses onnxruntime, which Silero pins to ONE thread - ideal for its tiny
    512-sample frames. The TorchScript model (onnx=False) is dramatically slower on many-core CPUs
    because PyTorch spreads these microscopic ops over every core, so in that mode we cap PyTorch
    to `torch_threads` threads (this is a process-wide PyTorch setting).
    """

    frame_samples = 512  # required by Silero at 16 kHz

    def __init__(self, onnx: bool = True, torch_threads: int = 1):
        import torch
        from silero_vad import load_silero_vad

        if onnx:
            try:
                import onnxruntime  # noqa: F401
            except ImportError:
                log.warning("onnxruntime is not installed - falling back to the slower TorchScript Silero model")
                onnx = False
        if not onnx and torch_threads:
            torch.set_num_threads(int(torch_threads))
        self._torch = torch
        self.model = load_silero_vad(onnx=onnx)
        self.device = "cpu (onnx)" if onnx else "cpu (torch)"

    def reset(self) -> None:
        self.model.reset_states()

    def speech_prob(self, frame: np.ndarray) -> float:
        with self._torch.no_grad():
            return float(self.model(self._torch.from_numpy(frame), SAMPLE_RATE).item())
