from __future__ import annotations

import numpy as np

from ..base import VAD
from ..registry import register


@register("vad", "energy")
class EnergyVAD(VAD):
    """Zero-dependency fallback: adaptive-noise-floor energy gate. Fine for quiet
    rooms, much weaker than Silero in real noise."""

    frame_samples = 480  # 30 ms

    def __init__(self, margin_db: float = 12.0, floor_db: float = -60.0, slope: float = 3.0):
        self.margin_db, self.floor_db, self.slope = margin_db, floor_db, slope
        self.noise_db = None

    def reset(self) -> None:
        self.noise_db = None

    def speech_prob(self, frame: np.ndarray) -> float:
        db = 20.0 * np.log10(float(np.sqrt(np.mean(frame ** 2))) + 1e-8)
        if self.noise_db is None:
            self.noise_db = max(db, self.floor_db)
        if db < self.noise_db:
            self.noise_db = 0.9 * self.noise_db + 0.1 * db  # follow drops quickly
        else:
            self.noise_db += 0.001 * (db - self.noise_db)  # drift up slowly
        x = (db - (self.noise_db + self.margin_db)) / self.slope
        return float(1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))))
