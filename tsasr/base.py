"""Interfaces every backend implements. Audio is always mono float32 @ 16 kHz."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000


@dataclass
class Transcript:
    text: str
    language: Optional[str] = None
    extra: dict = field(default_factory=dict)


class Component(ABC):
    registry_kind = ""
    registry_name = ""
    device: str = "cpu"  # backends set this to where the model actually runs

    def close(self) -> None:  # optional cleanup
        pass


class VAD(Component):
    """Frame-level voice activity detector.

    The pipeline cuts audio into frames of `frame_samples` and asks for a speech
    probability per frame. Hysteresis / min-silence / max-length live in the
    pipeline, so a VAD backend only has to answer "is this frame speech?".
    """

    frame_samples: int = 512

    def reset(self) -> None:
        pass

    @abstractmethod
    def speech_prob(self, frame: np.ndarray) -> float: ...


class SpeakerEncoder(Component):
    """Turns an utterance into a fixed-size, L2-normalised voice embedding."""

    min_seconds: float = 1.0  # below this, embeddings are unreliable

    @property
    def tag(self) -> str:
        """Identifies the embedding space. Stored embeddings are keyed by it, so
        switching encoders automatically re-embeds the stored enrollment audio."""
        return self.registry_name

    @abstractmethod
    def embed(self, audio: np.ndarray) -> np.ndarray: ...

    @staticmethod
    def _norm(v) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        return v / (np.linalg.norm(v) + 1e-9)


class STT(Component):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> Transcript: ...


class Denoiser(Component):
    @abstractmethod
    def enhance(self, audio: np.ndarray) -> np.ndarray: ...
