"""Near-real-time pipeline:

    PCM chunks -> frame VAD -> utterance segmenter -> (denoise) -> speaker verify
               -> accepted? -> STT -> transcript event

Only utterances that match the enrolled voice ever reach the STT model.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np

from .audio import pcm16_to_float
from .base import SAMPLE_RATE, VAD

if TYPE_CHECKING:
    from .engine import Engine


@dataclass
class PipelineParams:
    vad_threshold: float = 0.5
    min_speech_ms: int = 250        # utterances with less voiced audio are dropped
    min_silence_ms: int = 500       # silence that closes an utterance
    speech_pad_ms: int = 200        # context kept before/after speech
    max_utterance_s: float = 20.0   # force a cut for very long monologues
    speaker_verify: bool = True     # False -> target-speaker filter OFF: everyone who speaks is transcribed
    similarity_threshold: float = 0.5   # cosine similarity needed to count as the target
    min_verify_s: Optional[float] = None  # None -> use the encoder's own minimum
    short_policy: str = "tile"      # utterance shorter than min_verify_s: tile | reject | accept
    language: Optional[str] = None
    denoise_for_verify: bool = False
    denoise_for_stt: bool = False
    emit_rejected: bool = True
    # Speaker-change cutting: while someone keeps talking without pauses, re-check who is speaking
    # on a sliding window and cut the utterance when the speaker changes (so a neighbour who never
    # stops talking cannot swallow the target's speech into one long rejected segment).
    speaker_cut: bool = True
    speaker_check_ms: int = 400     # how often to re-check the speaker during continuous speech
    speaker_window_ms: int = 1500   # audio length used for each check
    speaker_flip_checks: int = 2    # consecutive disagreeing checks needed before cutting
    speaker_cut_margin: float = 0.1  # window threshold = similarity_threshold - margin (short windows are noisy)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineParams":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


TUNABLE = {f.name for f in fields(PipelineParams)}


class Segmenter:
    """Turns a stream of audio into complete utterances using frame-level VAD."""

    def __init__(self, vad: VAD, params: PipelineParams, sample_rate: int = SAMPLE_RATE):
        self.vad, self.p, self.sr = vad, params, sample_rate
        self.fs = vad.frame_samples
        self.frame_ms = 1000.0 * self.fs / sample_rate
        self._buf = np.zeros(0, dtype=np.float32)
        self._frame_idx = 0
        self._preroll: deque = deque()
        # optional: audio window -> True (target) / False (other) / None (can't tell)
        self.classifier: Optional[Callable[[np.ndarray], Optional[bool]]] = None
        self._reset_utt()

    def _reset_utt(self) -> None:
        self._in_speech = False
        self._frames: list = []
        self._silence = 0
        self._voiced = 0
        self._start_frame = 0
        self._label: Optional[bool] = None   # speaker label of the current utterance
        self._flips = 0
        self._since_check = 0
        self._agree_end = 0                   # frame count (in _frames) where the label last agreed

    def feed(self, audio: np.ndarray) -> list:
        """Returns events: ("start", t) | ("segment", t0, t1, audio) | ("discard",)"""
        events: list = []
        audio = np.asarray(audio, dtype=np.float32)
        self._buf = np.concatenate([self._buf, audio]) if self._buf.size else audio
        n = self._buf.size // self.fs
        for i in range(n):
            self._on_frame(self._buf[i * self.fs:(i + 1) * self.fs], events)
        self._buf = self._buf[n * self.fs:].copy()
        return events

    def flush(self) -> list:
        events: list = []
        if self._in_speech:
            self._finish(events)
        return events

    def _on_frame(self, frame: np.ndarray, events: list) -> None:
        p = self.p
        prob = self.vad.speech_prob(frame)
        idx = self._frame_idx
        self._frame_idx += 1
        # hysteresis: easier to stay in speech than to enter it
        speech = prob >= (p.vad_threshold - 0.15 if self._in_speech else p.vad_threshold)

        if not self._in_speech:
            if speech:
                self._in_speech = True
                self._start_frame = idx - len(self._preroll)
                self._frames = list(self._preroll) + [frame]
                self._preroll.clear()
                self._silence, self._voiced = 0, 1
                events.append(("start", self._start_frame * self.fs / self.sr))
            else:
                self._preroll.append(frame)
                keep = max(1, int(p.speech_pad_ms / self.frame_ms))
                while len(self._preroll) > keep:
                    self._preroll.popleft()
            return

        self._frames.append(frame)
        if speech:
            self._silence, self._voiced = 0, self._voiced + 1
        else:
            self._silence += 1
        if speech and self.classifier is not None and p.speaker_cut and p.speaker_verify:
            self._check_speaker(events)
        dur_s = len(self._frames) * self.fs / self.sr
        if self._silence * self.frame_ms >= p.min_silence_ms or dur_s >= p.max_utterance_s:
            self._finish(events)

    def _check_speaker(self, events: list) -> None:
        p = self.p
        self._since_check += 1
        nwin = max(1, int(p.speaker_window_ms / self.frame_ms))
        if self._since_check * self.frame_ms < p.speaker_check_ms or len(self._frames) < nwin:
            return
        self._since_check = 0
        lab = self.classifier(np.concatenate(self._frames[-nwin:]))
        if lab is None:
            return
        if self._label is None or lab == self._label:
            self._label, self._flips = lab, 0
            self._agree_end = len(self._frames)
            return
        self._flips += 1
        if self._flips < p.speaker_flip_checks:
            return
        # Speaker changed: close everything up to where the old speaker last matched, and
        # carry the rest over into a fresh utterance (it will be labelled on its own).
        cut = self._agree_end
        head, tail = self._frames[:cut], self._frames[cut:]
        t0 = self._start_frame * self.fs / self.sr
        if head and len(head) * self.frame_ms >= p.min_speech_ms:
            audio = np.concatenate(head)
            events.append(("segment", t0, t0 + audio.size / self.sr, audio))
        else:
            events.append(("discard",))
        self._start_frame += cut
        self._frames = tail
        self._voiced = len(tail)
        self._silence = 0
        self._label, self._flips, self._since_check = lab, 0, 0
        self._agree_end = len(tail)
        events.append(("start", self._start_frame * self.fs / self.sr))

    def _finish(self, events: list) -> None:
        p = self.p
        pad = int(p.speech_pad_ms / self.frame_ms)
        trim = max(0, self._silence - pad)
        frames = self._frames[: len(self._frames) - trim] if trim else self._frames
        if frames and self._voiced * self.frame_ms >= p.min_speech_ms:
            audio = np.concatenate(frames)
            t0 = self._start_frame * self.fs / self.sr
            events.append(("segment", t0, t0 + audio.size / self.sr, audio))
        else:
            events.append(("discard",))
        self._reset_utt()


def extract_speech(audio: np.ndarray, vad: VAD, params: PipelineParams) -> np.ndarray:
    """Offline helper: keep only the speech parts of `audio` (used for enrollment)."""
    vad.reset()
    seg = Segmenter(vad, params)
    events = seg.feed(audio) + seg.flush()
    parts = [e[3] for e in events if e[0] == "segment"]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


class Pipeline:
    """One live session (one websocket). Heavy models are shared via the Engine."""

    def __init__(self, engine: "Engine", profile_id: str, vad: VAD, params: PipelineParams,
                 language: Optional[str] = None):
        self.engine, self.profile_id, self.p = engine, profile_id, params
        self.language = language
        self.seg = Segmenter(vad, params)
        self.seg.classifier = self._window_is_target

    # ---- public API -----------------------------------------------------
    def update(self, **kw) -> None:
        """Live-tune parameters (threshold slider etc.)."""
        for k, v in kw.items():
            if k not in TUNABLE:
                continue
            cur = getattr(self.p, k)
            if isinstance(cur, bool):
                v = bool(v)
            elif isinstance(cur, int):
                v = int(v)
            elif isinstance(cur, float):
                v = float(v)
            setattr(self.p, k, v)
        if "language" in kw:
            self.language = kw["language"] or None

    def feed_pcm16(self, pcm: bytes) -> List[dict]:
        return self._handle(self.seg.feed(pcm16_to_float(pcm)))

    def flush(self) -> List[dict]:
        return self._handle(self.seg.flush()) + [{"type": "flushed"}]

    def process_audio(self, audio: np.ndarray) -> List[dict]:
        """Offline use: run a whole clip (mono float32 @ 16 kHz) through VAD -> verify -> STT."""
        return self._handle(self.seg.feed(audio) + self.seg.flush())

    # ---- internals ------------------------------------------------------
    def _handle(self, events: list) -> List[dict]:
        out: List[dict] = []
        for ev in events:
            if ev[0] == "start":
                out.append({"type": "speech_start", "t": round(ev[1], 2)})
            elif ev[0] == "discard":
                out.append({"type": "speech_discard"})
            else:
                res = self.process_utterance(ev[3], ev[1], ev[2])
                if res["type"] == "rejected" and not self.p.emit_rejected:
                    res = {"type": "speech_discard"}
                out.append(res)
        return out

    def _window_is_target(self, audio: np.ndarray) -> Optional[bool]:
        """Cheap per-window speaker check used to cut continuous speech at speaker changes."""
        eng, p = self.engine, self.p
        spk = eng.get("speaker")
        if audio.size < int((p.min_verify_s or spk.min_seconds) * SAMPLE_RATE):
            return None
        score = float(np.dot(spk.embed(audio), eng.profile_embedding(self.profile_id)))
        return score >= p.similarity_threshold - p.speaker_cut_margin

    def _score(self, audio: np.ndarray):
        """-> (similarity or None, reason or None)"""
        eng, p = self.engine, self.p
        spk = eng.get("speaker")
        need = int((p.min_verify_s or spk.min_seconds) * SAMPLE_RATE)
        if audio.size < need:
            if p.short_policy == "reject":
                return None, "too_short"
            if p.short_policy == "accept":
                return None, "short_accepted"
            audio = np.tile(audio, int(np.ceil(need / audio.size)))[:need]
        emb = spk.embed(audio)
        ref = eng.profile_embedding(self.profile_id)
        return float(np.dot(emb, ref)), None

    def process_utterance(self, audio: np.ndarray, t0: float, t1: float) -> dict:
        eng, p = self.engine, self.p
        info = {"t0": round(t0, 2), "t1": round(t1, 2), "duration": round(t1 - t0, 2)}

        den = eng.get("denoiser") if (p.denoise_for_verify or p.denoise_for_stt) else None
        clean = den.enhance(audio) if den is not None else None
        verify_audio = clean if (clean is not None and p.denoise_for_verify) else audio
        stt_audio = clean if (clean is not None and p.denoise_for_stt) else audio

        if p.speaker_verify:
            score, reason = self._score(verify_audio)
        else:
            score, reason = None, None
        if reason == "too_short":
            return {"type": "rejected", "reason": reason, "score": None, **info}
        if score is not None and score < p.similarity_threshold:
            return {"type": "rejected", "reason": "other_speaker", "score": round(score, 3), **info}

        t = time.perf_counter()
        result = eng.get("stt").transcribe(stt_audio, self.language or p.language)
        stt_ms = int((time.perf_counter() - t) * 1000)
        from .engine import BACKEND_UI  # local import: engine imports this module
        spec = eng.specs["stt"]
        mp =BACKEND_UI.get("stt", {}).get(spec["name"], {}).get("model_param", "")
        stt_model = str(spec["params"].get(mp) or spec["name"])
        if "/" in stt_model or "\\" in stt_model:  # HF id / path -> last component
            stt_model = stt_model.replace("\\", "/").rstrip("/").split("/")[-1]
        if not result.text:
            return {"type": "rejected", "reason": "empty", "score": None if score is None else round(score, 3), **info}
        return {"type": "transcript", "text": result.text, "language": result.language,
                "score": None if score is None else round(score, 3), "stt_ms": stt_ms, "stt_model": stt_model, **info}
