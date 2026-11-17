"""Owns the active components, hot-swapping, enrollment and per-session pipelines."""
from __future__ import annotations

import dataclasses
import importlib
import threading
from typing import Dict, Optional

import numpy as np

from . import backends  # noqa: F401  (registers built-in backends)
from . import registry
from .base import SAMPLE_RATE
from .config import resolve_path
from .pipeline import Pipeline, PipelineParams, extract_speech
from .profiles import ProfileStore
from .registry import KINDS


class Engine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        for mod in cfg.get("plugins") or []:  # user-supplied backend modules
            importlib.import_module(mod)

        self._lock = threading.RLock()
        self._inst: Dict[str, object] = {}
        self._emb_cache: Dict[tuple, np.ndarray] = {}
        self.store = ProfileStore(resolve_path(cfg.get("profiles_dir", "profiles")))
        self.defaults = PipelineParams.from_dict(cfg.get("pipeline"))

        self.specs: Dict[str, dict] = {}
        use = cfg.get("use") or {}
        for kind in KINDS:
            key = use.get(kind) or ("none" if kind == "denoiser" else None)
            if key is None:
                raise ValueError(f"config `use.{kind}` is required")
            self.specs[kind] = self._resolve(kind, preset=key)

    # ---- component management ----------------------------------------
    def _resolve(self, kind: str, preset: Optional[str] = None, name: Optional[str] = None,
                 params: Optional[dict] = None) -> dict:
        presets = (self.cfg.get("presets") or {}).get(kind) or {}
        if preset and preset in presets:
            p = presets[preset]
            return {"name": p["name"], "params": dict(p.get("params") or {}), "preset": preset}
        name = name or preset  # a bare backend name is accepted too
        if name not in registry.available(kind):
            raise KeyError(f"Unknown {kind} preset/backend {name!r}")
        return {"name": name, "params": dict(params or {}), "preset": None}

    def get(self, kind: str):
        inst = self._inst.get(kind)
        if inst is None:
            with self._lock:
                inst = self._inst.get(kind)
                if inst is None:
                    s = self.specs[kind]
                    inst = registry.create(kind, s["name"], s["params"])
                    self._inst[kind] = inst
        return inst

    def select(self, kind: str, preset: Optional[str] = None, name: Optional[str] = None,
               params: Optional[dict] = None) -> None:
        """Switch a component at runtime. Loads the new model first, then swaps; running
        sessions pick it up on their next utterance."""
        if kind not in KINDS:
            raise KeyError(f"unknown kind {kind!r}")
        spec = self._resolve(kind, preset, name, params)
        new = registry.create(kind, spec["name"], spec["params"])  # slow part, outside the lock
        with self._lock:
            self._inst[kind] = new
            self.specs[kind] = spec
            if kind == "speaker":
                self._emb_cache.clear()

    def make_vad(self):
        s = self.specs["vad"]
        return registry.create("vad", s["name"], s["params"])  # stateful -> one per session

    def warmup(self) -> None:
        for kind in KINDS:
            self.get(kind)

    # ---- enrollment ------------------------------------------------------
    def _embed_enrollment(self, speech: np.ndarray):
        spk = self.get("speaker")
        win, hop = int(3.0 * SAMPLE_RATE), int(1.5 * SAMPLE_RATE)
        need = int(spk.min_seconds * SAMPLE_RATE)
        if speech.size <= int(win * 1.5):
            chunks = [speech]
        else:
            chunks = [speech[i:i + win] for i in range(0, speech.size - win + 1, hop)]
        embs = []
        for c in chunks:
            if c.size < need:
                c = np.tile(c, int(np.ceil(need / c.size)))[:need]
            embs.append(spk.embed(c))
        E = np.stack(embs)
        centroid = E.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims = E @ centroid
        report = {"chunks": len(chunks), "consistency": round(float(sims.mean()), 3),
                  "min_similarity": round(float(sims.min()), 3)}
        return centroid.astype(np.float32), report

    def enroll(self, pid: str, audio: np.ndarray, append: bool = False,
               display_name: Optional[str] = None) -> dict:
        self.store._dir(pid)  # validates id
        if append and self.store.exists(pid):
            audio = np.concatenate([self.store.load_audio(pid),
                                    np.zeros(int(0.5 * SAMPLE_RATE), np.float32), audio])
        d = self.defaults
        params = PipelineParams(vad_threshold=d.vad_threshold, min_speech_ms=200, min_silence_ms=400,
                                speech_pad_ms=100, max_utterance_s=60.0)
        speech = extract_speech(audio, self.make_vad(), params)
        secs = speech.size / SAMPLE_RATE
        min_s = float((self.cfg.get("enrollment") or {}).get("min_speech_seconds", 3.0))
        if secs < min_s:
            raise ValueError(f"Only {secs:.1f}s of speech found; need at least {min_s:.0f}s "
                             f"(10-30s of clean, single-speaker speech works best).")
        with self._lock:
            tag = self.get("speaker").tag
            self.store.save_audio(pid, speech)
            self.store.clear_embeddings(pid)
            for k in [k for k in self._emb_cache if k[0] == pid]:
                del self._emb_cache[k]
            emb, report = self._embed_enrollment(speech)
            self.store.save_embedding(pid, tag, emb)
            self._emb_cache[(pid, tag)] = emb
            old = self.store.meta(pid)
            self.store.save_meta(pid, {"name": display_name or old.get("name") or pid,
                                       "created": old.get("created") or self.store.now(),
                                       "updated": self.store.now(), "duration_s": round(secs, 1),
                                       "last_enroll": report})
        warning = None
        if report["chunks"] > 1 and report["consistency"] < 0.45:
            warning = "Enrollment audio looks inconsistent - it may contain more than one speaker."
        return {"profile": pid, "speech_seconds": round(secs, 1), "report": report, "warning": warning}

    def profile_embedding(self, pid: str) -> np.ndarray:
        tag = self.get("speaker").tag
        key = (pid, tag)
        emb = self._emb_cache.get(key)
        if emb is not None:
            return emb
        with self._lock:
            emb = self.store.load_embedding(pid, tag)
            if emb is None:  # encoder was switched since enrollment -> re-embed stored audio
                emb, _ = self._embed_enrollment(self.store.load_audio(pid))
                self.store.save_embedding(pid, tag, emb)
            self._emb_cache[key] = emb
        return emb

    def delete_profile(self, pid: str) -> None:
        with self._lock:
            self.store.delete(pid)
            for k in [k for k in self._emb_cache if k[0] == pid]:
                del self._emb_cache[k]

    # ---- sessions / introspection -----------------------------------------
    def new_session(self, profile_id: str, language: Optional[str] = None) -> Pipeline:
        if not self.store.exists(profile_id):
            raise KeyError(f"profile {profile_id!r} not found - enroll a target voice first")
        self.get("speaker"), self.get("stt")  # make sure models are loaded
        self.profile_embedding(profile_id)
        params = dataclasses.replace(self.defaults)
        return Pipeline(self, profile_id, self.make_vad(), params, language or params.language)

    def describe(self) -> dict:
        presets_cfg = self.cfg.get("presets") or {}
        return {
            "active": {k: dict(self.specs[k]) for k in KINDS},
            "available": {k: registry.available(k) for k in KINDS},
            "presets": {k: {key: v.get("label", key) for key, v in (presets_cfg.get(k) or {}).items()}
                        for k in KINDS},
            "profiles": self.store.list(),
            "pipeline": dataclasses.asdict(self.defaults),
        }
