"""Owns the active components, hot-swapping, enrollment and per-session pipelines."""
from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import backends  # noqa: F401  (registers built-in backends)
from . import registry
from .base import SAMPLE_RATE
from .config import resolve_path
from .custom_models import ModelHistory, check_device, check_hf_model
from .device import describe_device, hardware_summary
from .pipeline import TUNABLE, Pipeline, PipelineParams, extract_speech
from .profiles import ProfileStore
from .registry import KINDS


# What the web UI offers for tweaking a backend's model / device after picking it.
# model_param = name of the constructor argument holding the model; models = suggestions
# (any other id or local path can be typed); devices = valid `device` values (None = no device option).
_DEV = ["auto", "cpu", "cuda", "xpu", "mps"]
BACKEND_UI = {
    "stt": {
        "faster_whisper": {"model_param": "model_size_or_path", "devices": ["auto", "cpu", "cuda"],
                           "models": ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo",
                                      "distil-large-v3"]},
        "mlx_whisper": {"model_param": "model", "devices": None,
                        "models": ["mlx-community/whisper-small-mlx", "mlx-community/whisper-large-v3-mlx",
                                   "mlx-community/whisper-large-v3-turbo"]},
        "funasr": {"model_param": "model", "devices": _DEV,
                   "models": ["iic/SenseVoiceSmall", "paraformer-zh"]},
        # custom=True: the user brings their own model (type an id / folder, or upload a .zip) and the
        # models that loaded successfully are remembered ("previously used" in the UI)
        "hf_transformers": {"model_param": "model", "devices": _DEV, "custom": True,
                            "models": ["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small",
                                       "openai/whisper-medium", "openai/whisper-large-v3",
                                       "facebook/mms-1b-all"]},
        "openai_http": {"model_param": "model", "devices": None, "models": ["whisper-1"]},
    },
    "speaker": {
        "speechbrain_ecapa": {"model_param": "source", "devices": _DEV,
                              "models": ["speechbrain/spkrec-ecapa-voxceleb"]},
    },
}


class Engine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        for mod in cfg.get("plugins") or []:  # user-supplied backend modules
            importlib.import_module(mod)

        self._lock = threading.RLock()
        self._inst: Dict[str, object] = {}
        self._emb_cache: Dict[tuple, np.ndarray] = {}
        self.store = ProfileStore(resolve_path(cfg.get("profiles_dir", "profiles")))
        self.history = ModelHistory(resolve_path(cfg.get("history_file", "models/model_history.json")))
        self.defaults = PipelineParams.from_dict(cfg.get("pipeline"))

        self.specs: Dict[str, dict] = {}
        use = cfg.get("use") or {}
        for kind in KINDS:
            key = use.get(kind) or ("none" if kind == "denoiser" else None)
            if key is None:
                raise ValueError(f"config `use.{kind}` is required")
            self.specs[kind] = self._resolve(kind, preset=key)
        self._config_specs = {k: dict(v) for k, v in self.specs.items()}  # what config.yaml asks for

        # Choices made in the web UI survive a restart: they live in a small state file that
        # overrides config.yaml for exactly the things the user changed (models, pipeline overrides).
        self._state_path = Path(resolve_path(cfg.get("state_file", "models/active_state.json")))
        self._pipeline_overrides: Dict[str, object] = {}
        self._from_state: set = set()
        self._load_state()

        # idle offload: models are loaded lazily by get(), so dropping them is safe at any time
        self._last_used = time.monotonic()
        self._active_sessions = 0

    # ---- idle offload ---------------------------------------------------
    def session_opened(self) -> None:
        with self._lock:
            self._active_sessions += 1
            self._last_used = time.monotonic()

    def session_closed(self) -> None:
        with self._lock:
            self._active_sessions = max(0, self._active_sessions - 1)
            self._last_used = time.monotonic()

    def unload(self) -> bool:
        """Free all loaded models (and the GPU memory they hold). They reload on next use."""
        with self._lock:
            if not self._inst:
                return False
            insts, self._inst = list(self._inst.values()), {}
        for inst in insts:
            try:
                inst.close()
            except Exception:
                logging.getLogger("tsasr.engine").exception("error while closing a model")
        del insts
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logging.getLogger("tsasr.engine").info("models unloaded (idle) - GPU memory released")
        return True

    def unload_if_idle(self, idle_s: float) -> bool:
        with self._lock:
            idle = self._active_sessions == 0 and self._inst and time.monotonic() - self._last_used >= idle_s
        return self.unload() if idle else False

    # ---- persisted UI choices ------------------------------------------
    def _load_state(self) -> None:
        log = logging.getLogger("tsasr.engine")
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            log.warning("ignoring unreadable state file %s", self._state_path)
            return
        for kind, spec in (data.get("specs") or {}).items():
            if kind in KINDS and isinstance(spec, dict) and spec.get("name") in registry.available(kind) \
                    and isinstance(spec.get("params"), dict):
                self.specs[kind] = {"name": spec["name"], "params": dict(spec["params"]),
                                    "preset": spec.get("preset") if spec.get("preset") in
                                    ((self.cfg.get("presets") or {}).get(kind) or {}) else None}
                self._from_state.add(kind)
        try:
            self.update_defaults(save=False, **(data.get("pipeline") or {}))
        except (ValueError, TypeError):
            log.warning("ignoring invalid pipeline settings in %s", self._state_path)
        if self._from_state or self._pipeline_overrides:
            log.info("restored saved settings from %s (models: %s)", self._state_path,
                     ", ".join(f"{k}={self.specs[k]['name']}" for k in sorted(self._from_state)) or "none")

    def _save_state(self) -> None:
        data = {"specs": {k: self.specs[k] for k in KINDS if self.specs[k] != self._config_specs[k]},
                "pipeline": self._pipeline_overrides}
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self._state_path)
        except OSError:
            logging.getLogger("tsasr.engine").exception("could not save %s", self._state_path)

    def update_defaults(self, save: bool = True, **kw) -> Dict[str, object]:
        """Change pipeline defaults for every *new* session (and so for the OpenAI API) and remember them.
        Only keys the user changed are stored, so later edits of the others in config.yaml still apply."""
        applied: Dict[str, object] = {}
        for k, v in kw.items():
            if k not in TUNABLE:
                raise ValueError(f"unknown pipeline setting {k!r}")
            cur = getattr(self.defaults, k)
            if k == "min_verify_s":
                v = None if v in (None, "") else float(v)
            elif k == "language":
                v = str(v).strip() or None if v is not None else None
            elif k == "short_policy":
                if v not in ("tile", "reject", "accept"):
                    raise ValueError("short_policy must be tile, reject or accept")
            elif isinstance(cur, bool):
                v = bool(v)
            elif isinstance(cur, int):
                v = int(v)
            elif isinstance(cur, float):
                v = float(v)
            applied[k] = v
        for k, v in applied.items():
            setattr(self.defaults, k, v)
        self._pipeline_overrides.update(applied)
        if save and applied:
            self._save_state()
        return applied

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
        self._last_used = time.monotonic()
        inst = self._inst.get(kind)
        if inst is None:
            with self._lock:
                inst = self._inst.get(kind)
                if inst is None:
                    s = self.specs[kind]
                    log = logging.getLogger("tsasr.engine")
                    log.info("loading %s: %s %s ...", kind, s["name"], s["params"])
                    t0 = time.perf_counter()
                    try:
                        inst = registry.create(kind, s["name"], s["params"])
                    except Exception:
                        if kind not in self._from_state:
                            raise
                        # the model saved from the UI no longer loads (deleted / moved): use config.yaml's
                        log.exception("saved %s model failed to load - falling back to config.yaml", kind)
                        self._from_state.discard(kind)
                        self.specs[kind] = dict(self._config_specs[kind])
                        self._save_state()
                        s = self.specs[kind]
                        inst = registry.create(kind, s["name"], s["params"])
                    self._inst[kind] = inst
                    log.info("loaded %s in %.1fs on %s", kind, time.perf_counter() - t0, describe_device(inst.device))
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
            self._from_state.discard(kind)
            if kind == "speaker":
                self._emb_cache.clear()
            self._save_state()
        info = BACKEND_UI.get(kind, {}).get(spec["name"], {})
        model = spec["params"].get(info.get("model_param", ""))
        if info.get("custom") and model:  # only successful loads get here -> remember them
            self.history.add(spec["name"], str(model), str(spec["params"].get("device", "auto")))

    def client_params(self, kind: str, name: str, params: Optional[dict]) -> dict:
        """What a network client may set when it picks an engine: the model and the device, nothing else.
        All other arguments (compute_type, base_url, trust_remote_code, file paths ...) come from config.yaml:
        the active params if this engine is already active, otherwise the first preset that uses it."""
        active = self.specs[kind]  # KeyError for an unknown kind -> reported as a 400
        if active["name"] == name:
            base = dict(active["params"])
        else:
            presets = (self.cfg.get("presets") or {}).get(kind) or {}
            base = next((dict(p.get("params") or {}) for p in presets.values() if p["name"] == name), {})
        info = BACKEND_UI.get(kind, {}).get(name)
        params = params or {}
        if info is None:  # not configurable from the UI: keep the config values
            return base
        mp = info["model_param"]
        if mp in params:
            model = str(params[mp]).strip()
            if not model or len(model) > 300 or any(ord(c) < 32 for c in model):
                raise ValueError("Invalid model name.")
            base[mp] = check_hf_model(model) if info.get("custom") else model
        if info.get("devices") and "device" in params:
            dev = check_device(params["device"])
            if dev.split(":")[0] not in info["devices"]:
                raise ValueError(f"{name} does not support device {dev!r}")
            base["device"] = dev
        return base

    def make_vad(self):
        s = self.specs["vad"]
        return registry.create("vad", s["name"], s["params"])  # stateful -> one per session

    def warmup(self) -> None:
        logging.getLogger("tsasr.engine").info("detected hardware: %s", hardware_summary())
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

    def refresh_profile(self, pid: str) -> None:
        """Drop the cached embedding so it is re-read from disk (another process may have re-enrolled)."""
        with self._lock:
            for k in [k for k in self._emb_cache if k[0] == pid]:
                del self._emb_cache[k]

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
            "backend_ui": BACKEND_UI,
            # params of each config preset: the UI uses them as the starting point for a backend
            "preset_specs": {k: {key: {"name": v["name"], "params": v.get("params") or {}}
                                 for key, v in (presets_cfg.get(k) or {}).items()} for k in KINDS},
            "profiles": self.store.list(),
            "history": self.history.list(),  # custom models used before (most recent first)
            "pipeline": dataclasses.asdict(self.defaults),
            # where each loaded model actually runs (None = not loaded yet)
            "devices": {k: (describe_device(getattr(self._inst[k], "device", "cpu")) if k in self._inst else None)
                        for k in KINDS},
            "hardware": hardware_summary(),
        }
