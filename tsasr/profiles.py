"""On-disk store for enrolled target voices.

profiles/<id>/enroll.wav        speech-only enrollment audio (kept so we can re-embed
                                when the speaker encoder is switched)
profiles/<id>/emb_<tag>.npy     cached embedding per encoder
profiles/<id>/meta.json         display name, duration, enrollment report
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

from .base import SAMPLE_RATE

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_TAG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class ProfileStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, pid: str) -> Path:
        if not _ID_RE.match(pid or ""):
            raise ValueError("Profile id may only contain letters, digits, '_' and '-' (max 64 chars).")
        return self.root / pid

    def exists(self, pid: str) -> bool:
        return (self._dir(pid) / "enroll.wav").is_file()

    def save_audio(self, pid: str, audio: np.ndarray) -> None:
        d = self._dir(pid)
        d.mkdir(parents=True, exist_ok=True)
        sf.write(str(d / "enroll.wav"), audio, SAMPLE_RATE, subtype="PCM_16")

    def load_audio(self, pid: str) -> np.ndarray:
        if not self.exists(pid):
            raise KeyError(f"profile {pid!r} not found - enroll a voice first")
        audio, _ = sf.read(str(self._dir(pid) / "enroll.wav"), dtype="float32")
        return audio

    def _emb_file(self, pid: str, tag: str) -> Path:
        return self._dir(pid) / f"emb_{_TAG_RE.sub('_', tag)}.npy"

    def save_embedding(self, pid: str, tag: str, emb: np.ndarray) -> None:
        np.save(self._emb_file(pid, tag), emb)

    def load_embedding(self, pid: str, tag: str) -> Optional[np.ndarray]:
        f = self._emb_file(pid, tag)
        return np.load(f) if f.is_file() else None

    def clear_embeddings(self, pid: str) -> None:
        for f in self._dir(pid).glob("emb_*.npy"):
            f.unlink()

    def meta(self, pid: str) -> Dict:
        f = self._dir(pid) / "meta.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}

    def save_meta(self, pid: str, meta: Dict) -> None:
        (self._dir(pid) / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def list(self) -> List[Dict]:
        out = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and (d / "enroll.wav").is_file():
                m = self.meta(d.name)
                out.append({"id": d.name, "name": m.get("name", d.name),
                            "duration_s": m.get("duration_s"), "created": m.get("created")})
        return out

    def delete(self, pid: str) -> None:
        d = self._dir(pid)
        if d.is_dir():
            shutil.rmtree(d)

    @staticmethod
    def now() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")
