"""User-supplied models: input checks, safe zip upload, and the 'previously used' memory.

Security note: over the network a client may only choose the *model* and *device* of an engine
(see Engine.client_params). Every other backend argument (trust_remote_code, base_url, file paths ...)
comes from config.yaml, so the API cannot be used to inject arbitrary kwargs. A model is still
code-adjacent data (pickled weights): only load models you trust, and set `server.token` if anyone
else can reach the server.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import List

from .config import resolve_path

_HF_ID = re.compile(r"^[A-Za-z0-9][\w.\-]*(/[\w.\-]+)?$")
_DEVICE = re.compile(r"^(auto|cpu|mps|xpu|cuda(:\d+)?)$")
_IGNORED = {"__MACOSX", ".DS_Store"}
MAX_UNZIPPED_BYTES = 20 * 1024 ** 3


def check_device(device: str) -> str:
    device = str(device).strip().lower()
    if not _DEVICE.match(device):
        raise ValueError("device must be auto, cpu, cuda, cuda:N, xpu or mps")
    return device


def check_hf_model(model: str) -> str:
    """A Hugging Face repo id (org/name) or an existing model folder on the server (needs config.json)."""
    local = Path(resolve_path(model))
    if local.is_dir():
        if not (local / "config.json").is_file():
            raise ValueError(f"{local} has no config.json - it does not look like a Hugging Face model folder.")
        return str(local)
    if not _HF_ID.match(model):
        raise ValueError("Not a valid Hugging Face repo id (org/name) and not an existing folder on the server.")
    return model


def extract_model_zip(zip_path: str, dest_root: Path, name: str) -> Path:
    """Safely unpack an uploaded zip into dest_root/<name>/ and return the folder holding config.json."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")[:64] or "model"
    dest_root.mkdir(parents=True, exist_ok=True)
    dest, n = dest_root / safe, 1
    while dest.exists():
        n += 1
        dest = dest_root / f"{safe}-{n}"
    dest.mkdir()
    root = dest.resolve()
    try:
        with zipfile.ZipFile(zip_path) as z:
            if sum(i.file_size for i in z.infolist()) > MAX_UNZIPPED_BYTES:
                raise ValueError("Archive is too large once unpacked (limit 20 GB).")
            for info in z.infolist():
                target = (root / info.filename).resolve()
                if target != root and root not in target.parents:
                    raise ValueError("Archive contains an unsafe path.")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
        entries = [p for p in dest.iterdir() if p.name not in _IGNORED]
        model_dir = entries[0] if len(entries) == 1 and entries[0].is_dir() else dest
        if not (model_dir / "config.json").is_file():
            raise ValueError("config.json not found in the archive. Zip the model folder itself "
                             "(config.json + weights + tokenizer/preprocessor files).")
        return model_dir
    except zipfile.BadZipFile:
        shutil.rmtree(dest, ignore_errors=True)
        raise ValueError("That file is not a valid .zip archive.") from None
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise


class ModelHistory:
    """Remembers custom models that loaded successfully (shared by every client of the server)."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _read(self) -> List[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, ValueError):
            return []

    def _write(self, items: List[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def list(self) -> List[dict]:
        with self._lock:
            return sorted(self._read(), key=lambda e: e.get("last_used", ""), reverse=True)

    def add(self, backend: str, model: str, device: str) -> None:
        with self._lock:
            items = self._read()
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            for e in items:
                if e.get("backend") == backend and e.get("model") == model:
                    e.update(device=device, last_used=now, uses=int(e.get("uses", 0)) + 1)
                    break
            else:
                items.append({"backend": backend, "model": model, "device": device, "last_used": now, "uses": 1})
            self._write(items)

    def remove(self, backend: str, model: str) -> None:
        with self._lock:
            self._write([e for e in self._read() if not (e.get("backend") == backend and e.get("model") == model)])
