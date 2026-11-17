from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p) -> str:
    """Make a path absolute, relative to the project root."""
    p = Path(p)
    return str(p if p.is_absolute() else ROOT / p)


def load_config(path: Optional[str] = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("server", {})
    cfg.setdefault("presets", {})
    cfg.setdefault("use", {})
    cfg.setdefault("pipeline", {})
    cfg.setdefault("plugins", [])
    return cfg
