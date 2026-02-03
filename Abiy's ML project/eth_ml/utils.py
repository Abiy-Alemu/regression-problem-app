"""
Utility functions for configuration and model persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


# Project root is the parent of this `eth_ml` package directory
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"


def get_models_dir() -> Path:
    """Return the default directory used to store trained models."""
    return MODELS_DIR


def save_joblib(obj: Any, path: Path | str) -> None:
    """Persist an object to disk using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path | str) -> Any:
    """Load a joblib-persisted object."""
    path = Path(path)
    return joblib.load(path)


def save_json(data: Any, path: Path | str) -> None:
    """Save a Python object as JSON (UTF-8, pretty-printed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path | str) -> Any:
    """Load JSON data into a Python object."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

