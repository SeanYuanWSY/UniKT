"""Shim: re-export restore/RestoredModel from the llm_explain package."""

from __future__ import annotations

import sys
from pathlib import Path

_LE = str(Path(__file__).resolve().parents[1] / "llm_explain")
if _LE not in sys.path:
    sys.path.insert(0, _LE)

from restore import RestoredModel, load_user_samples, restore  # noqa: F401,E402


def restore_any(run_dir: str, device: str = "cuda") -> RestoredModel:
    return restore(run_dir, device=device)
