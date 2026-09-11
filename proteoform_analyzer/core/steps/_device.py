"""Torch device selection helper (ZeroGPU-aware).

The in-process torch steps (ESM2 embedding for the UMAP tab, ESM2 zero-shot ddG)
historically pinned ``device = "cpu"``. On a Hugging Face ZeroGPU Space a
physical GPU is available *only* inside a ``@spaces.GPU``-decorated call; the
Spaces ``app.py`` sets ``PROTEOFORM_ZEROGPU=1`` and wraps those functions, so
this helper lets them move onto CUDA when it is genuinely present and fall back
to CPU everywhere else (local installs, CI, dry-runs).

Behaviour:
  * ``PROTEOFORM_ZEROGPU`` unset or ``"0"`` -> always ``"cpu"`` (unchanged
    local behaviour; no torch import forced on the caller's fast path).
  * ``PROTEOFORM_ZEROGPU="1"`` -> ``"cuda"`` when ``torch.cuda.is_available()``
    else ``"cpu"``.

This module never imports torch at import time.
"""
from __future__ import annotations

import os


def torch_device(default: str = "cpu") -> str:
    """Return the torch device string to use for in-process model inference."""
    if os.environ.get("PROTEOFORM_ZEROGPU", "0") != "1":
        return default
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return default
