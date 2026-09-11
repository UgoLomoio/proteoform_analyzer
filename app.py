"""Hugging Face Spaces entry point for the Proteoform Analyzer (ZeroGPU-ready).

This file lives at the *repository root* of the Space. The actual application
code is the ``proteoform_analyzer`` Python package in the sibling folder of the
same name. Hugging Face Spaces automatically runs ``app.py`` at the repo root,
so this module:

  1. imports ``spaces`` *before* torch / CUDA (required by ZeroGPU) and installs
     a no-op shim when ``spaces`` is not available (so the same file runs
     unchanged on a normal machine, in CI, or in a local dry-run);
  2. puts the repo root on ``sys.path`` and imports the package as
     ``proteoform_analyzer`` so the package's relative imports resolve;
  3. builds the Gradio Blocks app, enables the request queue, and launches with
     ZeroGPU-safe settings (``ssr_mode=False``, ``server_name="0.0.0.0"``).

ZeroGPU notes
-------------
ZeroGPU allocates a physical GPU only for the duration of a function decorated
with ``@spaces.GPU``. In this app the only work that runs a torch model *in the
same process* is the ESM2 sequence embedding (UMAP tab) and the ESM2 zero-shot
ddG scorer; those are decorated so they transparently use CUDA on a ZeroGPU
Space and CPU everywhere else. The heavier structure tools (Boltz-2 folding /
docking, RFAntibody, BoltzGen, DiffSBDD) run as *external APIs or subprocesses*
and are therefore not GPU-accelerated by ZeroGPU in-process; configure the Boltz
API key (Space secret ``BOLTZ_API_KEY``) for those, or they skip cleanly. See
DEPLOY_HF_ZEROGPU.md for the full matrix.
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# 1) ZeroGPU: ``import spaces`` MUST happen before any torch / CUDA import.
#    When the ``spaces`` package is unavailable (local machine, CI, dry-run),
#    install a minimal shim that provides a no-op ``@spaces.GPU`` decorator so
#    the identical codebase runs without Hugging Face infrastructure.
# ---------------------------------------------------------------------------
try:
    import spaces  # noqa: F401  (import-for-side-effects + decorator source)

    _HAS_SPACES = True
except Exception:  # pragma: no cover - exercised only off-platform
    import types

    def _gpu_noop(*dargs, **dkwargs):
        """Mimic ``spaces.GPU``: usable both bare and with keyword arguments.

        Supports ``@spaces.GPU``, ``@spaces.GPU()`` and
        ``@spaces.GPU(duration=...)`` without requiring the real package.
        """
        # Called as a bare decorator: @spaces.GPU
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]

        # Called with arguments: @spaces.GPU(duration=120)
        def _wrap(fn):
            return fn

        return _wrap

    spaces = types.ModuleType("spaces")
    spaces.GPU = _gpu_noop  # type: ignore[attr-defined]
    sys.modules["spaces"] = spaces
    _HAS_SPACES = False

# Make the ESM2 in-process steps GPU-aware only when a real ZeroGPU runtime is
# present. The package reads this env var (see core/steps/_device.py) to decide
# whether to move torch models onto CUDA. Off-platform it stays "cpu" so the
# dry-run and local installs behave exactly as before.
os.environ.setdefault("PROTEOFORM_ZEROGPU", "1" if _HAS_SPACES else "0")

# ---------------------------------------------------------------------------
# 2) Import the application package. On a Space the repo root holds this app.py
#    plus the proteoform_analyzer/ package folder; adding the repo root to
#    sys.path lets ``import proteoform_analyzer`` resolve with its relative
#    imports intact.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from proteoform_analyzer.gui import build_app  # noqa: E402


# ---------------------------------------------------------------------------
# 3) Wrap the ESM2 in-process torch entry points with @spaces.GPU so ZeroGPU
#    allocates a GPU for their duration. This is done by monkey-patching the
#    step modules *after* import; it is a no-op decorator off-platform.
# ---------------------------------------------------------------------------
def _install_gpu_wrappers() -> None:
    try:
        from proteoform_analyzer.core.steps import esm2 as _esm2

        if hasattr(_esm2, "_embed_sequences") and not getattr(
            _esm2._embed_sequences, "_zerogpu_wrapped", False
        ):
            _wrapped = spaces.GPU(duration=120)(_esm2._embed_sequences)
            _wrapped._zerogpu_wrapped = True  # type: ignore[attr-defined]
            _esm2._embed_sequences = _wrapped
    except Exception:
        pass

    try:
        from proteoform_analyzer.core.steps import ddg as _ddg

        if hasattr(_ddg, "_esm2_zeroshot_ddg") and not getattr(
            _ddg._esm2_zeroshot_ddg, "_zerogpu_wrapped", False
        ):
            _wrapped = spaces.GPU(duration=120)(_ddg._esm2_zeroshot_ddg)
            _wrapped._zerogpu_wrapped = True  # type: ignore[attr-defined]
            _ddg._esm2_zeroshot_ddg = _wrapped
    except Exception:
        pass


_install_gpu_wrappers()

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RFANTIBODY_ROOT = ROOT / "_vendored" / "RFantibody"
WEIGHTS_DIR = RFANTIBODY_ROOT / "weights"
DOWNLOAD_SCRIPT = RFANTIBODY_ROOT / "scripts" / "download_weights.sh"  # adjust path

def ensure_rfantibody_weights() -> None:
    if WEIGHTS_DIR.exists() and any(WEIGHTS_DIR.iterdir()):
        return

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Example: bash download_weights.sh --output <dir>
    cmd = [
        "bash",
        str(DOWNLOAD_SCRIPT),
        "--output",
        str(WEIGHTS_DIR),
    ]

    subprocess.run(cmd, check=True, cwd=str(RFANTIBODY_ROOT))

# Call early, before loading any RFantibody model
ensure_rfantibody_weights()


# Build the Gradio app at import time so ``gradio``'s auto-reload and the Spaces
# runtime can both find a module-level ``demo``/``app`` object.
demo = build_app()
demo.queue()  # required for long-running pipeline + ZeroGPU scheduling
app = demo  # alias some tooling looks for


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        ssr_mode=False,  # ZeroGPU / Spaces requirement
        show_error=True,
    )
