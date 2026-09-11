"""Shared Boltz backend resolver + hosted-API adapters.

Boltz-2 folding, Boltz-2 docking (protein+ligand co-fold), and BoltzGen binder
design all need a GPU. As of v3.2.0 there are four possible ways to get one, and
this module centralizes *which* one a given job uses so all three call sites
behave identically.

Backends, in resolution order (see :func:`resolve_backend`):

1. ``"api"`` — official Boltz hosted API (``api.boltz.bio``), used when an
API key is configured/in-env AND the ``boltz_api`` client
imports. Covers fold, dock, and design.
2. ``"local"`` — a local ``boltz`` binary (fold/dock), or an importable/PATH
``boltzgen`` (design).
3. ``"graft"`` — **folding only**: ptm-psi backbone-identical grafting (last
resort; TM=1.0, no structural signal). Dock/design have no
graft equivalent and resolve to ``"none"`` instead.
4. ``"none"`` — nothing available; caller skips the step cleanly.

The ``boltz_api`` and ``boltzgen`` packages are **optional** :
they are never imported at module load, and their absence simply removes that
tier from consideration rather than raising.
"""
from __future__ import annotations

import os
import json
import shutil
import logging
import importlib

log = logging.getLogger("proteoform_analyzer.boltz_backend")

# ---------------------------------------------------------------------------
# Availability probes (all side-effect free, never raise)
# ---------------------------------------------------------------------------

def _boltz_api_importable() -> bool:
    """True if the official ``boltz_api`` client package is importable."""
    try:
        importlib.import_module("boltz_api")
        return True
    except Exception:
        return False

def api_available(config) -> bool:
    """True if a hosted-API key is configured/in-env AND the client imports."""
    key = config.boltz2.resolved_api_key()
    return bool(key) and _boltz_api_importable()

def local_fold_available(config) -> bool:
    """True if a local ``boltz`` binary can be resolved for fold/dock."""
    from .boltz2_fold import _resolve_local_binary
    return _resolve_local_binary(config) is not None

def local_boltzgen_available() -> bool:
    """True if ``boltzgen`` is importable or on PATH (local binder design)."""
    try:
        importlib.import_module("boltzgen")
        return True
    except Exception:
        pass
    return shutil.which("boltzgen") is not None

def local_design_available(config) -> bool:
    """True if a local binder-design engine (BoltzGen) is available in this
    environment."""
    return local_boltzgen_available()

# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

_VALID_KINDS = ("fold", "dock", "design")

def resolve_backend(config, kind: str) -> str:
    """Return the backend id to use for ``kind`` ∈ {fold, dock, design}.

    Order: api → local → (graft for fold only) → none.
    ``prefer_local`` promotes the local tier above the API tier.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")

    prefer_local = bool(getattr(config.boltz2, "prefer_local", False))

    def _local_ok() -> bool:
        return local_fold_available(config) if kind in ("fold", "dock") \
               else local_design_available(config)

    tiers = ["api", "local"]
    if prefer_local:
        tiers = ["local", "api"]

    for tier in tiers:
        if tier == "api" and api_available(config):
            return "api"
        if tier == "local" and _local_ok():
            return "local"

    if kind == "fold" and bool(getattr(config.boltz2, "allow_graft_fallback", True)):
        if graft_available(config):
            return "graft"

    return "none"

def graft_available(config) -> bool:
    """True iff the graft fold-fallback could actually run."""
    try:
        from ._graft_seed import graft_available as _ga
        return bool(_ga(config))
    except Exception:
        return False

def backend_summary(config) -> dict:
    """Diagnostic snapshot of what each backend would resolve to."""
    return {
        "api_key_present": bool(config.boltz2.resolved_api_key()),
        "boltz_api_importable": _boltz_api_importable(),
        "local_fold": local_fold_available(config),
        "local_boltzgen": local_boltzgen_available(),
        "fold": resolve_backend(config, "fold"),
        "dock": resolve_backend(config, "dock"),
        "design": resolve_backend(config, "design"),
    }

# ---------------------------------------------------------------------------
# Hosted-API client
# ---------------------------------------------------------------------------

def _make_client(config):
    """Construct a ``boltz_api`` client from config. Raises on failure."""
    import inspect
    boltz_api = importlib.import_module("boltz_api")
    Boltz = getattr(boltz_api, "Boltz")
    key = config.boltz2.resolved_api_key()
    kwargs = {"api_key": key}
    base = getattr(config.boltz2, "api_base_url", None)
    if base:
        try:
            params = inspect.signature(Boltz).parameters
        except (TypeError, ValueError):
            params = {}
        if "base_url" in params:
            kwargs["base_url"] = base
        elif "base" in params:
            kwargs["base"] = base
    try:
        return Boltz(**kwargs)
    except TypeError:
        kwargs.pop("base_url", None)
        kwargs.pop("base", None)
        return Boltz(**kwargs)

def _run_dir_from_result(run_dir):
    """Normalize an SDK run() return value to a local directory path or None."""
    import pathlib
    if isinstance(run_dir, str):
        return run_dir
    if isinstance(run_dir, pathlib.Path):
        return str(run_dir)
    for attr in ("path", "run_dir", "output_dir", "local_dir", "directory"):
        val = getattr(run_dir, attr, None)
        if isinstance(val, str) and val:
            return val
    return None

def _copy_tree_into(src: str, out_dir: str) -> None:
    """Copy the contents of directory ``src`` into ``out_dir``."""
    if not (src and os.path.isdir(src)):
        return
    if os.path.abspath(src) == os.path.abspath(out_dir):
        return
    for name in os.listdir(src):
        s = os.path.join(src, name)
        d = os.path.join(out_dir, name)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy(s, d)

def _yaml_spec_to_entities(spec: dict) -> list[dict]:
    """Convert an internal Boltz YAML spec to the hosted-API ``entities`` list."""
    entities = []
    for chain in spec.get("sequences", []):
        if "protein" in chain:
            p = chain["protein"]
            entities.append({
                "type": "protein",
                "value": p["sequence"],
                "chain_ids": [p["id"]],
            })
        elif "ligand" in chain:
            lig = chain["ligand"]
            if "smiles" in lig:
                entities.append({
                    "type": "ligand_smiles",
                    "value": lig["smiles"],
                    "chain_ids": [lig.get("id", "L")]
                })
            elif "ccd" in lig:
                entities.append({
                    "type": "ligand_ccd",
                    "value": lig["ccd"],
                    "chain_ids": [lig.get("id", "L")]
                })
    return entities

def _find_cif(out_dir: str) -> str | None:
    """Find a predicted mmCIF in a downloaded API run directory."""
    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.endswith("_model_0.cif") or f.endswith(".cif"):
                return os.path.join(root, f)
    return None

def _find_cached_api_cif(job_name: str) -> str | None:
    """Find a CIF in boltz-api's persistent local run cache.

    boltz-api stores runs under the project working directory as:
    boltz-experiments/<job_name>/outputs/files/prediction/*.cif.
    """
    from pathlib import Path

    # This module is: <project>/proteoform_analyzer/core/steps/_boltz_backend.py
    project_dir = Path(__file__).resolve().parents[3]

    candidates = [
        project_dir / "boltz-experiments" / job_name,
        Path.cwd() / "boltz-experiments" / job_name,
    ]

    for job_dir in candidates:
        if not job_dir.is_dir():
            continue

        cif_files = sorted(
            job_dir.glob("outputs/files/prediction/*.cif"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if cif_files:
            return str(cif_files[0])

    return None

# ---------------------------------------------------------------------------
# Stale run-directory handling
# ---------------------------------------------------------------------------

_STALE_RUN_MARKER = "belongs to a different request"


def is_stale_run_dir_error(exc: BaseException) -> bool:
    """True if the SDK refused ``name`` because a cached run dir exists."""
    return _STALE_RUN_MARKER in str(exc).lower()


def _quarantine_stale_run_dir(exc: BaseException) -> str | None:
    """Rename the stale SDK run directory mentioned in the error message.

    The boltz-api SDK caches runs under ``boltz-experiments/<name>/`` keyed by
    request content. When the same job name is submitted with a different
    request (e.g. a mature sequence replacing the precursor after a pipeline
    upgrade), the SDK raises ``Run directory <path> belongs to a different
    request``. The old directory may still hold valuable predictions, so it is
    renamed to ``<name>.superseded[N]`` rather than deleted. Returns the new
    path, or None if the directory could not be located/renamed.
    """
    import re
    from pathlib import Path

    m = re.search(r"Run directory\s+(.+?)\s+belongs to a different request",
                  str(exc), re.IGNORECASE)
    candidates = []
    if m:
        candidates.append(Path(m.group(1).strip()))
    else:
        # Fallback: reconstruct from the job name if the message format
        # changes but still contains the marker.
        name_m = re.search(r"for\s+([\w\-.]+)\s*:", str(exc))
        job = name_m.group(1) if name_m else None
        if job:
            project_dir = Path(__file__).resolve().parents[3]
            candidates.append(project_dir / "boltz-experiments" / job)
            candidates.append(Path.cwd() / "boltz-experiments" / job)

    for stale in candidates:
        if not stale.is_dir():
            continue
        for i in range(100):
            suffix = ".superseded" if i == 0 else f".superseded{i}"
            dest = stale.with_name(stale.name + suffix)
            if not dest.exists():
                try:
                    stale.rename(dest)
                except OSError as e:
                    log.warning("could not quarantine stale run dir %s: %s",
                                stale, e)
                    return None
                log.warning(
                    "Boltz SDK run directory %s was created by a different "
                    "request (e.g. a pre-maturation sequence). Renamed it to "
                    "%s and resubmitting; the old predictions are preserved.",
                    stale, dest,
                )
                return str(dest)
    return None


# ---------------------------------------------------------------------------
# Auth-error detection
# ---------------------------------------------------------------------------

_AUTH_MARKERS = (
    "401", "unauthorized", "invalid or missing api key",
    "invalid api key", "api key", "authentication", "not authorized"
)

def is_auth_error(exc: BaseException) -> bool:
    """True if an exception looks like a Boltz API auth failure."""
    text = str(exc).lower()
    return any(m in text for m in _AUTH_MARKERS)

# ---------------------------------------------------------------------------
# Folding via API
# ---------------------------------------------------------------------------

def run_api_fold(config, spec: dict, out_dir: str, name: str = "fold"):
    """Fold one complex via the hosted API and return its local mmCIF path.

    The boltz-api SDK may raise after reporting a successful cached prediction
    when an archive URL is absent. In that case, reuse the already materialized
    CIF in the cached job directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    client = _make_client(config)
    entities = _yaml_spec_to_entities(spec)
    prediction_input = {"entities": entities}

    try:
        run_dir = client.predictions.structure_and_binding.run(
            model=config.boltz2.api_model,
            input=prediction_input,
            name=name,
        )

        source_dir = _run_dir_from_result(run_dir)
        if source_dir:
            _copy_tree_into(source_dir, out_dir)

    except Exception as exc:
        if is_stale_run_dir_error(exc) and _quarantine_stale_run_dir(exc):
            # Retry once now that the stale cache directory is out of the way.
            run_dir = client.predictions.structure_and_binding.run(
                model=config.boltz2.api_model,
                input=prediction_input,
                name=name,
            )
            source_dir = _run_dir_from_result(run_dir)
            if source_dir:
                _copy_tree_into(source_dir, out_dir)
        elif "did not return an archive url" not in str(exc).lower():
            raise

        log.warning(
            "Boltz SDK could not fetch archive for completed '%s'; "
            "checking the local cached run directory.",
            name,
        )

    # Normal case: the SDK returned a run directory and its contents were
    # copied into the pipeline's run-specific output directory.
    cif = _find_cif(out_dir)
    if cif:
        return cif

    # Cached-job recovery: SDK 0.50.0 may throw because the API response lacks
    # archive_url even though the predicted CIF already exists on disk.
    cif = _find_cached_api_cif(name)
    if cif:
        log.info("Found cached Boltz CIF for '%s': %s", name, cif)
        return cif

    raise RuntimeError(
        f"hosted API returned no mmCIF structure for '{name}'; "
        f"checked {out_dir} and the boltz-api local cache"
    )

def run_api_dock(config, spec: dict, out_dir: str, name: str = "dock"):
    """Co-fold a protein+ligand complex via the hosted API."""
    os.makedirs(out_dir, exist_ok=True)
    client = _make_client(config)
    entities = _yaml_spec_to_entities(spec)
    binder = None
    for e in entities:
        if e["type"].startswith("ligand"):
            binder = e["chain_ids"][0]
            break
    prediction_input = {"entities": entities}
    if binder:
        prediction_input["binding"] = {
            "type": "ligand_protein_binding",
            "binder_chain_id": binder
        }
    try:
        run_dir = client.predictions.structure_and_binding.run(
            model=config.boltz2.api_model,
            input=prediction_input,
            name=name
        )
    except Exception as exc:
        if not (is_stale_run_dir_error(exc) and _quarantine_stale_run_dir(exc)):
            raise
        # Retry once now that the stale cache directory is out of the way.
        run_dir = client.predictions.structure_and_binding.run(
            model=config.boltz2.api_model,
            input=prediction_input,
            name=name
        )
    _copy_tree_into(_run_dir_from_result(run_dir), out_dir)
    cif = _find_cif(out_dir)
    from .docking import _parse_boltz_confidence
    metrics = _parse_boltz_confidence(out_dir)
    return cif, metrics

def _design_resource(client):
    """Return the Protein Design resource of the SDK, or None."""
    protein = getattr(client, "protein", None)
    if protein is not None and getattr(protein, "design", None) is not None:
        return protein.design
    for attr in ("protein_design",):
        r = getattr(client, attr, None)
        if r is not None:
            return r
    return None

_DESIGN_TERMINAL_OK = {
    "completed", "succeeded", "success", "finished", "done"
}
_DESIGN_TERMINAL_BAD = {
    "failed", "error", "errored", "cancelled", "canceled", "stopped"
}

def _poll_design(design, run_id, max_wait_s: int = 3600, interval_s: int = 15):
    """Poll ``design.retrieve(run_id)`` until a terminal status."""
    import time
    waited = 0
    while True:
        rec = design.retrieve(run_id)
        status = (
            getattr(rec, "status", None)
            or (rec.get("status") if isinstance(rec, dict) else None)
            or ""
        )
        status = str(status).lower()
        if status in _DESIGN_TERMINAL_OK:
            return rec
        if status in _DESIGN_TERMINAL_BAD:
            raise RuntimeError(
                f"hosted API protein-design run {run_id} ended with status '{status}'"
            )
        if waited >= max_wait_s:
            raise TimeoutError(
                f"hosted API protein-design run {run_id} did not finish "
                f"within {max_wait_s}s (last status '{status or 'unknown'}')"
            )
        time.sleep(interval_s)
        waited += interval_s

def _write_design_results(results, out_dir: str) -> list:
    """Persist protein-design result structures into ``out_dir``."""
    outputs = []
    try:
        items = list(results)
    except TypeError:
        items = [results]
    for i, item in enumerate(items):
        def _g(obj, key):
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)
        for key, ext in (("cif", "cif"), ("structure", "cif"), ("pdb", "pdb")):
            text = _g(item, key)
            if isinstance(text, str) and text.strip() and (
                "ATOM" in text or "_atom_site" in text or "data_" in text
            ):
                dp = os.path.join(out_dir, f"design_{i:03d}.{ext}")
                with open(dp, "w") as fh:
                    fh.write(text)
                outputs.append(dp)
                break
        for key in ("path", "file", "local_path", "output_path"):
            p = _g(item, key)
            if isinstance(p, str) and os.path.isfile(p):
                dp = os.path.join(out_dir, os.path.basename(p))
                if os.path.abspath(p) != os.path.abspath(dp):
                    shutil.copy(p, dp)
                outputs.append(dp)
    return outputs

def run_api_design(config, target_pdb: str, out_dir: str, num_designs: int = 20):
    """Design protein binders for a target PDB via the hosted Protein Design API."""
    os.makedirs(out_dir, exist_ok=True)
    client = _make_client(config)
    design = _design_resource(client)
    if design is None:
        raise RuntimeError(
            "installed boltz_api client exposes no Protein Design endpoint "
            "(expected client.protein.design). Upgrade the 'boltz-api' package "
            "to a version that supports protein binder design."
        )

    with open(target_pdb) as fh:
        target_text = fh.read()

    design_input = {
        "target": {"structure": target_text, "format": "pdb"},
        "num_designs": num_designs,
        "num_samples": num_designs,
    }

    submitted = design.start(
        model=config.boltz2.api_model,
        input=design_input
    )
    run_id = (
        getattr(submitted, "id", None)
        or (submitted.get("id") if isinstance(submitted, dict) else None)
    )
    if not run_id:
        raise RuntimeError("hosted API protein-design start returned no run id")

    _poll_design(design, run_id)
    results = design.list_results(run_id)
    outputs = _write_design_results(results, out_dir)
    return outputs

# ---------------------------------------------------------------------------
# Structure provenance
# ---------------------------------------------------------------------------

def write_structure_provenance(
    paths: dict,
    method: str,
    backbone_identical: bool,
    detail: str = ""
) -> str:
    """Record how structures were produced so the GUI can warn appropriately."""
    rd = paths.get("results", "")
    if not rd:
        return ""
    p = os.path.join(rd, "structure_provenance.json")
    try:
        with open(p, "w") as f:
            json.dump({
                "method": method,
                "backbone_identical": bool(backbone_identical),
                "detail": detail
            }, f, indent=2)
    except Exception as e:
        log.warning("could not write structure provenance: %s", e)
    return p

GRAFT_WARNING = (
    "Structures were built by ptm-psi side-chain grafting onto the wild-type "
    "backbone because no folding backend (Boltz API key or a local boltz "
    "install) was available. The backbone is IDENTICAL to WT for every variant, "
    "so TM-scores are 1.0 by construction and there is no structural, pocket, or "
    "docking signal. Provide a Boltz API key (BOLTZ_API_KEY) or a local boltz "
    "install for real folding."
)