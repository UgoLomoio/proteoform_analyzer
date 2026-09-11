"""Step: ligand design.

Two engines (config.ligand_design_engine):
  - "diffsbdd" (default, local): DiffSBDD diffusion model generates small-molecule
    ligands for a protein pocket.  Uses the vendored DiffSBDD code + checkpoint.
  - "boltzgen" (optional): BoltzGen universal binder design. Different modality
    (designs protein/peptide binders); user-selectable. Runs via the Boltz API
    (boltz2.api_key / $BOLTZ_API_KEY) or a local ``boltzgen`` CLI install; the
    Biomni HPC backend was removed in v3.4.0.
"""
from __future__ import annotations

import os
import sys
import shutil
import logging
import numpy as np
import pandas as pd

from ..pipeline import StepResult

log = logging.getLogger("proteoform_analyzer.ligand_design")


def _vendored_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "_vendored")


# Heavy third-party deps required by the vendored DiffSBDD code
# (_vendored/diffsbdd/lightning_modules.py + generate_ligands.py, module-level).
# Import names mapped to their pip package names for a clear install hint.
_DIFFSBDD_DEPS = {
    "torch": "torch",
    "pytorch_lightning": "pytorch-lightning",
    "torch_scatter": "torch-scatter",
    "wandb": "wandb",
    "openbabel": "openbabel-wheel",  # provides `from openbabel import openbabel`
}


def _diffsbdd_dep_status():
    """Probe DiffSBDD heavy deps by *actually importing* them.

    ``importlib.util.find_spec`` only checks that a package is on disk; it does
    NOT catch packages that are installed but fail at import time. ``torch_scatter``
    is the classic offender: it runs ``torch.ops.load_library(...)`` at import,
    which raises ``OSError``/``RuntimeError`` when the compiled extension was
    built against a different torch/CUDA ABI than the installed torch. Such a
    package passes ``find_spec`` but blows up deep inside DiffSBDD's shell-out
    with a cryptic ``torch_scatter/__init__.py line 16`` traceback (exactly the
    error users hit locally).

    We therefore import each dep and classify it as:
      - "missing": not installed at all (ModuleNotFoundError / no spec)
      - "broken":  installed but import raises (ABI/version mismatch, etc.)

    Returns (missing, broken) where each is a list of
    ``(pip_name, detail)`` tuples.
    """
    import importlib
    import importlib.util
    missing, broken = [], []
    for import_name, pip_name in _DIFFSBDD_DEPS.items():
        # First: is it even on disk?
        try:
            spec = importlib.util.find_spec(import_name)
        except ModuleNotFoundError:
            spec = None
        except Exception as e:
            # A broken parent package can make find_spec itself raise.
            broken.append((pip_name, f"{type(e).__name__}: {e}"))
            continue
        if spec is None:
            missing.append((pip_name, "not installed"))
            continue
        # Second: does it actually import? (catches load_library/ABI failures)
        try:
            importlib.import_module(import_name)
        except ModuleNotFoundError as e:
            missing.append((pip_name, f"{type(e).__name__}: {e}"))
        except BaseException as e:
            # Intentionally broad: torch_scatter can raise OSError/RuntimeError
            # (and load_library failures can surface as low-level errors) at
            # import. Any failure here means the dep is unusable for DiffSBDD.
            detail = str(e).splitlines()[0] if str(e) else type(e).__name__
            broken.append((pip_name, f"{type(e).__name__}: {detail}"))
    return missing, broken


def _missing_diffsbdd_deps() -> list[str]:
    """Backwards-compatible: pip names of deps that are missing OR broken."""
    missing, broken = _diffsbdd_dep_status()
    return [p for p, _ in missing] + [p for p, _ in broken]


def _run_diffsbdd(config, paths: dict) -> StepResult:
    """Generate ligands with DiffSBDD for one variant pocket."""
    diffsbdd_dir = os.path.join(_vendored_path(), "diffsbdd")
    if diffsbdd_dir not in sys.path:
        sys.path.insert(0, diffsbdd_dir)

    ckpt = config.diffsbdd_checkpoint or os.path.join(diffsbdd_dir, "checkpoints",
                                                      "crossdocked_ca_cond.ckpt")
    if not os.path.exists(ckpt):
        return StepResult("ligand_design", "skipped", f"DiffSBDD checkpoint not found: {ckpt}")

    # Preflight: DiffSBDD's vendored code imports a heavy stack at module load
    # (torch, pytorch_lightning, torch_scatter, wandb, openbabel). We probe each
    # by actually importing it, so a package that is installed-but-broken (e.g.
    # torch_scatter failing torch.ops.load_library due to an ABI/version
    # mismatch) is caught here and turned into a clean skip with an actionable
    # message, instead of crashing deep inside the shell-out.
    missing, broken = _diffsbdd_dep_status()
    if missing or broken:
        parts = ["DiffSBDD ligand design was skipped because its extra "
                 "dependencies are not usable in this environment."]
        if missing:
            miss_names = ", ".join(p for p, _ in missing)
            parts.append(
                f"Not installed: {miss_names}. Install with "
                f"`pip install {miss_names}` or the optional extra "
                "`pip install proteoform-analyzer[design]`.")
        if broken:
            broke_desc = "; ".join(f"{p} ({d})" for p, d in broken)
            broke_names = " ".join(p for p, _ in broken)
            parts.append(
                f"Installed but failed to import: {broke_desc}. This is an "
                "environment/ABI mismatch, not a bug in this tool — most often "
                "torch-scatter compiled against a different torch/CUDA build. "
                f"Reinstall it to match your torch, e.g. `pip uninstall -y "
                f"{broke_names}` then install the wheel matching your exact "
                "torch version and CUDA/CPU build "
                "(see https://github.com/rusty1s/pytorch_scatter#installation).")
        parts.append("Alternatively switch ligand_design_engine to 'boltzgen' "
                     "(Boltz API or a local 'boltzgen' install).")
        log.warning("ligand_design: %s", " ".join(parts))
        return StepResult("ligand_design", "skipped", " ".join(parts))

    out_dir = paths["ligand_design"]
    os.makedirs(out_dir, exist_ok=True)

    # pick a mutant PDB to design for (first non-WT).
    # Structures resolve through boltz-experiments first (see _structure_source).
    from ._structure_source import resolve_structure_pdb
    pdb_dir = paths["pdbs_monomer"] if config.is_monomer else paths["pdbs"]
    target = None
    target_name = None
    for f in sorted(os.listdir(pdb_dir)):
        if f.endswith(".pdb") and not f.lower().startswith("wt"):
            target_name = f.replace(".pdb", "")
            target = (resolve_structure_pdb(paths, target_name, "ligand_design")
                      or os.path.join(pdb_dir, f))
            break
    if target is None:
        # use WT
        for f in sorted(os.listdir(pdb_dir)):
            if f.endswith(".pdb"):
                target_name = f.replace(".pdb", "")
                target = (resolve_structure_pdb(paths, target_name, "ligand_design")
                          or os.path.join(pdb_dir, f))
                break
    if target is None:
        return StepResult("ligand_design", "skipped", "No target PDB found")

    # pocket residues: use config or detect from binding-site center.
    # DiffSBDD expects "chain:resid" format (e.g. "A:96").
    pocket_residues = config.diffsbdd_pocket_residues
    if not pocket_residues:
        # default: residues near the geometric center, from chain A only
        # (avoid duplicates across symmetric chains)
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("t", target)
        coords = []
        res_ids = []
        for model in struct:
            for chain in model:
                if chain.id != "A":  # single chain to avoid duplicates
                    continue
                for res in chain:
                    if "CA" in res:
                        coords.append(res["CA"].get_coord())
                        res_ids.append(f"{chain.id}:{res.id[1]}")
        coords = np.array(coords)
        center = coords.mean(axis=0)
        dists = np.linalg.norm(coords - center, axis=1)
        # take 8 closest residues
        idx = np.argsort(dists)[:8]
        pocket_residues = [res_ids[i] for i in idx]
        log.info("Auto-detected pocket residues: %s", pocket_residues)

    # DiffSBDD generate_ligands.py expects: pdbfile, resi_list, checkpoint, outfile
    try:
        import torch
        from openbabel import openbabel
        openbabel.obErrorLog.StopLogging()
        from lightning_modules import LigandPocketDDPM
        from utils import get_resi_list, get_pocket
        from generate_ligands import sample  # may not exist; we replicate the call
    except Exception as e:

        log.info("DiffSBDD import path failed (%s); falling back to generate_ligands.py", e)
        import subprocess
        from pathlib import Path

        diffsbdd_root = Path(diffsbdd_dir).resolve()
        gen_script = diffsbdd_root / "generate_ligands.py"
        ckpt_path = Path(ckpt).resolve()
        target_path = Path(target).resolve()
        output_dir = Path(out_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        out_sdf_path = output_dir / f"{target_name}_generated.sdf"

        cmd = [
            sys.executable,
            str(gen_script),
            str(ckpt_path),
            "--pdbfile", str(target_path),
            "--resi_list", *[str(r) for r in pocket_residues],
            "--outfile", str(out_sdf_path),
            "--n_samples", str(config.diffsbdd_n_samples),
        ]

        log.info("DiffSBDD command: %s", " ".join(cmd))
        log.info(
            "DiffSBDD paths | cwd=%s | script=%s | ckpt=%s | pdb=%s | out=%s",
            diffsbdd_root, gen_script, ckpt_path, target_path, out_sdf_path
        )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(diffsbdd_root),
        )

        stderr = result.stderr or ""
        stdout = result.stdout or ""

        if result.returncode != 0 or not out_sdf_path.exists():
            err_lines = [
                line for line in stderr.splitlines()
                if "BiopythonWarning" not in line and "warnings.warn" not in line
            ]
            err_msg = "\n".join(err_lines[-10:]).strip()
            if not err_msg:
                err_msg = stderr[-500:].strip() or stdout[-500:].strip() or "Unknown DiffSBDD error"

            log.error(
                "DiffSBDD failed | returncode=%s | stderr=%s",
                result.returncode,
                err_msg,
            )

            return StepResult(
                "ligand_design",
                "skipped",
                f"DiffSBDD script failed: {err_msg[:300]}",
            )

        outputs = []
        for sdf_file in sorted(output_dir.glob("*.sdf")):
            outputs.append(str(sdf_file))

        if str(out_sdf_path) not in outputs:
            outputs.insert(0, str(out_sdf_path))

        # Build viewable target+ligand complex PDBs (for the 3D Structure
        # Viewer) and compute druggability metrics (QED/SA/logP/Lipinski/MW)
        # shown in the viewer's structure-summary card.
        n_complexes = 0
        try:
            from ..ligand_metrics import build_complexes_and_metrics
            complexes, mdf = build_complexes_and_metrics(
                str(target_path), str(out_sdf_path), str(output_dir),
                target_name=target_name)
            outputs.extend(complexes)
            n_complexes = len(complexes)
            if len(mdf):
                metrics_csv = os.path.join(str(output_dir), "ligand_metrics.csv")
                mdf.to_csv(metrics_csv, index=False)
                outputs.append(metrics_csv)
        except Exception as e:
            log.warning("ligand_design: complex building / ligand metrics "
                        "failed (SDF outputs unaffected): %s", e)

        return StepResult(
            "ligand_design",
            "ok",
            f"DiffSBDD generated {len([o for o in outputs if o.endswith('.sdf')])} "
            f"ligand file(s) for {target_name} (pocket: {pocket_residues}); "
            f"{n_complexes} target+ligand complex(es) written for the 3D viewer",
            outputs=outputs,
        )

    except Exception as e2:
        return StepResult(
            "ligand_design",
            "skipped",
            f"DiffSBDD script error: {e2}",
        )

    # If imports succeeded, run generation directly (replicate generate_ligands logic)
    try:
        model = LigandPocketDDPM.load_from_checkpoint(ckpt, map_location="cpu")
        model.eval()
        # ... (full generation logic would go here; the script path above is the
        #      reliable route since DiffSBDD's generate_ligands.py is the canonical entry)
        return StepResult("ligand_design", "skipped",
                          "DiffSBDD direct-API path not fully wired; use script path")
    except Exception as e:
        return StepResult("ligand_design", "skipped", f"DiffSBDD generation failed: {e}")


def _find_wt_target(paths: dict):
    """Locate the WT PDB used as the binder-design target.

    Structures resolve through boltz-experiments first (see _structure_source).
    """
    from ._structure_source import resolve_structure_pdb
    pdb_dir = paths["pdbs"]
    if not os.path.isdir(pdb_dir):
        return None
    for f in sorted(os.listdir(pdb_dir)):
        if f.endswith(".pdb") and f.lower().startswith("wt"):
            stem = f.replace(".pdb", "")
            return (resolve_structure_pdb(paths, stem, "ligand_design")
                    or os.path.join(pdb_dir, f))
    return None


def _run_local_boltzgen(config, paths, out_dir, target) -> StepResult:
    """Local BoltzGen binder design via the ``boltzgen`` CLI."""
    import yaml
    import shutil as _shutil
    import subprocess as _subprocess

    spec = {
        "version": 1,
        "targets": [{"pdb": target}],
        "protocol": "protein-anything",
    }
    spec_path = os.path.join(out_dir, "boltzgen_design_spec.yaml")
    with open(spec_path, "w") as f:
        yaml.dump(spec, f)

    boltzgen_bin = _shutil.which("boltzgen")
    if boltzgen_bin is None:
        return StepResult("ligand_design", "skipped",
                          "Local boltzgen resolved but 'boltzgen' not on PATH")
    local_out = os.path.join(out_dir, "local")
    os.makedirs(local_out, exist_ok=True)
    cmd = [boltzgen_bin, "run", spec_path, "--output", local_out,
           "--protocol", "protein-anything",
           "--num_designs", str(50),
           "--budget", str(5)]
    try:
        _subprocess.run(cmd, timeout=7200,
                        stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL)
        produced = [os.path.join(local_out, f) for f in os.listdir(local_out)
                    if f.endswith((".pdb", ".cif"))] if os.path.isdir(local_out) else []
        if produced:
            return StepResult("ligand_design", "ok",
                              f"Local BoltzGen produced {len(produced)} design(s).",
                              outputs=[spec_path] + produced)
        return StepResult("ligand_design", "skipped",
                          "Local BoltzGen produced no designs")
    except Exception as e:
        return StepResult("ligand_design", "skipped", f"Local BoltzGen run failed: {e}")


def _run_boltzgen(config, paths: dict) -> StepResult:
    """Protein binder design (BoltzGen).

    Backend chosen by the shared resolver: hosted Boltz API (Protein Design)
    -> local engine (``boltzgen``) -> none. Binder design has **no graft
    fallback** (there is nothing to graft), so backend=="none" skips cleanly
    with an actionable message. The Biomni-HPC backend was removed in v3.4.0.
    """
    from . import _boltz_backend as bb
    backend = bb.resolve_backend(config, "design")
    log.info("Boltz binder-design backend resolved to: %s", backend)
    if backend == "none":
        return StepResult(
            "ligand_design", "skipped",
            "Protein binder design has no available backend. Provide a Boltz API "
            "key (boltz2.api_key or $BOLTZ_API_KEY) or install 'boltzgen' locally. "
            "(Binder design has no graft fallback.)")

    out_dir = paths["ligand_design"]
    os.makedirs(out_dir, exist_ok=True)

    target = _find_wt_target(paths)
    if target is None:
        return StepResult("ligand_design", "skipped", "No WT PDB for binder-design target")

    # --- Hosted API path: Boltz Protein Design on the WT target -----------
    if backend == "api":
        try:
            api_out = os.path.join(out_dir, "api")
            files = bb.run_api_design(config, target, api_out,
                                      num_designs=50)
            n = len(files)
            if n:
                return StepResult(
                    "ligand_design", "ok",
                    f"Boltz API binder design produced {n} design(s) for the WT target.",
                    outputs=[target] + list(files))
            return StepResult("ligand_design", "skipped",
                              "Boltz API binder design returned no designs")
        except Exception as e:
            # Auth failure: warn once and fall back to local BoltzGen if available.
            if bb.is_auth_error(e):
                log.warning("Boltz API key invalid or unauthorized for binder design; "
                            "falling back to local BoltzGen.")
                if bb.local_boltzgen_available():
                    return _run_local_boltzgen(config, paths, out_dir, target)
                return StepResult("ligand_design", "skipped",
                                  "Boltz API binder design unavailable (invalid or "
                                  "missing API key) and no local 'boltzgen' install. "
                                  "Provide a valid Boltz API key or install 'boltzgen'.")
            return StepResult("ligand_design", "skipped",
                              f"Boltz API binder design failed: {e}")

    # --- Local path: BoltzGen only -----------------------------------------
    if bb.local_boltzgen_available():
        return _run_local_boltzgen(config, paths, out_dir, target)

    return StepResult("ligand_design", "skipped",
                      "Local binder-design engine resolved but 'boltzgen' is not "
                      "runnable in this environment.")


def run_ligand_design(config, paths: dict) -> StepResult:
    """Run ligand design with the configured engine."""
    if config.ligand_design_engine == "boltzgen":
        return _run_boltzgen(config, paths)
    return _run_diffsbdd(config, paths)
